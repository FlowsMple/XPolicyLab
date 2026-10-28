"""CPU contracts; these do not establish real-checkpoint or simulator parity."""

import io

import numpy as np
import pytest
import torch
from torch import nn
from XPolicyLab.policy.OpenWAM.rl import (
    JointReplay,
    OpenWAMNFTPolicy,
    action_nft_loss,
    flatten_replay_for_transport,
    restore_replay_pipeline,
)
from XPolicyLab.policy.OpenWAM import robotwin as robotwin_bridge
from XPolicyLab.policy.OpenWAM.robotwin import (
    eef20_to_steps,
    native_obs_to_packed,
    packed_to_native,
    pose_order,
    state_to_eef20,
)


def replay_data(batch=2):
    return {
        "nft_actions": torch.ones(batch, 3, 4),
        "nft_v": torch.zeros(batch, 3, 4),
        "nft_x0": torch.zeros(batch, 3, 4),
        "nft_sigma": torch.full((batch,), 0.5),
        "nft_valid_dims": torch.tensor([[True, True, False, False]]).expand(batch, -1),
    }


def test_preference_gradient_and_executed_dimensions():
    data = replay_data()
    velocity = torch.zeros_like(data["nft_v"], requires_grad=True)
    loss, _ = action_nft_loss(
        velocity, data, torch.tensor([5.0, -5.0]), torch.ones(2), clip_ratio=None
    )
    loss.backward()
    # Descent increases positive-sample velocity, moving x0 toward target.
    assert (velocity.grad[0, 0, :2] < 0).all()
    assert (velocity.grad[1, 0, :2] > 0).all()
    assert torch.count_nonzero(velocity.grad[:, 1:]) == 0
    assert torch.count_nonzero(velocity.grad[:, :, 2:]) == 0


def test_padding_and_invalid_samples_do_not_change_update():
    data = replay_data()
    velocity = torch.zeros_like(data["nft_v"], requires_grad=True)
    loss, _ = action_nft_loss(
        velocity,
        data,
        torch.tensor([5.0, -5.0]),
        torch.tensor([1.0, 0.0]),
        clip_ratio=None,
    )
    loss.backward()
    assert torch.count_nonzero(velocity.grad[1]) == 0
    original = loss.detach()
    data["nft_x0"][:, :, 2:] = 10000
    data["nft_x0"][:, 1:] = 10000
    new_loss, _ = action_nft_loss(
        velocity,
        data,
        torch.tensor([5.0, -5.0]),
        torch.tensor([1.0, 0.0]),
        clip_ratio=None,
    )
    torch.testing.assert_close(original, new_loss)


def test_zero_advantage_has_zero_gradient_at_on_policy_point():
    data = replay_data()
    velocity = torch.zeros_like(data["nft_v"], requires_grad=True)
    loss, _ = action_nft_loss(velocity, data, torch.zeros(2), torch.ones(2))
    loss.backward()
    assert torch.count_nonzero(velocity.grad) == 0


@pytest.mark.parametrize("field,value", [("nft_sigma", 0), ("nft_valid_dims", False)])
def test_invalid_replay_rejected(field, value):
    data = replay_data()
    data[field] = torch.full_like(data[field], value)
    with pytest.raises(ValueError):
        action_nft_loss(data["nft_v"], data, torch.ones(2), torch.ones(2))


def captured_replay():
    capture = JointReplay(1)
    actions = torch.ones(2, 3, 4)
    inputs = {
        "latents": torch.ones(2, 3, 4),
        "context": torch.ones(2, 2, 4),
        "prompt": ["a", "b"],
        "tiled": False,
        "tile_size": [4, 4],
    }
    capture.record(
        index=1,
        schedule=None,
        inputs=inputs,
        actions=actions,
        velocity=actions * 2,
        video_timestep=torch.ones(2),
        action_timestep=torch.ones(2),
    )
    capture.finish(actions)
    actions.zero_()
    inputs["latents"].zero_()
    return capture.data


def test_replay_is_immutable_and_serializable():
    data = captured_replay()
    assert data["nft_actions"].eq(1).all()
    assert data["nft_pipeline"]["latents"].eq(1).all()
    assert data["nft_x0"].eq(1).all()
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    torch.testing.assert_close(torch.load(buffer, weights_only=True), data)
    with pytest.raises(ValueError, match="not captured"):
        JointReplay(2).finish(torch.ones(1))


def test_flattened_replay_batches_on_the_trajectory_batch_axis():
    from rlinf.data.schema.embodied_types import Trajectory

    data = captured_replay()
    flat = flatten_replay_for_transport(data)
    assert "nft_pipeline" in data
    assert all(torch.is_tensor(value) for value in flat.values())
    stacked = {key: value.unsqueeze(0) for key, value in flat.items()}
    halves = [
        Trajectory(forward_inputs={key: value[:, index : index + 1] for key, value in stacked.items()})
        for index in range(2)
    ]
    with pytest.raises(TypeError):
        Trajectory.to_batch(
            [
                Trajectory(
                    forward_inputs={
                        key: (value.unsqueeze(0) if torch.is_tensor(value) else value)
                        for key, value in data.items()
                    }
                )
            ]
        )
    batched = Trajectory.to_batch(halves)["forward_inputs"]
    assert batched["nft_pipe.latents"].shape[:2] == (1, 2)
    merged = {key: value.reshape(-1, *value.shape[2:]) for key, value in batched.items()}
    restore_replay_pipeline(merged)
    policy = OpenWAMNFTPolicy(TinyJointArchitecture(), 10, [1, 1, 0, 0])
    output = policy(merged)["v_theta"]
    assert output.shape[0] == 2
    assert "nft_pipeline" in merged
    direct = flatten_replay_for_transport(captured_replay())
    torch.testing.assert_close(policy(direct)["v_theta"], policy(captured_replay())["v_theta"])


class TinyJointArchitecture(nn.Module):
    """Small coupled model: gradients must pass through frozen video weights."""

    def __init__(self):
        super().__init__()
        self.action_backbone = nn.Linear(4, 4)
        self.video = nn.Linear(4, 4)

    @property
    def backbones(self):
        return {"action": self.action_backbone, "video": self.video}

    def forward(self, actions, action_t, **kwargs):
        return None, self.video(self.action_backbone(actions))


def test_replay_gradient_reaches_only_action_weights_and_roundtrip():
    architecture = TinyJointArchitecture()
    policy = OpenWAMNFTPolicy(architecture, 10, [1, 1, 0, 0]).train()
    data = captured_replay()
    output = policy(data)["v_theta"]
    output.square().mean().backward()
    assert not architecture.training
    assert isinstance(architecture.action_backbone.weight, nn.Parameter)
    assert not isinstance(architecture.video.weight, nn.Parameter)
    assert architecture.action_backbone.weight.grad.abs().sum() > 0
    assert architecture.video.weight.grad is None
    restored = OpenWAMNFTPolicy(TinyJointArchitecture(), 10, [1, 1, 0, 0])
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored(data)["v_theta"], output)


def test_fsdp_units_follow_real_blocks_not_the_whole_policy():
    class DiTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

    class Architecture(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_backbone = nn.Linear(2, 2)
            self.block = DiTBlock()

        @property
        def backbones(self):
            return {}

    policy = OpenWAMNFTPolicy(Architecture(), 1, [1, 0])
    assert policy._no_split_modules == ["DiTBlock"]
    tiny = OpenWAMNFTPolicy(TinyJointArchitecture(), 1, [1, 0])
    assert tiny._no_split_modules == ["OpenWAMNFTPolicy"]


def test_mixed_metadata_rejected():
    data = captured_replay()
    data["nft_metadata"][1, 0] = 32
    policy = OpenWAMNFTPolicy(TinyJointArchitecture(), 10, [1, 1, 0, 0])
    with pytest.raises(ValueError, match="Mixed replay"):
        policy(data)


@pytest.mark.parametrize("native_order", ["wxyz", "xyzw"])
def test_rotated_pose_roundtrip_and_native_boundary(native_order):
    q = np.array([0.7, 0.2, -0.3, 0.4], dtype=np.float32)
    q /= np.linalg.norm(q)
    pose = np.r_[1.0, 2.0, 3.0, q]
    state = {
        "left_ee_pose": pose,
        "right_ee_pose": pose,
        "left_ee_joint_state": [0.2],
        "right_ee_joint_state": [0.8],
    }
    step = eef20_to_steps(state_to_eef20(state)[None])[0]
    np.testing.assert_allclose(step["left_ee_pose"], pose, atol=1e-6)
    official = robotwin_bridge.conversion.robotwin_endpose_to_eef20d(pose, pose, [0.2], [0.8])
    np.testing.assert_allclose(state_to_eef20(state), official, atol=1e-6)
    executed = packed_to_native(
        np.concatenate(
            [
                step["left_ee_pose"],
                np.asarray(step["left_ee_joint_state"]).reshape(1),
                step["right_ee_pose"],
                np.asarray(step["right_ee_joint_state"]).reshape(1),
            ]
        ),
        "wxyz",
    )
    np.testing.assert_allclose(
        executed, robotwin_bridge.conversion.eef20d_to_ee16d(official), atol=1e-5
    )
    raw_pose = pose_order(pose, "wxyz", native_order)
    raw = {
        "endpose": {
            "left_endpose": raw_pose,
            "right_endpose": raw_pose,
            "left_gripper": 0.2,
            "right_gripper": 0.8,
        }
    }
    packed = native_obs_to_packed(raw, native_order)
    native = packed_to_native(packed, native_order)
    np.testing.assert_allclose(native, np.r_[raw_pose, 0.2, raw_pose, 0.8], atol=1e-6)
    with pytest.raises(ValueError, match="unit quaternion"):
        pose_order(np.zeros(7), "wxyz", native_order)
