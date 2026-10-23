"""CPU contracts; these do not establish real-checkpoint or simulator parity."""

import io

import numpy as np
import pytest
import torch
from torch import nn
from XPolicyLab.policy.OpenWAM.rl import JointReplay, OpenWAMNFTPolicy, action_nft_loss
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
    assert architecture.action_backbone.weight.grad.abs().sum() > 0
    assert architecture.video.weight.grad is None
    restored = OpenWAMNFTPolicy(TinyJointArchitecture(), 10, [1, 1, 0, 0])
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored(data)["v_theta"], output)


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
