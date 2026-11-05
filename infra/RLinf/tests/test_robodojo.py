"""CPU contract tests; the simulator boundary is a deterministic test double."""

from copy import deepcopy
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from XPolicyLab.infra.RLinf.envs.robodojo.extension import ENV_ID
from XPolicyLab.infra.RLinf.envs.robodojo.env import (
    RoboDojoEnv,
    pack_observations,
    potential_reward,
)
from XPolicyLab.infra.RLinf.envs.robodojo.process import SimulatorProcess
from XPolicyLab.utils.process_data import unpack_robot_state

DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def observation(value=0):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # red must stay red
    return {
        "state": unpack_robot_state(
            np.full(14, value, dtype=np.float32), "joint", DIMS
        ),
        "vision": {
            k: {"color": rgb.copy()}
            for k in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
    }


class Backend:
    metadata = {"robot_action_dim_info": DIMS, "num_layouts": 7}

    def __init__(self, script):
        self.script, self.calls, self.closed = script, [], False

    def snapshot(self, row=None):
        row = row or {}
        return {
            "observations": [observation(self.index), observation(self.index)],
            "progress": row.get("progress", [0.0, 0.0]),
            "official_score": [0.0, 0.0],
            "success": row.get("success", [False, False]),
            "failed": [False, False],
            "truncated": [False, False],
        }

    def reset(self, layout_ids):
        self.layout_ids = layout_ids
        self.index = 0
        return self.snapshot()

    def step(self, actions, env_indices):
        self.calls.append((deepcopy(actions), list(env_indices)))
        row = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return self.snapshot(row)

    def close(self):
        self.closed = True


def env(script=None, **kwargs):
    cfg = OmegaConf.create(
        dict(max_episode_steps=4, gamma=1.0, shaping_weight=0.5, enable_shaping=True, **kwargs)
    )
    backend = Backend(script or [{}])
    return RoboDojoEnv(cfg, 2, backend=backend), backend


def test_partial_success_masks_actions_and_caches_terminal_obs():
    task, backend = env([{"success": [True, False]}, {}, {}])
    task.reset()
    actions = torch.arange(2 * 3 * 14).reshape(2, 3, 14).float()
    obs, rewards, term, trunc, infos = task.chunk_step(actions)
    assert [indices for _, indices in backend.calls] == [[0, 1], [1], [1]]
    np.testing.assert_array_equal(
        backend.calls[0][0][0]["left_arm_joint_state"], np.arange(6)
    )
    assert backend.calls[0][0][0]["left_ee_joint_state"][0] == 6
    assert term[0].all() and not term[1].any() and not trunc.any()
    assert rewards.sum() == 1 and rewards[0, 0] == 1
    assert (
        obs[-1]["states"][0, 0] == 1
    )  # terminal frame, not later shared-physics frames
    assert obs[-1]["states"][1, 0] == 3
    assert infos[-1]["final_observation"] is obs[-1]
    _, rewards, _, trunc, _ = task.chunk_step(actions)
    assert rewards.sum() == 0
    assert trunc[1].all()
    assert len(backend.calls) == 4
    task.close()
    assert backend.closed


def test_progress_regression_and_recovery_are_current_state():
    task, _ = env(
        [{"progress": [0.85, 0]}, {"progress": [0.15, 0]}, {"progress": [0.85, 0]}]
    )
    task.reset()
    outputs = [task.chunk_step(torch.zeros(2, 1, 14)) for _ in range(3)]
    rewards = [o[1][0].item() for o in outputs]
    assert rewards == pytest.approx([0.425, -0.35, 0.35])
    assert outputs[-1][-1][-1]["episode"]["regressions"][0] == 1
    assert outputs[-1][-1][-1]["episode"]["recoveries"][0] == 1


def test_potential_discount_telescopes_and_timeout_keeps_potential():
    gamma, weight = 0.9, 0.5
    path = [0.0, 0.85, 0.15, 0.85, 0.0]
    values = [
        potential_reward(
            path[i], path[i + 1], False, i == 3, gamma=gamma, weight=weight
        )
        for i in range(4)
    ]
    assert sum(gamma**i * r for i, r in enumerate(values)) == pytest.approx(
        0.0, abs=1e-7
    )
    assert potential_reward(
        0.15, 0.85, False, False, gamma=gamma, weight=weight
    ) == pytest.approx(0.3075)
    assert potential_reward(
        0.85, 0.85, True, True, gamma=gamma, weight=weight
    ) == pytest.approx(0.575)


def test_chunk_shaping_emitted_once_and_same_for_success_inside_chunk():
    task, _ = env(
        [{"progress": [0.15, 0]}, {"progress": [0.85, 0]}, {"success": [True, False]}]
    )
    task.reset()
    _, rewards, term, _, _ = task.chunk_step(torch.zeros(2, 3, 14))
    assert rewards[0].tolist() == [0.0, 0.0, 1.0]
    assert term[0].tolist() == [False, False, True]


def test_reset_group_ids_and_no_partial_reset():
    task, backend = env(group_size=2)
    task.reset(seed=12)
    assert backend.layout_ids[0] == backend.layout_ids[1]
    ids = backend.layout_ids.copy()
    task.reset(seed=12)
    assert backend.layout_ids == ids
    with pytest.raises(NotImplementedError):
        task.reset(env_ids=[0])


@pytest.mark.parametrize(
    "actions", [torch.zeros(2, 3, 13), torch.full((2, 3, 14), float("nan"))]
)
def test_bad_actions_do_not_reach_simulator(actions):
    task, backend = env()
    task.reset()
    with pytest.raises(ValueError):
        task.chunk_step(actions)
    assert not backend.calls


def test_rgb_and_joint_roundtrip():
    packed = pack_observations([observation(2)], DIMS)
    assert packed["states"].shape == (1, 14)
    assert packed["main_images"][0, 0, 0].tolist() == [255, 0, 0]
    assert packed["wrist_images"].shape == (1, 2, 8, 8, 3)


def test_ee_observation_and_action_reach_native_pose_control(monkeypatch):
    task, backend = env(action_type="ee")
    native = observation()
    poses = np.array([0.3, 0.1, 0.2, 1, 0, 0, 0, 0.5] * 2, dtype=np.float32)
    native["state"].update(unpack_robot_state(poses, "ee", DIMS))
    snapshot = backend.snapshot

    def ee_snapshot(row=None):
        result = snapshot(row)
        result["observations"] = [native, native]
        return result

    monkeypatch.setattr(backend, "snapshot", ee_snapshot)
    obs, _ = task.reset()
    np.testing.assert_array_equal(obs["states"][0], poses)
    actions = obs["states"].unsqueeze(1).clone()
    actions[0, 0, 0] += 0.01
    task.chunk_step(actions)
    sent = backend.calls[0][0][0]
    assert "left_arm_joint_state" not in sent
    np.testing.assert_array_equal(sent["left_ee_pose"], actions[0, 0, :7])
    np.testing.assert_array_equal(sent["right_ee_pose"], poses[8:15])


def test_registry_and_native_action_passthrough():
    from XPolicyLab.infra.RLinf.register import register
    from rlinf.envs import get_env_cls
    from rlinf.envs.action_utils import prepare_actions

    register()
    cfg = OmegaConf.create({"init_params": {"id": ENV_ID}})
    assert get_env_cls("isaaclab", cfg) is RoboDojoEnv
    actions = torch.rand(2, 3, 14)
    result = prepare_actions(actions, "isaaclab", "xpolicylab", 3, 14)
    torch.testing.assert_close(result, actions)


def test_native_timeout_bootstrap_distinguishes_success():
    from rlinf.data.schema.embodied_types import EnvPart, EnvTransition
    from XPolicyLab.infra.RLinf.envs.robodojo.rollout import bootstrap_robodojo_timeout

    trunc = torch.tensor([[False, True], [False, False]])
    term = torch.tensor([[False, False], [True, True]])
    transition = EnvTransition(
        rewards=torch.tensor([[0.0, 0.2], [1.0, 0.0]]),
        dones=trunc | term,
        truncations=trunc,
        terminations=term,
    )
    part = EnvPart(
        sources=[], transition=transition, bootstrap_values=torch.tensor([[2.0], [9.0]])
    )
    result = bootstrap_robodojo_timeout(part, 0.9)
    torch.testing.assert_close(
        result.transition.rewards, torch.tensor([[0.0, 2.0], [1.0, 0.0]])
    )
    assert part.transition.rewards[0, 1] == 0.2  # transport data not mutated
    # Native v0.4 does not add the bonus a second time when auto_reset=false.
    native = result.compute_rewards(
        env_reward_weight=1.0,
        reward_weight=0.0,
        auto_reset=False,
        bootstrap_type="standard",
        gamma=0.9,
    )
    torch.testing.assert_close(native, result.transition.rewards)


@pytest.fixture
def simulator_factory(tmp_path):
    (tmp_path / "fake_sim.py").write_text("""
import os
import sys
import time
class Backend:
    def __init__(self, cfg): self.cfg = cfg
    def metadata(self): return {"pid": os.getpid(), "torch_loaded": "torch" in sys.modules, "args": sys.argv[3:]}
    def reset(self, ids):
        if self.cfg.get("crash"): os._exit(3)
        if self.cfg.get("raise"): raise ValueError("bad layout")
        if self.cfg.get("sleep"): time.sleep(10)
        print("Simulator stdout does not corrupt RPC", flush=True)
        return ids
    def close(self): pass
""")
    return tmp_path


@pytest.mark.parametrize("render_gpu,kit_args", [(None, []), (7, ["--/renderer/activeGpu=7"])])
def test_rpc_separate_process_and_clean_shutdown(simulator_factory, render_gpu, kit_args):
    sim = SimulatorProcess(
        sys.executable, simulator_factory, {}, timeout=2, factory="fake_sim:Backend", render_gpu=render_gpu
    )
    assert not sim.metadata["torch_loaded"]
    assert sim.metadata["args"] == kit_args
    assert sim.reset([1, 2]) == [1, 2]
    sim.close()
    assert sim.process.poll() == 0
    sim.close()


@pytest.mark.parametrize(
    "failure, error",
    [("raise", RuntimeError), ("crash", RuntimeError), ("sleep", TimeoutError)],
)
def test_rpc_errors_and_timeouts_reap_child(simulator_factory, failure, error):
    sim = SimulatorProcess(
        sys.executable,
        simulator_factory,
        {failure: True},
        timeout=0.5,
        factory="fake_sim:Backend",
    )
    with pytest.raises(error):
        sim.reset([1])
    assert sim.process.poll() is not None


def test_rollout_publishes_native_terminal_value_with_timeout_bonus():
    from rlinf.data.schema.embodied_types import EnvPart, EnvTransition
    from XPolicyLab.infra.RLinf.envs.robodojo.rollout import RoboDojoRolloutWorker
    from types import SimpleNamespace

    worker = RoboDojoRolloutWorker.__new__(RoboDojoRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"train": {"init_params": {"id": ENV_ID}}}})
    worker.model_cfg = OmegaConf.create({"model_type": "openpi", "add_value_head": True})
    worker.algorithm_cfg = OmegaConf.create({"gamma": 0.9})
    worker.collect_transitions = False
    worker.enable_rlt = False
    received = []
    worker._predict_rollout_actions = lambda obs: (None, {"prev_values": obs["value"]})
    part = EnvPart(
        sources=[],
        requires_inference=True,
        next_obs={"value": torch.tensor([[2.0]])},
        transition=EnvTransition(
            rewards=torch.zeros(1, 2), truncations=torch.tensor([[False, True]])
        ),
    )
    worker._publish_env_part(
        part, None, None, SimpleNamespace(put=lambda p, **kwargs: received.append(p))
    )
    assert received[0].transition.rewards[0, 1] == pytest.approx(1.8)
    assert received[0].final_prev_values.item() == 2.0


def test_env_worker_closes_train_and_eval_even_after_one_error():
    from XPolicyLab.infra.RLinf.workers.env import XPolicyLabEnvWorker
    from types import SimpleNamespace

    closed = []

    def fail():
        closed.append("train")
        raise ValueError("cleanup error")

    train = SimpleNamespace(close=fail)
    worker = SimpleNamespace(
        env_list=[train],
        eval_env_list=[train, SimpleNamespace(close=lambda: closed.append("eval"))],
    )
    with pytest.raises(RuntimeError, match="cleanup error"):
        XPolicyLabEnvWorker.close_environments(worker)
    assert closed == ["train", "eval"]


def test_native_loss_mask_keeps_terminal_action_and_masks_padding():
    from rlinf.utils.metric_utils import compute_loss_mask

    task, _ = env([{"success": [True, False]}, {}, {}])
    task.reset()
    _, _, term1, trunc1, _ = task.chunk_step(torch.zeros(2, 3, 14))
    _, _, term2, trunc2, _ = task.chunk_step(torch.zeros(2, 3, 14))
    dones = torch.stack([torch.zeros_like(term1), term1 | trunc1, term2 | trunc2])
    mask, _ = compute_loss_mask(dones)
    assert mask[:, 0].tolist() == [[True, False, False], [False, False, False]]
    assert mask[:, 1].tolist() == [[True, True, True], [True, False, False]]
    assert mask.any(dim=-1)[:, 0].tolist() == [True, False]


def test_sparse_default_ignores_progress_and_gamma_and_sends_success_once():
    for gamma in (.1, .99, 1.):
        backend = Backend([{'progress': [.15, 0]}, {'progress': [.15, 0]},
                           {'progress': [0, 0]}, {'success': [True, False]}])
        task = RoboDojoEnv(OmegaConf.create(dict(max_episode_steps=10, gamma=gamma)), 2, backend=backend)
        task.reset()
        for expected in (0, 0, 0, 1, 0):
            _, reward, _, _, infos = task.chunk_step(torch.zeros(2, 1, 14))
            assert reward[0].sum() == expected
            parts = infos[-1]['reward_components']
            assert parts['base_reward'][0] == expected
            assert not parts['shaping_reward'].any()
        task.close()

def test_shaping_requires_explicit_opt_in_and_algorithm_gamma():
    for cfg in (dict(shaping_weight=.5), dict(enable_shaping=True, shaping_weight=.5)):
        with pytest.raises(ValueError):
            RoboDojoEnv(OmegaConf.create(dict(max_episode_steps=4, **cfg)), 2, backend=Backend([{}]))

def test_opt_in_shaping_is_separate_from_simulator_base_reward():
    task, _ = env([{'progress': [.15, 0]}])
    task.reset()
    _, reward, _, _, infos = task.chunk_step(torch.zeros(2, 1, 14))
    parts = infos[-1]['reward_components']
    assert not parts['base_reward'].any()
    assert parts['shaping_reward'][0] == pytest.approx(.075)
    torch.testing.assert_close(parts['returned_reward'], reward.sum(-1))
