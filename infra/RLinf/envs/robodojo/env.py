"""RLinf v0.4 synchronous RoboDojo stack_bowls environment.

Finished slots retain terminal observations and receive no further actions.
Isaac still advances the shared scene for live slots; no partial reset is used.
Rewards are emitted once per chunk, matching RLinf's chunk-level discount.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf
import torch

from .extension import ENV_ID as ENV_ID
from .process import SimulatorProcess


def potential_reward(previous, current, success, terminated, *, gamma, weight):
    """True terminal states have zero potential; time limits retain potential."""
    final = np.where(terminated, 0.0, current)
    return np.asarray(success, dtype=np.float32) + weight * (gamma * final - previous)


def pack_observations(observations, dimensions, action_type="joint"):
    from XPolicyLab.utils.process_data import decode_image_bit, pack_robot_state

    def rgb(obs, name):
        value = obs["vision"][name]["color"]
        if isinstance(value, (bytes, bytearray)) or np.asarray(value).ndim == 1:
            value = decode_image_bit(value)
        value = np.asarray(value)
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(
                f"{name}: expected RGB HWC uint8, got {value.shape}/{value.dtype}."
            )
        return value.copy()

    states = np.stack(
        [
            pack_robot_state(obs, action_type, dimensions, source_type="obs")
            for obs in observations
        ]
    ).astype(np.float32)
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite RoboDojo robot observation.")
    return {
        "states": torch.from_numpy(states),
        "main_images": torch.from_numpy(
            np.stack(
                [
                    rgb(o, "cam_head" if "cam_head" in o["vision"] else "cam_high")
                    for o in observations
                ]
            )
        ),
        "wrist_images": torch.from_numpy(
            np.stack(
                [
                    [rgb(o, "cam_left_wrist"), rgb(o, "cam_right_wrist")]
                    for o in observations
                ]
            )
        ),
        "task_descriptions": ["Stack the three bowls together."] * len(observations),
    }


class RoboDojoEnv:
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        *,
        backend=None,
    ):
        del total_num_processes, worker_info
        self.cfg, self.num_envs = cfg, int(num_envs)
        if cfg.get("auto_reset", False) or cfg.get("ignore_terminations", False):
            raise ValueError(
                "RoboDojo requires auto_reset=false and ignore_terminations=false."
            )
        self.action_type = cfg.get("action_type", "joint")
        if self.action_type not in ("joint", "ee"):
            raise ValueError(f"Unsupported RoboDojo action type: {self.action_type}")
        self.group_size = int(cfg.get("group_size", 1))
        if (
            self.num_envs <= 0
            or self.group_size <= 0
            or self.num_envs % self.group_size
        ):
            raise ValueError(
                "RoboDojo batch size must be a positive multiple of group_size."
            )
        self.device = torch.device("cpu")
        self.auto_reset = self.ignore_terminations = False
        self.is_start = True
        self.max_steps = int(cfg.max_episode_steps)
        self.enable_shaping = bool(cfg.get("enable_shaping", False))
        if self.enable_shaping and "gamma" not in cfg:
            raise ValueError("Explicit shaping requires gamma supplied by the RL algorithm.")
        self.gamma = float(cfg.get("gamma", 1.0))
        self.weight = float(cfg.get("shaping_weight", 0.0))
        if not 0 <= self.gamma <= 1 or self.weight < 0 or self.max_steps <= 0:
            raise ValueError(
                "Invalid RoboDojo reward discount/weight or episode length."
            )
        if not self.enable_shaping and self.weight != 0:
            raise ValueError("Set enable_shaping=true explicitly before using a nonzero shaping_weight.")
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + seed_offset)
        if backend is None:
            params = OmegaConf.to_container(cfg.init_params, resolve=True)
            params.pop("id", None)
            params.update(
                num_envs=self.num_envs,
                max_episode_steps=self.max_steps,
                seed=int(cfg.get("seed", 0)) + seed_offset,
            )
            backend = SimulatorProcess(
                cfg.sim_python,
                cfg.robodojo_root,
                params,
                timeout=cfg.get("rpc_timeout", 300),
                render_gpu=cfg.get("render_gpu"),
            )
        self.backend = backend
        try:
            self.dimensions = self.backend.metadata["robot_action_dim_info"]
            arm_dims = self.dimensions["arm_dim"]
            pose_dims = 7 * len(arm_dims) if self.action_type == "ee" else sum(arm_dims)
            self.action_dim = pose_dims + sum(self.dimensions["ee_dim"])
            self.layout_count = int(self.backend.metadata["num_layouts"])
            if self.layout_count <= 0:
                raise ValueError("No stack_bowls layouts available.")
            self.reset_state_ids = None
            self.update_reset_state_ids()
            self._observations = None
        except BaseException:
            self.close()
            raise

    def update_reset_state_ids(self):
        if (
            self.cfg.get("use_fixed_reset_state_ids", False)
            and self.reset_state_ids is not None
        ):
            return
        groups = self.rng.integers(
            self.layout_count, size=self.num_envs // self.group_size
        )
        self.reset_state_ids = np.repeat(groups, self.group_size).tolist()

    def reset(self, seed=None, env_ids=None, options=None):
        del options
        if env_ids is not None:
            raise NotImplementedError("RoboDojo supports full-batch resets only.")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.reset_state_ids = None
        self.update_reset_state_ids()
        result = self.backend.reset(self.reset_state_ids)
        self._accept(result)
        if np.any(self.terminated | self.truncated):
            raise RuntimeError("RoboDojo returned an already finished initial state.")
        self.elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.peak_progress = self.progress.copy()
        self.regressed = np.zeros(self.num_envs, dtype=bool)
        self.recoveries = np.zeros(self.num_envs, dtype=np.int64)
        self.regressions = np.zeros(self.num_envs, dtype=np.int64)
        return pack_observations(self._observations, self.dimensions, self.action_type), {}

    def _accept(self, result, active=None):
        if active is None:
            active = np.ones(self.num_envs, dtype=bool)
        if len(result["observations"]) != self.num_envs:
            raise ValueError("Simulator observation batch size mismatch.")
        fields = {
            key: np.asarray(
                result[key],
                dtype=np.float32 if key in ("progress", "official_score") else bool,
            )
            for key in ("progress", "official_score", "success", "failed", "truncated")
        }
        for key, values in fields.items():
            if values.shape != (self.num_envs,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid RoboDojo {key}: expected finite batch vector."
                )
        if np.any((fields["progress"] < 0) | (fields["progress"] > 1)):
            raise ValueError("RoboDojo progress must be in [0, 1].")
        fields["base_reward"] = np.asarray(result.get("base_reward", fields["success"]), dtype=np.float32)
        if fields["base_reward"].shape != (self.num_envs,) or not np.isfinite(fields["base_reward"]).all():
            raise ValueError("Invalid simulator base_reward batch.")
        fields["terminated"] = fields["success"] | fields["failed"]
        if np.any(fields["terminated"] & fields["truncated"]):
            raise ValueError("RoboDojo cannot terminate and truncate the same slot.")
        for key, values in fields.items():
            if not hasattr(self, key):
                setattr(self, key, values.copy())
            else:
                getattr(self, key)[active] = values[active]
        if self._observations is None:
            self._observations = deepcopy(result["observations"])
        else:
            for i in np.flatnonzero(active):
                self._observations[i] = deepcopy(result["observations"][i])

    def chunk_step(self, chunk_actions):
        from XPolicyLab.utils.process_data import unpack_robot_state

        if self._observations is None:
            raise RuntimeError("Call reset before chunk_step.")
        actions = (
            chunk_actions.detach().cpu().numpy()
            if torch.is_tensor(chunk_actions)
            else np.asarray(chunk_actions)
        )
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[2] != self.action_dim
            or actions.shape[1] <= 0
        ):
            raise ValueError(
                f"Expected actions [B,H,{self.action_dim}], got {actions.shape}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("RoboDojo actions must be finite.")
        horizon = actions.shape[1]
        initially_active = ~(self.terminated | self.truncated)
        previous = self.progress.copy()
        base_reward = np.zeros(self.num_envs, dtype=np.float32)
        term = np.zeros((self.num_envs, horizon), dtype=bool)
        trunc = term.copy()
        last_step = np.zeros(self.num_envs, dtype=int)
        for j in range(horizon):
            active = ~(self.terminated | self.truncated)
            indices = np.flatnonzero(active).tolist()
            if indices:
                unpacked = [
                    unpack_robot_state(
                        actions[i, j], self.action_type, self.dimensions, source_type="obs"
                    )
                    for i in indices
                ]
                self._accept(self.backend.step(unpacked, indices), active)
                base_reward[active] += self.base_reward[active]
                self.elapsed_steps[active] += 1
                self.truncated |= (
                    self.elapsed_steps.numpy() >= self.max_steps
                ) & ~self.terminated
                last_step[active] = j
                dropped = active & (self.progress < self.peak_progress - 1e-6)
                self.regressions += dropped & ~self.regressed
                self.regressed |= dropped
                restored = (
                    active
                    & self.regressed
                    & (self.progress >= self.peak_progress - 1e-6)
                )
                self.recoveries += restored
                self.regressed[restored] = False
                self.peak_progress = np.maximum(self.peak_progress, self.progress)
            term[:, j], trunc[:, j] = self.terminated, self.truncated
        shaping_reward = np.zeros(self.num_envs, dtype=np.float32)
        if self.enable_shaping:
            shaping_reward = potential_reward(
                previous, self.progress, False, self.terminated,
                gamma=self.gamma, weight=self.weight,
            )
        scale = initially_active * float(self.cfg.get("reward_coef", 1.0))
        base_reward *= scale
        shaping_reward *= scale
        boundary_reward = base_reward + shaping_reward
        rewards = np.zeros((self.num_envs, horizon), dtype=np.float32)
        rewards[np.arange(self.num_envs), last_step] = boundary_reward
        self.returns += boundary_reward
        obs = pack_observations(self._observations, self.dimensions, self.action_type)
        episode = {
            "success_once": self.success.copy(),
            "return": self.returns.copy(),
            "episode_len": self.elapsed_steps.numpy().copy(),
            "official_score": self.official_score.copy(),
            "progress": self.progress.copy(),
            "regressions": self.regressions.copy(),
            "recoveries": self.recoveries.copy(),
        }
        info = {
            "reward_components": {
                "base_reward": torch.from_numpy(base_reward.copy()),
                "shaping_reward": torch.from_numpy(np.asarray(shaping_reward, dtype=np.float32).copy()),
                "returned_reward": torch.from_numpy(boundary_reward.copy()),
            },
            "episode": {k: torch.from_numpy(v) for k, v in episode.items()},
            "final_observation": obs,
            "_final_observation": torch.from_numpy(self.terminated | self.truncated),
        }
        return (
            [obs],
            torch.from_numpy(rewards),
            torch.from_numpy(term),
            torch.from_numpy(trunc),
            [info],
        )

    def close(self):
        self.backend.close()
