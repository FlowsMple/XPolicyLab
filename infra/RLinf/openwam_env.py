"""Opt-in RoboTwin EE execution; never send Cartesian actions to qpos control."""

import multiprocessing as mp
import os
from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.workers.env.env_worker import EnvWorker

from XPolicyLab.policy.OpenWAM.robotwin import native_obs_to_packed, packed_to_native


def is_recoverable_ee_execution_failure(exc):
    """Planner failures that happen before either arm moves.

    mplib TOPP raises RuntimeError after a path exists. The EE branch of
    take_action also divides by a 0-length successful plan on the first
    control iteration, before set_arm_joints or scene.step.
    """
    if isinstance(exc, RuntimeError) and "Fail to parameterize path" in str(exc):
        return True
    return isinstance(exc, ZeroDivisionError)


def build_ee_vector_env(task_config, num_envs, seeds, quaternion_order):
    # Loaded only in simulator workers. Uses RoboTwin's RLinf_support branch.
    from robotwin.envs import vector_env as native

    class EESubEnv(native.SubEnv):
        def create_instruction(self):
            descriptions = native.generate_episode_descriptions(
                self.task_name, self.episode_info_list, 1, self.env_seed
            )
            return descriptions[0][self.instruction_type][0]

        def _observation(self):
            raw = self.task.get_obs()
            cameras = raw["observation"]
            return {
                "full_image": cameras["head_camera"]["rgb"],
                "left_wrist_image": cameras["left_camera"]["rgb"],
                "right_wrist_image": cameras["right_camera"]["rgb"],
                "state": native_obs_to_packed(raw, quaternion_order),
                "instruction": self.task.get_instruction(),
            }

        def get_obs(self):
            with self.lock:
                return self._observation()

        def step(self, actions):
            actions = np.asarray(actions)
            if actions.shape != (1, 16):
                raise ValueError(f"EE experiment expects [1,16], got {actions.shape}.")
            if self.get_instruction() is None:
                self.reset(env_seed=None)
            with self.lock:
                before = self.task.take_action_cnt
                planner_failure = False
                try:
                    self.task.take_action(
                        packed_to_native(actions[0], quaternion_order),
                        action_type="ee",
                    )
                except Exception as exc:
                    if not is_recoverable_ee_execution_failure(exc):
                        raise
                    # take_action already counted this step and wrote the video
                    # frame. The failure is before either arm moves.
                    planner_failure = True
                    print(
                        f"EE plan failed ({type(exc).__name__}: {exc}); holding both arms for this step",
                        flush=True,
                    )
                executed = self.task.take_action_cnt - before
                success = bool(self.task.eval_success or self.task.check_success())
                truncated = (
                    self.task.take_action_cnt >= self.task.step_lim and not success
                )
                return {
                    "obs": self._observation(),
                    "reward": float(success),
                    "terminated": success,
                    "truncated": truncated,
                    "info": {
                        "success": success,
                        "executed_action_count": executed,
                        "planner_failure": planner_failure,
                    },
                }

        def step_with_full_obs(self, actions):
            result = self.step(actions)
            result["obs"] = [result["obs"]]
            return result

    class EEVectorEnv(native.VectorEnv):
        def _init_envs(self):
            self.args["action_dim"] = 16
            for index in range(self.n_envs):
                env = EESubEnv(
                    env_id=index,
                    task_name=self.task_name,
                    args=deepcopy(self.args),
                    env_seed=self.env_seeds[index],
                    instruction_type=self.instruction_type,
                    global_lock=self.global_lock,
                )
                env.setup_task()
                self.envs.append(env)

    return EEVectorEnv(task_config=task_config, n_envs=num_envs, env_seeds=seeds)


class OpenWAMRoboTwinEnv(RoboTwinEnv):
    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        order = self.cfg.get("native_quaternion_order")
        if order not in ("xyzw", "wxyz"):
            raise ValueError(
                "Set native_quaternion_order after checking your RoboTwin version."
            )
        os.environ["ASSETS_PATH"] = self.cfg.assets_path
        task = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        if not task["data_type"].get("endpose") or not task["camera"].get(
            "collect_wrist_camera"
        ):
            raise ValueError("OpenWAM requires endpose and both wrist cameras.")
        self.venv = build_ee_vector_env(
            task, self.num_envs, self.reset_state_ids.tolist(), order
        )

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        if chunk_step != 1:
            raise ValueError("First EE experiment executes exactly one step per chunk.")
        # Reward the actual single executed transition, including success at step 0.
        return step_reward.reshape(self.num_envs, 1)

    def sample_action_space(self):
        raise NotImplementedError(
            "Random Cartesian commands are not a valid debug policy."
        )


class OpenWAMEnvWorker(EnvWorker):
    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        if env_cfg.env_type != "robotwin":
            raise ValueError("OpenWAM EE experiment requires env_type=robotwin.")
        return super()._setup_env_and_wrappers(
            OpenWAMRoboTwinEnv, env_cfg, num_envs_per_stage
        )
