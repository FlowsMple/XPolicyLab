"""Capture initial real RGB/EE observations for the standalone replay probe."""

import argparse
import os
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from XPolicyLab.infra.RLinf.openwam_env import build_ee_vector_env
    from XPolicyLab.policy.OpenWAM.robotwin import native_obs_to_packed

    order = os.environ["ROBOTWIN_QUATERNION_ORDER"]
    os.environ["ASSETS_PATH"] = os.environ["ROBOTWIN_ASSETS_PATH"]
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent / "config"), version_base="1.1"
    ):
        cfg = compose(config_name="robotwin_adjust_bottle_xpolicylab_openwam_eval")
    task = OmegaConf.to_container(cfg.env.eval.task_config, resolve=True)
    env = build_ee_vector_env(task, 1, [0], order)
    try:
        env.reset(env_seeds=[0])
        # Access this one native environment only, before any rollout threads.
        obs = env.envs[0].get_obs()
        raw = env.envs[0].task.get_obs()
        packed = native_obs_to_packed(raw, order)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            main_images=obs["full_image"][None],
            wrist_images=np.stack([obs["left_wrist_image"], obs["right_wrist_image"]])[
                None
            ],
            states=packed[None],
            task_descriptions=np.array([obs["instruction"]]),
        )
        print(f"Saved initial observation: {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
