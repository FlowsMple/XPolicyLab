"""Bounded, weight-free physical simulator checks. Always selects physical GPU 0.

Run using the simulator's Python with the simulator checkout as working directory.
No Ray, model, checkpoint, data download, or training process is started.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def select_gpu_zero():
    if any(name in sys.modules for name in ("torch", "sapien", "isaacsim")):
        raise RuntimeError("GPU selection must precede simulator/CUDA imports.")
    row = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=uuid,name,pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    uuid, name, pci = row
    os.environ.update(
        CUDA_VISIBLE_DEVICES=uuid,
        NVIDIA_VISIBLE_DEVICES=uuid,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        OMNI_KIT_ALLOW_ROOT="1",
    )
    return {"physical_index": 0, "uuid": uuid, "name": name, "pci_bus_id": pci}


def inspect_observation(obs, output, label):
    import numpy as np
    from PIL import Image

    def array(value):
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    state = array(obs["states"])
    assert state.ndim == 2 and state.shape[0] == 1 and np.isfinite(state).all(), (
        state.shape
    )
    cameras = {"head": array(obs["main_images"])[0]}
    wrists = array(obs["wrist_images"])[0]
    assert len(wrists) == 2, wrists.shape
    cameras.update(left_wrist=wrists[0], right_wrist=wrists[1])
    images = {}
    for name, rgb in cameras.items():
        assert rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8, (
            name,
            rgb.shape,
            rgb.dtype,
        )
        assert rgb.max() > rgb.min(), f"Flat/empty camera: {name}"
        Image.fromarray(rgb).save(output / f"{label}_{name}.png")
        images[name] = {
            "shape": list(rgb.shape),
            "range": [int(rgb.min()), int(rgb.max())],
        }
    return {"state": state.tolist(), "cameras": images}


def validate_robodojo(args, report):
    import numpy as np
    from omegaconf import OmegaConf

    from XPolicyLab.infra.RLinf.envs.robodojo.env import RoboDojoEnv

    cfg = OmegaConf.create(
        {
            "max_episode_steps": args.steps,
            "gamma": 0.99,
            "shaping_weight": 0.0,
            "seed": 0,
            "use_fixed_reset_state_ids": True,
            "auto_reset": False,
            "ignore_terminations": False,
            "sim_python": sys.executable,
            "robodojo_root": str(args.root),
            "rpc_timeout": 600,
            "init_params": {
                "task_name": "stack_bowls",
                "env_cfg_type": "arx_x5",
                "layout_seed": args.layout_seed,
                "device": "cuda:0",
            },
        }
    )
    backend = None
    recorder = None
    if args.reward_probes:
        from XPolicyLab.infra.RLinf.envs.robodojo.process import SimulatorProcess

        params = OmegaConf.to_container(cfg.init_params)
        params.update(num_envs=1, max_episode_steps=20, seed=0)
        if args.record_video:
            params["record_frames"] = 8
        cfg.max_episode_steps = 20
        backend = SimulatorProcess(
            sys.executable,
            args.root,
            params,
            timeout=600,
            factory="src.rlinf_bridge.diagnostics:RewardProbeBackend",
        )
        from XPolicyLab.infra.RLinf.tests.tools.validation_video import CaptureBackend

        backend = CaptureBackend(backend)
    env = RoboDojoEnv(cfg, 1, backend=backend)
    try:
        report["backend_metadata"] = env.backend.metadata
        if args.record_video:
            assert env.backend.metadata.get("render_zero_delay"), (
                "Zero-delay Kit camera settings are not active"
            )
        env.reset_state_ids = [args.seed]
        obs, _ = env.reset()
        report["reset"] = inspect_observation(obs, args.output, "reset")
        progress_history = [float(env.progress[0])]
        if args.record_video:
            from XPolicyLab.infra.RLinf.tests.tools.validation_video import VideoRecorder

            recorder = VideoRecorder(args.output / "reward_validation.mp4")
            recorder.append(
                backend.last_result,
                0,
                0.0,
                progress_history[0],
                float(env.official_score[0]),
                progress_history,
            )
        initial = obs["states"].clone()
        trace = []
        report["trace"] = trace
        horizon = 1 if args.reward_probes else args.chunk_length
        count = 4 if args.reward_probes else (args.steps + horizon - 1) // horizon
        for i in range(count):
            # Joint actions are absolute targets; zeros are not a neutral action.
            actions = initial[:, None, :].repeat(1, horizon, 1)
            started = time.monotonic()
            observations, reward, term, trunc, infos = env.chunk_step(actions)
            assert np.isfinite(reward.numpy()).all()
            trace.append(
                {
                    "step": i + 1,
                    "seconds": time.monotonic() - started,
                    "reward": reward.tolist(),
                    "terminated": term.tolist(),
                    "truncated": trunc.tolist(),
                    "episode": {k: v.tolist() for k, v in infos[-1]["episode"].items()},
                    "pose_sync_errors": backend.last_result.get("pose_sync_errors")
                    if args.reward_probes
                    else None,
                }
            )
            if args.reward_probes:
                inspect_observation(observations[-1], args.output, f"probe_{i + 1}")
                progress_history.append(float(env.progress[0]))
                if recorder:
                    recorder.append(
                        backend.last_result,
                        i + 1,
                        float(reward.sum()),
                        float(env.progress[0]),
                        float(env.official_score[0]),
                        progress_history,
                    )
            if term.any() or trunc.any():
                break
        report["final"] = inspect_observation(observations[-1], args.output, "final")
        if args.reward_probes:
            assert term.any() and not trunc.any(), (
                "Controlled three-bowl fixture should succeed"
            )
            np.testing.assert_allclose(
                [t["episode"]["progress"][0] for t in trace], [0.15, 0.0, 0.15, 0.85]
            )
            np.testing.assert_allclose(
                [t["reward"][0][0] for t in trace],
                [0.0, 0.0, 0.0, 1.0],
                atol=1e-6,
            )
            assert trace[2]["episode"]["regressions"] == [1]
            assert trace[2]["episode"]["recoveries"] == [1]
            report["probe_scope"] = (
                "Direct rigid-body placement; real task predicates and RL reward; not a policy trajectory or a settled-stack stability test."
            )
        else:
            assert trunc.any() and not term.any(), (
                "Hold episode should reach time limit, not success/failure."
            )
            assert len(trace) == count
            assert env.elapsed_steps.tolist() == [args.steps]
            expected_start = (args.steps - 1) % horizon
            assert not trunc[0, :expected_start].any()
            assert trunc[0, expected_start:].all()
        before = observations[-1]["states"].clone()
        after, reward, _, _, _ = env.chunk_step(initial[:, None, :])
        assert not reward.any(), "Finished episode emitted reward twice"
        report["post_terminal_reward"] = reward.tolist()
        assert (after[-1]["states"] == before).all(), "Terminal observation changed"
        obs2, _ = env.reset()
        report["repeat_reset"] = inspect_observation(obs2, args.output, "reset_repeat")
        report["reset_max_joint_difference"] = float(
            (obs2["states"] - initial).abs().max()
        )
        report["checked"] = [
            "RGB/state",
            "finished-slot reward and observation",
            "repeat reset",
        ]
        report["checked"] += (
            ["controlled progress/regression/recovery/success"]
            if args.reward_probes
            else ["absolute joint hold", "timeout mask"]
        )
        if recorder:
            report["video"] = {
                "path": str(recorder.path),
                "fps": recorder.fps,
                "frames": recorder.frames,
                "segments": recorder.segments,
                "scope": "Live rendered frames of controlled pose fixtures; no physics integration between held frames.",
            }
            report["checked"] += [
                "PhysX/USD pose agreement",
                "visual bowl-count sequence 3,2,3,2,1",
            ]
    finally:
        try:
            if recorder:
                # Keep the primary assertion if encoding fails during cleanup.
                recorder.close(check=sys.exc_info()[0] is None)
        finally:
            env.close()


def validate_robotwin(args, report):
    import numpy as np
    import sapien
    from scripts.eval_policy_xpolicylab import (
        class_decorator,
        load_task_args,
        robotwin_obs_to_xpolicylab,
        xpolicylab_action_to_robotwin,
    )

    from XPolicyLab.infra.RLinf.envs.robodojo.env import pack_observations
    from XPolicyLab.utils.process_data import (
        get_robot_action_dim_info,
        unpack_robot_state,
    )

    cfg, _ = load_task_args(
        {
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "policy_name": "validation",
        }
    )
    cfg.update(render_freq=0, eval_video_log=False, collect_data=False)
    dims = get_robot_action_dim_info("aloha_agilex")
    env = class_decorator("adjust_bottle")
    try:
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **cfg)
        env.step_lim = args.steps
        env.set_instruction("Adjust the bottle.")

        def observe():
            raw = env.get_obs()
            packed = pack_observations(
                [robotwin_obs_to_xpolicylab(raw, task_env=env)], dims
            )
            return raw, packed

        raw, obs = observe()
        report["reset"] = inspect_observation(obs, args.output, "reset")
        initial = obs["states"].numpy()[0].copy()
        report["trace"] = []
        for i in range(args.steps):
            target = unpack_robot_state(initial, "joint", dims)
            native, action_type = xpolicylab_action_to_robotwin(
                target, action_type="joint", current_observation=raw
            )
            env.take_action(native, action_type=action_type)
            raw, obs = observe()
            report["trace"].append(
                {
                    "step": i + 1,
                    "success": bool(env.check_success()),
                    "action_count": int(env.take_action_cnt),
                }
            )
        report["final"] = inspect_observation(obs, args.output, "final")
        assert env.take_action_cnt == args.steps
        assert not env.eval_success, "Unexpected success during hold"
        saved = env.bottle.get_pose()
        origin = np.asarray(saved.p).copy()
        point = np.asarray(env.bottle.get_functional_point(0)[:3])
        side = -1 if env.qpose_tag == 0 else 1
        report["success_predicate_probes"] = []
        try:
            for name, target, expected in (
                ("correct_side_and_height", [side * 0.25, point[1], 0.95], True),
                ("too_low", [side * 0.25, point[1], 0.85], False),
                ("wrong_side", [-side * 0.25, point[1], 0.95], False),
            ):
                env.bottle.actor.set_pose(
                    sapien.Pose(origin + np.asarray(target) - point, saved.q)
                )
                actual = bool(env.check_success())
                report["success_predicate_probes"].append(
                    {"case": name, "success": actual}
                )
                assert actual == expected, name
        finally:
            env.bottle.actor.set_pose(saved)
        assert not env.check_success()
        report["predicate_probe_scope"] = (
            "Direct bottle placement; native success predicate only, not a policy rollout."
        )
        report["checked"] = [
            "native simulator RGB/state",
            "XPolicyLab action conversion",
            "absolute joint hold",
        ]
        report["checked"].append("native success positive/negative pose fixtures")
        report["rlinf_vector_env_checked"] = False
    finally:
        env.close_env()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator", choices=("robotwin", "robodojo"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RoboDojo layout index / RoboTwin random seed",
    )
    parser.add_argument("--layout-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-probes", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 800:
        parser.error("steps must be in [1, 800]")
    if not 1 <= args.chunk_length <= 50:
        parser.error("chunk-length must be in [1, 50]")
    if args.reward_probes and args.simulator != "robodojo":
        parser.error("--reward-probes requires robodojo")
    if args.record_video and not (
        args.reward_probes and args.seed == 0 and args.layout_seed == 0
    ):
        parser.error(
            "--record-video requires --reward-probes, --seed 0 and --layout-seed 0 (white-bowl image check)"
        )
    if args.record_video and not shutil.which("ffmpeg"):
        parser.error("--record-video requires ffmpeg on PATH")
    args.root = Path(__file__).resolve().parents[5]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.root)
    sys.path[:0] = [
        str(args.root),
        str(args.root / "XPolicyLab"),
        str(args.root / "description/utils"),
    ]
    report = {
        "simulator": args.simulator,
        "seed": args.seed,
        "layout_seed": args.layout_seed,
        "chunk_length": args.chunk_length,
        "status": "failed",
        "python": sys.executable,
        "root": str(args.root),
    }
    started = time.monotonic()
    try:
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(300, repeat=True)
        report["gpu"] = select_gpu_zero()
        import torch

        assert torch.cuda.device_count() == 1, (
            "Expected exactly one visible CUDA device"
        )
        report["torch"] = torch.__version__
        globals()[f"validate_{args.simulator}"](args, report)
        report["status"] = "passed"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: report[k] for k in ("status", "simulator", "elapsed_seconds")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
