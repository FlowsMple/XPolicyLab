"""Real-checkpoint replay/gradient probe, without Ray or a running simulator.

Run as python -m XPolicyLab.policy.OpenWAM.rlinf_probe --help.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unavailable"


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device) if torch.is_tensor(value) else value


def sample(value, index):
    if isinstance(value, dict):
        return {key: sample(item, index) for key, item in value.items()}
    return value[index : index + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--obs",
        type=Path,
        help="NPZ: RGB main_images/wrist_images, EE16 wxyz states, task_descriptions",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Repeat observations to test batch vs microbatch replay",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    report = {
        "status": "started",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint": str(args.checkpoint.resolve()),
        "xpolicylab_revision": revision(root),
        "rlinf_revision": revision(root / "infra/RLinf/RLinf"),
        "observation_source": str(args.obs)
        if args.obs
        else "SYNTHETIC: shape/gradient test only",
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    started = time.monotonic()
    try:
        for package in ("transformers", "diffusers", "ray", "omegaconf"):
            report[package] = importlib.metadata.version(package)
        report["config_sha256"] = hashlib.sha256(
            (args.checkpoint / "config.yaml").read_bytes()
        ).hexdigest()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if str(args.device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; the real-checkpoint probe needs a configured GPU environment."
                )
            report["gpu"] = torch.cuda.get_device_name(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        from XPolicyLab.utils.process_data import unpack_robot_state

        from .model import Model
        from .rl import action_nft_loss

        model = Model(
            {
                "ckpt_dir": str(args.checkpoint.resolve()),
                "env_cfg_type": "aloha_agilex",
                "action_type": "ee",
                "openwam_benchmark": "robotwin",
                "experimental_nft": True,
                "rl": {"modes": ["eval", "nft"]},
                "num_action_chunks": 1,
                "replan_steps": 1,
                "device": args.device,
                "seed": args.seed,
            }
        )
        model.policy.eval()
        if args.gradient_checkpointing:
            model.policy.gradient_checkpointing_enable()
        if args.obs:
            with np.load(args.obs, allow_pickle=False) as saved:
                data = dict(saved)
        else:
            data = {
                "main_images": np.zeros((1, 256, 256, 3), dtype=np.uint8),
                "wrist_images": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8),
                "states": np.array(
                    [[0.1, 0.1, 0.3, 1, 0, 0, 0, 1, -0.1, 0.1, 0.3, 1, 0, 0, 0, 1]],
                    dtype=np.float32,
                ),
                "task_descriptions": ["adjust the bottle"],
            }
        obs = []
        for i, state in enumerate(data["states"]):
            obs.append(
                {
                    "state": unpack_robot_state(
                        state, "ee", model.robot_action_dim_info, source_type="obs"
                    ),
                    "instruction": str(data["task_descriptions"][i]),
                    "env_idx": i,
                    "vision": {
                        "cam_high": {"color": data["main_images"][i]},
                        "cam_left_wrist": {"color": data["wrist_images"][i, 0]},
                        "cam_right_wrist": {"color": data["wrist_images"][i, 1]},
                    },
                }
            )
        obs = obs * args.repeat
        # The final Euler step can have exactly zero self-consistency error;
        # use the first step for a meaningful gradient plumbing probe.
        actions, extras = model.predict_action_batch_rl(
            obs, mode="eval", nft_step_index=0
        )
        if actions.shape != (len(obs), 1, 16) or not np.isfinite(actions).all():
            raise AssertionError(f"Invalid execution actions: {actions.shape}")
        report["action_shape"] = list(actions.shape)
        report["actions"] = actions.tolist()
        replay = to_device(extras["forward_inputs"], args.device)
        with torch.no_grad():
            prediction = model.nft_forward(replay)["v_theta"].float()
        old = replay["nft_v"].float()
        report["replay_max_abs_error"] = (prediction - old).abs().max().item()
        report["replay_mean_abs_error"] = (prediction - old).abs().mean().item()
        report["replay_relative_rms_error"] = (
            (prediction - old).square().mean().sqrt()
            / old.square().mean().sqrt().clamp_min(1e-8)
        ).item()
        torch.testing.assert_close(prediction, old, atol=args.atol, rtol=args.rtol)
        with torch.no_grad():
            micro_prediction = torch.cat(
                [
                    model.nft_forward(sample(replay, i))["v_theta"].float()
                    for i in range(len(obs))
                ]
            )
        report["microbatch_replay_max_abs_error"] = (
            (micro_prediction - old).abs().max().item()
        )
        torch.testing.assert_close(
            micro_prediction, old, atol=args.atol, rtol=args.rtol
        )
        report["replay_pass"] = True
        if args.backward:
            model.policy.train()
            model.policy.zero_grad(set_to_none=True)
            velocity = model.nft_forward(replay)["v_theta"]
            # Deliberately non-zero advantage: tests gradient plumbing, not learning.
            loss, _ = action_nft_loss(
                velocity,
                replay,
                torch.ones(len(obs), device=args.device),
                torch.ones(len(obs), device=args.device),
                clip_ratio=None,
            )
            if not torch.isfinite(loss):
                raise AssertionError("Non-finite probe loss")
            loss.backward()
            total = 0.0
            for name, parameter in model.policy.named_parameters():
                grad = parameter.grad
                if not parameter.requires_grad and grad is not None:
                    raise AssertionError(f"Frozen parameter has gradient: {name}")
                if grad is not None:
                    if not torch.isfinite(grad).all():
                        raise AssertionError(f"Non-finite gradient: {name}")
                    total += grad.float().square().sum().item()
            if total <= 0:
                raise AssertionError(
                    "No action gradient; inspect detach/no_grad/freeze paths"
                )
            report.update(
                backward_pass=True, gradient_norm=total**0.5, probe_loss=loss.item()
            )
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            report["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(args.device) / 2**30
            )
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Probe {report['status']}: {args.output}")


if __name__ == "__main__":
    main()
