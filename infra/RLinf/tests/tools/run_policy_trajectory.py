"""One bounded real-policy episode, retaining its layout and every transition.

Run with the RoboDojo interpreter. Start the ordinary XPolicyLab policy server
separately in its own interpreter, with its explicitly selected GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def run(args, report):
    from client_server.ws.model_client import WsModelClient
    import numpy as np
    from omegaconf import OmegaConf

    from XPolicyLab.infra.RLinf.envs.robodojo.env import RoboDojoEnv
    from XPolicyLab.infra.RLinf.envs.robodojo.process import SimulatorProcess
    from XPolicyLab.infra.RLinf.tests.tools.trajectory_recording import TrajectoryWriter, policy_frame
    from XPolicyLab.infra.RLinf.tests.tools.validate_simulator import inspect_observation
    from XPolicyLab.infra.RLinf.tests.tools.validation_video import CaptureBackend, VideoRecorder
    from XPolicyLab.utils.process_data import pack_robot_state

    params = dict(
        task_name="stack_bowls",
        env_cfg_type="arx_x5",
        layout_seed=args.layout_seed,
        device="cuda:0",
        num_envs=1,
        max_episode_steps=args.max_steps,
        seed=args.seed,
        record_dir=str(args.output),
    )
    backend = CaptureBackend(
        SimulatorProcess(
            sys.executable,
            args.root,
            params,
            timeout=600,
            factory="src.rlinf_bridge.recording:RecordingBackend",
        )
    )
    env = RoboDojoEnv(
        OmegaConf.create(
            dict(
                max_episode_steps=args.max_steps,
                gamma=0.99,
                shaping_weight=0.0,
                seed=args.seed,
                use_fixed_reset_state_ids=True,
                auto_reset=False,
                ignore_terminations=False,
            )
        ),
        1,
        backend=backend,
    )
    client = video = dataset = None
    trace = []
    report["trace"] = trace
    try:
        env.reset_state_ids = [args.layout_index]
        obs, _ = env.reset()
        report["backend_metadata"] = backend.metadata
        report["initial"] = inspect_observation(obs, args.output, "initial")
        raw = backend.last_result["observations"][0]
        layout = json.loads((args.output / "initial_layout.json").read_text())
        dataset = TrajectoryWriter(args.output / "episode.hdf5", env.dimensions, raw["instruction"], layout, 25)
        video = VideoRecorder(args.output / "trajectory.mp4", fps=25)
        history = [float(env.progress[0])]
        initial_clock = backend.last_result["physics_time"]
        video.write_frame(policy_frame(raw, "INITIAL", 0, args.max_steps, 0, history[0], 0, history, 0))
        client = WsModelClient(
            url=args.policy_url,
            evaluation_id="wedojo-pi05-trajectory",
            trial_id=f"layout-{args.layout_seed}-{args.layout_index}",
            request_timeout_s=600,
            max_connect_seconds=600,
        )
        client.call(func_name="reset")
        inferences = 0
        done = False
        with (args.output / "trace.jsonl").open("w") as stream:
            while not done and len(trace) < args.max_steps:
                client.call(func_name="update_obs_batch", obs=backend.last_result["observations"])
                started = time.monotonic()
                chunks = client.call(func_name="get_action_batch", obs=[0])
                elapsed = time.monotonic() - started
                if not isinstance(chunks, (list, tuple)) or len(chunks) != 1 or not chunks[0]:
                    raise ValueError("Policy must return one nonempty action chunk")
                inferences += 1
                print(f"INFERENCE {inferences}: {len(chunks[0])} actions in {elapsed:.3f}s", flush=True)
                for action in chunks[0]:
                    vector = pack_robot_state({"state": action}, "joint", env.dimensions)
                    if vector.shape != (env.action_dim,) or not np.isfinite(vector).all():
                        raise ValueError("Policy returned invalid joint targets")
                    previous_raw = backend.last_result["observations"][0]
                    previous = float(env.progress[0])
                    observations, rewards, term, trunc, infos = env.chunk_step(vector[None, None, :])
                    reward, progress = float(rewards.sum()), float(env.progress[0])
                    result = backend.last_result
                    raw = result["observations"][0]
                    episode = infos[-1]["episode"]
                    phase = (
                        "SUCCESS"
                        if bool(env.success[0])
                        else (
                            "FAILED"
                            if bool(env.terminated[0])
                            else "TIMEOUT"
                            if bool(env.truncated[0])
                            else "REGRESSION"
                            if progress < previous - 1e-6
                            else "RECOVERY"
                            if int(episode["recoveries"][0]) > report.get("recoveries", 0)
                            else "PROGRESS"
                            if progress > 0
                            else "RUNNING"
                        )
                    )
                    report["recoveries"] = int(episode["recoveries"][0])
                    history.append(progress)
                    metrics = dict(
                        step=len(trace) + 1,
                        reward=reward,
                        base_reward=float(infos[-1]["reward_components"]["base_reward"][0]),
                        shaping_reward=float(infos[-1]["reward_components"]["shaping_reward"][0]),
                        progress=progress,
                        official_score=float(env.official_score[0]),
                        terminated=bool(term.any()),
                        truncated=bool(trunc.any()),
                        physics_time=result["physics_time"],
                    )
                    row = dict(
                        **metrics,
                        phase=phase,
                        rigid_poses=result["rigid_poses"],
                        support_forces=result.get("support_forces"),
                        native_success=[bool(x) for x in result.get("native_success", [])],
                        unsupported_native_success=[bool(x) for x in result.get("unsupported_native_success", [])],
                        action=vector.tolist(),
                        state=observations[-1]["states"][0].tolist(),
                        regressions=int(episode["regressions"][0]),
                        recoveries=int(episode["recoveries"][0]),
                    )
                    dataset.append(previous_raw, vector, metrics)
                    trace.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    video.write_frame(
                        policy_frame(
                            raw,
                            phase,
                            len(trace),
                            args.max_steps,
                            reward,
                            progress,
                            metrics["official_score"],
                            history,
                            result["physics_time"] - initial_clock,
                        )
                    )
                    done = bool(term.any() or trunc.any())
                    if progress != previous or done:
                        inspect_observation(observations[-1], args.output, f"event_{len(trace):04d}_{phase.lower()}")
                        print(
                            f"EVENT step={len(trace)} phase={phase} progress={progress:.2f} reward={reward:+.5f}",
                            flush=True,
                        )
                    if done or len(trace) >= args.max_steps:
                        break
        report.update(
            success=bool(env.success[0]),
            terminated=bool(env.terminated[0]),
            truncated=bool(env.truncated[0]),
            steps=len(trace),
            policy_inferences=inferences,
            total_reward=float(env.returns[0]),
            regressions=int(env.regressions[0]),
            recoveries=int(env.recoveries[0]),
            simulated_seconds=result["physics_time"] - initial_clock,
            video_frames=video.frames,
            video_fps=video.fps,
        )
        inspect_observation(observations[-1], args.output, "final")
        assert dataset.rows == len(trace) and video.frames == len(trace) + 1
        assert result["native_action_count"] == len(trace)
        assert report["simulated_seconds"] > 0
        after, reward, _, _, _ = env.chunk_step(vector[None, None, :])
        assert not reward.any(), "Repeated terminal reward"
        assert (after[-1]["states"] == observations[-1]["states"]).all()
        report["post_terminal_reward"] = reward.tolist()
        report["status"] = "completed"
    finally:
        try:
            if client:
                client.close()
        finally:
            try:
                if dataset:
                    dataset.close()
            finally:
                try:
                    if video:
                        video.close(check=sys.exc_info()[0] is None)
                finally:
                    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:19605")
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--layout-seed", type=int, default=0)
    parser.add_argument("--layout-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 800:
        parser.error("max-steps must be in [1,800]")
    args.root = Path(__file__).resolve().parents[5]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.root)
    sys.path[:0] = [str(args.root), str(args.root / "XPolicyLab")]
    from XPolicyLab.infra.RLinf.tests.tools.validate_simulator import select_gpu_zero

    report = dict(
        status="failed",
        policy_url=args.policy_url,
        layout_seed=args.layout_seed,
        layout_index=args.layout_index,
        max_steps=args.max_steps,
        scope="One real closed-loop policy episode; no rigid-body pose injection",
    )
    if args.policy_config:
        import yaml

        text = args.policy_config.resolve().read_text()
        report["policy_config"] = yaml.safe_load(text)
        (args.output / "policy_server.yaml").write_text(text)
    started = time.monotonic()
    try:
        report["simulator_gpu"] = select_gpu_zero()
        run(args, report)
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report.get(k) for k in ("status", "steps", "success", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
