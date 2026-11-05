"""CPU integration: real RL accounting and recording, scripted simulator/policy."""

import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from XPolicyLab.infra.RLinf.tests.tools.run_policy_trajectory import run
from XPolicyLab.utils.process_data import unpack_robot_state


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires system ffmpeg")
def test_complete_episode_stops_mid_chunk_and_closes_every_resource(tmp_path, monkeypatch):
    from XPolicyLab.infra.RLinf.envs.robodojo import process as robodojo_process

    dimensions = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    instances = {}

    class Simulator:
        metadata = {"robot_action_dim_info": dimensions, "num_layouts": 1}

        def __init__(self, python, root, config, **kwargs):
            self.index, self.closed = 0, False
            self.output = Path(config["record_dir"])
            instances["simulator"] = self

        def snapshot(self):
            image = np.zeros((16, 16, 3), np.uint8)
            image[..., 0] = 240
            obs = {
                "state": unpack_robot_state(np.zeros(14), "joint", dimensions),
                "instruction": np.str_("Stack the three bowls together."),
                "vision": {name: {"color": image} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
            }
            return {
                "observations": [obs],
                "progress": [0.85 if self.index == 2 else 0.15 if self.index else 0],
                "official_score": [100 if self.index == 2 else 15 if self.index else 0],
                "success": [self.index == 2],
                "failed": [False],
                "truncated": [False],
                "rigid_poses": {},
                "physics_time": self.index * 0.04,
                "native_action_count": self.index,
            }

        def reset(self, ids):
            (self.output / "initial_layout.json").write_text(json.dumps({"Rigid": {}}))
            return self.snapshot()

        def step(self, actions, indices):
            self.index += 1
            assert self.index <= 2, "Executed an extra action after success"
            return self.snapshot()

        def close(self):
            self.closed = True

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            instances["client"] = self

        def call(self, func_name, obs=None):
            if func_name == "get_action_batch":
                action = unpack_robot_state(np.zeros(14), "joint", dimensions)
                return [[action, action, action]]

        def close(self):
            self.closed = True

    monkeypatch.setattr(robodojo_process, "SimulatorProcess", Simulator)
    # Script only the network boundary; the GPU experiment uses the real WS client.
    module = ModuleType("client_server.ws.model_client")
    module.WsModelClient = Client
    monkeypatch.setitem(sys.modules, "client_server.ws.model_client", module)
    args = SimpleNamespace(
        root=tmp_path, output=tmp_path, seed=0, layout_seed=0, layout_index=0, max_steps=4, policy_url="ws://unused"
    )
    report = {}
    run(args, report)
    assert report["status"] == "completed" and report["success"]
    assert report["steps"] == 2 and report["video_frames"] == 3
    assert report["post_terminal_reward"] == [[0.0]]
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2
    assert instances["simulator"].closed and instances["client"].closed
