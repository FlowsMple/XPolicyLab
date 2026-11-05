import json

import h5py
import numpy as np
import pytest

from XPolicyLab.infra.RLinf.tests.tools.trajectory_recording import CAMERAS, TrajectoryWriter, policy_frame
from XPolicyLab.utils.process_data import decode_image_bit, unpack_robot_state


def test_trajectory_retains_layout_and_aligned_rgb_state_action_rows(tmp_path):
    dimensions = {"arm_dim": [2, 2], "ee_dim": [1, 1]}
    path = tmp_path / "episode.hdf5"
    layout = {"Rigid": {"bowl": [{"label": "bowl0", "default_pos": [0, 0, 0.8]}]}}
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[..., 0] = 240
    raw = {
        "state": unpack_robot_state(np.arange(6, dtype=float), "joint", dimensions),
        "vision": {name: {"color": rgb} for name in CAMERAS},
    }
    writer = TrajectoryWriter(path, dimensions, np.str_("Stack the bowls."), layout, 25)
    try:
        for i in range(2):
            writer.append(
                raw,
                np.arange(6, dtype=float) + i + 10,
                {"reward": float(i), "physics_time": i * 0.04, "terminated": bool(i)},
            )
    finally:
        writer.close()
    with h5py.File(path) as source:
        assert json.loads(source["additional_info/scene_layout_json"][()]) == layout
        assert source["additional_info/frequency"][()] == 25
        np.testing.assert_array_equal(source["state/left_arm_joint_states"][:], [[0, 1], [0, 1]])
        np.testing.assert_array_equal(source["action/right_arm_joint_states"][:], [[13, 14], [14, 15]])
        assert source["metrics/terminated"][:].tolist() == [False, True]
        for camera in CAMERAS:
            bits = source[f"vision/{camera}/colors"][0]
            assert b"XPL-RGB1" in bits.tobytes()
            decoded = decode_image_bit(bits)
            assert decoded[8, 8, 0] > 230 and decoded[8, 8, 2] < 10
    with pytest.raises(FileExistsError):
        TrajectoryWriter(path, dimensions, "", layout, 25)


def test_policy_video_frame_handles_a_full_episode_history():
    rgb = np.zeros((480, 640, 3), np.uint8)
    raw = {"vision": {name: {"color": rgb} for name in CAMERAS}}
    frame = policy_frame(raw, "SUCCESS", 800, 800, 0.575, 0.85, 100, [0] * 799 + [0.85, 0.85], 32)
    assert frame.shape == (800, 1024, 3) and frame.dtype == np.uint8
