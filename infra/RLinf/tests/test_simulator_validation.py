"""CPU checks for validation tooling; never claim a physical simulator pass."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from XPolicyLab.infra.RLinf.tests.tools.validate_simulator import inspect_observation


def test_simulator_import_does_not_require_policy_runtime():
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from XPolicyLab.infra.RLinf.envs.robodojo.process import SimulatorProcess; assert 'rlinf' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_observation_inspection_rejects_empty_camera(tmp_path):
    obs = {
        "states": torch.zeros(1, 14),
        "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 2, 8, 8, 3, dtype=torch.uint8),
    }
    with pytest.raises(AssertionError, match="Flat/empty"):
        inspect_observation(obs, tmp_path, "empty")


def test_observation_inspection_preserves_rgb(tmp_path):
    from PIL import Image

    rgb = torch.zeros(1, 8, 8, 3, dtype=torch.uint8)
    rgb[..., 0] = 255
    obs = {
        "states": torch.zeros(1, 14),
        "main_images": rgb,
        "wrist_images": rgb[:, None].repeat(1, 2, 1, 1, 1),
    }
    result = inspect_observation(obs, tmp_path, "red")
    assert result["cameras"]["head"]["shape"] == [8, 8, 3]
    assert np.asarray(Image.open(tmp_path / "red_head.png"))[0, 0].tolist() == [
        255,
        0,
        0,
    ]


def test_gpu_selection_uses_physical_zero_uuid_before_torch_import():
    script = Path(__file__).resolve().parent / "tools" / "validate_simulator.py"
    code = """
import importlib.util, os
from unittest.mock import patch
spec = importlib.util.spec_from_file_location("smoke", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with patch("subprocess.check_output", return_value="GPU-zero, RTX 4090, 00000000:63:00.0\\n") as call:
    meta = module.select_gpu_zero()
assert "--id=0" in call.call_args.args[0]
assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-zero"
assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "GPU-zero"
assert meta["physical_index"] == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", "SCRIPT = " + repr(str(script)) + "\n" + code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
