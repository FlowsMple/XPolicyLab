import pathlib

import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.policies import aloha_policy
from openpi.training import config as _config


def _camera(value: int) -> np.ndarray:
    return np.full((3, 8, 8), value, dtype=np.uint8)


def test_aloha_inputs_default_uses_three_cameras():
    data = {
        "state": np.ones((16,), dtype=np.float32),
        "images": {
            "cam_high": _camera(1),
            "cam_left_wrist": _camera(2),
            "cam_right_wrist": _camera(3),
        },
    }

    outputs = aloha_policy.AlohaInputs(adapt_to_pi=False)(data)

    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"][..., 0], 1)
    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"][..., 0], 2)
    np.testing.assert_array_equal(outputs["image"]["right_wrist_0_rgb"][..., 0], 3)
    assert bool(outputs["image_mask"]["base_0_rgb"])
    assert bool(outputs["image_mask"]["left_wrist_0_rgb"])
    assert bool(outputs["image_mask"]["right_wrist_0_rgb"])


def test_aloha_inputs_left_wrist_only_masks_other_slots():
    left = _camera(9)
    data = {
        "state": np.ones((16,), dtype=np.float32),
        "images": {"cam_left_wrist": left},
    }

    outputs = aloha_policy.AlohaInputs(adapt_to_pi=False, enabled_cameras=("cam_left_wrist",))(data)

    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"][..., 0], 9)
    assert np.all(outputs["image"]["base_0_rgb"] == 0)
    assert np.all(outputs["image"]["right_wrist_0_rgb"] == 0)
    assert not bool(outputs["image_mask"]["base_0_rgb"])
    assert bool(outputs["image_mask"]["left_wrist_0_rgb"])
    assert not bool(outputs["image_mask"]["right_wrist_0_rgb"])


def test_aloha_inputs_left_wrist_only_ignores_other_present_cameras():
    data = {
        "state": np.ones((16,), dtype=np.float32),
        "images": {
            "cam_high": _camera(1),
            "cam_left_wrist": _camera(2),
            "cam_right_wrist": _camera(3),
        },
    }

    outputs = aloha_policy.AlohaInputs(adapt_to_pi=False, enabled_cameras=("cam_left_wrist",))(data)

    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"][..., 0], 2)
    assert np.all(outputs["image"]["base_0_rgb"] == 0)
    assert np.all(outputs["image"]["right_wrist_0_rgb"] == 0)
    assert not bool(outputs["image_mask"]["base_0_rgb"])
    assert bool(outputs["image_mask"]["left_wrist_0_rgb"])
    assert not bool(outputs["image_mask"]["right_wrist_0_rgb"])


def test_aloha_inputs_left_wrist_only_requires_enabled_camera():
    data = {
        "state": np.ones((16,), dtype=np.float32),
        "images": {"cam_high": _camera(1)},
    }

    with pytest.raises(ValueError, match="No enabled camera images found"):
        aloha_policy.AlohaInputs(adapt_to_pi=False, enabled_cameras=("cam_left_wrist",))(data)


def test_left_wrist_config_keeps_delta_ee_and_filters_cameras():
    config = _config.get_config("pi05_real_tianji_marvin_ee_delta_full2_left_wrist")

    assert config.data.repo_id == "RoboDojo_real-pick_magnet_full2-tianji_marvin-ee"
    assert config.data.use_delta_ee_actions
    assert not config.data.use_delta_joint_actions
    assert config.data.action_output_dim == 16
    assert config.data.enabled_cameras == ("cam_left_wrist",)

    data_config = config.data.create(pathlib.Path("/tmp"), config.model)
    aloha_inputs = [t for t in data_config.data_transforms.inputs if isinstance(t, aloha_policy.AlohaInputs)]
    assert len(aloha_inputs) == 1
    assert aloha_inputs[0].enabled_cameras == ("cam_left_wrist",)
    assert any(isinstance(t, _transforms.DeltaEEActions) for t in data_config.data_transforms.inputs)

    repack = data_config.repack_transforms.inputs[0]
    assert isinstance(repack, _transforms.RepackTransform)
    assert set(repack.structure["images"]) == {"cam_left_wrist"}
    assert repack.structure["images"]["cam_left_wrist"] == "observation.images.cam_left_wrist"
