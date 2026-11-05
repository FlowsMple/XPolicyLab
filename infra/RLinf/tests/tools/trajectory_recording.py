"""Streaming RGB trajectory artifacts for a real single-episode policy rollout."""

import json

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .validation_video import _font

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class TrajectoryWriter:
    """Each row stores observation_t, action_t and transition_t metrics."""

    def __init__(self, path, dimensions, instruction, layout, frequency):
        self.dimensions, self.rows = dimensions, 0
        self.file = h5py.File(path, "x")
        self.file.create_dataset("data_format_version", data="v1.0")
        self.file.create_dataset("instruction", data=str(instruction), dtype=h5py.string_dtype("utf-8"))
        self.file.create_dataset("additional_info/frequency", data=frequency)
        self.file.create_dataset("additional_info/scene_layout_json", data=json.dumps(layout))
        self.file.attrs["transition_convention"] = (
            "observation_t, action_t, reward_t; terminal observation saved separately"
        )
        self.file.attrs["image_encoding"] = "XPL-RGB1 via encode_image_bit"

    def _append_array(self, name, value):
        value = np.asarray(value)
        if name not in self.file:
            self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape), dtype=value.dtype)
        dataset = self.file[name]
        dataset.resize(self.rows + 1, axis=0)
        dataset[self.rows] = value

    def append(self, observation, action, metrics):
        from XPolicyLab.utils.process_data import encode_image_bit, pack_robot_state, unpack_robot_state

        state = pack_robot_state(observation, "joint", self.dimensions)
        for group, vector in (("state", state), ("action", action)):
            for key, value in unpack_robot_state(vector, "joint", self.dimensions, source_type="dataset").items():
                self._append_array(f"{group}/{key}", value)
        for key, value in metrics.items():
            self._append_array(f"metrics/{key}", value)
        for camera in CAMERAS:
            rgb = np.asarray(observation["vision"][camera]["color"])
            name = f"vision/{camera}/colors"
            if name not in self.file:
                self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.uint8))
                self.file.create_dataset(f"vision/{camera}/shape", data=rgb.shape)
            dataset = self.file[name]
            dataset.resize(self.rows + 1, axis=0)
            dataset[self.rows] = np.frombuffer(encode_image_bit(rgb), dtype=np.uint8)
        self.rows += 1
        self.file.flush()

    def close(self):
        self.file.close()


def policy_frame(observation, phase, step, max_steps, reward, progress, score, history, sim_seconds):
    canvas = Image.new("RGB", (1024, 800), "#101820")
    draw = ImageDraw.Draw(canvas)
    white, muted, green = "#eff4fa", "#acb8c5", "#5bddad"
    draw.text((20, 16), "RoboDojo / stack_bowls / Pi05", font=_font(29), fill=white)
    draw.text(
        (20, 58), "Closed-loop policy | physics: GPU 0 renderer / CPU | policy: GPU 1", font=_font(17), fill=muted
    )
    for i, camera in enumerate(CAMERAS):
        rgb = np.asarray(observation["vision"][camera]["color"])
        size, xy = ((640, 480), (20, 100)) if i == 0 else ((288, 216), (690, 298 + (i - 1) * 236))
        canvas.paste(Image.fromarray(rgb).resize(size), xy)
        label = ("Head camera", "Left wrist camera", "Right wrist camera")[i]
        draw.text((xy[0], xy[1] + size[1]), label, font=_font(16), fill=muted)
    draw.text((690, 100), phase, font=_font(27), fill="#ff9c73" if phase == "REGRESSION" else green)
    for y, text in zip(
        (149, 185, 221, 257),
        (
            f"Progress: {progress:.2f}",
            f"Base reward: {reward:+.2f}",
            f"Native score: {score:.0f}/100",
            f"Action: {step}/{max_steps}",
        ),
    ):
        draw.text((690, y), text, font=_font(21), fill=white)
    draw.text((690, 284), "Shaping: disabled", font=_font(17), fill=muted)
    draw.text((20, 624), "Progress through the physical episode", font=_font(19), fill=white)
    draw.line([(45, 661), (45, 746), (625, 746)], fill="#465565", width=2)
    points = [(45 + i * 580 / max_steps, 746 - float(v) * 85 / 0.85) for i, v in enumerate(history)]
    if len(points) > 1:
        draw.line(points, fill=green, width=3)
    draw.text(
        (20, 752),
        f"Simulation elapsed: {sim_seconds:.2f}s | captured at 25 Hz | original OpenPI/JAX checkpoint 59999",
        font=_font(14),
        fill=muted,
    )
    draw.text(
        (20, 777), "Live policy actions and physical contacts; no object pose injection.", font=_font(14), fill=muted
    )
    return np.asarray(canvas)
