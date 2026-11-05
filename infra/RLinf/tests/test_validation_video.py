"""CPU-only image/encoder tests; synthetic fixtures are not simulator evidence."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw
from XPolicyLab.infra.RLinf.tests.tools.validation_video import (
    EXPECTED_BOWLS,
    VideoRecorder,
    visible_bowl_count,
)


def synthetic_frame(count):
    image = Image.new("RGB", (640, 480), (120, 25, 10))
    draw = ImageDraw.Draw(image)
    for x in (190, 320, 450)[:count]:
        draw.ellipse((x - 20, 190, x + 20, 230), fill="white")
    # Ignore small specular noise and white objects outside the table ROI.
    draw.rectangle((120, 180, 121, 181), fill="white")
    draw.rectangle((10, 10, 70, 70), fill="white")
    return np.asarray(image)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_table_image_count_excludes_noise_and_outside_objects(count):
    assert visible_bowl_count(synthetic_frame(count)) == count


def test_stale_camera_is_rejected_even_if_reward_reports_progress():
    # No encoder is needed: a mismatching frame must fail before any write.
    recorder = object.__new__(VideoRecorder)
    stale = synthetic_frame(3)
    with pytest.raises(AssertionError, match="PROGRESS camera mismatch"):
        recorder.append(
            {"probe_frames": [[stale, stale, stale]]},
            1,
            0.07425,
            0.15,
            15,
            [0, 0.15],
        )


def test_empty_frame_batch_is_rejected():
    recorder = object.__new__(VideoRecorder)
    with pytest.raises(AssertionError, match="INITIAL camera mismatch"):
        recorder.append({"probe_frames": []}, 0, 0, 0, 0, [0])


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="video integration check requires system ffmpeg/ffprobe",
)
def test_video_roundtrip_has_all_phases_and_preserves_rgb(tmp_path):
    path = tmp_path / "synthetic_encoder_test.mp4"
    recorder = VideoRecorder(path)
    progresses = [0, 0.15, 0, 0.15, 0.85]
    rewards = [0, 0.07425, -0.075, 0.07425, 0.925]
    try:
        for i, count in enumerate(EXPECTED_BOWLS):
            rgb = synthetic_frame(count)
            result = {"probe_frames": [[rgb, rgb, rgb]] * 2, "probe_physics_time": 1.0}
            recorder.append(
                result,
                i,
                rewards[i],
                progresses[i],
                100 if i == 4 else 15,
                progresses[: i + 1],
            )
    finally:
        recorder.close()
    recorder.close()  # Cleanup can be repeated safely.
    assert recorder.frames == 10
    assert [s["bowl_counts"] for s in recorder.segments] == [
        [c, c] for c in EXPECTED_BOWLS
    ]
    metadata = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    assert metadata["codec_name"] == "h264"
    assert (metadata["width"], metadata["height"]) == (1024, 800)
    assert int(metadata["nb_frames"]) == 10
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(800, 1024, 3)
    pixel = decoded[400, 300].astype(int)
    assert pixel[0] > pixel[2] + 80, pixel  # Red must not become blue.
