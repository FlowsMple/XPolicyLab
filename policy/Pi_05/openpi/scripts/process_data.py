import os
import numpy as np
import shutil
import argparse
import cv2
import h5py
import dataclasses
from pathlib import Path
from typing import Any, Literal
import random
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_yaml, load_json
from XPolicyLab.utils.process_data import decode_image_bit, pack_robot_state

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT_PATH = Path(__file__).parent.parent.parent.parent.parent

CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    robot_action_dim_info: dict = None,
) -> LeRobotDataset:
    
    MOTORS = [
        *[f"left_{i}" for i in range(robot_action_dim_info["arm_dim"][0])],
        *[f"left_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][0])],
        *[f"right_{i}" for i in range(robot_action_dim_info["arm_dim"][1])],
        *[f"right_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][1])]
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }

    for camera_name in CAMERA_ALIASES.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["height", "width", "channels"],
        }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def _load_compressed_images(group: h5py.Group, key: str) -> np.ndarray:
    return np.asarray(decode_image_bit(group[key][()]))


def _as_2d_float(array) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _group_to_arrays(group: h5py.Group) -> dict[str, np.ndarray]:
    return {key: _as_2d_float(group[key][()]) for key in group.keys()}


def _pose_dim_from_group(group: dict[str, np.ndarray], default: int = 7) -> int:
    for key in ("left_ee_poses", "right_ee_poses", "ee_poses"):
        if key in group:
            return int(np.asarray(group[key]).shape[-1])
    return default


def _pack_dim_info(action_type: str, robot_action_dim_info: dict, sample_group: dict | None = None) -> dict:
    """Keep ee_dim (gripper). For EE, pack pose using the recorded pose width, not joint DoF."""
    if action_type != "ee":
        return robot_action_dim_info
    pose_dim = _pose_dim_from_group(sample_group or {}, default=7)
    return {
        "arm_dim": [pose_dim] * len(robot_action_dim_info["arm_dim"]),
        "ee_dim": list(robot_action_dim_info["ee_dim"]),
    }


def _decode_instruction_value(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_instruction_value(value.item())
        result = []
        for item in value.tolist():
            result.extend(_decode_instruction_value(item))
        return [item for item in result if item]
    return [str(value)]


def _read_instructions(ep: h5py.File) -> list[str] | None:
    if "instructions" in ep:
        strings = _decode_instruction_value(ep["instructions"][()])
        return strings or None
    if "instruction" in ep:
        strings = _decode_instruction_value(ep["instruction"][()])
        return strings or None
    return None


def _read_frequency(ep: h5py.File, default: int = 30) -> int:
    if "additional_info" in ep and "frequency" in ep["additional_info"]:
        return int(np.asarray(ep["additional_info"]["frequency"][()]).item())
    return default


def load_data(ep_path, action_type: str, robot_action_dim_info: dict) -> dict[str, Any]:
    with h5py.File(ep_path, "r") as ep:
        if "state" not in ep:
            raise KeyError(f"{ep_path} is missing a 'state' group")
        if "action" not in ep:
            raise KeyError(f"{ep_path} is missing an 'action' group")

        packed = {
            "state": _group_to_arrays(ep["state"]),
            "action": _group_to_arrays(ep["action"]),
        }
        pack_info = _pack_dim_info(action_type, robot_action_dim_info, packed["state"])
        state = pack_robot_state(
            packed, action_type, pack_info, source_type="dataset", state_type="state"
        ).astype(np.float32)
        action = pack_robot_state(
            packed, action_type, pack_info, source_type="dataset", state_type="action"
        ).astype(np.float32)

        images = {}
        for source_name, output_name in CAMERA_ALIASES.items():
            if source_name in ep["vision"]:
                images[output_name] = _load_compressed_images(ep["vision"][source_name], "colors")
        instructions = _read_instructions(ep)
        frequency = _read_frequency(ep)

    return {
        "images": images,
        "state": state,
        "action": action,
        "velocity": None,
        "effort": None,
        "timestamps": None,
        "instructions": instructions,
        "frequency": frequency,
    }

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="Dataset bench name (e.g., RoboDojo)")
    parser.add_argument("ckpt_name", type=str, help="Run name; also selects raw task dir under data/<bench>/")
    parser.add_argument("env_cfg_type", type=str, help="Environment config type (e.g., arx_x5)")
    parser.add_argument("action_type", type=str, help="Action type for artifact naming (e.g., joint)")
    parser.add_argument(
        "expert_data_num",
        type=str,
        nargs="?",
        default=None,
        help="Optional number of episodes to process; non-numeric values are treated as raw_task_dirs.",
    )
    parser.add_argument(
        "raw_task_dirs",
        type=str,
        nargs="?",
        default=None,
        help="Optional raw task dir or comma-separated dirs under data/<bench>/; defaults to ckpt_name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="image",
        help="Whether to store images as videos or individual image files",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="If set, use this prompt for every episode (overrides HDF5 instruction).",
    )
    args = parser.parse_args()

    bench_name = args.bench_name
    ckpt_name = args.ckpt_name
    env_cfg_type = args.env_cfg_type
    action_type = args.action_type
    repo_id = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
    mode = args.mode
    instruction = args.instruction
    expert_data_num = None
    raw_task_dirs_arg = args.raw_task_dirs
    if args.expert_data_num is not None:
        try:
            expert_data_num = int(args.expert_data_num)
        except ValueError:
            if args.raw_task_dirs is not None:
                raise ValueError("raw_task_dirs was provided twice.") from None
            raw_task_dirs_arg = args.expert_data_num
    raw_task_dirs = [item.strip() for item in (raw_task_dirs_arg or ckpt_name).split(",") if item.strip()]

    env_cfg = load_yaml(os.path.join(ROOT_PATH, "./env_cfg", f"{env_cfg_type}.yml"))
    robot_type = env_cfg['config']['robot']

    robot_action_dim_info = load_json(os.path.join(ROOT_PATH, "env_cfg/robot", "_robot_info.json"))[robot_type]

    episode_files = []
    for raw_task_dir in raw_task_dirs:
        load_data_dir = Path(ROOT_PATH) / "data" / str(bench_name) / raw_task_dir / str(env_cfg_type)
        task_episode_files = sorted(load_data_dir.glob("data/episode_*.hdf5"))
        if not task_episode_files:
            task_episode_files = sorted(load_data_dir.glob("*.hdf5"))
        episode_files.extend(task_episode_files)
    if expert_data_num is not None:
        episode_files = episode_files[:expert_data_num]
    if not episode_files:
        raise FileNotFoundError(
            f"No episode_*.hdf5 files found under data/{bench_name}/<task>/{env_cfg_type}"
        )

    fps = 30
    with h5py.File(episode_files[0], "r") as first_ep:
        fps = _read_frequency(first_ep, default=30)
        sample_state = _group_to_arrays(first_ep["state"]) if "state" in first_ep else {}
    pack_info = _pack_dim_info(action_type, robot_action_dim_info, sample_state)

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        mode=mode,
        dataset_config=DEFAULT_DATASET_CONFIG,
        robot_action_dim_info=pack_info,
    )

    for ep_file in tqdm(episode_files, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file, action_type, robot_action_dim_info)
            num_frames = data["state"].shape[0]
            if instruction:
                episode_instruction = instruction
            elif data["instructions"]:
                episode_instruction = random.choice(data["instructions"])
            else:
                episode_instruction = "Do your job."

            for i in range(num_frames):
                frame = {
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                    "task": episode_instruction,
                }
                for camera_name, images in data["images"].items():
                    frame[f"observation.images.{camera_name}"] = images[i]

                dataset.add_frame(frame)
            
            dataset.save_episode()
            dataset.hf_dataset = dataset.create_hf_dataset()
            tqdm.write(f"Finished {ep_file.name} with {num_frames} frames")
        except Exception as e:
            tqdm.write(f"Error processing episode {ep_file}: {e}")

if __name__ == "__main__":
    main()
