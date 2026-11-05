"""Compose shipped configs without starting Ray, CUDA, or a model."""

import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from XPolicyLab.infra.RLinf.core.configuration import config_dirs, register_config_paths
from XPolicyLab.infra.RLinf.core.validation import validate_adapter_config

register_config_paths()
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    p.stem
    for directory in config_dirs()
    for p in directory.glob("*.yaml")
    if not p.stem.endswith("_sft")
]


@pytest.mark.parametrize("name", CONFIGS)
def test_compose(name, monkeypatch):
    for key in (
        "POLICY_MODEL_PATH",
        "ROBOTWIN_ASSETS_PATH",
        "ROBODOJO_ROOT",
        "ROBODOJO_SIM_PYTHON",
        "ROBOTWIN_QUATERNION_ORDER",
    ):
        monkeypatch.setenv(key, "/not-installed")
    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name=name)
    OmegaConf.to_container(cfg, resolve=True)
    from rlinf import config as native
    from rlinf.scheduler.cluster.utils import parse_rank_config
    from XPolicyLab.infra.RLinf.register import register

    # The shipped single-node placements describe the world sizes without Ray.
    sizes = {
        component.strip(): len(parse_rank_config(ranks))
        for components, ranks in cfg.cluster.component_placement.items()
        for component in components.split(",")
    }
    placement = Mock()
    placement.get_world_size.side_effect = sizes.__getitem__
    monkeypatch.setattr(native, "Cluster", Mock())
    monkeypatch.setattr(native, "HybridComponentPlacement", Mock(return_value=placement))
    register()
    validate_adapter_config(cfg)
    if "pi05" in name and not cfg.runner.only_eval:
        cfg.algorithm.recompute_logprobs = True
    native.validate_cfg(cfg)
    model = cfg.rollout.model if cfg.runner.only_eval else cfg.actor.model
    if "pi05" in name:
        assert model.xpolicylab.rl_backend == "rlinf_v0_4"
        if not cfg.runner.only_eval:
            assert cfg.actor.fsdp_config.sharding_strategy == "full_shard"
        assert model.model_type == "openpi"


@pytest.mark.parametrize("sharding", ["no_shard", "shard_grad_op", "full_shard"])
def test_pi05_uses_native_fsdp_validation(monkeypatch, sharding):
    from rlinf.config import validate_fsdp_cfg

    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name="robotwin_adjust_bottle_ppo_xpolicylab_pi05")
    cfg.actor.fsdp_config.sharding_strategy = sharding
    cfg.actor.fsdp_config.use_orig_params = False
    assert validate_fsdp_cfg(cfg.actor).fsdp_config.use_orig_params
    cfg.actor.fsdp_config.mixed_precision.param_dtype = "bf16"
    with pytest.raises(ValueError, match="mixed_precision_rules"):
        validate_fsdp_cfg(cfg.actor)


@pytest.mark.parametrize(
    "override",
    [
        "env.train.auto_reset=true",
        "env.enable_offload=true",
        "env.train.max_episode_steps=799",
        "env.train.group_size=3",
        "algorithm.reward_type=action_level",
        "actor.model.xpolicylab.action_type=ee",
    ],
)
def test_robodojo_rejects_incompatible_runtime(override, monkeypatch):
    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(
            config_name="robodojo_stack_bowls_ppo_xpolicylab_pi05", overrides=[override]
        )
    with pytest.raises(ValueError):
        validate_adapter_config(cfg)


def test_gamma_consistency_only_required_for_explicit_shaping(monkeypatch):
    monkeypatch.setenv('RLINF_PATH', str(ROOT / 'RLinf'))
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base='1.1'):
        cfg = compose(config_name='robodojo_stack_bowls_ppo_xpolicylab_pi05', overrides=['env.train.gamma=0.5'])
    validate_adapter_config(cfg)
    cfg.env.train.enable_shaping = True
    cfg.env.train.shaping_weight = .5
    with pytest.raises(ValueError):
        validate_adapter_config(cfg)


@pytest.mark.parametrize(
    "script,args,address,entry",
    [
        ("run.sh", [], "local", "run.py"),
        ("run_openwam.sh", ["train"], "local", "run.py"),
        ("run_openwam.sh", ["train", "--help"], "", "check_config.py"),
        ("run_openwam.sh", ["eval", "--cfg", "job"], "", "check_config.py"),
    ],
)
def test_launcher_skips_ray_start(tmp_path, monkeypatch, script, args, address, entry):
    # The OpenWAM launcher creates its import shim relative to the script.
    launcher = tmp_path / "XPolicyLab" / "infra" / "RLinf" / "scripts" / script
    launcher.parent.mkdir(parents=True)
    for name in {"run.sh", script}:
        (launcher.parent / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    for key in ("POLICY_MODEL_PATH", "ROBOTWIN_ASSETS_PATH", "ROBOTWIN_RUNTIME_PATH"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("ROBOTWIN_QUATERNION_ORDER", "wxyz")
    ray = tmp_path / "ray"
    ray.write_text('#!/bin/sh\necho "unexpected Ray call: $*" >&2\nexit 7\n')
    ray.chmod(0o755)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\n[ "$1" = "-c" ] || echo "${1##*/} $RAY_ADDRESS"\n')
    python.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("RAY_ADDRESS", address)
    launch = subprocess.run(["bash", str(launcher), *args], capture_output=True, text=True)
    assert launch.returncode == 0, launch.stderr
    assert launch.stdout.split() == [entry] + ([address] if address else [])

def test_openwam_requires_matching_calibration(monkeypatch):
    monkeypatch.setenv("RLINF_PATH", str(ROOT / "RLinf"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.1"):
        cfg = compose(config_name="robotwin_adjust_bottle_xpolicylab_openwam_eval")
    cfg.rollout.model.xpolicylab.openwam_benchmark = None
    with pytest.raises(ValueError, match="calibration"):
        validate_adapter_config(cfg)
