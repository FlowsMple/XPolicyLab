"""Evaluate OpenWAM with the same EE boundary used by the NFT experiment."""

import hydra
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

from XPolicyLab.infra.RLinf.openwam_env import OpenWAMEnvWorker
from XPolicyLab.infra.RLinf.register import register
from XPolicyLab.infra.RLinf.rollout_worker import XPolicyLabRolloutWorker


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_adjust_bottle_xpolicylab_openwam_eval",
)
def main(cfg):
    register()
    cfg = validate_cfg(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = XPolicyLabRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = OpenWAMEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
