"""OpenWAM NFT worker and the RoboTwin end-effector action boundary."""


def actor_worker(cfg):
    if cfg.actor.model.xpolicylab.required_mode == "nft":
        from .nft_worker import OpenWAMNFTWorker

        return OpenWAMNFTWorker


def env_class(env_cfg):
    if env_cfg.env_type == "robotwin":
        from .robotwin_env import OpenWAMRoboTwinEnv

        return OpenWAMRoboTwinEnv


def validate(cfg, model, env, mode):
    if mode == "train" and model.xpolicylab.required_mode == "nft":
        if cfg.algorithm.loss_type != "embodied_nft":
            raise ValueError("OpenWAM NFT requires embodied_nft loss.")
    from XPolicyLab.infra.RLinf.core.extensions import simulator_extension

    extension = simulator_extension(env)
    simulator = extension.NAME if extension else env.env_type
    benchmark = str(model.xpolicylab.get("openwam_benchmark") or "robodojo").lower()
    if benchmark != simulator:
        raise ValueError(f"OpenWAM calibration must match the {simulator} simulator.")
