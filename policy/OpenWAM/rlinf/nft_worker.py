"""OpenWAM's recorded-joint-state NFT objective on the existing RLinf runner."""

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

from XPolicyLab.policy.OpenWAM.rl import action_nft_loss, restore_replay_pipeline


class OpenWAMNFTWorker(EmbodiedNFTFSDPPolicy):
    def init_worker(self):
        alg = self.cfg.algorithm
        expected = {
            "nft_xcur_source": "rollout",
            "nft_target_space": "x0",
            "nft_tau": 1.0,
            "adv_type": "grpo",
            "nft_loss_form": "mse",
        }
        for key, value in expected.items():
            if alg.get(key) != value:
                raise ValueError(f"OpenWAM first experiment requires {key}={value!r}.")
        if alg.get("recompute_v", False) or self.cfg.rollout.get(
            "recompute_logprobs", False
        ):
            raise ValueError(
                "OpenWAM NFT uses recorded predictions, not log-prob recomputation."
            )
        if self.cfg.actor.model.num_action_chunks != 1:
            raise ValueError("OpenWAM first experiment requires num_action_chunks=1.")
        fsdp = self.cfg.actor.fsdp_config
        if (
            fsdp.strategy != "fsdp"
            or fsdp.sharding_strategy != "no_shard"
            or not fsdp.use_orig_params
        ):
            raise NotImplementedError(
                "First experiment requires FSDP1/no_shard/use_orig_params=true."
            )
        if self._world_size != 1:
            raise NotImplementedError(
                "Validate a single actor rank before enabling distributed training."
            )
        if (
            self.cfg.actor.enable_offload
            or self.cfg.rollout.enable_offload
            or fsdp.cpu_offload
        ):
            raise NotImplementedError(
                "Offload is not validated for OpenWAM's inference-side module references."
            )
        super().init_worker()

    def _precompute_nft_training_inputs(self):
        # The inherited run loop uses prev_logprobs only as transport shapes.
        # Do not call its single-stream schedule or resampling helpers.
        # Time and batch are already flattened. Group the transport keys
        # before the inherited microbatch splitter walks this dict.
        data = restore_replay_pipeline(self.rollout_batch["forward_inputs"])
        required = {
            "nft_pipeline",
            "nft_metadata",
            "nft_actions",
            "nft_x0",
            "nft_v",
            "nft_sigma",
            "nft_valid_dims",
            "nft_video_t",
            "nft_action_t",
        }
        if required - data.keys():
            raise ValueError(
                f"Missing joint replay fields: {sorted(required - data.keys())}"
            )

    def nft_forward_and_loss(self, batch):
        with self.amp_context:
            outputs = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=batch["forward_inputs"],
                compute_values=False,
            )
        loss, metrics = action_nft_loss(
            outputs["v_theta"],
            batch["forward_inputs"],
            batch["advantages"],
            batch["loss_mask"],
            beta=float(self.cfg.algorithm.nft_beta),
            advantage_clip=float(self.cfg.algorithm.adv_clip_max),
            clip_ratio=self.cfg.algorithm.get("nft_clip_ratio", None),
        )
        return loss, {key: value.item() for key, value in metrics.items()}
