"""Experimental on-policy, action-only NFT with recorded joint WAM conditions."""

import json
from typing import ClassVar

import torch
from torch import nn


class JointReplay:
    """Capture one shared denoising index; all stored tensors are batch-first."""

    def __init__(self, step_index):
        self.step_index = int(step_index)
        self.data = None

    def record(
        self,
        *,
        index,
        schedule,
        inputs,
        actions,
        velocity,
        video_timestep,
        action_timestep,
    ):
        if index != self.step_index:
            return
        if velocity is None:
            raise ValueError("Selected step has no action velocity.")
        tensors, metadata = {}, {}
        batch = actions.shape[0]
        for key, value in inputs.items():
            if key in ("prompt", "seed", "input_image"):
                continue
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[0] != batch:
                    raise ValueError(
                        f"Replay input {key} must have batch axis {batch}."
                    )
                tensors[key] = value.detach().cpu().clone()
            else:
                metadata[key] = value
        payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
        if len(payload) > 4096:
            raise ValueError("Replay metadata exceeds 4096 bytes.")
        encoded = torch.zeros(batch, 4096, dtype=torch.uint8)
        encoded[:, : len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
        self.data = {
            "nft_step": torch.full((batch,), index, dtype=torch.int64),
            "nft_pipeline": tensors,
            "nft_metadata": encoded,
            "nft_actions": actions.detach().cpu().clone(),
            "nft_v": velocity.detach().float().cpu().clone(),
            "nft_video_t": video_timestep.detach().cpu().clone(),
            "nft_action_t": action_timestep.detach().cpu().clone(),
        }

    def finish(self, actions):
        if self.data is None:
            raise ValueError("Selected denoising step was not captured.")
        self.data["nft_x0"] = actions.detach().float().cpu().clone()


def action_nft_loss(
    v_theta,
    replay,
    advantages,
    loss_mask,
    *,
    beta=0.1,
    advantage_clip=5.0,
    clip_ratio=0.1,
):
    """NFT x0 energies on executed actions and valid representation dimensions.

    The recorded video state is a fixed condition. This is a conditional
    surrogate, not a likelihood of the complete joint generation path.
    """
    if beta <= 0 or advantage_clip <= 0:
        raise ValueError("beta and advantage_clip must be positive.")
    old = replay["nft_v"].float().detach()
    x = replay["nft_actions"].float()
    target = replay["nft_x0"].float()
    sigma = replay["nft_sigma"].float().reshape(-1, 1, 1)
    valid = replay["nft_valid_dims"].bool()[:, None, :]
    if not valid.any(dim=-1).all():
        raise ValueError("Every sample must have active action dimensions.")
    if not torch.isfinite(sigma).all() or not ((sigma > 0) & (sigma <= 1)).all():
        raise ValueError("Recorded action sigma must be finite and in (0,1].")
    # First experiment executes exactly one action per replan.
    v_theta, old, x, target = [v[:, :1] for v in (v_theta.float(), old, x, target)]
    if not (v_theta.shape == old.shape == x.shape == target.shape):
        raise ValueError("NFT prediction, replay and target shapes must match.")
    denom = valid.sum(dim=(1, 2)).clamp_min(1)
    delta = v_theta - old

    def energies(change):
        values = []
        for sign in (1, -1):
            pred = x - sigma * (old + sign * beta * change)
            error = pred - target
            scale = (error.abs() * valid).sum((1, 2)) / denom
            weight = scale.detach().clamp_min(1e-5).reciprocal()
            values.append((error.square() * valid).sum((1, 2)) / denom * weight)
        return values

    adv = advantages.reshape(v_theta.shape[0], -1)
    mask = loss_mask.reshape(v_theta.shape[0], -1)
    if adv.shape[1] != 1 or mask.shape[1] != 1:
        raise ValueError(
            "First NFT experiment requires one action and one advantage per chunk."
        )
    pref = (adv[:, 0].clamp(-advantage_clip, advantage_clip) / advantage_clip + 1) / 2
    pos, neg = energies(delta)
    loss = (pref * pos + (1 - pref) * neg) * (advantage_clip / beta)
    clip_frac = loss.new_zeros(())
    if clip_ratio is not None:
        if clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive or None.")
        delta_norm = (delta.square() * valid).sum((1, 2)).sqrt().clamp_min(1e-8)
        old_norm = (old.square() * valid).sum((1, 2)).sqrt().clamp_min(1e-8)
        coef = (clip_ratio * old_norm / delta_norm).clamp(max=1)[:, None, None]
        clipped = torch.where(coef < 1, (delta * coef).detach(), delta)
        cpos, cneg = energies(clipped)
        loss = torch.maximum(
            loss, (pref * cpos + (1 - pref) * cneg) * (advantage_clip / beta)
        )
        clip_frac = (coef < 1).float().mean()
    mask = mask[:, 0].to(loss)
    loss = (loss * mask).sum() / mask.sum().clamp_min(1)
    return loss, {
        "actor/nft_loss": loss.detach(),
        "actor/clip_frac": clip_frac.detach(),
        "actor/E_pos_mean": pos.detach().mean(),
        "actor/E_neg_mean": neg.detach().mean(),
        "actor/adv_abs_mean": adv.detach().abs().mean(),
        "actor/zero_adv_fraction": (adv == 0).float().mean(),
        "actor/replay_v_mse": (
            (delta.detach().square() * valid).sum((1, 2)) / denom
        ).mean(),
    }


class OpenWAMNFTPolicy(nn.Module):
    """Register the actual architecture once and replay its action predictions."""

    _no_split_modules: ClassVar[list[str]] = ["OpenWAMNFTPolicy"]

    def __init__(self, architecture, num_steps, valid_dims):
        super().__init__()
        self.architecture = architecture
        self.num_steps = int(num_steps)
        self.register_buffer(
            "valid_dims", torch.as_tensor(valid_dims, dtype=torch.bool)
        )
        self.use_gradient_checkpointing = False
        # Freeze weights only: input gradients through video blocks must survive.
        architecture.requires_grad_(False)
        architecture.action_backbone.requires_grad_(True)
        for backbone in architecture.backbones.values():
            for name, module in backbone.named_modules():
                # Precomputed conditioning encoders need no replay gradients.
                if name.split(".")[0] in ("vae", "text_encoder"):
                    continue
                if getattr(module, "_openwam_no_grad_wrapped", False):
                    raise ValueError(
                        "Joint backbone was no_grad-wrapped; freeze weights without disabling autograd."
                    )

    def train(self, mode=True):
        super().train(mode)
        # Disable stochastic layers in both rollout and replay, keep autograd on.
        self.architecture.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.use_gradient_checkpointing = True

    def sync_runtime(self):
        param = next(self.architecture.action_backbone.parameters())
        self.architecture._device, self.architecture._dtype = param.device, param.dtype
        for backbone in self.architecture.backbones.values():
            if hasattr(backbone, "_device"):
                backbone._device = param.device
            if hasattr(backbone, "_dtype"):
                backbone._dtype = param.dtype

    def forward(self, replay):
        self.sync_runtime()
        param = next(self.architecture.action_backbone.parameters())
        rows = replay["nft_metadata"].detach().cpu()
        if not torch.equal(rows, rows[:1].expand_as(rows)):
            raise ValueError("Mixed replay metadata within a minibatch is unsupported.")
        metadata = json.loads(bytes(rows[0].tolist()).rstrip(b"\0"))
        pipeline = dict(metadata)
        for key, value in replay["nft_pipeline"].items():
            pipeline[key] = value.to(
                device=param.device,
                dtype=param.dtype if value.is_floating_point() else value.dtype,
            )
        _, velocity = self.architecture(
            replay["nft_actions"].to(param),
            replay["nft_action_t"].to(param),
            **pipeline,
            timestep=replay["nft_video_t"].to(param),
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        if velocity is None:
            raise RuntimeError("OpenWAM forward returned no action prediction.")
        return {"v_theta": velocity}
