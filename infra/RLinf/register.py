"""Register the XPolicyLab embodied adapter with RLinf."""

from __future__ import annotations


def register() -> None:
    """Register ``model_type: xpolicylab`` in this process."""
    from omegaconf import DictConfig

    from rlinf.models import register_model

    def _build_xpolicylab(cfg: DictConfig, torch_dtype):
        from XPolicyLab.infra.RLinf import get_model

        return get_model(cfg, torch_dtype)

    register_model(
        "xpolicylab",
        _build_xpolicylab,
        category="embodied",
        force=True,
    )
