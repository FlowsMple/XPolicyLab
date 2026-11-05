# XPolicyLab · RLinf

在仿真宿主的 XPolicyLab 根目录运行 `bash infra/RLinf/scripts/run.sh <配置名>`。训练与评估共用入口，由配置中的 `runner.only_eval` 选择。配置名和 Hydra 覆盖参数可直接沿用。

- [Pi0.5 和 Abot 的 PPO、GRPO、断点恢复](doc/PI05_ABOT_QUICKSTART_zh.md)
- [OpenWAM 的 NFT、宿主切换、录像与结果目录](doc/OPENWAM_QUICKSTART_zh.md)
- [Pi0.5 的 PyTorch SFT](doc/PI05_SFT.md)
- [英文使用手册](doc/GUIDE.md) · [策略、仿真扩展与目录迁移](doc/EXTENSIONS.md)
- [RLinf 0.4 迁移](doc/MIGRATION_0_4.md) · [OpenPI 对齐](doc/OPENPI_TRAINING.md) · [适配审查](doc/ADAPTER_REVIEW.md) · [训练记录](doc/TRAINING_RESULTS.md)

`core/` 定义模型、观测和配置接口，`workers/` 连接 RLinf。仿真实现在 `envs/`，策略专用实现和配方在 [policy/](../../policy)。`scripts/` 放启动入口，`tests/` 放回归测试和独立验证工具，`doc/` 放文档。`RLinf/` 是锁定版本的上游子模块。训练结果由 RLinf 按 `runner.logger.log_path` 写入。
