# OpenPI PyTorch 训练核对

审计对象是 Pi_05 在 RoboTwin、RoboDojo 的 PPO、GRPO 和共用模型权重的独立 PyTorch SFT。Pi_0 和 Pi_0_Fast 的 XPolicyLab 策略入口没有声明 PPO/GRPO。上游的原生 Pi0 和 Pi0-FAST 是不同的算法入口。

既有 Pi_05 工作区代码已经复用 RLinf 的 `Pi0RL._predict_train`、`_predict_eval` 和 `default_forward`。训练采样保留去噪链、采样步以及变换后的三路 RGB、状态和提示词。Actor 用这些张量重算 likelihood 和 critic。任务坐标变换、归一化统计量及官方旧 checkpoint 键名转换由适配层衔接。

修复前的差异包括通用模型名绕过 OpenPI 的 FSDP 规则和重算白名单，以及 SFT 和 RL 对合法分片的额外限制。以原生 `EmbodiedFSDPActor.train_micro_batch`、`validate_fsdp_cfg` 和 `Pi0.sft_forward` 为准。SFT 应使用当前上游的全局 RNG 图像增广，并保持配置选定的 gradient checkpointing。RL 的 rollout 和 replay 都使用无随机增广的预处理。

验证分三层。CPU 比较 rollout/replay 的 log-prob 和 value，并在更新后检查概率变化。配置测试检查原生模型分支、分片及 FP32 模块的保护。GPU 检查真实 checkpoint 的采样、重算和梯度。各层的执行结果见缺陷记录。

## 训练边界

- **配置和分派。** XPolicyLab 的 openpi 注册项仅在配置指定 Pi_05 时构造任务桥接模型。普通 OpenPI（包括独立 SFT）仍交给原生工厂。PPO/GRPO 的采样模式由 HF worker 传递。训练配置可以通过 `algorithm.recompute_logprobs=true` 启用更新前的额外校验重算，默认每个 microbatch 的可微前向始终存在。
- **分布式和精度。** FSDP1 的原生校验处理 full_shard、shard_grad_op、no_shard、use_orig_params 和低精度模块规则。参数转换和冻结复用上游 `_apply_openpi_param_dtypes`、`_freeze_after_load`。小型 SigLIP/Gemma 的 SFT→RL 测试覆盖 FP32 参数和 BF16 激活组合。[上游分片变更](https://github.com/RLinf/RLinf/commit/30e0236981c6a1dc19202de64116834c12673828)
- **梯度和恢复。** PPO/GRPO 通过原生 actor 聚合 loss、反传并更新。梯度裁剪使用当前 RLinf 的 FSDP 实现。独立 SFT 继承原生 worker 的训练步及 checkpoint，并额外保存任务统计量。上游对低精度梯度原地裁剪、SFT 无显式生成器时的图像增广，以及 DCP 每个 rank 的 RNG 恢复均直接来自锁定子模块。[梯度裁剪](https://github.com/RLinf/RLinf/commit/fab7f407fbc611085eb163962e3c89c0ab424c1a) · [SFT 增广](https://github.com/RLinf/RLinf/commit/ff903c5f094498e3e9e7268d54e51037eb1f33b4) · [DCP RNG](https://github.com/RLinf/RLinf/commit/dec2377bb8fa726827b36b12bb99cbfba12657dd)
- **任务差异。** XPolicyLab 保留任务 tokenizer、三路相机次序、关节增量变换和 norm_stats。Replay 保存已变换的张量；官方入口通常保存原始观测并再次变换。数值检查比较的是同一批样本的 rollout 和 actor 输出。CPU 回归覆盖三个随机采样器、joint/single-step likelihood、critic 梯度，以及无 critic 时使用原生 actor loss 的 GRPO 更新。PPO 默认从 VLM 取 value；suffix critic 只允许 joint_logprob，避免 rollout 与重算平均不同去噪步的 value。独立 SFT 与 RL 训练期间混入 SFT 数据是两个入口，本次验证范围为独立 SFT 及 PPO/GRPO。

以 PyTorch 2.11.0 CPU 直接验证了两项更新。对 BF16 参数设置四个值均为 2 的梯度，以 total_norm=4、max_grad_norm=1 调用 RLinf 的 clip_grad_by_total_norm_，原始 .grad 的四个值均变为 0.5。对同一幅 224×224 RGB 渐变图调用 preprocess_observation，不传 generator；训练模式下种子 8 可复现，换到 9 会改变图像，评估模式下两个种子的结果逐元素相同。

## 双卡数值结果

L20-3 物理 GPU 6、7，PyTorch 2.11.0+cu128，两个 NCCL rank，用现有 `pi05_robotwin_lerobot_v30_xpolicylab_pytorch` 权重、真实 tokenizer 及 RoboTwin_lerobot_v30 的 norm_stats。配方为 RoboTwin Pi0.5 PPO，保留 flow_sde、5 个去噪步、冻结 VLM 和 VLM critic，每个 rank 使用一条合成三视角观测，执行动作块缩为 4×14。测试经扩展注册和公开 get_model 入口构造模型，用 FSDP1 FULL_SHARD、上游 wrap policy 及 actor_critic loss 反传，再执行一次 AdamW 更新。

1. **采样。** 固定每个 rank 的 Python 和 Torch 种子为 100 + rank。相同变换后的观测分别通过完整 XPolicyLab 入口和直接调用原生 Pi0RL._predict_train。逐元素检查环境动作、prev_logprobs、prev_values，两个 rank 的绝对误差均为 0。
2. **Actor。** 针对同一条去噪链，比较适配器 FSDP 前向与独立重新加载、直接包裹 backbone 的原生 FSDP 前向。log-prob 和 value 的最大绝对差在两个 rank 上均为 0。这里共用了任务变换与权重加载，以隔离训练桥接造成的影响。
3. **Rollout 与 actor 是另一项比较。** 更新前，同一权重在两条路径间的单元素最大 log-prob 差分别为 0.00485480 和 0.02795696，最大 value 差均为 0.00439501。原生未分片 actor 也复现了相同的 log-prob 差。首次探针以 0.0002 检查 rollout 对 actor，阈值未通过；随后增加了原生对照，未将该差异当作零。
4. **更新。** 两个 rank 都完成反向传播。裁剪前全局梯度范数约 228.0963，AdamW step 后两个 rank 的 log-prob 均有变化。这轮合成观测验证覆盖一次训练更新。后续 Ray 驱动的真实仿真、存档和恢复见下一节。

## RoboTwin adjust_bottle 真实仿真与断点恢复

2026-11-04 在 L20-3 使用当前工作区和 RLinf `e2aeeadd`，通过正式入口 `train_rl.py` 启动独立 Ray 集群。Pi0.5 使用 `pi05_robotwin_lerobot_v30_xpolicylab_pytorch` 初始权重、真实 tokenizer 和 `RoboTwin_lerobot_v30` 统计量。模拟器为 RoboTwin `RLinf_support` 提交 `0008ae6`，使用 Aloha AgileX 双臂、MPLib、三路 RGB 与 14 维关节动作。场景使用真实 SAPIEN 渲染，隔离的模拟器副本将渲染设备显式绑定到 env worker 的 CUDA 0，并在建场景时校验物理 PCI 为 `0000:e2:00.0`。

- **设备和批次。** 物理 GPU 6 运行一个 actor rank，GPU 7 运行 rollout 与环境。PyTorch 2.11.0+cu128 的 FSDP1 在单 actor rank 上实际使用 `NO_SHARD`。每条轨迹 200 步，每个动作块 50 步，5 个去噪步。PPO 使用 2 个环境、GAE 和 value head，rollout_epoch=1、global_batch_size=4。GRPO 多组验证使用 8 个环境、group_size=8、rollout_epoch=4、global_batch_size=32，每个训练步采集 32 条轨迹。两个算法均使用 update_epoch=1、micro_batch_size=1 和 algorithm.recompute_logprobs=true。
- **恢复方法。** 各算法第一次运行设置 `runner.max_steps=1 runner.save_interval=1`。训练进程退出后重新启动，设置 `runner.max_steps=2 runner.resume_dir=<实验目录>/<算法>/checkpoints/global_step_1`。存档格式为 PyTorch DCP，保存模型、优化器、调度器和各 rank 的 RNG。实验关闭额外的全量权重导出。核验时直接读取 DCP 内的动作输出层、Adam 一阶矩和 step，同时核对独立进程日志、仿真视频和环境回报。
- **复现位置。** L20-3 持久目录为 `/personal/zijian/test/RLinf/RoboTwin/XPolicyLab/infra/RLinf/results/adjust_bottle_pi05_20261104.FLaptA`，保存 RLinf 日志、checkpoint 和视频。本节提到的运行脚本、证据文件及 `diagnostics/` 位于 `/personal/zijian/test/RLinf/validation-artifacts/RoboTwin/adjust_bottle_pi05_20261104.FLaptA/`。其中 `source.tgz` 为训练源码快照，SHA-256 为 `89ed5cc1e67b708617216a79a39f04ac5d250afc95f4db72ed3358bc6bbf7dd2`。`prepare.py` 准备模拟器渲染绑定，`run_phase.sh` 和 `run_grpo_phase.sh` 分别记录 PPO 与多组 GRPO 的 Hydra 覆盖项，`verify_results.py` 从日志、视频和存档提取证据。

### PPO

首轮 2/2 条轨迹成功，平均回报 1.0，actor 梯度范数 361.0，保存 `ppo/checkpoints/global_step_1`。独立进程的恢复日志明确指向这份存档；恢复后再次完成 2 条轨迹，2/2 成功，梯度范数 339.2，并保存 `ppo/checkpoints/global_step_2`。

DCP 内 `backbone.action_out_proj.weight` 的 Adam step 从 2 增至 4。该层对初始权重的首轮最大绝对变化为 1.00136×10⁻⁵，恢复后相对第 1 步又变化 1.00043×10⁻⁵（32,768 个元素中 32,767 个改变）。每份存档含 9,177,008,180 字节。两次视频各 5 帧，对应初始观测及 4 个动作块；解码后的首末帧像素平均绝对差分别为 29.59 和 32.33。完整逐步记录见远端 `ppo-evidence.json`。

### GRPO

首个诊断配置在单组中采集 4 条轨迹，4/4 成功。组内回报一致，原生奖励过滤将整组剔除：loss_mask_fraction=0、grad_norm=0、Adam 一阶矩最大绝对值为 0。虽然存档存在且 Adam 计数增加，动作头变化只有约 5.96×10⁻⁸，本次诊断不计入有效更新。诊断日志、视频和存档的数值核验记录保留在 `diagnostics/single-group/`，零梯度存档已清理。随后将组大小和每步采样轮数恢复为工作区标准配方的 8 和 4，从原始任务权重重新开始。成功奖励、噪声幅度和组回报过滤阈值均保持原配方设置。

多组首轮共 32 条真实轨迹，29 条成功。三个组均为 8/8，另一个组为 5/8；混合回报组参与更新，有效掩码比例 0.211，actor 梯度范数 97.21。动作输出层相对初始权重的最大绝对变化为 1.99983×10⁻⁵，Adam step 为 4，一阶矩最大绝对值 0.01416。首轮存档在 `grpo/checkpoints/global_step_1`。


恢复阶段的新进程加载上述 DCP 后，又完成 32 条轨迹，31 条成功。三个组为 8/8，另一个组为 7/8；有效掩码比例 0.195，actor 梯度范数 66.745。第二份存档在 `grpo/checkpoints/global_step_2`。动作头的 Adam step 从 4 增至 8，相对第 1 步有 32,767/32,768 个权重元素变化，最大绝对变化为 1.84393×10⁻⁵。每份 GRPO 存档含 9,143,872,397 字节，优化器、调度器及 RNG 均可在 DCP 元数据中定位。四组视频在两个阶段分别写入 `grpo/video/phase-1/` 和 `grpo/video/phase-2/`，详见远端 `grpo-evidence.json`。

另从四份 DCP 直接读取了 `fsdp_checkpoint.lr_schedulers.0.last_epoch`，PPO 和 GRPO 均从 1 延续到 2。两种算法的最终可恢复目录分别是 `ppo/checkpoints/global_step_2`、`grpo/checkpoints/global_step_2`。

实验结束后清理了节点内存盘上的临时副本；远端保留四份有效 DCP、源码包、启动脚本、训练日志、视频及证据 JSON。以上成功数来自训练采样，其中 PPO 共 4 条、有效 GRPO 实验共 64 条。抽查了 PPO 恢复阶段及 GRPO 首轮视频，首帧可见瓶身横置，后续帧可见夹爪抓住瓶身并调整姿态。此次验收覆盖真实 rollout、梯度更新、DCP 保存和新进程恢复；独立评估成功率和长期收敛仍需另行测量。
