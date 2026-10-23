# OpenWAM 接入 RLinf：评估结论与适配计划

更新日期：2026-10-23。

评估对象：`policy/OpenWAM/`、其内置 OpenWAM 实现、`infra/RLinf/` 桥接层，以及本地 RLinf 原项目的模型接入代码。

代码基线：XPolicyLab `b2594786`；RLinf `0067f7d5ce525c18bda439ed8dbb638abafd6a06`，后者提交日期为 2026-10-08。结论以此次检查的本地源码为准，不代表上游最新版本已经完成相同接入。

本文主体保留适配前的源码评估与设计依据；其中“当前未实现”等表述对应上述基线。2026-10-23 已加入首轮实验代码，最新实现状态见下一节；仍未通过 GPU 验证。

## 本次实施更新（2026-10-23）

已按 RoboTwin EEF → 动作分支 NFT 路线实现首版：显式 NFT 能力门禁、真实模型注册、联合去噪状态记录/重放、有效维度和首个执行步损失、RoboTwin EE 环境、独立评估入口、短训练配置及预检脚本。RLinf 子模块不变；内置 OpenWAM 增加可选 observer。

完整操作手册：[OpenWAM RLinf 实验步骤](policy/OpenWAM/RLINF_EXPERIMENTS.md)。其中包含依赖与源码准备、四元数语义核对、真实观测采集、GPU 梯度预检、原始基线、两轮 smoke、checkpoint 恢复、训练后对比和反馈模板。

本地验证为 30 个 CPU 检查及配置展开；真实权重、GPU、FSDP、仿真、学习效果尚未运行。首版限制为 DualSystem joint self-attention、同步 CFG=1、每次执行一个 EE 动作、单 actor rank/no_shard、同一 Python 环境。PPO/GRPO 概率重算、SFT 与多机训练没有在本次实现。

原生 RoboTwin 与 OpenWAM helper 的四元数约定存在待核对差异，因此 runtime 顺序必须显式指定；转换单测通过不代表实际 checkpoint 的数据语义已确认。请先完成实验手册 A–C，再启动长训练。

2026-10-23 补充：参考上游 [RL for Video Generation Models](https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/video_models.rst)，增加 Diffusion-NFT 候选训练路线；随后检查本地具身 NFT worker、OpenPI NFT 示例和 XPolicyLab 转发接口，补充第 6.7 节的可行性复评。上游 `main` 链接可变；本地静态结论仍对应上述代码基线，不代表通过了运行验证。

## 1. 目标、范围与结论

**目标是在现有 XPolicyLab → RLinf → RoboTwin 链路中接入 OpenWAM，逐步支持推理评估和在线强化学习。**

当前 XPolicyLab RLinf 示例以 RoboTwin 为环境，Pi_05、Abot_M0 的配置使用 `aloha_agilex`、`action_type: joint`、14 维执行动作。本次工作应沿用这条环境链路，不以新增 RoboDojo 环境为前提。

可行性判断：

- **推理评估可接入。** OpenWAM 已有原生 RoboTwin 接口和批量生成能力，但当前 XPolicyLab OpenWAM 封装面向 RoboDojo，不能直接作为 RoboTwin RLinf 策略使用。
- **SFT 有可复用基础。** 原生架构提供输入准备和视频/动作 flow-matching loss。接到 RLinf 仍需补数据入口、模型注册、分布式调用及 checkpoint 管理。
- **GRPO 原理上可行，当前尚未实现。** 必须新增随机采样、采样信息保存及带梯度的概率重算，并处理视频动作耦合。不能通过更改模式声明直接获得训练能力。
- **PPO 可在上述基础上扩展。** 还需要 value head、bootstrap 和 GAE 对齐。
- **Diffusion-NFT 是待验证的另一条训练路线。** 可研究以 RoboTwin 实际执行奖励指导联合视频动作去噪更新，避免预先把 PPO/GRPO 的概率比值目标当作唯一选择；尚不能认定其支持 OpenWAM，或能直接复用媒体生成训练入口。详见第 6.6 节。
- **NFT 已有可复用的具身训练基础。** 本地包含 `EmbodiedNFTFSDPPolicy`、OpenPI NFT 配置及 XPolicyLab `nft_forward` 转发。缺口集中在 OpenWAM 双流状态、schedule、有效维度及能力门禁，不能再把 NFT 描述为只有媒体生成参考。建议在推理和梯度通过后优先做动作分支 NFT 最小实验，保留概率重算路线；此优先级是工程判断，不是学习效果结论。
- **优先复用现有外部注册与桥接机制。** 不需要先向 RLinf 内置模型列表加入 OpenWAM；是否需要扩展桥接或 worker 的维度处理，要由实际 checkpoint 和概率目标决定。

### 1.1 对先前评估的修正

此前提到 RoboDojo，是因为当前 [OpenWAM Model](policy/OpenWAM/model.py) 和 [train.sh](policy/OpenWAM/train.sh) 专门封装了 RoboDojo dual-X5、绝对 EE 控制，并使用该环境的标定。它们描述的是现有封装的约束，不是本次 RLinf 的目标环境。

因此：

1. 不把新增 RoboDojo/IsaacLab 环境列为接入步骤。
2. 不把 RoboDojo 的 dual-X5 坐标变换用于 RoboTwin。
3. 不把 80 → EEF20 → EE16 当成所有接入方案的固定流程；是否需要这条转换由 checkpoint 决定。
4. 也不因为现有 RLinf 示例是 joint14，就认定 OpenWAM 发布权重同样输出 joint14。

## 2. 已确认事实与尚未确认事项

### 2.1 已确认事实

- 现有 [RLinf 启动脚本](infra/RLinf/run.sh) 配置了 `RLINF_EXT_MODULE=XPolicyLab.infra.RLinf.register`，使用外部注册加载 `model_type: xpolicylab`。
- [桥接模型构建器](infra/RLinf/__init__.py) 加载 `XPolicyLab.policy.<policy_name>.model.Model`；它不会自动读取并合并该策略的 `deploy.yml`，模型必要字段必须由 RLinf 配置或模型默认值提供。
- [桥接包装器](infra/RLinf/action_model.py) 从策略对象的 `action_type`、`robot_action_dim_info` 读取数据契约，从 `policy` 或 `model` 注册可见的 `nn.Module`。
- 当前 OpenWAM `Model` 只在局部变量中保存动作类型和维度信息；实际网络由 `_engine` 间接持有，没有暴露在上述标准属性上。
- 当前 OpenWAM 没有覆盖 RL 采样、概率重算、SFT 接口，继承的模式声明为 `RL_MODES = ("eval",)`。
- 原生 [RoboTwin dataloader 配置](policy/OpenWAM/OpenWAM/configs/dataloader/robotwin.yaml) 支持 joint 与 EEF，但默认是 `eef`，并使用统一 80 维空间。
- 原生 [RoboTwin 评估说明](policy/OpenWAM/OpenWAM/benchmarks/robotwin/README.md)将发布权重描述为 EEF 模式；[policy_config.yml](policy/OpenWAM/OpenWAM/benchmarks/robotwin/policy_config.yml)默认 `action_type: ee`、`state_dim: 20`。
- 原生 OpenWAM RoboTwin 客户端可将 EEF20 转为 EE16，交给 `take_action(action_type="ee")`。这是原生客户端的能力，不等于当前 RLinf RoboTwin VectorEnv 已支持相同模式。
- 当前 OpenWAM 批量部署路径在 `no_grad` 下运行，只返回最终反归一化动作，没有提供 RL 所需的采样概率和重算输入。

### 2.2 尚未确认事项

- 实际要使用的 checkpoint 路径、训练数据、机器人、joint/EEF 模式、统计量、统一空间映射。
- 实际 checkpoint 的架构变体、跨模态 attention mask、proprio 使用方式及推理 schedule；源码默认值不能替代 checkpoint 中保存的配置。
- 当前安装的 `robotwin.envs.vector_env.VectorEnv` 能否返回 EE 状态、选择 EE 执行模式，以及终止前实际执行多少步。
- OpenWAM 在目标软件环境中的 import、加载、单卡推理、梯度和分布式可用性。
- 吞吐、显存、训练稳定性和最终成功率。本文不提供未经实测的资源配置或工期保证。

## 3. 阅读到的现有模型接入模式

### 3.1 OpenPI：主要算法参考

关键源码：

- [OpenPI RL 策略](infra/RLinf/RLinf/rlinf/models/embodiment/openpi/tasks/rl.py)：`_predict_train`、`default_forward`。
- [OpenPI RL sampler](infra/RLinf/RLinf/rlinf/models/embodiment/openpi/sampling/rl_sampler.py)：转移均值/标准差、Gaussian log-prob。
- [XPolicyLab Pi_05 模型](policy/Pi_05/model.py)与 [RL 实现](policy/Pi_05/rl.py)。

可借鉴内容：

- 将训练随机采样与评估采样分开。
- 保存动作去噪链、选中的去噪步、条件输入及模型空间动作。
- actor 使用相同样本重新计算 log-prob，梯度回到真实网络。
- PPO 增加价值估计；GRPO 可以不构建 value head。

不能直接复制的部分：OpenPI 的时间网格、prefix cache、有效动作切片，不能默认适用于 OpenWAM 的 shifted sigma schedule 和联合视频动作架构。

### 3.2 GR00T / ABot：flow policy 与桥接参考

关键源码：

- [GR00T N1.6](infra/RLinf/RLinf/rlinf/models/embodiment/gr00t/gr00t_n1d6/gr00t_action_model.py)：随机去噪、链保存、概率重算和 value head。
- [RLinf ABot action head](infra/RLinf/RLinf/rlinf/models/embodiment/abot_m0/action_head_rl.py)：`get_rl_action`、`forward`。
- [XPolicyLab Abot_M0 模型](policy/Abot_M0/model.py)与 [RL 实现](policy/Abot_M0/rl.py)。

它们证明现有桥接可以承载具有真实概率重算能力的 flow policy。OpenWAM 应复用这类接口组织方式，但仍需实现自己的联合条件和采样数学。

另外，RLinf actor 对部分内置模型有专门的 `prev_logprobs` 参数传递和输出处理。`model_type: xpolicylab` 不会自动获得所有内置模型分支，移植时应遵守当前通用桥接契约。

### 3.3 FastWAM / DreamZero / Cosmos3：工程参考与能力边界

- [FastWAM](infra/RLinf/RLinf/rlinf/models/embodiment/fastwam/fastwam_policy.py)实现推理和 SFT；其 `default_forward` 不实现 PPO/GRPO 重算。
- [DreamZero](infra/RLinf/RLinf/rlinf/models/embodiment/dreamzero/dreamzero_policy.py)实现推理和 SFT；rollout 中的零 log-prob 不能用于证明策略梯度可训练。
- [Cosmos3](infra/RLinf/RLinf/rlinf/models/embodiment/cosmos3/cosmos3_policy.py)主要提供 SFT forward，推理另走服务端适配。

这些实现适合参考视频模型的加载、SFT、精度和部署组织方式；不能据此认定 OpenWAM 只需增加一个 wrapper 就能支持在线 RL。

### 3.4 OpenPI NFT：比媒体生成更直接的具身训练参考

- 本地 [OpenPI NFT 配置](infra/RLinf/RLinf/examples/embodiment/config/libero_object_0_nft_openpi.yaml)在 LIBERO 环境使用 `loss_type: embodied_nft`、组内优势和 `flow_ode`；上游 [同名配置](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_object_0_nft_openpi.yaml)也可查阅。这说明 NFT 并不限于 OCR 单步环境，但 OpenPI 并非 OpenWAM 的联合视频动作架构。
- [具身 NFT worker](infra/RLinf/RLinf/rlinf/workers/actor/fsdp_nft_policy_worker.py)已有奖励优势到去噪更新的训练循环及参考权重管理。[XPolicyLab 训练入口](infra/RLinf/train_rl.py)已经选择该 worker。
- [桥接包装器](infra/RLinf/action_model.py)已有 `ForwardType.NFT` 分发和 `nft_forward` 转发；[Pi_05 模型](policy/Pi_05/model.py)和 [RL 模块](policy/Pi_05/rl.py)已有对应的状态保存与速度预测实现。
- 本地 [VidAR action-similarity 配置](infra/RLinf/RLinf/examples/diffusion/config/vidar_robotwin_nft_action_similarity.yaml)提供另一种具身数据奖励参考，但其环境配置的 `env_type` 为 `diffusion`；[奖励实现](infra/RLinf/RLinf/rlinf/envs/sim/diffusion/rewards/embodied/action_similarity.py)比较生成视频经 IDM 预测的动作与数据集动作。它不是在 RoboTwin 仿真中执行动作后按任务成功给奖励的在线闭环。

上述结论来自配置和源码检查，没有运行这些示例，也不据此宣称 WAM 在线 RL 已经验证。

## 4. 必须先决定的动作接入路线

### 4.1 路线 A：匹配的 RoboTwin joint checkpoint

适用前提：实际 checkpoint 确实以目标 RoboTwin 机器人的 joint 动作为训练目标，proprio、维度、顺序和统计量也匹配。

这条路线最接近当前 Pi_05 / Abot_M0 的 joint 示例。对于现有 aloha-agilex 示例，执行动作是双臂关节与夹爪共 14 维；其他机器人仍应通过共享维度函数解析，不能将 14 硬编码为 OpenWAM 通用常量。

需要确认：

- 模型原始动作宽度与统一空间宽度；joint checkpoint 也可能使用统一空间。
- joint 的 `unify_action_map`。当前 EEF 的 `0:10`、`34:44` 映射不能原样用于 joint。
- 使用 checkpoint 保存的 joint 统计量，不使用 RoboDojo 或 EEF 统计量替代。
- 夹爪值、关节顺序以及执行接口是否与现有 VectorEnv 一致。

**仅修改 YAML 的 `action_mode` 或 `action_dim`，无法把 EEF 权重变成 joint 权重。** 如果没有可用的 joint checkpoint，需要另做数据/权重适配或 SFT，并评估行为质量。

### 4.2 路线 B：RoboTwin EEF checkpoint

这条路线与当前 OpenWAM 原生发布权重说明更一致，但要扩展现有 RoboTwin 链路的 EE 支持。

需要明确三层表示：

1. 模型空间：以 checkpoint 为准，默认配置是统一 80 维。
2. 原始策略空间：EEF20，每臂 `xyz3 + rot6d6 + gripper1`。
3. 环境执行空间：EE16，每臂 `xyz3 + quaternion4 + gripper1`。

必须处理四元数顺序：

- 原生 OpenWAM RoboTwin [转换工具](policy/OpenWAM/OpenWAM/benchmarks/utils/action_conversion.py)使用 `xyzw`。
- XPolicyLab 标准 EE pose 使用 `wxyz`。
- 若经 XPolicyLab 标准观测/动作字典传递，需在明确的 RoboTwin 边界完成顺序转换；`pack_robot_state` / `unpack_robot_state` 只打包和切片，不会自动转换四元数顺序或坐标系。

原生 [RoboTwin 客户端](policy/OpenWAM/OpenWAM/benchmarks/robotwin/openwam2robotwin_interface.py)是状态提取、相机映射、prompt 和动作转换的参考。其存在并不意味着 RLinf 当前的 `states` 已包含 EEF 信息。

实施时应扩展或配置现有 RoboTwin VectorEnv/环境适配，使其返回匹配的 EE proprio 并正确执行 EE 动作；不需要另建 RoboDojo 环境。不要复用 `arx_x5_calibration()` 的硬编码基座变换。

## 5. 建议的模型与桥接结构

### 5.1 模型侧职责

建议在 `policy/OpenWAM/` 下增加专门的 RL 模块，例如 `rl.py`，组织以下职责。该文件目前尚未创建。

- 一个真正的 `nn.Module`，注册 OpenWAM architecture，以及可选的 value head。
- checkpoint 驱动的归一化、统一空间映射和有效维度 mask。
- RoboTwin 条件构建：三相机、指令、joint 或 EEF proprio。
- 训练采样、评估采样和带梯度的概率重算。
- 原生 SFT loss 的转发，可用于离线训练或最小梯度诊断。

`Model` 应暴露 `self.action_type`、`self.robot_action_dim_info`，并把上述模块放到 `self.policy` 或 `self.model`。不要把非 `nn.Module` 的 engine 当成可训练 backbone，也不要另外加载一份只用于采样的网络而失去权重同步。

现有 RoboDojo serving 行为如需保留，应通过明确的 benchmark/契约分支选择，避免加入 RoboTwin 后破坏原来的部署入口。

### 5.2 对外接口

按 [ModelTemplate](model_template.py)和[当前 wrapper](infra/RLinf/action_model.py)实现：

以下 RL 接口主要针对现有概率重算路线；NFT 若被选用，应优先沿用已有 `nft_forward` 分发并扩展其数据契约，核对现有 loss 的假设，不应伪造 log-prob 来套用 PPO/GRPO。

- `predict_action_batch_rl(obs_list, mode, compute_values, **kwargs)`：返回环境动作 `[B,H,D_env]` 和 extras。
- extras 包含 `prev_logprobs`、PPO 所需的 `prev_values [B,1]`、以及 batch-first tensor 组成的 `forward_inputs`。
- `compute_logprob_value(forward_inputs, compute_values, **kwargs)`：重算相同样本，返回 `logprobs`、可选 `values [B]`、约定形状的 entropy。
- `sft_forward(...)`：若实现，按所选 SFT worker 的契约返回 loss；当前 XPolicyLab 模板描述为标量，部分上游模型返回 loss dict，不能混用后假定自动兼容。
- 仅在对应路径实现并验证后，向 `RL_MODES` 声明 `grpo`、`ppo` 或 `sft`。

当前 wrapper 的 rollout 动作经 NumPy 转换后变成 tensor，因此训练采样返回动作应遵守其 CPU/NumPy 边界；actor 梯度应通过保存的 `forward_inputs` 重算，不能期待环境动作张量保留 rollout 计算图。

### 5.3 桥接与环境侧职责

- [obs.py](infra/RLinf/obs.py)目前将主相机映射为 `cam_high`；需让 OpenWAM 的 RoboTwin 分支识别该契约，或增加明确可配置的相机映射。
- wrist 顺序保持左、右，与训练时拼图位置一致。
- 图像保持 RGB；不增加 BGR/RGB 交换，也不在 `model.py` 手工解码。
- RoboTwin 原生 `qpos`、OpenWAM dataloader 的 `joint`、XPolicyLab 的 `joint` 是不同层的命名，需要显式映射。
- RLinf 的 [RoboTwin action 分支](infra/RLinf/RLinf/rlinf/envs/action_utils.py)当前直接透传动作，不自动执行 EEF20→EE16 或四元数重排。
- [RoboTwinEnv](infra/RLinf/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py)使用外部 `robotwin.envs.vector_env.VectorEnv`。实际状态语义、执行模式和进程隔离能力要在目标安装版本核对，不能从启动脚本注释推断已具备。

## 6. RL 采样与训练目标的设计要求

第 6.1 节的随机转移与概率重算要求针对 PPO/GRPO 概率比值路线；第 6.2～6.5 节中的联合条件、梯度、表示和执行长度问题也需在 NFT 路线中重新落实。NFT 候选方案见第 6.6 节。

### 6.1 当前推理路径不提供可用的 RL log-prob

源码入口：

- [engine.generate_batch](policy/OpenWAM/OpenWAM/openwam/deploy/engine.py)。
- [BaseWAMArchitecture](policy/OpenWAM/OpenWAM/openwam/model/architectures/base.py)中的 `generate_batch`、`_run_joint_denoising`。
- [ActionScheduler](policy/OpenWAM/OpenWAM/openwam/model/action_backbone/scheduler.py)。

当前循环是 flow-matching Euler 更新，部署接口在 `no_grad` 下执行，并把最终动作转成 NumPy、反归一化。不能将 SFT MSE 的负值、任意常数或初始噪声概率充当最终动作的 log-prob。

训练路径需显式定义随机转移，例如动作去噪某一步的条件 Gaussian；其均值、标准差、采样与重算必须来自同一个定义。OpenWAM 的 shifted sigma、视频/动作时间步及最后一步行为应保持一致，不能直接套 OpenPI 的均匀时间网格。

当前适配器未传 seed，批量生成底层每样本默认 seed 为 42。训练必须提供独立且可恢复的随机流，并与环境 reset seed 区分；相同观测下始终复用相同噪声会削弱 GRPO 组内探索。

### 6.2 联合视频动作条件

默认 [dual_system 配置](policy/OpenWAM/OpenWAM/configs/model/dual_system.yaml)使用联合 self-attention 和 `mutual` mask。实际 checkpoint 必须另行确认。

在该模式下，每层的视频和动作 token 一起做 attention，每轮同时推进两个 latent。因此动作转移重算通常需要当时的视频 latent、动作 latent、两个时间步、文本条件、首帧条件及 proprio。

可评估的方案包括保存选中转移的完整输入，或保存能够按明确规则重建联合状态的轨迹信息。只保存动作链、然后重新随机生成视频，不能重现原来的条件概率。

必须区分两类优化目标：

- 选中去噪转移的条件概率/代理目标。
- 最终环境动作的边缘概率，或严格定义的完整生成路径概率。

前者可以参考现有 flow policy 的工程实现，但不能直接宣称等于后者。若保存的视频状态是在旧策略下产生的，更新时将其视为固定条件，也必须明确这属于什么目标或近似。完整路径中确定性且依赖参数的视频转移不能被无说明地忽略。

### 6.3 冻结参数与梯度

建议第一版限制可训练参数范围，例如先训练动作分支；这是控制实验规模的建议，不是已验证的最优策略。

在 mutual attention 下，冻结视频参数不等于视频输出与动作无关，也不等于可以删除视频计算或随意缓存其输出。若所选目标需要梯度穿过冻结模块的输入，不能把整段计算放入 `no_grad`。OpenWAM 的 `freeze_modules` 会包装部分模块 forward，实际调用路径必须检查。

直接将 checkpoint 的 mask 改成 `isolated` / `action_sees_video` 可能改变已训练策略行为，不应作为无需验证的显存优化。

### 6.4 概率维度与执行维度

定义并分别记录 `D_model`、`D_raw`、`D_env`，不要只用一个 `action_dim` 描述三者。

- 模型空间的 padding/unmapped 维度应按概率目标进行屏蔽或约束，不能让无效噪声主导 loss。
- EEF 的 rot6d→quaternion、夹爪裁剪或二值化可能是非可逆映射。应保存投影前样本并明确潜在动作的优化目标，不把投影后的数值直接送入原空间 Gaussian 公式。
- [actor worker](infra/RLinf/RLinf/rlinf/workers/actor/embodied_fsdp_actor_worker.py)向 loss 传递执行动作维度；[loss 预处理](infra/RLinf/RLinf/rlinf/algorithms/utils.py)按该维度 reshape log-prob。
- 例如 `[B,32,20]` 按 16 重排会变成 `[B,40,16]`。action/token 级语义会改变；chunk 级求和即使碰巧相同，也不能证明 mask、诊断和目标都正确。
- 第一版应明确选择概率聚合方式，并让 rollout、重算、loss、entropy 与终止 mask 一致。需要时扩展桥接/worker 的维度契约；不要通过随意截断、补零或更改形状掩盖语义差异。

### 6.5 执行 horizon

原生默认 `num_frames: 33` 对应 32 步动作，`video_stride: 4` 对应 9 帧视频条件/预测窗口。视频帧数、生成动作长度、实际执行长度不能混淆，实际值仍以 checkpoint 为准。

当前 wrapper 只检查返回 horizon 不大于 `num_action_chunks`，但 rollout worker 用配置 horizon 计算循环次数。因此第一版应让生成/截取、配置和执行 horizon 一致，并检查终止导致的部分执行如何参与 loss；不能因 wrapper 接受较短 chunk 就认为完整链路正确。

### 6.6 候选路线：以 RoboTwin 奖励驱动 Diffusion-NFT

**定位：算法适配研究，尚未实现或验证。** 上游视频生成教程提供了奖励驱动去噪更新的参考；它不证明 OpenWAM 已具备机器人在线 RL 能力，也不能替代第 4 节的 joint/EEF 接入选择。动作接入路线与训练算法路线是两个独立决策。

#### 与概率重算路线的区别

现有方案为随机去噪转移定义概率，并通过旧/新 log-prob 构造 PPO/GRPO 更新。NFT 候选方案则研究用相对奖励指导去噪预测的训练目标，不以同一种概率比值为前提。

组内奖励归一化与模型更新目标应分开理解：采用 NFT 并不排斥组内相对优势。不能把普通 SFT loss 改名为 NFT，也不能将任意负 MSE 当成 log-prob 交给现有 PPO/GRPO loss。

#### 必须明确的机器人任务语义

媒体生成示例的一次生成可直接获得评分；RoboTwin 中动作执行后会改变环境，任务通常需要多个动作 chunk。生成的视频不是实际环境执行结果，OCR 奖励也不对应机器人任务成功。

第一版实验应书面固定以下选择：

1. **优化对象**：仅动作去噪预测，还是联合视频动作预测；若含视频项，明确其作用、权重及任务奖励为何能监督它，避免用大量视频维度淹没动作目标。
2. **奖励与信用分配**：使用真实任务成功、环境回报或明确的 shaping；定义 episode 回报如何分配给生成 chunk，以及终止、截断和部分执行的 mask。视频帧索引不等于环境动作步索引，不能直接照搬媒体评分的帧分组配置。
3. **候选组**：定义同组是否从相同任务和可恢复的初始状态出发；评估环境复制/恢复能力及随机性。不同动作执行后环境已变化，不能把连续生成的动作直接视为同一状态下可比较的候选。
4. **探索**：使用独立、可恢复的生成随机流，确认候选存在行为和回报差异；若成功奖励全为零，需先解决有效探索或奖励设计。

#### 模型与训练链路适配

- 明确 OpenWAM 的视频/动作时间步、shifted sigma、预测参数化和训练目标空间；不能仅因同属 flow/diffusion 模型就套用其他模型的公式与 schedule。
- 定义保存或重建的联合输入、生成目标和参考预测。mutual attention 下，仅保存动作而重新随机生成视频仍可能改变训练条件；不计算 log-prob 并不意味着不需要联合状态。
- 固定参考模型的更新规则、同步时机及恢复状态，避免一个 batch 中目标和参考权重不一致。
- 在模型空间定义有效动作 mask、归一化和 loss 聚合，再将动作转换到执行空间；不能对裁剪或非可逆转换后的动作直接套用原空间目标。
- 检查当前 embodied rollout、buffer、actor 和外部桥接能否传输 NFT 所需信息及调用专用 loss。需要时增加独立入口或契约；不能直接把 OpenWAM 放进媒体生成配置，就假定 RoboTwin 多步交互已经接通。
- 可训练范围、冻结模块的输入梯度、真实网络注册、FSDP、权重同步与 checkpoint 生命周期仍需验证。NFT 的显存、吞吐及学习效果必须实测，不能预先承诺更简单或更省资源。

#### 选择条件与最小验收

先完成阶段 0～3，再比较两条训练路线。只有在 NFT 目标、奖励分配、联合状态和 worker 改动明确后，才进入最小实现；若这些条件尚不成立，保留其为候选，不据此删除已设计的概率重算路线。

NFT 的最小验收包括：固定权重和随机状态可复现训练目标；参考状态可恢复；有效维度与部分执行 mask 正确；在受控样本上检查奖励/优势符号与损失梯度方向一致；目标参数获得有限梯度且更新同步到 rollout；少量真实任务优化后，与固定 seeds 的未训练基线比较成功率、回报和动作异常。loss 下降或生成视频看起来合理，都不能单独证明机器人策略学会了任务。

NFT 路线不适用“旧/新 log-prob 一致、初始 ratio 接近 1”作为自身验收；若后续增加 PPO，仍须另行实现其概率与价值估计链路。

### 6.7 源码复评：可复用基础、具体缺口与推荐最小实验

**结论：可以立项适配；当前不能直接开启 OpenWAM 在线 RL。** 推理与 SFT 的主要工作是接口及运行时适配；NFT 和概率重算路线还涉及训练目标与联合状态设计。没有发现必须更换 RLinf 框架的结构性障碍，但尚不能承诺成功率提升、显存或吞吐。

#### 已核实的可复用基础

OpenWAM 的 `BaseWAMArchitecture` 是可训练的 `nn.Module`，提供 `prepare_inputs`、`compute_loss` 和联合 `forward`。默认 dual-system 的具体 forward 返回视频与动作预测，因此无需从部署接口的最终 NumPy 动作反推梯度；应直接在注册到 actor 的网络上计算目标。

动作 scheduler 使用 `(1-sigma) * clean + sigma * noise` 加噪，以 `noise-clean` 为速度目标。零 SDE 噪声的 NFT `x0` 重建形式 `x_t-sigma*v` 在这个表示下有对应关系，这是尝试 NFT 的数学接口基础；并不自动证明联合视频动作条件下的奖励优化目标有效。

#### 当前不能直接兼容的五处契约

1. **时间表和单位。** NFT worker 的 `_build_schedule_and_timesteps` 当前固定 `linspace(1,0,num_steps+1)`；OpenWAM 动作 scheduler 使用 shifted sigma，并以 `sigma * 1000` 表示模型时间步，联合生成还允许视频和动作不同步。只传 `num_steps` 不够，应让模型提供或保存实际双流时间步及步长。例如四步、shift=5 时，动作 sigma 为 `1,0.9375,0.833333,0.625,0`，不同于 worker 的 `1,0.75,0.5,0.25,0`；这仅是公式示例，不是实际 checkpoint 设置。
2. **联合状态与重采样。** worker 的 `resample` 路径只重建单个 `nft_xcur` 并重算 `v_old`。OpenWAM 需要相应的视频 latent、动作 latent、首帧、文本、proprio 和两个时间步。必须明确是重建联合样本还是固定视频状态的条件代理目标，不能只重采样动作后默默沿用不匹配的视频时间条件。
3. **维度与 mask。** worker 在能量聚合中直接对张量维度求均值/和，随后对齐 `loss_mask`；这里没有现成的 OpenWAM 80 维语义 mask。应在能量、权重和裁剪计算中一致处理有效维度，或显式选取对应维度，并对齐实际执行的动作步。模型 80 维、EEF20 与执行 EE16 不能混用。
4. **能力门禁仍按概率路线设计。** `capabilities.py` 只接受 `eval/sft/ppo/grpo`，且 wrapper 对训练模式要求 `compute_logprob_value` 与 `prev_logprobs`。因此已有 `nft_forward` 不等于独立 NFT 模式已接通。需区分采样能力、概率重算能力和 NFT 能力；worker 当前仍用 `prev_logprobs` 的形状推导 batch，也应解耦或明确为形状元数据，不能伪称真实采样概率。
5. **参数和环境接入。** 当前 OpenWAM `Model` 没有暴露 `self.policy/self.model`、`self.action_type`、`self.robot_action_dim_info`，wrapper 会落到 `nn.Identity` 和默认 joint 语义；相机名也存在 `cam_head` 与 `cam_high` 差异。必须先建立 RoboTwin 分支，移除该分支对 RoboDojo X5 标定的依赖，并注册实际 architecture，才能讨论训练。

时间表差异不意味着 NFT 训练采样必须永远沿用推理网格。若采用 `resample`，可以另行定义训练时间分布，但须统一 sigma、模型时间单位、联合状态构造和目标公式；若重放 rollout 的选中步骤，则必须使用采样时的实际时间条件，不能用相同索引查询另一张时间表。

#### 建议的第一个 NFT 实验

这是控制变量的实验建议，不是已验证算法：

1. 使用已经通过原生客户端评估的 RoboTwin checkpoint；发布权重若为 EEF，则先打通 EE 环境链路，不为迁就现有 joint 示例强改配置。
2. 保持 checkpoint 的 attention mask、相机、归一化和推理 schedule；先限制训练参数到动作分支，仍保留必要的视频计算和输入梯度，避免用整段 `no_grad` 破坏梯度路径。
3. 优先验证 `nft_xcur_source: rollout`、`nft_target_space: x0`、`nft_tau: 1.0` 的动作条件代理目标：保存选中去噪步的联合输入、旧动作预测和最终模型空间动作，以真实执行回报构造优势。先不引入视频动作联合重采样或慢更新参考模型；这些配置只有完成上述 schedule/mask/接口适配后才可用。
4. 继续使用随机初始噪声提供候选差异，无需仅为概率比值而先添加每步 Gaussian 转移。验证同组初始任务条件、随机流以及奖励差异；确定性 Euler 更新不代表不同初始噪声会得到相同动作。
5. 依次检查：同权重同输入的动作速度重算一致；正/负优势的受控梯度方向；有效维度与终止 mask；动作参数真实更新且同步到 rollout；固定 seeds 的真实成功率与回报。该条件代理目标不等于完整联合生成分布的精确策略梯度，需用实验判断是否有效。

通过最小实验后，再决定是否扩大训练参数范围、使用联合重采样/EMA，或实现阶段 4A 的概率重算方案。不要把“先做 NFT”解释为 GRPO/PPO 不可行，也不要把一个低层 flow 公式对齐当作端到端训练已经成立。

## 7. 分布式运行、依赖与 checkpoint

### 7.1 FSDP 与权重同步

- 真实网络必须注册到 wrapper 的 backbone，参数名、state dict 和优化器参数组保持一致。
- 混合冻结参数、不同精度参数、MoT 的分层调用需要专门验证 wrap 边界。部分计算绕过模块常规 `forward`，不能盲目按普通 Transformer block 包裹。
- 检查 gradient checkpointing 的启用接口是否真正传到架构；当前外层 wrapper 不会自动实现所有底层运行时方法。
- offload 后检查架构保存的 device/dtype 元数据、缓存 tensor 与实际参数位置是否一致。
- actor 更新必须能同步到 rollout 使用的同一份网络；测试应检查参数及输出变化，而非只检查通信完成。

### 7.2 依赖

当前 [OpenWAM 依赖](policy/OpenWAM/OpenWAM/pyproject.toml)要求 `transformers>=5.5,<6`；本地 [RLinf embodied extra](infra/RLinf/RLinf/pyproject.toml)限制 `transformers<=4.57.6`。这是完整 extras 安装时的版本冲突，不等于 RLinf 通用接口在原理上不支持 OpenWAM。

应参考 [install_rl_runtime.sh](infra/RLinf/install_rl_runtime.sh)在策略环境安装所需 runtime，保留 OpenWAM 依赖并进行 import/运行验证。脚本采用 RLinf `--no-deps` 安装，也意味着必须补查实际使用模块的依赖完整性；安装命令结束不等于依赖验证完成。

不要直接复制 Abot 的 Torch 升级选项或其他策略的环境。是否升级 Torch、FlashAttention 等由目标组合测试决定。

### 7.3 checkpoint 生命周期

分别验证三个场景：

1. 从 OpenWAM 原生目录初始化：`config.yaml`、`checkpoint_step_*.safetensors`、统计量和所需资产。
2. 从 RLinf checkpoint 恢复训练：模型、优化器、scheduler、步数和所需 RNG 状态。
3. 将 RL 后权重导回 OpenWAM 部署：处理 wrapper 前缀、可选 value head，保留配置、统计量及 tokenizer 等资产。

“能加载原生权重”不能替代后两项验收。路径解析继续复用 `XPolicyLab.utils.checkpoint_resolver`。

## 8. 推荐适配顺序与阶段验收

以下是建议实施顺序；未完成阶段不应通过模式声明跳过。

### 阶段 0：固定 checkpoint 和数据/动作契约

工作：选择实际 RoboTwin checkpoint，记录训练模式、机器人、统计量、原始/统一维度、映射、坐标和四元数顺序、相机、prompt、架构、mask、schedule 与 horizon。检查目标 VectorEnv 的真实能力。

决策：选择第 4 节的 joint 或 EEF 路线。若采用文档中描述的 EEF 发布权重，则规划现有 RoboTwin 链路的 EE 扩展；不默认存在匹配的 joint 权重。

验收：同一条真实观测可在原生 RoboTwin 客户端与候选 RLinf 输入链中产生一致的模型条件；动作可以按相同语义交给目标环境。

### 阶段 1：完成模型侧包装和推理一致性

工作：补标准属性、真实网络注册、RoboTwin 条件构建、相机映射及 checkpoint 驱动后处理。复用已有批量生成，关闭未验证的 compile、DiT cache、CFG 等加速/引导路径。

验收：参数枚举非空且归属正确；固定噪声时 B=1 与原生推理一致；批量拆分、环境索引和 reset 无串扰；输出维度、horizon 与所选契约一致。

这里的 dummy/debug 只验证协议，不能替代真实 checkpoint 推理。

### 阶段 2：打通 RoboTwin 小规模评估

工作：基于现有 `adjust_bottle` 示例创建 OpenWAM eval 配置，启用所需相机；按路线补齐 joint/EE 状态和执行支持。核对 eval runner 与入口脚本匹配，不能仅因为已有 eval YAML 就假定训练入口能直接运行它。

验收：少量真实 episode 能 reset、执行 chunk、结束并记录成功/失败；与原生客户端在相同控制设置下比较输入及动作，定位表示差异。

### 阶段 3：验证梯度、FSDP 和权重生命周期

工作：可用原生 SFT loss 做最小梯度诊断，也可用已经定义清楚的 RL 重算路径；先单卡，再小规模多卡。检查冻结范围、优化器、wrap、offload、同步、保存和恢复。

验收：目标参数获得有限非零梯度并变化，冻结参数保持不变；更新能到达 rollout；重载后的同输入输出在约定误差内一致。

此阶段不强制先完成一次完整 SFT 训练。SFT 是可复用的诊断/训练能力，是否需要另行训练由 checkpoint 质量和实验目标决定。

### 阶段 4 前置决策：选择训练目标

保留阶段 0～3 的公共基础。按第 6.6～6.7 节评估 NFT 与现有概率重算方案，记录所选目标、环境奖励分配、候选组定义、保存状态和桥接/worker 改动。当前建议优先验证第 6.7 节的动作分支 NFT 小实验，无需同时实现两条路线，也不以候选 NFT 尚未验证为由阻塞推理评估。

### 阶段 4A：概率路线——实现并验证 GRPO 采样与重算

工作：选定概率目标，增加随机 sampler、联合输入保存、有效维度处理、schedule 对齐及带梯度重算。通过契约检查后才启用 `RL_MODES` 中的 `grpo`。

验收：同权重下旧 log-prob 与重算结果一致，初始 ratio 接近 1；检查误差分布及最大偏差，不只看均值。误差容限按 fp32/bf16 和实际 kernel 设定并记录。

还应验证：相同观测的不同随机流产生动作差异；group 内任务/初始条件按 GRPO 定义对齐；部分终止样本正确 mask；梯度、loss 和更新后概率有限。通过后再跑少量完整优化迭代。

### 阶段 4B：NFT 路线——实现并验证任务奖励驱动更新（可选）

仅在前置评估选择 NFT 时实施。按第 6.6 节定义联合去噪目标、参考更新规则、轨迹数据和奖励分配，增加相应训练接口及 loss；不通过伪造 `prev_logprobs` 或声明 `grpo` 能力绕过现有门禁。新增模式名称及入口以实际框架扩展为准，本文不预设已有 `nft` 模式。

验收：完成第 6.6 节的目标复现、梯度方向、mask、参考恢复及同步检查，通过后再运行少量真实 RoboTwin 优化迭代。记录吞吐和显存，判断是否适合扩大实验。

### 阶段 5：验证所选路线的学习效果，再按需扩展 PPO

工作：固定验证 seeds 和任务设置，与未训练基线比较成功率、回报、动作异常和吞吐；确认变化可重复后再扩大环境数及训练范围。

PPO 扩展：先具备阶段 4A 的概率链路，再加入 value head，明确 value 对应观测/随机条件的含义，验证 `prev_values`、bootstrap、GAE、终止与截断、critic 参数组和 checkpoint。完成阶段 4B 不代表已具备这些能力。

验收：不能仅以 loss 下降判定学习有效；应保留固定条件下的任务指标。GRPO 跑通也不等于 PPO 已完成。

### 阶段 6：性能优化与交付

正确性稳定后再评估 compile、缓存、梯度检查点、精度、offload、批量和多节点。每项优化都与已验证基线比较，保留可以复现的配置与运行记录。

## 9. 预计涉及的文件与边界

以下为拟修改范围，不表示已实施：

- `policy/OpenWAM/model.py`：RoboTwin 分支、标准属性、网络注册、RL/SFT 接口转发。
- `policy/OpenWAM/rl.py`（拟新增）：RL 模块、sampler、联合条件缓存、概率重算和可选 value head。
- `policy/OpenWAM/deploy.yml`：必要的显式选项，保持 demo 的完整基础键集；RLinf 不自动合并此文件。
- `policy/OpenWAM/README.md`：补充 RoboTwin RL 的 checkpoint、配置、安装、运行和能力边界。
- `infra/RLinf/config/`：OpenWAM eval、GRPO，后续 PPO 配置；以现有 RoboTwin 示例为模板，按真实 checkpoint 填写。
- `infra/RLinf/obs.py`、`action_model.py`：只在通用相机/表示/概率契约确实不足时扩展，保证现有 Pi_05、Abot_M0 行为。
- `infra/RLinf/train_rl.py` 或配套 eval/SFT 入口：按实际 runner 需要处理，不把 eval、SFT 当作现有训练入口的自动能力。
- `infra/RLinf/tests/` 与 OpenWAM 对应测试：覆盖新契约、维度和表示变换、采样重算、参数注册。
- RoboTwin VectorEnv/环境适配：仅在所选路线需要额外状态或执行支持时修改，应保持为 RoboTwin 扩展。
- 内置 OpenWAM architecture：只有现有公开方法不足以保存/重算联合输入时，才做最小扩展。
- NFT 路线若被选用：优先复用现有 `nft_forward`、`EmbodiedNFTFSDPPolicy` 和入口分发；预计涉及 `capabilities.py`、wrapper 采样门禁、rollout 数据字段、模型侧双时间步与联合输入，以及 worker 的 schedule/mask 契约。参考管理和 loss 已有基础，需适配验证，无需预设全部重写。
- RLinf 原项目：优先通过外部扩展解决；若必须改 worker/loss，则记录原因，按其贡献要求单独验证。

## 10. 实施时需要保留的仓库约定

- XPolicyLab 从父工作区导入；`env_cfg/` 位于 checkout 外，不能拼出 `XPolicyLab/env_cfg/`。
- 动作维度通过共享 helper 获取，机器人信息遵守根目录 `AGENTS.md` 的两份注册约定。
- 图像全程 RGB；模型不解码；新增离线代码只用共享 `decode_image_bit` / `encode_image_bit` 处理存储图像位流。
- checkpoint 路径使用共享 resolver；`policy_name` 与目录名一致。
- 新增配置从当前示例复制后逐项核对，不能照搬其他模型的统计量、动作模式、50 步 horizon 或资源规模。
- 不用假 log-prob 绕过模式门禁；尚未实现的能力保持禁用。
- 修改 RLinf 原项目实现前，按其 `AGENTS.md` 检索相关已有 PR/issue 并报告重叠工作；本文的只读评估不等于已完成该实施前检查。

## 11. 当前验证记录

已完成：

- 阅读本地 OpenWAM 模型、RoboTwin dataloader/原生客户端、训练损失和联合推理代码。
- 阅读 RLinf 主要模型示例及 XPolicyLab 注册、能力门禁、观测/动作桥接、rollout 和 actor loss。
- 静态检查 OpenWAM 标准属性和接口覆盖情况；用标准库执行了继承模式门禁探针，确认默认仅 eval，直接启用 GRPO 会被拒绝。
- 核对配置中的 RoboTwin 环境、动作模式、默认映射、四元数顺序和版本约束。

尚未完成：真实 checkpoint 加载、真实 RoboTwin episode、Torch 反向传播、GPU/FSDP、概率一致性测试、显存吞吐测试、RL 学习曲线及 checkpoint 往返验证。

2026-10-23 补充记录：阅读上游视频模型教程与 OpenPI NFT 配置；检查本地 NFT worker、OpenPI/Pi_05 实现、XPolicyLab 门禁/转发和 OpenWAM 联合 forward/scheduler，记录第 6.7 节的接口差异。使用标准库 AST 检查七个相关 Python 文件可解析，并确认 OpenWAM 未覆盖四个训练接口、未赋值桥接需要的标准属性；独立门禁探针确认 `nft` 名称当前被拒绝。未实现 OpenWAM NFT loss、机器人奖励分配或训练接入，也未运行该路线的实验。AST 与门禁检查不等于模型 import、反向传播或端到端测试。

本机检查时为 macOS ARM，当前系统 Python 未安装 Torch、NumPy、OmegaConf、pytest 等依赖，也未提供用于验证的实际 checkpoint。本次没有安装训练环境，未把静态探针描述为端到端测试。

## 12. 开始实施前的交接清单

- [ ] 选定并记录实际 RoboTwin checkpoint，检查其保存配置与统计量。
- [ ] 决定 joint 或 EEF 路线，核对目标 VectorEnv 与控制接口。
- [ ] 固定 RGB 相机映射、prompt、proprio、四元数顺序、坐标系和夹爪约定。
- [ ] 分别定义模型、原始策略、环境执行、log-prob 的表示与维度。
- [ ] 选择概率重算或 NFT 训练路线，明确目标、奖励分配、候选组及联合视频动作状态的保存/重建方法，决定可训练范围。
- [ ] 若选择 NFT，固定上游实现提交并检查本地兼容性，明确专用 loss、参考更新/恢复和 worker 数据契约。
- [ ] 准备匹配的策略 runtime、RoboTwin runtime、GPU 和少量验证任务。
- [ ] 按阶段 0 → 1 → 2 → 3 → 训练目标决策 → 4A 或 4B → 5 → 6 实施，并保存各阶段验收证据。

核心实施原则：先固定 RoboTwin checkpoint 与动作契约，再证明推理和梯度链路正确，最后验证所选训练目标的正确性与真实任务学习效果；NFT 保持候选身份，直到完成对应验收。
