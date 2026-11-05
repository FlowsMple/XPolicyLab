# OpenWAM 在 RoboDojo 和 RoboTwin 上训练

两个宿主使用同一套 OpenWAM NFT actor、损失和 RLinf 0.4 训练入口。切换时要一起切换 **宿主工作目录、任务配置和对应的 EEF 权重**。RoboDojo 的 `arx_x5` 和 RoboTwin 的 `aloha_agilex` 使用不同的标定与归一化统计。

下面的 Bash 示例可在 L20-3 上按顺序运行。GPU 6 负责 actor，GPU 7 负责 rollout 和仿真。

使用关节动作策略时，见 [Pi0.5 与 Abot 的 PPO、GRPO 训练](PI05_ABOT_QUICKSTART_zh.md)。

本文命令使用重构后的目录。在所选宿主的 XPolicyLab 根目录用 `test -f infra/RLinf/scripts/run.sh` 检查版本。此前 L20-3 实验使用旧布局，在该工作区使用新命令前需同步本版 XPolicyLab。[目录与外部工具迁移](EXTENSIONS.md#迁移现有调用)

## 1. 选择宿主

先登录服务器并准备策略环境。Luminis 只运行 RoboDojo 的 Isaac Sim 子进程。

```bash
ssh L20-3
```

```bash
base=/personal/zijian/test/RLinf
export PATH="$base/runtime/openwam-env/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7
export RAY_ADDRESS=local RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RLINF_CODE_WORKING_DIR=0
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4294967296
```

**A. RoboDojo · `stack_bowls`**

```bash
export ROBODOJO_ROOT="$base/RoboDojo"
cd "$ROBODOJO_ROOT/XPolicyLab"
export ROBODOJO_SIM_PYTHON="$base/runtime/luminis-python"
export POLICY_MODEL_PATH=/personal/yuechen/OpenWAM/models/New_OpenWAM_RoboDojo_SFT_60k
cfg=robodojo_stack_bowls_nft_xpolicylab_openwam
sim_args=(env.train.render_gpu=7 env.eval.render_gpu=7 env.train.rpc_timeout=360)
```

`luminis-python` 激活 Conda 的 `Luminis` 后执行仿真子进程。`render_gpu` 是 Kit 的图形设备编号，L20-3 上已按 UUID 核对为物理 GPU 7。宿主提供 `src/rlinf_bridge/runtime.py`、支持 `external_control=True` 的 evaluator、接触检测、机器人资产与训练和评估布局。

**B. RoboTwin · `adjust_bottle`**

```bash
export ROBOTWIN_RUNTIME_PATH="$base/runtime/robotwin-support"
cd "$ROBOTWIN_RUNTIME_PATH/XPolicyLab"
export ROBOTWIN_PATH="$ROBOTWIN_RUNTIME_PATH"
export ROBOTWIN_ASSETS_PATH=/personal/zijian/RoboTwin
export ASSETS_PATH="$ROBOTWIN_ASSETS_PATH" ROBOT_PLATFORM=ALOHA
export ROBOTWIN_QUATERNION_ORDER=wxyz
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export POLICY_MODEL_PATH=/personal/zijian/rlinf/checkpoints/OpenWAM-Alpha-Sim-RoboTwin-Full
cfg=robotwin_adjust_bottle_nft_xpolicylab_openwam_split
sim_args=()
```

这个工作目录使用 `RLinf_support` 的 `0008ae6`、当前适配器和已绑定 GPU 7 的 SAPIEN 渲染器。L20-3 上旧的 `$base/RoboTwin/XPolicyLab` 保留原检出，其 `infra/RLinf/results/` 与上述工作目录共用训练输出。两种宿主均可从各自的 XPolicyLab 根目录运行下一节。

在其他机器上，使用本机的策略 Python 和匹配的 checkpoint 目录。目录应包含 `config.yaml`、权重、`normalization_stats.npy` 和 tokenizer。把这版 XPolicyLab 放在有对应 `env_cfg/` 的宿主下，用 `git submodule update --init infra/RLinf/RLinf` 初始化已登记的子模块。RoboTwin 还需使该解释器可以导入同一份原生 `envs` 和 `robotwin`，并按本机设备设置渲染器与 placement。若原生运行时在另一个 checkout，见 [RoboTwin 安装与启动](OPENWAM_EXPERIMENTS.md)。

## 2. 检查配置并启动

选好 A 或 B 后，在**同一个 Bash 会话**中运行。每次新训练单独设一个 `log_path`，这样 TensorBoard 配置和指标不会与其他实验混写。

```bash
out="$PWD/infra/RLinf/results/$(date +%Y%m%d-%H%M%S)-$cfg"
args=('~cluster.component_placement' '+cluster.component_placement={actor:6,env:7,rollout:7}'
      "runner.logger.log_path=$out" runner.logger.experiment_name=openwam_nft
      runner.save_interval=1 +actor.fsdp_config.save_full_model_weights=false
      "${sim_args[@]}")

# 仅打印 Hydra 配置，检查模型路径、env_type、placement 和日志目录。
bash infra/RLinf/scripts/run.sh "$cfg" "${args[@]}" --cfg job --resolve

# 采样一轮、更新 actor，保存 global_step_1。
bash infra/RLinf/scripts/run.sh "$cfg" "${args[@]}" runner.max_steps=1
```

这两套配置的默认采样长度均为 16 个动作。若要采满任务时限，RoboDojo 使用 `env.train.max_episode_steps=800`（rollout 长度随之变化）；RoboTwin 同时指定 `env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200`。将覆盖项追加到训练命令即可。`runner.max_steps` 控制训练迭代数。

## 3. 从 checkpoint 继续

保留相同的宿主、基础权重、模型和优化器配置。将 `runner.resume_dir` 指向完整的 `global_step_N` 目录。目标总步数设为 2，下面的示例会从第 1 步再训练一轮。

```bash
resume="$out/openwam_nft/checkpoints/global_step_1"
test -f "$resume/actor/dcp_checkpoint/.metadata" &&
  bash infra/RLinf/scripts/run.sh "$cfg" "${args[@]}" "runner.resume_dir=$resume" runner.max_steps=2
```

恢复到新日志目录也可以，`resume_dir` 仍填写原 checkpoint 的绝对路径。L20-3 上已有的 RoboDojo 第 2 步存档在 `$base/RoboDojo/XPolicyLab/infra/RLinf/results/dojo_openwam_20261104.ud2vq3/nft/checkpoints/global_step_2`。

## 4. 手动启动会产生什么

示例中的 `out` 是 RLinf 的 `runner.logger.log_path`。在 L20-3 上按第 2 节分别启动两个宿主，两次都完成真实 rollout、actor 更新和第 1 步保存。两次的运行目录顶层都只有以下三项。

```text
<out>/
├── metrics.log
├── tensorboard/
│   ├── config.yaml
│   └── events.out.tfevents.*
└── openwam_nft/
    └── checkpoints/global_step_1/actor/dcp_checkpoint/
        ├── .metadata
        └── *.distcp
```

- **继续训练需要**整个 `global_step_N` 目录，包括隐藏的 `.metadata` 和全部分片。仍须提供初始化模型所需的基础权重和统计。指标和 TensorBoard 文件便于对照参数、查看曲线。
- **完整权重导出可选。** 上游 FSDP 默认还保存 `actor/model_state_dict/full_weights.pt`。示例用 `+actor.fsdp_config.save_full_model_weights=false` 节省磁盘；去掉该覆盖项即可采用上游默认。
- **额外日志和视频按需生成。** 上游示例 shell 用 `tee` 写 `run_embodiment.log`；这里的 `run.sh` 直接向终端输出。两者调用相同的 RLinf 指标和 checkpoint 保存逻辑。启用逐 worker 日志会生成 `worker_logs/`。RoboTwin 可用 `env.train.video_cfg.save_video=true "env.train.video_cfg.video_base_dir=$out/video"` 录制训练画面。RoboDojo 当前训练适配器不支持该开关；此前的完整回合 MP4 来自独立评估。
- **不指定 `log_path` 时**，当前 OpenWAM 配方直接使用 `infra/RLinf/results/`。只改 `experiment_name` 会分开 checkpoint，但 `metrics.log` 和 `tensorboard/` 仍共用父目录。上游示例启动器也会为每次运行分配独立日志目录。[上游启动器](../RLinf/examples/embodiment/run_embodiment.sh) · [指标记录](../RLinf/rlinf/utils/metric_logger.py) · [checkpoint 保存](../RLinf/rlinf/runners/embodied_runner.py)

### L20-3 上之前看到的文件

`*.exit`、`controller*.log`、`check_state.py`、源码压缩包、`initial_obs.npz`、`steps.jsonl` 和核验 JSON 是之前测试时额外保存的。手动运行第 2 节的命令不会创建这些文件。它们已按原跑次归档到 **`/personal/zijian/test/RLinf/validation-artifacts/<RoboDojo 或 RoboTwin>/<原跑次名>/`**。策略 Python 和 Luminis 启动器放在 `$base/runtime/`，训练无需进入归档目录。旧 RoboTwin `results/dojo_openwam_20261104.ud2vq3` 是指向 RoboDojo 原跑次的兼容软链接，只占一个目录项。

此前 `global_step_2` 在 `stack_bowls` 的完整录像位于 `$base/RoboDojo/XPolicyLab/infra/RLinf/results/stack_bowls_step2_video.UNBvAq/OpenWAM-global_step_2-RoboDojo-stack_bowls.mp4`。同名验证归档包含本回合的得分、逐步动作和录制脚本。[训练与恢复记录](TRAINING_RESULTS.md#robodojo-stack_bowls--openwam-nft)

### 可选：重新录制已有的 RoboDojo 第 2 步存档

L20-3 保留了当次独立评估的配方。以下命令固定使用上述 `global_step_2`、评估布局集 1、种子 42 和 GPU 6、7，运行到成功、终止或 800 步时限。最后一个参数是**尚不存在**的输出目录。

```bash
base=/personal/zijian/test/RLinf
recorder="$base/validation-artifacts/RoboDojo/stack_bowls_step2_video.UNBvAq/evaluate.sh"
mkdir -p "$base/evaluations"
bash "$recorder" "$base/evaluations/stack_bowls-step2-$(date +%Y%m%d-%H%M%S)"
```

评估完成后，在指定目录打开 `OpenWAM-global_step_2-RoboDojo-stack_bowls.mp4`。同目录的 `result.json` 记录得分和终止原因，`steps.jsonl` 记录逐步动作。这三个文件只随这一独立录制命令产生。
