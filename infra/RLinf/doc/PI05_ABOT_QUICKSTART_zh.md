# Pi0.5 与 Abot 的 PPO、GRPO 训练

同一个 `infra/RLinf/scripts/run.sh` 可以启动 Pi0.5 或 Abot。两种模型都预测双臂 **14 维关节动作，每块 50 步**。RoboDojo 使用 `arx_x5` 执行 `stack_bowls`，RoboTwin 使用 `aloha_agilex` 执行 `adjust_bottle`。两者的初始权重、关节标定和统计量须与宿主匹配。

本文命令使用重构后的目录。在所选宿主的 XPolicyLab 根目录用 `test -f infra/RLinf/scripts/run.sh` 检查版本。此前 L20-3 实验使用旧布局，在该工作区使用新命令前需同步本版 XPolicyLab。[目录与外部工具迁移](EXTENSIONS.md#迁移现有调用)

## 1. 选择一次实验

登录 `ssh L20-3`，在一个新的 Bash 会话中运行下面各节。修改前三个变量即可切换组合。GPU 6 运行 actor；GPU 7 运行 rollout 和仿真。

```bash
host=robodojo     # robodojo 或 robotwin
policy=pi05       # pi05 或 abot
algo=ppo          # ppo 或 grpo

base=/personal/zijian/test/RLinf
export PATH="/personal/zijian/rlinf/RLinf/.venv-abot-robotwin/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7
export RAY_ADDRESS=local RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RLINF_CODE_WORKING_DIR=0
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4294967296
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

例如，保留上面的选择会启动 `robodojo_stack_bowls_ppo_xpolicylab_pi05`。要在 RoboTwin 上跑 Abot GRPO，把三项改为 `host=robotwin`、`policy=abot`、`algo=grpo` 后执行下一节，配置名将是 `robotwin_adjust_bottle_grpo_xpolicylab_abot`。

## 2. 选择宿主、权重与模型参数

```bash
extra=()
case "$host" in
  robodojo)
    export ROBODOJO_ROOT="$base/RoboDojo"
    cd "$ROBODOJO_ROOT/XPolicyLab"
    export ROBODOJO_SIM_PYTHON="$base/runtime/luminis-python"
    task=stack_bowls
    extra+=(env.train.render_gpu=7 env.eval.render_gpu=7 env.train.rpc_timeout=360)
    ;;
  robotwin)
    export ROBOTWIN_PATH="$base/runtime/robotwin-support"
    cd "$ROBOTWIN_PATH/XPolicyLab"
    export ROBOTWIN_ASSETS_PATH=/personal/zijian/RoboTwin
    export ASSETS_PATH="$ROBOTWIN_ASSETS_PATH" ROBOT_PLATFORM=ALOHA
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    task=adjust_bottle
    ;;
esac

case "$policy" in
  pi05)
    export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONPATH="$base/runtime/pi05-python-deps:$PWD/policy/Pi_05/openpi/src:$PWD/policy/Pi_05/openpi/packages/openpi-client/src"
    extra+=(algorithm.recompute_logprobs=true)
    if [[ "$host" == robodojo ]]; then
      export POLICY_MODEL_PATH=/personal/zijian/rlinf/checkpoints/pi05_robodojo_sim_arx_x5_joint_0_openpi_pytorch
      export PI05_TRAIN_CONFIG=pi05_base_aloha_full_sim_arx-x5_seed_0 PI05_REPO_ID=arx_x5_sim
      extra+=("+actor.model.xpolicylab.norm_stats_path=$POLICY_MODEL_PATH/arx_x5_sim")
    else
      export POLICY_MODEL_PATH=/personal/zijian/rlinf/checkpoints/pi05_robotwin_lerobot_v30_xpolicylab_pytorch
    fi
    ;;
  abot)
    if [[ "$host" == robodojo ]]; then
      export POLICY_MODEL_PATH=/personal/final_ckpt/Abot_M0/RoboDojo-sim-arx_x5-joint-0
      ckpt_step=150000
    else
      export POLICY_MODEL_PATH=/personal/zijian/RoboTwin/ckpt_to_upload/Abot_M0/RoboTwin-cotrain-aloha_agilex-joint-0
      ckpt_step=50000
    fi
    extra+=("actor.model.xpolicylab.checkpoint_num=$ckpt_step" actor.model.xpolicylab.unnorm_key=robodojo_sim)
    ;;
esac
cfg="${host}_${task}_${algo}_xpolicylab_${policy}"
```

Pi0.5 加载 PyTorch `model.safetensors`。RoboDojo 这份权重的 `norm_stats.json` 在 `arx_x5_sim/`，所以显式传入其目录；RoboTwin 的默认目录是 `assets/RoboTwin_lerobot_v30/`。Abot 的 `POLICY_MODEL_PATH` 指向含 `config.yaml`、`dataset_statistics.json` 和 `checkpoints/` 的运行目录。`checkpoint_num` 匹配 `steps_<N>_pytorch_model.pt`，`unnorm_key` 来自这份统计 JSON；此处两个 checkpoint 都使用 `robodojo_sim`。Abot 还需能读取其 YAML 所指的 Qwen-VL 权重。

在其他机器上，使用与 RLinf 0.4 匹配的 PyTorch 2.11 策略环境。Pi0.5 需要本仓库的 OpenPI 代码、对应 tokenizer 和数据变换；Abot 需要其视觉骨干和动作统计。L20-3 上的 `pi05-python-deps` 只为策略进程补充 OpenPI 运行依赖，RoboDojo 仿真由单独的 Luminis Python 启动。[Pi0.5 安装](../../../policy/Pi_05/README.md#installation) · [Abot 安装](../../../policy/Abot_M0/README.md#installation)

## 3. 启动训练

下面使用一个 actor rank、microbatch 1。RoboTwin GRPO 的 global batch 为 Pi0.5 32、Abot 16，其余组合采用 4。PPO 用 GAE 和 value head；GRPO 按同一布局的轨迹组计算相对优势。示例将训练与评估分开，在每轮保存可恢复的 DCP。

```bash
group=1; n=2; epochs=1; batch=4
if [[ "$algo" == grpo ]]; then group=2; n=4; fi
if [[ "$host" == robotwin ]]; then
  horizon=200
  if [[ "$algo" == grpo ]]; then
    n=8; group=4; epochs=4; batch=16
    if [[ "$policy" == pi05 ]]; then group=8; batch=32; fi
  fi
else
  horizon=100
  if [[ "$algo" == grpo ]]; then horizon=800; fi
fi

out="$PWD/infra/RLinf/results/$(date +%Y%m%d-%H%M%S)-$cfg"
args=('~cluster.component_placement' '+cluster.component_placement={actor:6,env:7,rollout:7}'
      "runner.logger.log_path=$out" "runner.logger.experiment_name=$algo"
      runner.save_interval=1 runner.val_check_interval=-1
      actor.enable_offload=false rollout.enable_offload=false env.enable_offload=false
      "env.train.total_num_envs=$n" "algorithm.group_size=$group" "env.train.rollout_epoch=$epochs"
      "env.train.max_episode_steps=$horizon" "env.train.max_steps_per_rollout_epoch=$horizon"
      algorithm.update_epoch=1 actor.micro_batch_size=1 "actor.global_batch_size=$batch"
      +actor.fsdp_config.save_full_model_weights=false "${extra[@]}")
bash infra/RLinf/scripts/run.sh "$cfg" "${args[@]}" --cfg job --resolve
bash infra/RLinf/scripts/run.sh "$cfg" "${args[@]}" runner.max_steps=1
```

`runner.max_steps=1` 表示一次训练迭代。RoboDojo 的 PPO 示例将每个回合限制到 100 个环境动作以便快速验收；完整的 `stack_bowls` 请设 `horizon=800` 后重新构造 `args`。RoboTwin 完整回合为 200 步。对这组关节策略，长度须为 50 的倍数，每个 actor rank 收到的动作块总数须能整除训练批量。检查 GRPO 的组内回报、`loss_mask_fraction` 和 `actor/grad_norm`；全组回报相同可能产生零优势，过滤后没有有效样本时，上游的掩码后 `rewards` 均值显示为 `nan`。可结合原始 `success_once`、动作头 Adam 一阶矩和梯度判断；即使梯度为零，Adam 计步和权重衰减也可能继续。

## 4. 断点恢复与输出

```bash
resume="$out/$algo/checkpoints/global_step_1"
test -f "$resume/actor/dcp_checkpoint/.metadata" &&
  bash infra/RLinf/scripts/run.sh "$cfg" "${args[@]}" "runner.resume_dir=$resume" runner.max_steps=2
```

新进程仍需相同的策略环境、基础权重、统计量和配置。将完整的 `global_step_1` 目录传给 `resume_dir`。上例以 2 为目标总步数，再训练一轮并写出 `global_step_2`。每个 `out` 的标准产物为 `metrics.log`、`tensorboard/` 和 `<ppo 或 grpo>/checkpoints/`；测试日志、脚本不属于这些产物。[输出目录说明](OPENWAM_QUICKSTART_zh.md#4-手动启动会产生什么)

## 5. 如何验收一次训练

1. 在 `tensorboard/config.yaml` 核对 `env_type`、`required_mode`、初始权重和 `num_action_chunks=50`。Pi0.5 用 `model_type=openpi` 进入原生 PyTorch 更新逻辑。Abot 用 `model_type=xpolicylab` 和策略自身的 log-prob 前向。
2. 在 `metrics.log` 对照 `episode_len`、`num_trajectories`、`success_once`、`actor/grad_norm`。RoboDojo 的 `official_score` 包含叠起部分碗的得分。当前 GRPO 的稀疏奖励取决于任务成功。PPO 即使没有成功也可能经由 critic 和超时 bootstrap 得到非零梯度。
3. 在 DCP 中检查动作头。Pi0.5 核验 `action_out_proj.weight`，Abot 核验 `action_model.action_decoder.layer2.weight`。对照初始权重或前一个 checkpoint，并检查同一参数的 Adam `step` 和 `exp_avg`。恢复时日志应出现 `Resuming training from checkpoint directory`，新存档的 Adam 计数应接续旧值。

## 6. L20-3 实测记录

2026-11-04，RLinf 0.4.0（`e2aeeadd`）、物理 GPU 6/7。两套宿主的八种组合均从手册的标准入口完成了真实 rollout、actor 训练循环和 `global_step_1` 保存。以下成功数均为训练采样。另运行适配器和 OpenPI CPU 回归，128 项通过。

**PPO。** RoboTwin 的 Pi0.5 成功 2/2，Abot 成功 0/2；两者的梯度范数分别是 272.0、337.3，动作头均有非零 Adam 一阶矩。RoboDojo 的两种策略分别完成 2 条 100 步轨迹，成功数均为 0，首轮梯度范数分别为 143.8、789.0。随后各自用新进程加载 `global_step_1`，都保存了 `global_step_2`。动作头的 Adam 计数从 1 增到 2，相对第 1 步的最大权重变化分别是 `5.01e-6`、`5.00e-6`。这些短回合验证了可恢复的更新，未检验成功率提升。

**GRPO 的小批量诊断。** RoboTwin 首批各采样 8 条，Pi0.5 全成功，Abot 全失败。两个跑次的 `loss_mask_fraction` 和动作头梯度均为 0。RoboDojo 的默认示例按两组各两条、800 步上限分别采样，两种策略均成功 0/4，平均官方得分分别为 15 和 11.25。`loss_mask_fraction=1`，但组内优势、策略梯度及 Adam 一阶矩均为 0。这里的 16 次 Adam 计步不能作为奖励驱动更新的证据。

**RoboTwin GRPO 的扩大采样。** 8 个环境采样 4 轮，共 32 条轨迹。Abot 采用每组 4 条，成功 18/32，有效掩码比例 0.562、梯度范数 82.393，动作头 Adam 一阶矩最大值 0.01303，权重相对初始最大变化 `3.75e-5`。Pi0.5 在每组 4 条的对照中成功 32/32，奖励过滤后没有有效样本。改用第 3 节的每组 8 条、global batch 32 后，成功 27/32，有效掩码比例 0.414，梯度范数 100.4。动作头 Adam 计数 4、一阶矩最大值 0.01021，权重最大变化 `2.00e-5`。

**RoboDojo GRPO 的固定布局对照。** 保持原始成功奖励和 800 步上限，额外设置 `env.train.use_fixed_reset_state_ids=true env.train.seed=27 algorithm.group_size=4`，每种模型在同一训练布局各采 4 条。Pi0.5、Abot 仍均成功 0/4，平均官方得分分别为 3.75 和 0。加上默认示例，两种模型各有 8 条完整轨迹，均只验证到 rollout、优势计算、训练循环和存档；这些样本没有产生非零 GRPO 更新信号。

**存档位置。** 在相应宿主的 `XPolicyLab` 根目录下，`infra/RLinf/results/20261104-joint-PL77Hx-<cfg>/` 保留 RoboDojo 四个组合及 RoboTwin Abot GRPO 的标准训练产物。RoboDojo 的两个 PPO 均有第 1、2 步，两个 GRPO 保留默认示例的第 1 步。RoboTwin Abot GRPO 保留 18/32 那轮第 1 步。RoboTwin 的其余三个组合已有此前的可恢复存档，本次复测的重复分片已省去。[此前的存档](TRAINING_RESULTS.md#2026-11-04-abot-与-openwam-验证) · [Pi0.5 有效 GRPO 更新与恢复记录](OPENPI_TRAINING.md#robotwin-adjust_bottle-真实仿真与断点恢复)

L20-3 上的 `/personal/zijian/test/RLinf/validation-artifacts/joint-rl-PL77Hx/verified.json` 记录每次检查的动作头、权重差、Adam 时钟、奖励及保留的 checkpoint 路径；`runs.tar.gz` 收录启动参数、原始日志与 CPU 测试结果。两者都在训练 `results/` 之外。
