# OpenWAM → RLinf 首轮实验手册

更新：2026-10-23。状态：**实验代码已实现，CPU 契约测试通过；真实 OpenWAM 权重、GPU/FSDP、RoboTwin 仿真和学习效果尚未验证。** 本文中的服务器命令是待执行实验步骤，不是已完成实验记录。

对应代码基线：XPolicyLab `b2594786` 加本次修改，RLinf 子模块 `0067f7d5ce525c18bda439ed8dbb638abafd6a06`。请一起传输本次新增文件；仅 `git diff` 不包含未跟踪文件。不要先升级 RLinf 子模块或替换内置 OpenWAM。

## 1. 本次实现什么

面向 **OpenWAM RoboTwin EEF checkpoint + aloha_agilex + adjust_bottle**，使用 RLinf 具身训练循环及 NFT 动作损失。默认 RoboDojo 服务仍使用原来的标定和推理逻辑。

数据走向：三路 RGB / 双臂末端状态 → OpenWAM 联合视频动作生成 → EEF20 → EE16 → RoboTwin `take_action(action_type="ee")` → 任务成功奖励 → 组内优势 → 动作分支 NFT 更新。

- 只支持 `DualSystemSelfAttnArchitecture`、同步去噪、CFG=1。其他 checkpoint 会报错，不能仅通过删除检查来适配。
- 生成完整动作序列，每次只执行第一步；训练损失也只计算第一步及 checkpoint 声明的 20 个有效 EEF 维度。模型内部维度可能为 80，环境执行维度为 16，两者不可混用。
- 从原生去噪循环记录一个随机步骤：视频 latent、动作 latent、两条时间轴、文本/视觉条件、旧动作速度、最终归一化动作。更新时重放这些条件，不套用 OpenPI 的线性时间表。
- 仅动作 backbone 的参数可训练；视频、proprio 投影等其他参数冻结。视频网络仍参与有梯度的计算，不能用整段 `no_grad` 截断跨模态梯度。
- 目标为 `x0 = x_t - sigma_a * v` 的正负混合能量；`sigma_a` 来自实际 action timestep / scheduler 训练步数。组内优势裁剪后映射到 [0,1]，控制正负分支权重。这是**固定已采样视频条件下的实验性替代目标**，未证明等价于完整联合轨迹的策略梯度。
- 使用 `nft_tau=1`，每轮更新后同步当前权重。`prev_logprobs` 的零张量只满足 RLinf 传输形状协议；它不代表真实概率，也不进入此 NFT 损失。PPO/GRPO 概率重算接口仍未实现。`adv_type=grpo` 在这里仅表示组内相对优势。
- 首轮仅支持一个 actor rank、FSDP1 `no_shard`、`use_orig_params=true`，关闭 offload。actor 和 rollout 各持有一份模型，显存需求不能按单次推理估计。
- 仿真与模型使用**同一个 Python 环境**。原有其他策略的 `ROBOTWIN_SIM_PYTHON` 子进程机制不适用于本入口。暂不支持分离 Python、FSDP2、多 actor rank、异步去噪、跨多步动作 chunk 和 PPO。

上游 [PR #1597](https://github.com/RLinf/RLinf/pull/1597) 是 OpenWAM SFT 相关工作，可跟踪参考；本地实现不依赖该 PR，也不据此宣称官方已有可运行的 OpenWAM 在线 RL 样例。

## 2. 文件入口

- `infra/RLinf/run_openwam.sh`：`train` / `eval` 启动入口。
- `infra/RLinf/config/robotwin_adjust_bottle_nft_xpolicylab_openwam.yaml`：短训练配置。
- `infra/RLinf/config/robotwin_adjust_bottle_xpolicylab_openwam_eval.yaml`：独立评估配置。
- `infra/RLinf/openwam_env.py`：RoboTwin 末端状态、EE 执行和单步奖励适配。
- `infra/RLinf/openwam_nft_worker.py`：替换默认 NFT 重采样及 loss，复用原训练循环。
- `policy/OpenWAM/model.py`、`rl.py`、`robotwin.py`：模型桥接、重放损失、动作转换。
- `policy/OpenWAM/rlinf_probe.py`：真实权重的重放/梯度检查，输出 JSON。
- `infra/RLinf/capture_openwam_obs.py`：从仿真初态导出真实观测 NPZ。

RLinf 子模块未修改。内置 OpenWAM 仅在 `engine.generate_batch` 和架构去噪循环增加可选 observer；不传 observer 时保持原调用行为。

## 3. 服务器准备

### 3.1 同步源码与必要配置

建议目录结构：

```text
/workspace/RoboTwin/                  # 当前父项目，保留 env_cfg
  env_cfg/aloha_agilex.yml
  env_cfg/robot/_robot_info.json
  XPolicyLab/                        # 含本次修改、新文件、内置 OpenWAM
    infra/RLinf/RLinf/               # 固定为上述 commit
/workspace/RoboTwin-RLinf/           # 单独的 RoboTwin RLinf_support checkout
/data/openwam_robotwin_eef/          # 实际 checkpoint
/data/robotwin_assets_parent/assets/ # 仿真资源
```

`env_cfg/` 在 XPolicyLab 的父目录，不能只复制 XPolicyLab 后省掉它。另一个 `RoboTwin-RLinf` checkout 用于提供 `robotwin.envs.vector_env`。启动脚本把它放在 PYTHONPATH 首位并切到该目录，避免父项目的 `envs` 遮蔽它。

```bash
export XPL_ROOT=/workspace/RoboTwin/XPolicyLab
export RLINF_PATH="$XPL_ROOT/infra/RLinf/RLinf"
export ROBOTWIN_RUNTIME_PATH=/workspace/RoboTwin-RLinf
export POLICY_MODEL_PATH=/data/openwam_robotwin_eef
export ROBOTWIN_ASSETS_PATH=/data/robotwin_assets_parent
export EXP_ROOT=/data/experiments/openwam_nft_round1
mkdir -p "$EXP_ROOT"

# 若目标位置尚无该仿真 checkout：
git clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_RUNTIME_PATH"

git -C "$XPL_ROOT" rev-parse HEAD > "$EXP_ROOT/xpolicylab_revision.txt"
git -C "$RLINF_PATH" rev-parse HEAD > "$EXP_ROOT/rlinf_revision.txt"
git -C "$ROBOTWIN_RUNTIME_PATH" rev-parse HEAD > "$EXP_ROOT/robotwin_revision.txt"
git -C "$XPL_ROOT" status --short > "$EXP_ROOT/worktree_status.txt"
```

`RLinf_support` 是可变分支，务必保存实际 SHA。首次适配依据的是该分支的 [VectorEnv 接口](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/robotwin/envs/vector_env.py)；服务器版本变化可能需要调整。

### 3.2 Python / CUDA / 仿真依赖

采用 Linux x86_64 + NVIDIA CUDA；本机 CPU 测试环境不能作为服务器依赖锁。先在独立环境中把 RoboTwin 的资源、SAPIEN 渲染和 `mplib` 规划跑通，再加入模型及 RLinf。保留服务器已有能工作的仿真环境快照，避免直接覆盖其他实验的环境。

从零安装时可参考本地 `infra/RLinf/RLinf/requirements/install.sh` 中 `install_robotwin_env`，以及 [RoboTwin 安装说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)。该函数包含 SAPIEN/MPLib 的补丁；单独 `pip install sapien` 不等于完成安装。当前 RLinf 的 env-only 安装入口不支持 RoboTwin，**不要使用 `--env robotwin` 而省略 model 来期待自动安装成功**。

候选模型环境（待服务器验证）：Python 3.11、Torch 2.11 / CUDA 12.8。Torch 必须与驱动、torchvision、FlashAttention 等二进制扩展匹配。可在已具备 RoboTwin 的同一环境中执行：

```bash
# 先激活你为本实验准备的环境；以下命令中的 python 必须是它。
python -V
which python
nvidia-smi > "$EXP_ROOT/nvidia-smi.txt"

# 若该环境还没有匹配的 PyTorch；已有工作组合可先保留并记录版本。
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

cd "$XPL_ROOT"
bash policy/OpenWAM/install.sh
bash infra/RLinf/install_rl_runtime.sh
# 安装 RLinf 基础依赖；不要安装 embodied extra 覆盖模型的 transformers 版本。
python -m pip install -e "$RLINF_PATH"
python -m pip install pytest rich tensorboard imageio imageio-ffmpeg
python -m pip check
python -m pip freeze > "$EXP_ROOT/pip-freeze.txt"
```

内置 OpenWAM 要求 `transformers>=5.5,<6`、`diffusers>=0.38,<1`；RLinf 某些模型额外依赖要求 Transformers 4.x。依赖冲突应在此阶段解决，不要用 Pi_05/OpenVLA 的整套安装流程最后覆盖 OpenWAM。`install_rl_runtime.sh` 主要安装调度依赖，不负责安装仿真。

资源目录按 runtime `task_config/_embodiment_config.yml` 中 `file_path` 的相对路径检查。例如其中是 `assets/embodiments/aloha-agilex` 时，`ROBOTWIN_ASSETS_PATH` 应为 `assets` 的父目录。还需要物体资源、机器人模型、相机配置、SAPIEN/PhysX 和渲染库；缺失时先排查捕获观测步骤，不进入训练。

```bash
export PYTHONPATH="$ROBOTWIN_RUNTIME_PATH:$(dirname "$XPL_ROOT"):$RLINF_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ROBOT_PLATFORM=ALOHA
export ASSETS_PATH="$ROBOTWIN_ASSETS_PATH"
cd "$ROBOTWIN_RUNTIME_PATH"
python -c 'import torch, sapien, mplib, rlinf, openwam, transformers; print(torch.__version__, transformers.__version__); assert torch.cuda.is_available()'
python -c 'import robotwin.envs.vector_env as v; print(v.__file__)'
python -c 'from XPolicyLab.utils.process_data import get_robot_action_dim_info; print(get_robot_action_dim_info("aloha_agilex"))'
```

最后一项应得到双臂 `arm_dim=[6,6]`、`ee_dim=[1,1]`。初始化 import 的报错、GPU 可见性、Vulkan/EGL/PhysX 错误都应先解决并保留完整日志。本次未提供已验证的统一 Docker 镜像或完整 Linux lockfile。

### 3.3 Checkpoint 与四元数顺序：必须先对照

checkpoint 至少应有 `config.yaml`、模型 safetensors、对应归一化统计和加载器需要的 tokenizer/视频模型资产。使用官方 RoboTwin **EEF** 权重，不能用 RoboDojo 权重，也不能用 joint 权重。路径可以是本地目录，不需要改动文件命名规则。

```bash
python - <<'PY'
import os
from omegaconf import OmegaConf
c = OmegaConf.load(os.path.join(os.environ['POLICY_MODEL_PATH'], 'config.yaml'))
for key in ['dataloader', 'model', 'inference']:
    print(key, OmegaConf.to_yaml(c.get(key, {})))
PY
```

确认训练机器人/相机布局、绝对 EEF 控制、统计文件、unify map 和实际模型结构一致。适配器只检查可明确识别的部分，无法自动证明机器人、数据和物理坐标一致。

**四元数不是只看数值范围就能判定。** XPolicyLab 边界固定为 `wxyz`；OpenWAM 原生 RoboTwin EEF helper 假设 `xyzw`。本地 RoboTwin `_trans_endpose` 使用 `transforms3d.mat2quat`，输出为 `wxyz`，与 OpenWAM 客户端注释不同。请检查目标 runtime 的 observation 产生处及 EE 控制器消费处，并与真实 checkpoint 数据约定对照。

```bash
rg -n 'def _trans_endpose|mat2quat|get_left_ee_pose|get_right_ee_pose' "$ROBOTWIN_RUNTIME_PATH/envs"
rg -n 'action_type.*ee|quat|Pose\(' "$ROBOTWIN_RUNTIME_PATH/envs/_base_task.py"
# 以下仅在确认你的 runtime 两端均为 wxyz 后设置；若实际为 xyzw 则改为 xyzw。
export ROBOTWIN_QUATERNION_ORDER=wxyz
```

必须用一个**非单位旋转**样本对比原生客户端与适配器：观察 EEF20 的旋转列、位置单位、左右臂次序、夹爪开闭含义，执行后检查机器人朝向。round-trip 单测只能证明转换互逆，不能证明 checkpoint 训练时的约定正确。若训练数据本身按另一种顺序解释了四元数，需要按真实数据建立显式兼容变换；不能只切换上述 runtime 参数掩盖错误。

## 4. 依次执行实验

所有日志建议用 `set -o pipefail` 配合 `tee`，避免上游失败却误认为命令成功。每次实验使用独立 `experiment_name`。

### A. CPU 契约测试与配置展开

```bash
cd "$XPL_ROOT"
python -m pytest policy/OpenWAM/tests/test_rl.py infra/RLinf/tests/test_adapter.py \
  -k 'not abot' -q | tee "$EXP_ROOT/cpu-tests.log"
bash -n infra/RLinf/run_openwam.sh
bash infra/RLinf/run_openwam.sh train --cfg job --resolve > "$EXP_ROOT/train-resolved.yaml"
bash infra/RLinf/run_openwam.sh eval --cfg job --resolve > "$EXP_ROOT/eval-resolved.yaml"
```

预期：30 个测试通过。排除的 Abot 测试依赖它自己的完整模型环境，与本适配无关。`--cfg` 仅展开配置，不启动 Ray，也不证明 GPU 入口已跑通。独立评估请使用 `eval` 子命令，不能把训练配置的 `runner.only_eval` 改为 true 来代替。

### B. 真实观测、去噪重放与梯度

先只加载仿真，不加载模型：

```bash
cd "$ROBOTWIN_RUNTIME_PATH"
python "$XPL_ROOT/infra/RLinf/capture_openwam_obs.py" \
  --output "$EXP_ROOT/initial_obs.npz" 2>&1 | tee "$EXP_ROOT/capture.log"
```

输出 NPZ 包含 `main_images [B,H,W,3]`、`wrist_images [B,2,H,W,3]`（RGB uint8）、`states [B,16]`（绝对 EE、wxyz）、Unicode `task_descriptions [B]`。捕获的是 seed=0 初态，不执行动作。

然后运行真实权重预检：

```bash
python -m XPolicyLab.policy.OpenWAM.rlinf_probe \
  --checkpoint "$POLICY_MODEL_PATH" --obs "$EXP_ROOT/initial_obs.npz" \
  --backward --gradient-checkpointing \
  --output "$EXP_ROOT/probe.json" 2>&1 | tee "$EXP_ROOT/probe.log"
```

默认把观测重复两次，检查 B=2 rollout、B=2 replay 和 B=1 microbatch replay 的一致性。固定检查第一个去噪步骤，避免末步自一致误差恰好为零导致无梯度。默认 `atol=rtol=0.02` 是 bf16 调试起点，必须保留实测最大/平均误差；不要通过不断放宽阈值掩盖条件丢失。

验收：JSON `status=passed`、`replay_pass=true`、`backward_pass=true`，梯度 finite 且非零，冻结参数无梯度，动作 `[B,1,16]` 有限；保留峰值显存与耗时。本检查人为使用非零优势，仅证明计算链路能产生梯度，**不证明真实奖励能提供学习信号**。它不执行 optimizer step、不修改原 checkpoint，也不验证 FSDP。

暂时无法启动仿真时可以省略 `--obs`，但 JSON 会标记 `SYNTHETIC`；这只用于形状/梯度检查，不能代替真实观测和物理执行对照。

### C. 未训练基线评估

```bash
cd "$XPL_ROOT"
bash infra/RLinf/run_openwam.sh eval \
  runner.logger.log_path="$EXP_ROOT" \
  runner.logger.experiment_name=baseline_eval \
  2>&1 | tee "$EXP_ROOT/baseline_eval.log"
```

默认 2 个环境，单次评估按配置运行 200 个执行步，保存评估视频。检查机器人动作、左右腕相机、任务描述、成功/超时标志和结束后的 reset；用原生 OpenWAM RoboTwin 评估对照相同机器人、任务、种子和 `replan_steps=1`。官方默认 horizon 的成绩不能直接与这个每步重规划版本相比。

若原生推理能成功而此适配明显异常，先排查转换、相机布局、归一化、seed 和重规划设置；不要进入长训练。

### D. 两轮训练链路检查

```bash
bash infra/RLinf/run_openwam.sh train \
  runner.logger.log_path="$EXP_ROOT" \
  runner.logger.experiment_name=nft_smoke \
  2>&1 | tee "$EXP_ROOT/nft_smoke.log"
```

默认 2 个环境、组大小 2、16 个执行步/采样轮次、2 个外层训练轮次、microbatch=1、global_batch=2、lr=1e-6。**两轮不是两次 optimizer step**：一次采样产生多个 minibatch。检查采样→传输→重放→反传→权重同步→checkpoint 保存是否完整。

`actor/nft_loss`、`actor/grad_norm`、`actor/adv_abs_mean`、`actor/zero_adv_fraction`、`actor/replay_v_mse`、`actor/clip_frac` 应有记录。`replay_v_mse` 在 minibatch 更新后可增大，不能要求整个 epoch 都为零。

16 步通常不足以完成任务；组内全成功或全失败会导致优势为零。此时梯度可以为零，loss 非零也不代表正在有效学习。初始 on-policy 点零优势应产生零梯度；配置将 weight decay 设为 0，避免把无奖励时的衰减误认为奖励学习。不要修改成功奖励或伪造正样本让日志“好看”。

验收必须包括：无 NaN/Inf、无 shape/mask 异常、进程完整结束、保存文件存在。真实奖励产生非零优势后，再要求动作参数有更新，并核查冻结参数未改变。

### E. 恢复与训练后独立评估

当前 RLinf 保存路径为：

```text
$EXP_ROOT/nft_smoke/checkpoints/global_step_<N>/actor/
  dcp_checkpoint/                  # 模型 + optimizer + scheduler 的训练恢复状态
  model_state_dict/full_weights.pt # 给 rollout 直接加载的完整 state_dict
```

以实际日志和目录中的 N 为准，不假定一定从 1 开始编号。

```bash
export SAVED_STEP="$EXP_ROOT/nft_smoke/checkpoints/global_step_<实际N>"
test -f "$SAVED_STEP/actor/model_state_dict/full_weights.pt"

bash infra/RLinf/run_openwam.sh eval \
  runner.ckpt_path="$SAVED_STEP/actor/model_state_dict/full_weights.pt" \
  runner.logger.log_path="$EXP_ROOT" runner.logger.experiment_name=smoke_after_eval \
  2>&1 | tee "$EXP_ROOT/smoke_after_eval.log"

# 增加总外层轮数；resume_dir 指向含 actor/ 的 global_step_N 目录。
bash infra/RLinf/run_openwam.sh train \
  runner.resume_dir="$SAVED_STEP" runner.max_epochs=3 \
  runner.logger.log_path="$EXP_ROOT" runner.logger.experiment_name=nft_resume \
  2>&1 | tee "$EXP_ROOT/resume.log"
```

`POLICY_MODEL_PATH` 始终指向原始 OpenWAM 目录以提供结构、统计和 tokenizer；`runner.ckpt_path` 在构建后覆盖 RLinf 包装器权重。`full_weights.pt` 不是原生 OpenWAM 发布目录，不能直接拿它当 OpenWAM server 的 ckpt_dir。训练恢复不承诺复现仿真状态及随机数流，所以不能宣称逐位续跑一致。

### F. 有效学习实验

A–E 通过后，至少让采样覆盖完整 episode，再增加环境/组数。首个对照可以保留组大小 2，增大采样到 200 步：

```bash
bash infra/RLinf/run_openwam.sh train \
  env.train.max_steps_per_rollout_epoch=200 \
  runner.max_epochs=20 runner.save_interval=5 \
  runner.logger.log_path="$EXP_ROOT" runner.logger.experiment_name=nft_learning_seed1234 \
  2>&1 | tee "$EXP_ROOT/learning.log"
```

20 轮只是候选预算，不保证收敛。先确认奖励方差及非零优势比例，再尝试 group_size=4、total_num_envs=4；两者必须整除，且注意 replay 视频 latent 也占用大量 CPU 内存和传输带宽。

用独立 eval seeds 比较原 checkpoint 与训练后 checkpoint。增加评估样本可同时设置 `env.eval.total_num_envs=4 env.eval.rollout_epoch=5`；前后配置必须一致，并报告实际完成 episode 数，而不是仅报告配置中的环境数。至少重复 3 个训练 seed；报告每个 seed 的成功数/总数、均值与波动、训练 GPU 小时。

首版不支持通过切换 `full_shard`、FSDP2 或 offload 来临时救显存。先测 B 阶段单模型峰值；若 actor+rollout 同卡 OOM，可在复制的配置中把 actor 放 GPU0、env/rollout 放 GPU1（各自保持一个 rank），重新做 B–E。这个分卡组合也需要实际验证。勿直接减小 checkpoint 的视频长度、分辨率或 denoise_steps 后与原基线混比。

## 5. 失败定位与反馈

- **导入/配置失败**：发送完整 traceback、`which python`、pip freeze、resolved YAML、三个源码 SHA；优先排查 Transformers/FlashAttention/Torch、runtime checkout 遮蔽和父目录 env_cfg。
- **capture 失败**：检查 assets 父目录、SAPIEN 渲染、mplib、相机和 endpose 开关。模型还没加载，先定位仿真问题。
- **重放误差大**：核对条件 tensor、两条 timestep、CFG、dtype、train/eval、B=2/B=1 差异及 observer 位置。不要直接启动训练。
- **无梯度**：检查 action backbone 的 trainable 参数、是否被原生 freeze 的 no_grad 包装、选取的去噪步及 x0 自一致误差；真实训练再检查优势是否全零。
- **动作异常但重放通过**：优先查四元数、左右臂、绝对/相对动作、统计量与相机；神经网络数值一致不代表物理语义正确。
- **零成功率**：先和同 checkpoint 的原生推理比较。基线也全失败时，稀疏奖励 NFT 可能没有学习信号，不能简单增加轮数保证改善。
- **保存/恢复失败**：提供 checkpoint 目录树、state_dict 缺失/额外 key、首次保存与加载日志；不要用 `strict=False` 跳过未知的不匹配。

请按以下模板反馈，并附 `probe.json`、完整报错日志、展开配置、训练指标及代表性视频：

```text
实验编号 / 日期：
XPolicyLab SHA + 修改包 / RLinf SHA / RoboTwin SHA：
checkpoint 名称 / config hash / 数据机器人与动作约定：
GPU 型号、数量、显存 / driver / CUDA / Python / Torch：
四元数顺序的代码依据：
A CPU 测试、配置展开：
B 真实观测 probe：batch/microbatch 误差、gradient_norm、峰值显存、耗时：
C baseline：成功数/总 episode、异常动作/视频：
D smoke：是否保存、loss/grad_norm、非零优势比例、NaN、GPU/CPU 峰值：
E 恢复与权重加载：
F 训练设置、种子、训练前后成功率、GPU 小时：
第一个失败阶段与完整 traceback（不要只截最后一行）：
```

## 6. 本地验证记录

2026-10-23，macOS ARM，临时 CPU 环境：Python 3.13.13、Torch 2.14.1、NumPy 2.5.4、pytest 9.1.1。

- `policy/OpenWAM/tests/test_rl.py`：10 个检查通过，覆盖正/负/零优势梯度、有效动作/维度掩码、非法输入、重放快照、state_dict 往返、冻结权重以及非单位旋转转换。
- `infra/RLinf/tests/test_adapter.py -k 'not abot'`：20 个检查通过，含新增 NFT 能力门禁/梯度和已有 PPO/GRPO/eval 桥接检查。
- 尝试完整 suite 时，Abot 相关用例因本机未安装其 Transformers 模型依赖失败；未将该结果记为通过，也未修改无关 Abot 实现。
- 两个 Hydra 配置的 compose / resolve 通过。Python 编译、启动脚本语法及新增模块静态检查见本次交付说明。
- **未运行**：真实 checkpoint、原生服务/调试客户端、SAPIEN、CUDA、FSDP、Ray 集群训练、权重同步、真实存档恢复或成功率实验。CPU 测试不覆盖这些系统边界。
