# Pi_05 天机 Marvin 真机改造说明

本文总结当前工作区里 `policy/Pi_05` 相对 `main` 的改动：为什么改、改完要达成什么、以及每个文件具体改了什么。

## 原因与目的

原先的 Pi_05 适配面向 Aloha 仿真关节空间（`arx_x5`、`action_type=joint`）。数据转换把左右臂关节和夹爪拼成一条状态，再用“下一帧状态当作当前动作”伪造 action；训练配置把动作切成固定 14 维关节，并用关节增量（前 6 维相对、夹爪绝对）。这套假设和天机 Marvin 真机数据对不上：

- HDF5 里已经有独立的 `state` 和 `action` 组，动作不该再由状态平移伪造。
- 末端控制的臂维度是位姿（xyz + 四元数，宽 7），不是机器人关节自由度。`ee_dim` 仍是夹爪，不能跟着臂维度一起改。
- 双臂末端打包后是 16 维（每臂 7 维位姿 + 1 维夹爪）。模型内部 `action_dim=32`，输出必须切回 16 维，不能再用 Aloha 的 14 维关节切片。
- 位姿增量不能沿用关节的逐维相减。位置是平移差，旋转是相对四元数 `q_action * q_state^{-1}`，夹爪保持绝对量。
- 真机采集频率写在 `additional_info/frequency`（常见 30 Hz），不能再写死 50。
- 语言标注可能在 `instructions` 或 `instruction` 下，类型是 bytes、字符串或数组。
- 后续实验要在同一套末端增量动作上换数据规模、只训左腕相机、以及按 stride / 15 fps 抽帧，并复用已有归一化统计。

目标是：用框架自己的 `pack_robot_state` 读真机 HDF5，支持绝对末端、末端增量、绝对关节三条训练线，并让部署侧按 16 维末端动作还原。

## 动作布局

末端（`action_type=ee`）单臂 8 维，双臂 16 维：

| 区间 | 含义 | 增量训练时 |
| --- | --- | --- |
| 0:3 / 8:11 | xyz | 减去当前状态位置 |
| 3:7 / 11:15 | 四元数 wxyz | `q_action * q_state^{-1}`，并归一化到 w ≥ 0 |
| 7 / 15 | 夹爪 | 保持绝对值 |

关节配置 `pi05_real_tianji_marvin_joint` 也是 16 维（每臂 7 关节 + 1 夹爪），但不做增量。`ee_dim` 始终表示夹爪宽度，末端位姿宽度写进 `arm_dim`。

## 具体修改

### 数据转换

`openpi/scripts/process_data.py`

- LeRobot 导入从 `lerobot.common.datasets` 改为当前环境的 `lerobot.datasets`。
- 图像读取改为 `decode_image_bit(group[key][()])`，先取出 HDF5 标量再解码。
- 删除按关节名硬拼、以及把状态平移一帧当作动作的逻辑。改为读取 `state` / `action` 两组，交给 `pack_robot_state`。
- `action_type=ee` 时，位姿宽度从 `left_ee_poses` / `right_ee_poses` / `ee_poses` 的最后一维读取（默认 7），只替换 `arm_dim`，`ee_dim` 保持夹爪。
- 指令兼容 `instructions` 与 `instruction`，并解开 bytes / 字符串 / ndarray。`--instruction` 默认改为 `None`：传入时覆盖整条数据；否则从 HDF5 里随机抽一条；都没有才用 `"Do your job."`。同一条 episode 内 prompt 固定，不再每帧重抽。
- 帧率从首条 episode 的 `additional_info/frequency` 读取，缺省 30，不再写死 50。数据集在读到样本维度之后再创建。
- 找不到 `episode_*.hdf5` 时直接报错，避免建出一个空数据集。

`process_data.sh`

- `--mode` 挪到可选位置参数之后，避免和 `expert_data_num` / `raw_task_dirs` 交错。
- 设置 `OPENPI_INSTRUCTION` 时追加 `--instruction`。

### 部署与推理

`model.py`

- `action_type=ee` 时把 `robot_action_dim_info["arm_dim"]` 改成每臂 7，`ee_dim` 原样保留。推理打包和还原因此按 xyz+quat+gripper，而不是关节自由度。

`deploy.yml`

- 默认环境改为 `tianji_marvin`，`action_type` 改为 `ee`。
- 训练配置改为 `pi05_real_tianji_marvin_ee`，`repo_id` 改为 `RoboDojo_real-pick_magnet-tianji_marvin-ee`。
- 换别的实验配置时，`train_config_name` 必须和 `train.sh` 使用的 OpenPI config 一致。

### 末端增量与相机掩码

`openpi/src/openpi/transforms.py`

- 新增 `DeltaEEActions` / `AbsoluteEEActions`。输入侧把绝对末端动作变成相对当前状态的增量；输出侧做逆变换。夹爪不参与。
- 四元数先归一化，再把 `w < 0` 的四元数取反，避免 `q` 和 `-q` 表示同一旋转时数值乱跳。

`openpi/src/openpi/policies/aloha_policy.py`

- `AlohaInputs` 增加 `enabled_cameras`。未启用或缺失的相机槽填黑图，并把 `image_mask` 设为 False。默认仍要求 `cam_high`，三相机布局不变。
- 新增 `BimanualEEOutputs`，把模型补齐后的动作切回 `action_output_dim`（末端为 16）。关节 Aloha 路径仍走原来的 `AlohaOutputs`。

`openpi/src/openpi/training/config.py`

- `LeRobotAlohaDataConfig` 增加 `use_delta_ee_actions`、`action_output_dim`、`enabled_cameras`。
- 关节增量和末端增量互斥。末端增量要求 `action_output_dim` 能被单臂 8 维整除。
- `adapt_to_pi=False`：真机数据不做 Aloha 关节空间翻转。

`openpi/src/openpi/transforms_test.py`、`openpi/src/openpi/policies/aloha_policy_test.py`

- 覆盖末端增量往返、单位状态下动作不变、默认三相机、仅左腕时其余槽位黑图且 mask 为 False，以及左腕配置确实挂上 `DeltaEEActions`。

### 训练配置

权重都从本地 π0.5 base 加载：

`/vepfs-cnbje63de6fae220/xspark_shared/model_weights/openpi-assets/checkpoints/pi05_base/params`

模型统一 `Pi0Config(pi05=True, action_dim=32)`，`fsdp_devices=2`，`seed=0`。

| config 名 | 数据 repo | 动作 | 步数 | batch | 备注 |
| --- | --- | --- | --- | --- | --- |
| `pi05_real_tianji_marvin_ee` | `RoboDojo_real-pick_magnet-tianji_marvin-ee` | 绝对末端 16 维 | 60000 | 144 | 当前 `deploy.yml` 默认 |
| `pi05_real_tianji_marvin_ee_delta` | 同上 | 末端增量 | 30000 | 144 | 与绝对末端同一份数据 |
| `pi05_real_tianji_marvin_ee_delta_all` | `..._pick_magnet_all-...-ee` | 末端增量 | 30000 | 144 | 9012+9017+9018+0919，1512 条 |
| `pi05_real_tianji_marvin_ee_delta_full` | `..._pick_magnet_full-...-ee` | 末端增量 | 30000 | 144 | 再加 9019，2003 条 |
| `pi05_real_tianji_marvin_joint` | `..._pick_magnet_full-...-joint` | 绝对关节 16 维 | 30000 | 144 | 同一批 full 数据的关节版 |
| `pi05_real_tianji_marvin_ee_delta_full2` | `..._pick_magnet_full2-...-ee` | 末端增量 | 30000 | 144 | full 再加 0922/0923，3003 条 |
| `..._full2_left_wrist` | 同上 full2 | 末端增量 | 30000 | 144 | 只启用 `cam_left_wrist` |
| `..._full2_left_wrist_stride2` | `...-ee-stride2` | 末端增量 | 30000 | 128 | 只训左腕；归一化复用 full2 |
| `..._full2_fps15` | `...-ee-fps15` | 末端增量 | 20000 | 128 | 15 fps 降采样；每 10000 步存档；归一化复用 full2 |
| `..._full2_stride2` | `...-ee-stride2` | 末端增量 | 20000 | 128 | 隔帧保留、仍标 30 fps；归一化复用 full2 |
| `..._full2_stride3` | `...-ee-stride3` | 末端增量 | 20000 | 128 | 每 3 帧保留；归一化复用 full2 |

复用归一化的配置把 `assets_dir` 指到 `policy/Pi_05/openpi/assets/pi05_real_tianji_marvin_ee_delta_full2`，`asset_id` 为 `RoboDojo_real-pick_magnet_full2-tianji_marvin-ee`。抽帧改变的是时间密度，状态和动作的数值分布沿用 30 fps full2，避免在短序列上重算统计。

### 训练脚本

`train.sh`

- 增加可选环境变量 `OPENPI_BATCH_SIZE`，设置后传给 `--batch-size`，用来覆盖 config 里的 batch。

`run_norm_then_train.sh`（新文件）

- 先跑 `scripts/compute_norm_stats.py`，再调用 `train.sh`。
- 默认 config 是 `pi05_real_tianji_marvin_ee`，repo 是 `RoboDojo_real-pick_magnet-tianji_marvin-ee`，`action_type=ee`，checkpoint 名 `pick_magnet`，GPU `2,3,4,5,6,7`。
- `OPENPI_SKIP_NORM=1` 跳过统计；`OPENPI_NORM_MAX_FRAMES` 限制统计帧数；`OPENPI_NORM_GPU` 指定统计用的卡（默认 2）。
- `HF_LEROBOT_HOME` 默认 `/vepfs-cnbje63de6fae220/zijian`。

## 使用时注意

- 末端训练、归一化、部署必须使用同一套 config 和同一份 norm stats。绝对末端和末端增量的统计不能混用。
- `deploy.yml` 里的 `train_config_name` 要改成实际训出来的那一个（例如 `pi05_real_tianji_marvin_ee_delta_full2`），`repo_id` 要和该 config 的 asset id 一致。
- 权重路径和 `run_norm_then_train.sh` 里的数据根目录是这台机器上的绝对路径，换环境需要改。
- 本地 `data/` 与 `env_cfg` 符号链接不属于这次代码改动，没有放进提交。
