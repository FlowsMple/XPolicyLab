# RLinf 近期训练结果（RoboTwin `adjust_bottle`）

整理日期：2026-10-22。数据来自 2026-10-10 至 2026-10-16 的训练日志，指标按 50 步（或注明的窗口）平均，避免单步波动。

任务都是 `adjust_bottle`，本体 `aloha-agilex`，关节动作 14 维，动作块 50，回合长度 200。机器是 8 张 L20（每张 44.4 GB），actor、rollout、env 放在同一组 GPU 上。成功率指日志里的 `success_once`；这些跑次里它和 `return` 相同，`episode_len` 保持 200。

## 结论

1. **Pi05 PPO（24 环境）是唯一同时有策略更新和定期评估的跑次。** 训练成功率从约 72% 升到约 81%。24 条固定种子上的 eval 从早期约 73% 到后期约 88%，第 100 步为 95.8%（23/24）。第 129 步为了改测 Abot GRPO 被停下，checkpoint 在 `global_step_120`。
2. **Pi05 GRPO 的策略在更新，训练成功率从约 82% 升到约 90%，但没有 eval。** 从 `global_step_30` 续到第 286 步，`approx_kl` 稳定在 0.06 左右。起点已经是强策略（约 72%–80%）。`val_check_interval` 在这次启动里是 `-1`。checkpoint 在 `global_step_280`。
3. **Abot PPO 的训练成功率从约 14% 升到约 71%，critic 也在变好，但日志里的策略比率全程是 1。** 611 步里 `actor/ratio = 1`、`approx_kl ≈ 0`、`clip_fraction = 0`。同一套日志里，Pi05 PPO/GRPO 和后来的 Abot GRPO 都有非零 KL。因此 71% 不能从 PPO 比率读成“策略已经按 PPO 更新学会了”。这次同样没有 eval。checkpoint 在 `global_step_610`。
4. **Abot GRPO 在低成功率上先塌掉，后半段才拉回来。** 前 20 步训练成功率约 19%，随后降到约 2%，eval 连续为 0。`loss_mask_fraction` 从 0.23 降到约 0.05，和 `filter_rewards`（组成功率超出 0.1–0.9 就丢掉）把整组失败滤掉一致，有效更新几乎没了。第 160 步之后训练成功率和 mask 一起回升；第 180、190 步 eval 为 25%（16 条里 4 条）。第 198 步手动停止，checkpoint 在 `global_step_190`。
5. **这组 L20 上跑不住配置里的 64 环境 GRPO。** 仿真用光追（32 spp）加 OIDN，显存要等仿真进程退出才释放。训练和评估两套场景同时常驻会把 44 GB 占满。Pi05 PPO 降到 24 环境后能做 eval；Abot GRPO 最后是 32 个训练环境、16 个评估环境，评估结束退出仿真进程，训练仿真每 40 次全量 reset 重启一次。

Dexbotic DM0 只跑了 2 步 PPO（`success_once = 0`），随后停掉，没有训练结论。

## 结果一览

| 跑次 | 步数 | 训练成功率 | 策略 KL | Eval | 结束 |
|---|---|---|---|---|---|
| Pi05 GRPO | 31→286（从 step 30 续） | 82% → 90% | ≈ 0.06 | 无 | 10-12 停下，改跑 Abot PPO。ckpt 280 |
| Pi05 PPO，64 环境 | 1→9 | ≈ 73% | ≈ 0.05 | 第一次 eval 时 OOM | 无 ckpt |
| Pi05 PPO，32 环境 | 1→29 | ≈ 72%–74% | ≈ 0.06 | 81.3%（32 条，step 10 和 20） | step 30 的 eval OOM。ckpt 20 |
| Pi05 PPO，24 环境 | 1→129 | 72% → 81% | ≈ 0.06 | 见下表，24 条 | 10-14 手动停。ckpt 120 |
| Abot PPO，32 环境 | 1→611 | 14% → 71% | ≈ 0（ratio = 1） | 无 | 10-13 手动停。ckpt 610 |
| Abot GRPO，32 环境 | 1→198 | 19% → 2%，160 步后回到约 20% | 塌缩时接近 0，回升后约 0.05 | 前 150 步为 0；180/190 步 25%（16 条） | 10-16 手动停。ckpt 190 |

训练成功率是窗口平均，不是单步峰值。单步可以到 95% 以上，也可以掉回个位数。

## Pi05 GRPO

目录：`XPolicyLab/infra/RLinf/results/pi05_grpo/`。

权重：`/personal/zijian/rlinf/checkpoints/pi05_robotwin_lerobot_v30_xpolicylab_pytorch`。配置是 GRPO，`group_size 8`，64 环境，`rollout_epoch 4`，每步 256 条轨迹，`global_batch_size 256`，`update_epoch 4`，`filter_rewards` 开，学习率 `5e-6`。实际启动把 `val_check_interval` 设成 `-1`（见该目录 `tensorboard/config.yaml`）。仓库 yaml 里这一项现在仍是 10。

| 步 | 训练成功率 | approx KL | loss mask | 步时 |
|---|---|---|---|---|
| 31–49 | 82.1% | 0.065 | 0.49 | ~327 s |
| 50–99 | 87.0% | 0.063 | 0.39 | ~319 s |
| 100–149 | 86.0% | 0.061 | 0.38 | |
| 150–199 | 84.3% | 0.057 | 0.36 | |
| 200–249 | 89.3% | 0.060 | 0.30 | |
| 250–286 | 89.7% | 0.057 | 0.31 | ~318 s |

前一次跑到第 37 步时，Cursor 远程会话断开超过 3 小时，进程树收到 SIGTERM，日志里没有 Python traceback。从 `global_step_30` 续上后，上面这张表才是连续结果。mask 随成功率下降，是高成功率组被 `rewards_upper_bound: 0.9` 滤掉，可学习的组变少。

每步约 5.3 分钟。没有留出 eval，90% 只说明训练采样变好了。

## Pi05 PPO

日志在共享文件 `XPolicyLab/infra/RLinf/results/metrics.log`（按 Global Step 回绕分成三次），启动日志在 `results/pi05_ppo/`。权重与 GRPO 相同。PPO，`group_size 1`，带 value head，`adv_type: gae`，学习率 `5e-6`，value 学习率 `1e-4`。

三次环境数：

- **64 环境**（每步 128 条）：9 步，训练成功率约 73%，value loss 从 0.34 降到 0.065。第一次 eval 的 `reset` 时 CUDA OOM。一张卡上 env 约 34 GB，rollout 约 7.5 GB，actor 约 2.8 GB。没有 checkpoint。
- **32 环境**（每步 64 条）：29 步。step 10 和 step 20 的 eval 都是 81.3%（32 条）。准备做 step 30 的 eval 时 `cannot create buffer`，同时 OIDN OOM。checkpoint 到 `global_step_20`。
- **24 环境第一次启动**没有进入训练：`AssertionError: 24 is not divisible by 32`，batch 和环境数不整除。改成 `global_batch_size 192` 后重训。这次的设置已经写进 `config/robotwin_adjust_bottle_ppo_xpolicylab_pi05.yaml`。

24 环境重训（每步 48 条训练轨迹，eval 24 条，步时约 81 s）：

| 步 | 训练成功率 | value loss | explained variance | approx KL |
|---|---|---|---|---|
| 1–19 | 72.4% | 0.092 | 0.10 | 0.064 |
| 40–59 | 78.6% | 0.056 | 0.27 | 0.068 |
| 80–99 | 81.8% | 0.050 | 0.25 | 0.084 |
| 120–129 | 81.0% | 0.049 | 0.29 | 0.061 |

Eval（`success_once` 与 `success_at_end` 相同）：

| Step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eval | 79.2% | 70.8% | 66.7% | 75.0% | 87.5% | 79.2% | 79.2% | 79.2% | 83.3% | 95.8% | 83.3% | 87.5% |

24 条里一条就是 4.2 个点，所以 66.7% 和 95.8% 差 7 条成功。前四次 eval 平均约 73%，后四次（90–120）平均 87.5%。方向和训练曲线一致，样本还小。2026-10-14 11:23 为启动 Abot GRPO 手动停止（`SIGTERM`），最后 checkpoint 是 `global_step_120`，121–129 步没保存。

## Abot PPO

目录：`XPolicyLab/infra/RLinf/results/abot_ppo/`。

权重：`/personal/zijian/RoboTwin/ckpt_to_upload/Abot_M0/RoboTwin-cotrain-aloha_agilex-joint-0`，`checkpoint_num: 50000`。PPO，32 环境，`group_size 1`，`rollout_epoch 2`，每步 64 条，`global_batch_size 256`，`update_epoch 1`，学习率 `5e-6`，value 学习率 `1e-4`。这次启动的 `val_check_interval` 是 `-1`。2026-10-12 14:23 开始，2026-10-13 11:10 按要求停下。步时约 81 s。

| 步 | 训练成功率 | value loss | explained variance | ratio | approx KL |
|---|---|---|---|---|---|
| 1–49 | 13.9% | 0.068 | < 0 | 1 | ~0 |
| 100–149 | 23.7% | 0.050 | 0.39 | 1 | ~0 |
| 200–249 | 19.6% | 0.037 | 0.45 | 1 | ~0 |
| 300–349 | 38.1% | 0.058 | 0.47 | 1 | ~0 |
| 400–449 | 61.5% | 0.051 | 0.53 | 1 | ~0 |
| 500–549 | 69.3% | 0.047 | 0.51 | 1 | ~0 |
| 600–611 | 71.0% | 0.046 | 0.55 | 1 | ~0 |

前 250 步大致在 14%–24%，之后才升到 70% 一带。critic 的 explained variance 从负值升到约 0.55，value loss 降到约 0.05。`actor/grad_norm` 全程大约 60。

策略侧的记录与此并行：611 步的 `actor/ratio` 都是 1，`clip_fraction` 都是 0，`approx_kl` 在 `1e-7` 量级。Abot GRPO 使用同样的 `update_epoch: 1`，在成功率回升时 KL 可以到 0.05。所以这次不能把 KL 接近 0 理解成“只更新了一个 epoch 所以比率还来不及变”。训练采样的成功率确实上升了；PPO 日志没有记下对应的策略比率变化。要下“Abot PPO 学会了”这个结论，需要先核对 logprob 是否真的跟着权重走，并补一次固定种子 eval。checkpoint：`abot_ppo/robotwin_ppo_xpolicylab_abot/checkpoints/global_step_610`。

同目录里还有两次很短的开头（2 步、6 步，每步 128 条），其中第一次的 `approx_kl` 是 0.018。611 步那次从第 1 步起比率就是 1。

## Abot GRPO

目录：`XPolicyLab/infra/RLinf/results/abot_grpo_n32/`。权重与 Abot PPO 相同。

最终能跑的配置（已写入 `config/robotwin_adjust_bottle_grpo_xpolicylab_abot.yaml`）：32 个训练环境，`group_size 4`，`rollout_epoch 4`，每步 128 条，`global_batch_size 128`，`update_epoch 1`，`filter_rewards` 开（0.1–0.9），评估 16 环境且 `enable_offload: true`，`sim_restart_resets: 40`。学习率 `5e-6`。步时约 150–170 s，每 10 步的仿真重启大约多 80 s。

更早的两次尝试没有形成曲线：

- 配置里的 64 环境、`group_size 8`：第一次 rollout 之后 `cannot create buffer`，进程退出。日志在 `results/abot_grpo/launch.log`。
- 缩到 32 环境、`group_size 4` 的 2 步冒烟（`abot_grpo_fit`）能跑完，训练成功率 7.8% 然后 10.2%，eval 3.1%。只说明链路通了。

`abot_grpo_n32` 本身分成两段曲线。中间几次是 OOM 后续训，step 81–83 在日志里重复了一份。

塌缩段（先 128 条，大约第 30 步起为了显存改成 24 环境、96 条）：

| 步 | 训练成功率 | loss mask | approx KL |
|---|---|---|---|
| 1–19 | 19.2% | 0.23 | 0.013 |
| 20–39 | 2.9% | 0.05 | 0.003 |
| 40–59 | 2.6% | 0.05 | 0.001 |
| 60–79 | 1.4% | 0.04 | ~0 |
| 80–89 | 2.1% | 0.05 | 0.004 |

Eval：step 10 为 3.1%（32 条），step 20–80 为 0。

从 `global_step_100` 再续、评估改成 16 条之后：

| 步 | 训练成功率 | loss mask | approx KL | Eval |
|---|---|---|---|---|
| 101–119 | 1.4% | 0.05 | 0.006 | step 110：0 |
| 120–139 | 2.6% | 0.06 | 0.006 | 0 |
| 140–159 | 2.3% | 0.06 | 0.004 | 0 |
| 160–179 | 11.4% | 0.17 | 0.013 | step 160、170：6.3% |
| 180–198 | 20.3% | 0.27 | 0.053 | step 180、190：25% |

第 198 步的单步 `approx_kl` 到了 0.66，窗口平均是 0.05。25% 的 eval 只有 16 条，4 条成功，和塌缩前的 32 条 eval 不能直接比。2026-10-16 训练仍在跑时被手动 Ctrl-C（`EXIT:130`）。可续训的权重是 `abot_grpo_n32/robotwin_grpo_xpolicylab_abot/checkpoints/global_step_190`。

## 显存

一张 L20 上同时有 FSDP actor、rollout 模型和 SAPIEN 光追场景。`envs/_base_task.py` 把相机 shader 设成 `rt`，spp 32，path depth 8，并开 OIDN。这几块显存随场景留在仿真进程里，`close` 场景并不会把它们还回去。

因此：

- 只打开顶层 `env.enable_offload` 不够。RLinf 读的是 `env.train` 和 `env.eval` 下面的 `enable_offload`。
- 训练仿真和评估仿真若都常驻，大约是 18 GB + 16 GB，再加约 10 GB 的 rollout，44 GB 就满了。Pi05 PPO 的 64 环境和 32 环境都死在下一次 eval 的 `reset` / `take_picture`。
- 给训练也做每步 offload 可以把显存降到约 27 GB，但每步要从约 150 s 变成约 230 s，因为场景要重建。
- Abot GRPO 采用的折中是：训练仿真常驻，每 40 次全量 reset（约 10 步）退出进程再拉起；评估仿真在评估结束时退出；评估环境数降到 16。评估峰值大约 39 GB，余量约 5 GB。训练仿真若连续活过 10 步，会以大约每步 0.3 GB 继续涨，step 110 的评估因此再次 OOM。重启间隔 40 是按这个涨幅留出来的。

## 日志位置

权威指标在各自目录的 `metrics.log`。Pi05 PPO 没有单独的 metrics 文件，三次都追加在 `results/metrics.log`。这个共享文件不写实验名，只写 Global Step，后面还接了 Abot 的短冒烟。

| 内容 | 路径 |
|---|---|
| Pi05 GRPO 指标、启动日志、ckpt | `XPolicyLab/infra/RLinf/results/pi05_grpo/` |
| Pi05 PPO 启动日志 | `results/pi05_ppo/launch.log`、`launch_n32.log`、`launch_n24_retrain.log` |
| Pi05 PPO 指标 | `results/metrics.log` |
| Pi05 PPO ckpt | `results/robotwin_ppo_xpolicylab_pi05_n32/checkpoints/`、`results/robotwin_ppo_xpolicylab_pi05_n24/checkpoints/` |
| Abot PPO | `results/abot_ppo/` |
| Abot GRPO | `results/abot_grpo_n32/` |
| 每次启动的完整配置 | 对应目录下的 `tensorboard/config.yaml` |

## 还没有的结果

- Pi05 GRPO 和 Abot PPO 都没有固定种子 eval。仓库 yaml 的 `val_check_interval: 10` 和这两次实际启动不一致。
- Abot PPO 的成功率上升还没有和 `ratio = 1` 对上。
- Abot GRPO 停在刚回升的位置，eval 只有 16 条，25% 还不稳定。
- 四个跑次都只覆盖 `adjust_bottle` 这一项任务。
