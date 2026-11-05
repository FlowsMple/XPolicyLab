# RLinf 0.4 版本核对

2026-11-04 核对了三个不同的提交。XPolicyLab 原 gitlink 是发布前的 `0067f7d5`，官方 v0.4 标签是 `2348ece3`。本工作区 RLinf 子模块与当日官方 main 均为 `e2aeeaddc09dbbdee179bde6bc6b03bb374e304b`，旧 gitlink 和新检出的 pyproject.toml 都声明 0.4.0。本次以最后一个提交作为可复现的目标。两个分支的历史已经分叉，不能只按版本号判断 API。

## 0.3 依赖的去向

v0.3 源码声明 Torch 2.6.0。其 OpenPi0ForRLActionPrediction 继承外部 openpi 的 PI0Pytorch，RoboTwin 环境位于 rlinf/envs/robotwin。当前 0.4.0 的 embodied 运行栈定位 Torch 2.11.0，Pi0RL 继承 RLinf 自有 Pi0，环境位于 rlinf/envs/sim/robotwin。XPolicyLab 原提交中的 Pi05RLPolicy 同样继承 PI0Pytorch 并自行计算去噪步概率；本次开始审查时，工作区已有的未提交修改已将这段实现替换成 Pi0RL。此次修复补上了模型类别、分片与精度的对齐。正式 v0.4 与本次 main 的 tasks/rl.py 内容相同，FSDP 和 SFT 训练路径则有实际差异。

## 修复内容

- 已同步 XPolicyLab 的 gitlink 和运行时检查。原 RL 启动器、配置检查和 SFT 入口各自写死了发布标签，正常子模块更新和手动更新会导致不同的失败。
- 保留 Ray 各进程调用的扩展注册、HF rollout 以及 OpenWAM 专用的 NFT 和 EE 环境。API reward 的按需导入已更新，EnvPart 超时回报已通过真实发布方法复验。
- 已核对 FSDP 分片、混合精度和 actor 重算的实际入口。优化器、梯度裁剪和 checkpoint 由子模块提供。历史训练结果使用旧实现，不能据此判定新版本的收敛情况。

可用 `git ls-files -s infra/RLinf/RLinf`、`git -C infra/RLinf/RLinf rev-parse HEAD` 和 `git ls-remote https://github.com/RLinf/RLinf.git refs/heads/main refs/tags/v0.4` 分别检查锁定版本、检出版本和官方引用。[官方发布说明](https://github.com/RLinf/RLinf/releases/tag/v0.4)

## 迁移后的入口

Pi_05 的六个 RL 配方使用 OpenPI 模型类别，四个训练配方默认 full_shard。RoboTwin 和 RoboDojo 继续使用各自的动作、状态与奖励约定。Abot_M0 的独立 PPO/GRPO 实现和 OpenWAM 的 NFT 仍使用通用外部模型类别。专用的 OpenWAM 启动脚本也检查同一 gitlink。RoboTwin 原生 env worker 在自己的解释器导入模拟器；旧脚本注释提到的 ROBOTWIN_SIM_PYTHON 在当前上游没有消费者。

可从 XPolicyLab 根目录无卡检查正式入口。下列路径仅供配置解析使用。

```bash
POLICY_MODEL_PATH=/tmp/model ROBOTWIN_ASSETS_PATH=/tmp/assets \
  bash infra/RLinf/scripts/run.sh robotwin_adjust_bottle_ppo_xpolicylab_pi05 \
  --cfg job --resolve algorithm.recompute_logprobs=true
bash infra/RLinf/scripts/train_pi05_sft.sh RoboTwin audit aloha_agilex joint 0 6,7 \
  --dry-run actor.fsdp_config.sharding_strategy=full_shard
```

两条命令均成功解析出模型类别 openpi、分片 full_shard。第一条还解析出 actor 更新前的可选重算开关为 true。
