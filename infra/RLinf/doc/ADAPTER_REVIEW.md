# XPolicyLab RLinf 适配缺陷与验证


## 2026-11-05 目录与扩展接口回归

- 原训练入口按策略名选择 worker，独立评估和训练使用不同的环境选择逻辑。现在两者读取同一份策略扩展，使用 `runner.only_eval` 分派 runner。RoboDojo PPO 的终态回报位于宿主扩展，OpenPI 的原生类别转换位于 Pi0.5 扩展。
- 在相同环境变量和上游子模块下，分别从搬迁前的配置副本和新的 Hydra 搜索路径解析 17 个 RL 配置及 1 个 SFT 配置，18 份解析后的用户配置逐项相等。
- 通过公共入口运行 148 项本地 CPU 测试，涵盖新策略和新宿主的发现、已有 worker 的选择、训练和评估分流、OpenPI 数据、模型重算和模拟器进程清理。[扩展回归用例](../tests/test_extensions.py)
- 在 L20-3 独立临时目录通过新启动器展开 RoboDojo OpenWAM NFT、RoboTwin Pi0.5 PPO 和 RoboTwin Abot eval。RLinf 的原生扩展加载器成功注册 RoboDojo 任务。策略环境成功导入统一 driver、NFT worker 和 EE 环境，Luminis 可单独导入 IPC 桥接。两张指定 GPU 当时已有任务。本轮验证范围是代码、配置和启动路径，真实 rollout 和续训的日期与结果见[训练记录](TRAINING_RESULTS.md)。

## v0.4 迁移审查

修复前在当前子模块运行 CPU 套件，109 项通过、2 项失败。以下按实际影响记录原因、修复和复验方法。

1. **正常启动被版本检查阻止。** 上游检出与三个硬编码的发布 SHA 不同。使 gitlink 成为版本依据，并把实际使用的提交写入 SFT metadata。用 RL 启动脚本和 SFT dry-run 复验。
2. **Pi0.5 合法的分片及 log-prob 重算配置被拒绝。** 外部模型名称无法触发上游 OpenPI 分类，适配层又复刻了旧的 no-shard 限制。让 Pi0.5 使用原生模型类别，并将当前上游的分片规则用于训练。在同一配置上验证三种 FSDP1 分片，检查 PPO/GRPO 的采样、重算和参数更新。
3. **API reward 本地服务按需启动会导入已删除的模块。** 将 train_rl 的导入改到当前上游的 sglang 包。核对真实入口导出并测试惰性导入。
4. **超时回报测试的 worker fixture 使用旧字段。** 新 rollout worker 在终态推理前读取 model_type；旧夹具没有这个字段。通过真实 _publish_env_part 构造 EnvPart，检查超时只获得一次折扣后的终态 value，并确认真正的成功终止没有奖励加成。
5. **RoboDojo Pi0.5 GRPO 的默认配置无法启动。** 单卡四个环境配了八条轨迹一组。RLinf 的 validate_cfg 在启动 worker 前报组大小不能整除。保留四个环境及 batch 大小，改为每组四条。在配置测试中运行真实 validate_cfg，仅替代 Ray 的硬件发现和 placement 查询。
6. **Pi0.5 解析第 0 步 checkpoint 时可能永久等待。** 当目录没有精确匹配且备选步数有多位数字时，旧的十进制缩放循环会反复将 0 乘 10。改用有限位数的整数乘法，测试精确第 0 步与缺少第 0 步时的既有回退。
7. **参数存储精度和计算精度被绑在一起。** 旧桥接将一个字符串同时用于 Pi0Config 与权重转换，无法复现上游 FP32 权重配 BF16 backbone 激活的组合。复用上游 dtype 解析、加载后冻结与参数转换，并用小型真实 Pi0 网络验证两者可以分开。

## 本地复验

11 月 4 日补测发现，OpenWAM 启动脚本的 `--help` 会执行 `ray start --head`，显式设置 `RAY_ADDRESS=local` 也会进入这条路径。脚本此前只识别 `--cfg` 并无条件探测默认集群。现将配置和帮助路由到公共 Hydra 入口，并让训练驱动处理显式 Ray 地址。回归用例在 `tests/test_configs.py::test_launcher_skips_ray_start`，用替身令误起 Ray 直接报错。另用真实 Hydra 分别展开了训练、评估配置并检查两者的帮助输出。

在 RoboTwin 的 XPolicyLab 工作区，用 Python 3.11、PyTorch 2.11.0（CPU）执行下列命令。HF_HOME 可以指定任意可写的测试缓存目录。

```bash
HF_HOME=/tmp/xpl-rlinf-hf PYTHONDONTWRITEBYTECODE=1 \
  bash infra/RLinf/scripts/test_cpu.sh -q -p no:cacheprovider
```

133 项通过。覆盖 16 个训练/评估配方、真实 LeRobot 数据读写、小模型的 SFT 梯度与权重转换、Pi0.5 和 Abot 的采样重算、无 critic 的原生 GRPO loss 更新，以及 OpenWAM NFT 的张量契约。配置测试使用实际 RLinf 校验，Ray 的集群与各组件进程数查询由配置中的 placement 代替。另已通过两个 shell dry-run、shell 语法、Ruff 的语法与未定义名称检查，以及 git diff --check。SGLang 服务只核对了导入路径与导出符号，未启动奖励模型服务。

清理前逐一核对了三个适配测试目录中的 12 个 Python 源文件。替换了过时的分片断言并更新旧 worker 夹具，删除了这三个目录的 13 个旧 pytest 字节码文件，并清除了仅含这 12 个测试模块历史结果的 .pytest_cache。12 个源文件均有当前实现对应的回归用例，OpenPI 数据集成测试单独放置以隔离可选依赖。OpenWAM 的真实权重探针仍由实验手册调用。

## GPU 复验

使用 L20-3 的 6、7 号卡、PyTorch 2.11.0+cu128、真实 Pi0.5 checkpoint 和合成观测完成一次双卡 FULL_SHARD PPO 更新。两个 rank 均通过原生对照。修复工作区的 PyTorch 数值、首轮阈值失败原因和未覆盖项见 [OpenPI 训练核对](OPENPI_TRAINING.md#双卡数值结果)。GPU 未单独运行 Abot_M0、OpenWAM 或 SFT 训练。测试结束后已删除远端工作目录与节点内存盘中本次创建的隔离副本。

## adjust_bottle 运行时的 Ray 集群选择

**指定 Ray 地址仍可能启动额外 head。** 在共享 L20-3 上，训练使用 `RAY_ADDRESS=local` 为每个新进程建立私有集群。原 `run.sh` 会先运行 `ray status`，其失败分支尝试在默认端口建立独立的 head；即使用户已指定地址也会进入该分支。现在只有未指定 `RAY_ADDRESS` 时才执行这段兼容逻辑。回归用例让 Ray CLI 返回 7，修复前启动器随之退出，修复后私有地址直接传到 `train_rl.py`。包含 OpenPI 数据和 OpenWAM 在内的本地回归套件 134 项通过。实际 GPU 实验直接调用同一个 Python 训练入口。

真实轨迹、更新与恢复结果记在 [OpenPI 训练核对](OPENPI_TRAINING.md#robotwin-adjust_bottle-真实仿真与断点恢复)。

该服务器的 `/tmp` 可用空间不足 1 GB，实验将 Ray 临时文件放入节点内存盘。一次多组 GRPO 启动使用过长的子目录，Ray 在创建 Plasma socket 时触发 Linux AF_UNIX 的 107 字节上限。将实验脚本的临时目录缩短为 `$src/r1` 和 `$src/r2` 后通过启动。该失败发生在采样前，日志保存在远端 `diagnostics/grpo-ray-socket.log`。
