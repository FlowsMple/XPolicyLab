# 接入策略和仿真环境

RoboTwin 和 RoboDojo 都可以作为顶层工作区，各自提供任务、机器人资产、`env_cfg/` 和仿真运行时。在选定宿主的 `XPolicyLab/` 根目录启动 RLinf，选与该宿主匹配的任务配置、权重和标定。以下命令可以在策略 Python 环境中先展开配置，去掉 `--cfg job --resolve` 即可按配置启动训练。

```bash
# RoboTwin
POLICY_MODEL_PATH=/path/to/aloha/weights ROBOTWIN_ASSETS_PATH=/path/to/robotwin/assets-parent \
  bash infra/RLinf/scripts/run.sh robotwin_adjust_bottle_ppo_xpolicylab_pi05 --cfg job --resolve

# RoboDojo
POLICY_MODEL_PATH=/path/to/arx/weights ROBODOJO_ROOT=/path/to/RoboDojo \
  ROBODOJO_SIM_PYTHON=/path/to/luminis-python \
  bash infra/RLinf/scripts/run.sh robodojo_stack_bowls_ppo_xpolicylab_pi05 --cfg job --resolve
```

GPU 分配、模型任务统计和运行时的实际设置见[关节策略训练](PI05_ABOT_QUICKSTART_zh.md)和 [OpenWAM 训练](OPENWAM_QUICKSTART_zh.md)。

## 接入策略

1. 在 `policy/<策略名>/model.py` 导出 `Model`，配置的 `xpolicylab.policy_name` 使用同一个目录名。按支持的模式声明 `RL_MODES`。PPO、GRPO 实现 `predict_action_batch_rl` 和 `compute_logprob_value`，并使同一批 rollout 数据能重算对应的 log-prob。PPO 还需提供 value。[模型契约](../../../model_template.py)
2. 把 Hydra 配方放到 `policy/<策略名>/rlinf/config/`。配置名在所有策略间须唯一，可以组合共享的 `model/xpolicylab`、宿主的 `env/<任务>` 和上游 FSDP 配置。Abot 仅提供配方和标准模型接口。[Abot 配方](../../../policy/Abot_M0/rlinf/config)
3. 需要特殊行为时，在同目录的 `rlinf/extension.py` 定义所需函数。注册发生在每个 Ray 进程中，包初始化和扩展模块顶层应能在其他策略的环境中导入。专有依赖在函数被选中后导入。
   - `configure_model(cfg, model_cfg)` 将上游模型参数写入策略构造参数。可用 `register()` 注册原生模型别名。[Pi0.5 示例](../../../policy/Pi_05/rlinf/extension.py)
   - `actor_worker(cfg)` 可返回专用 actor 类。`env_class(env_cfg)` 可为特定宿主选择策略需要的动作边界。返回 `None` 使用默认 worker 或环境。`validate(cfg, model, env, mode)` 校验每一侧的配置，此处 `mode` 为 `train` 或 `eval`。[OpenWAM 示例](../../../policy/OpenWAM/rlinf/extension.py)

## 接入仿真环境

- RLinf 已支持的原生环境可直接使用上游 `env_type` 和任务配置。RoboTwin 的通用策略路径使用原生 `RoboTwinEnv`。
- 自定义宿主在 `infra/RLinf/envs/<宿主>/` 实现环境，在 `config/env/` 提供任务 YAML。在 `extension.py` 声明 `NAME` 和 `ENV_KEY = (env_type, task_id)`。当 `task_id` 为 `None` 时匹配整个后端；具体任务优先。环境键重复会在发现阶段报错。
- 使用与该宿主的动作格式兼容的 RLinf 原生后端。RoboDojo 使用 `("isaaclab", init_params.id)`，在 `register()` 中把任务 ID 和环境类加入上游注册表。RLinf 0.4 校验 `env_type` 并据此处理动作；新增一种上游不认识的后端名还需要对应的原生支持。[注册示例](../envs/robodojo/extension.py) · [任务配置](../envs/robodojo/config/env/robodojo_stack_bowls.yaml)
- 环境提供 RLinf 所需的批量 reset、分块 step、终止/截断信号和清理。宿主需要独立 Python 时，由该适配器管理子进程。可提供 `validate(cfg, model, env, mode)` 校验动作维度、终止和奖励约定，也可提供 `rollout_worker(cfg)` 处理该仿真的采样语义。RoboDojo 用它为 PPO 的时间截断补上终态 value。[环境适配](../envs/robodojo/env.py) · [rollout](../envs/robodojo/rollout.py)

## 迁移现有调用

- 原适配目录顶层的启动器统一移入 `scripts/`。训练和评估共用 `scripts/run.sh`；原 Pi0.5 PyTorch SFT 入口为 `scripts/train_pi05_sft.sh`。启动脚本设置 `RLINF_EXT_MODULE=XPolicyLab.infra.RLinf.register`，公共模型工厂为 `XPolicyLab.infra.RLinf.get_model`。
- 从 Python 独立组合 RL 配置时，先调用 `XPolicyLab.infra.RLinf.core.configuration.register_config_paths()`，再以 `infra/RLinf/config/` 为 Hydra 主配置目录。命令行入口已完成此步。
- 外部宿主和独立录像工具若直接导入过内部模块，升级这份 XPolicyLab 时须一起更新。原 `robodojo_env`、`robodojo_process` 分别对应 `XPolicyLab.infra.RLinf.envs.robodojo.env`、`XPolicyLab.infra.RLinf.envs.robodojo.process`；`obs` 对应 `XPolicyLab.infra.RLinf.core.obs`；`validate_simulator` 和 `validation_video` 位于 `XPolicyLab.infra.RLinf.tests.tools`。RoboDojo 的独立奖励重算和支撑检查工具会用到这些内部导入。OpenWAM 重放探针的新入口是 `python -m XPolicyLab.infra.RLinf.tests.tools.openwam_replay`。
- 常规回归在 XPolicyLab 根目录运行 `bash infra/RLinf/scripts/test_cpu.sh -q`。验证工具集中于 `tests/tools/`，训练入口不会调用它们。训练日志目录、checkpoint 层级和 `runner.resume_dir` 的用法见[产物说明](OPENWAM_QUICKSTART_zh.md#4-手动启动会产生什么)。
