# Pi_05

**Contributor:** RoboDojo Team | **Paper:** π0.5: a Vision-Language-Action Model with Open-World Generalization | **arXiv:** [2504.16054](https://arxiv.org/abs/2504.16054) | **Original code:** [OpenPI](https://github.com/Physical-Intelligence/openpi)

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate
```

For policy-server evaluation, pass `uv` as the ninth argument to `eval.sh` to use this OpenPI environment.

### RLinf training environment

Run the following once from the **XPolicyLab root** on Linux with CUDA and `uv` installed. The example creates `.venv` for training, installs the bundled OpenPI dependencies, and selects RLinf’s PyTorch 2.11 CUDA build. For RoboTwin, also install the simulator dependencies in the training environment. [RLinf’s RoboTwin setup](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)

```bash
git submodule update --init infra/RLinf/RLinf
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" GIT_LFS_SKIP_SMUDGE=1 uv sync --project policy/Pi_05/openpi --locked --inexact --group lerobot
python -m pip install -e . "numpy<2"
bash infra/RLinf/scripts/install_rl_runtime.sh --upgrade-torch
```

For subsequent training runs, activate `.venv` from the XPolicyLab root with `source .venv/bin/activate`.

## Data Processing

Run the converter in the OpenPI environment to turn RoboDojo demonstrations into the LeRobot dataset read by `train.sh`. By default, the second argument names both the source task and the output dataset’s task. To give the dataset a different name, pass the source task as the fifth argument. To limit the episode count as well, pass the count fifth and the source task sixth, as shown below. Multiple source tasks can be separated by commas. [Dataset fields and camera names](../../README.md#official-lerobot-conversion)

```bash
cd XPolicyLab/policy/Pi_05
source openpi/.venv/bin/activate
# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Use the first 50 episodes and give the resulting dataset its own name
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

The example below uses OpenPI’s JAX trainer and the `stack_bowls` dataset prepared above. Replace `<gpu-ids>` with the GPU indices assigned to this run, separated by commas.

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 '<gpu-ids>'
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. Use `OPENPI_FSDP_DEVICES` to override the number of devices used for OpenPI sharding.

### RLinf PyTorch supervised fine-tuning (SFT)

From the XPolicyLab root, activate the RLinf training environment. Set `POLICY_MODEL_PATH` to your initial Pi0.5 PyTorch checkpoint (or pass `actor.model.model_path` on the command line). Set `PI05_NORM_STATS_PATH` to the normalization file for the training data. Adjust the example path if your file is stored elsewhere.

```bash
export PI05_NORM_STATS_PATH="$POLICY_MODEL_PATH/assets/robodojo_stack_bowls/norm_stats.json"
bash infra/RLinf/scripts/train_pi05_sft.sh RoboDojo stack_bowls arx_x5 joint 0 '<gpu-ids>'
```

Append `--dry-run` to inspect the configuration, or `--compute-norm-stats` to compute the dataset statistics on CPU and write them to a new `norm_stats.json` at `PI05_NORM_STATS_PATH`. SFT checkpoints go under `policy/Pi_05/checkpoints/pytorch_sft/`.

## Evaluation

To evaluate the JAX checkpoint from the training example, replace the two GPU placeholders with the devices for the policy server and simulator. `RoboDojo` is the simulator’s Conda environment.

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh RoboDojo stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 arx_x5 joint 0 '<policy-gpu-id>' '<sim-gpu-id>' uv RoboDojo
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

In `deploy.yml`, `train_config_name` selects the named OpenPI recipe used when training the checkpoint. For example, `pi05_base_aloha_full_sim_arx-x5_seed_0` is registered in the bundled OpenPI configuration. The `seed` argument on the training command sets the seed for the current run. Set `repo_id` to the directory containing `norm_stats.json` under the checkpoint’s `assets/`, and `checkpoint_num` to the training step to evaluate. [OpenPI training recipes](openpi/src/openpi/training/config.py)

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to a host-specific directory under the system temporary directory. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

## RLinf v0.4 backend

To run PPO or GRPO on RoboDojo or RoboTwin, follow the [RLinf setup and launch guide](../../infra/RLinf/doc/GUIDE.md).

For standalone evaluation of RL weights, select `rl_backend: rlinf_v0_4` and supply model weights and matching normalization statistics.
