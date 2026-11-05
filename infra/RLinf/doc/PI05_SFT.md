# Pi05 native PyTorch SFT (RLinf v0.4)

`train.sh` continues to run the original official OpenPI/JAX flow.
The additional `train_pi05_sft.sh` uses the RLinf **0.4.0** checkout at
`e2aeeaddc09dbbdee179bde6bc6b03bb374e304b`, for the network, flow-matching loss,
backward pass, optimizer, scheduler and FSDP checkpoints. It constructs native
`rlinf.models.embodiment.openpi.pi0.Pi0` with `pi05=true`, not the older
`openpi.models_pytorch.PI0Pytorch`. No upstream source is changed.

OpenPI supplies the existing task config, LeRobot loader, tokenizer, delta
actions and quantile normalization. Its JAX imports run on CPU; they do not
participate in the training network or reserve GPU memory.

## Environment and data

Use a **separate Linux CUDA policy environment** with RLinf v0.4 / torch 2.11
and the bundled OpenPI data/tokenizer dependencies. Keep `openpi/.venv` for the
original flow. `RL_PYTHON=/absolute/path/to/python` selects the new interpreter;
the launcher never invokes `uv run` or re-syncs the original environment.
The RL runtime installer now also installs `torchdata==0.11.0` and `tensorboard`.
This entry supports full SFT on a single node with one or more GPUs, without LoRA.

Supported robots: `arx_x5` and `aloha_agilex`, dual-arm `joint` actions.
Dimensions come from the shared training helper. Defaults: horizon 50, padded
action dimension 32, 200 prompt tokens. The selected OpenPI TrainConfig must
agree with the native model and initial checkpoint. SFT keeps FP32 master
parameters/optimizer updates, BF16 backbone activations and native FP32
projection/norm/vision-embedding rules. FSDP1 defaults to `no_shard` (override with `full_shard` or `shard_grad_op`) and
`use_orig_params=true`: each GPU holds the full model and optimizer state.

Use a prepared **LeRobot v3.0** export (`lerobot==0.4.4`) from the official
converter, from XPolicyLab:

```bash
python scripts/transform_lerobot_v30_format.py 'RoboDojo.stack_bowls.arx_x5' \
  --repo_id RoboDojo-stack_bowls-arx_x5-joint
```

Keys: `observation.state`, `action`, `observation.images.cam_high`,
`cam_left_wrist`, `cam_right_wrist`, plus task metadata. RGB stays RGB; LeRobot
owns decoding and OpenPI defaults to PyAV. Prompts come from task metadata.
Joints use current-state deltas; grippers remain absolute. LeRobot supplies
temporal windows and episode-end padding using the **model horizon**.

The new path uses the converter above; policy-local `process_data.sh` is
preserved for its original environment. Do not mix v2.1 exports with a v3.0
loader. `data.train_data_paths` accepts a local directory with `meta/info.json`
or a LeRobot cache/repository ID. Prefer absolute filesystem paths.

## Inspect, normalize and train

From the XPolicyLab root, inspect without GPU, data or weights:

```bash
bash infra/RLinf/scripts/train_pi05_sft.sh RoboTwin adjust_bottle aloha_agilex joint 0 0 --dry-run
bash infra/RLinf/scripts/train_pi05_sft.sh RoboDojo stack_bowls arx_x5 joint 0 0 --dry-run
```

Compute task statistics using the same delta transforms. This needs data and
the OpenPI tokenizer asset, but no model weights or GPU:

```bash
export PI05_NORM_STATS_PATH=/absolute/path/to/stack_bowls_stats/norm_stats.json
bash infra/RLinf/scripts/train_pi05_sft.sh RoboDojo stack_bowls arx_x5 joint 0 0 \
  --compute-norm-stats \
  data.train_data_paths=/absolute/path/to/lerobot_stack_bowls data.num_workers=0
```

Existing statistics are not overwritten. A matching existing `norm_stats.json`
can be supplied directly. LeRobot's raw `meta/stats.json` is not a substitute:
it has not applied the policy's delta transforms.

Once CUDA and base weights are available:

```bash
export POLICY_MODEL_PATH=/absolute/path/to/native_pi05_base
bash infra/RLinf/scripts/train_pi05_sft.sh RoboDojo stack_bowls arx_x5 joint 0 0 \
  data.train_data_paths=/absolute/path/to/lerobot_stack_bowls \
  actor.micro_batch_size=1 actor.global_batch_size=32 \
  runner.max_steps=30000 runner.save_interval=5000
```

The sixth argument can be `0,1` for two GPUs. Global batch must be divisible by
micro batch times GPU count; choose batches after measuring GPU memory.
Other overrides include `actor.optim.lr`, `data.num_workers` and
`actor.model.openpi.config_name`. `OPENPI_LEROBOT_REPO_ID` sets the default data
ID when no data-path override is supplied. `POLICY_MODEL_PATH` accepts native or
compatible safetensors/RLinf full weights, not a JAX `params/` directory.

## Checkpoints, resume and RL

Outputs are isolated from the original JAX run directory:

```text
policy/Pi_05/checkpoints/pytorch_sft/<bench>-<ckpt>-<robot>-joint-<seed>/
  norm_stats.json
  sft_config.yaml
  sft_metadata.json
  checkpoints/global_step_<N>/actor/
    dcp_checkpoint/                   # model, optimizer, scheduler
    model_state_dict/full_weights.pt  # native model weights
    norm_stats.json
    sft_config.yaml
    sft_metadata.json
```

Stats are copied before workers start and into every checkpoint. A nonempty
output requires explicit resume or a new `ckpt_name`. To resume, keep matching
model/data settings and add
`runner.resume_dir=/absolute/path/to/checkpoints/global_step_5000`.
Keep `POLICY_MODEL_PATH` available for construction; it can point to the resume
checkpoint's `actor` directory. `runner.max_steps` is the total target step.
RLinf restores training state, but its map-style OpenPI loader does not persist
the exact iterator/RNG position; resumed minibatches need not match an
uninterrupted run.

For RoboDojo evaluation/RL, from XPolicyLab with simulator environment/assets
configured as described in the RLinf guide:

```bash
export POLICY_MODEL_PATH=/absolute/path/to/global_step_30000/actor
export PI05_TRAIN_CONFIG=pi05_base_aloha_full_sim_arx-x5_seed_0
bash infra/RLinf/scripts/run.sh robodojo_stack_bowls_xpolicylab_pi05_eval \
  +rollout.model.xpolicylab.norm_stats_path="$POLICY_MODEL_PATH"
bash infra/RLinf/scripts/run.sh robodojo_stack_bowls_ppo_xpolicylab_pi05 \
  +actor.model.xpolicylab.norm_stats_path="$POLICY_MODEL_PATH"
```

Here RL `norm_stats_path` is the **directory** containing `norm_stats.json`.
Use the matching SFT TrainConfig. Do not set `runner.ckpt_path` to unwrapped SFT
weights: `POLICY_MODEL_PATH` loads them through the bridge and initializes any
new critic head. For policy-server evaluation, explicitly set
`rl_backend: rlinf_v0_4`, `model_path`, `train_config_name` and `norm_stats_path`
in a run-specific deploy config. Original JAX evaluation remains available.

## CPU verification and limits

Standard tests cover native SFT loss/backpropagation through small real
SigLIP/Gemma networks, optimizer updates, native model loading, SFT-to-RL weight
transfer, invalid configs and overwrite guards. For additional data tests, use
an isolated CPU environment, from XPolicyLab:

```bash
python -m pip install -r infra/RLinf/tests/requirements-cpu-test.txt
python -m pip install -r infra/RLinf/tests/requirements-cpu-sft-data.txt
python -m pip install --no-deps lerobot==0.4.4
bash infra/RLinf/scripts/test_cpu.sh -q infra/RLinf/tests/openpi/test_pi05_sft_data.py
```

LeRobot is installed without unrelated viewer/robot extras: its optional rerun
viewer requires NumPy 2, while this OpenPI checkout requires NumPy <2.
Data-test requirements pin compatible Arrow/datasets versions. These are CPU
test dependencies, not a complete GPU runtime lockfile.

Data tests write/read a temporary v3.0 dataset and check temporal padding, RGB,
prompts, joint deltas, quantile normalization, action padding and statistics
computation. Only the external tokenizer asset is replaced by a test double.
These tests cover the standalone SFT workflow on CPU. GPU SFT training and resume remain unverified. The separate PPO and GRPO runs used a real tokenizer, checkpoint and simulator and verified DCP resume. [RL run evidence](OPENPI_TRAINING.md#robotwin-adjust_bottle-真实仿真与断点恢复)
