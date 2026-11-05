# XPolicyLab / RLinf v0.4

The XPolicyLab gitlink pins RLinf 0.4.0 at `e2aeeadd`. Use
`git submodule update --init infra/RLinf/RLinf` after checking out XPolicyLab.
See the [version audit](MIGRATION_0_4.md), [OpenPI training comparison](OPENPI_TRAINING.md)
and [adapter review and verification](ADAPTER_REVIEW.md).

## Pi05 migration

`policy/Pi_05/rl.py` bridges XPolicyLab task transforms to native v0.4 `Pi0RL`.
The original OpenPI install still supplies task configuration, normalization and
tokenization. Ordinary evaluation/SFT do not import the RL backend; explicitly
select `rl_backend: rlinf_v0_4` when evaluating a native RL model.

Pi05 training uses RLinf model type `openpi`, so actor-side likelihood checks and
FSDP validation use the upstream OpenPI code. Set `algorithm.recompute_logprobs=true`
to re-evaluate old likelihoods before the optimizer update.

The bridge stores normalized observation tensors, chain samples and denoising
indices in the v0.4 replay format. RGB pixels are mapped from uint8 to [-1, 1]
once. State normalization/delta transforms remain those of the training config.
It deliberately uses pinned private `_predict_train`, `_predict_eval` and
checkpoint conversion helpers; changing the RLinf revision requires rerunning
these contract tests.

Supported weights: single-file and indexed/sharded `model.safetensors`, legacy
OpenPI names, native `full_weights.pt`, and XPolicyLab `backbone.*` full state
weights. Missing backbone tensors fail. Base weights can initialize new value or
noise heads. Old optimizer/FSDP states cannot be resumed across the architecture
change. For native RL checkpoints, use `runner.resume_dir` with the matching
training config. [Verified RoboTwin training and resume](#robotwin-adjust_bottle-training-and-resume)

For standalone evaluation of a full Pi05 RL model, set `POLICY_MODEL_PATH` to the
full weights or their directory and set
`+rollout.model.xpolicylab.norm_stats_path=/path/to/assets/REPO_ID` when statistics
are stored separately. Use matching train config and action/noise-head settings.
Alternatively `runner.ckpt_path` loads the **exact wrapper state dict** strictly
after construction: in that case PPO checkpoints require
`rollout.model.add_value_head=true` and the same critic configuration.

Pi05 RL defaults to FSDP1 `full_shard`. RLinf also supports `shard_grad_op`
and `no_shard` and enables `use_orig_params` for OpenPI. Shipped configs leave mixed-precision casts unset.
PPO uses `value_after_vlm=true`: native v0.4 suffix critics average different
steps during rollout and single-step recompute. Suffix critics therefore require
`joint_logprob=true`; the inconsistent combination raises an error. Custom
`value_hidden_sizes`, `safe_get_logprob=true` and Pi05 NFT are unsupported.

## RoboTwin adjust_bottle training and resume

Start in the XPolicyLab root inside a RoboTwin workspace. Use a Linux CUDA environment with PyTorch 2.11, the RLinf runtime and Pi0.5's OpenPI transforms and tokenizer. The env worker must be able to import `robotwin.envs.vector_env` from the RoboTwin `RLinf_support` runtime and load SAPIEN and MPLib. When using a separate runtime checkout, ensure its `envs` and `robotwin` packages resolve together in the env worker. On a shared host, bind the SAPIEN renderer to the env worker’s assigned GPU. The L20-3 run used runtime commit `0008ae6`; its rendering setup and exact launch scripts are recorded with the checkpoints. [Policy installation](../../../policy/Pi_05/README.md#installation) · [Verified run and artifact paths](OPENPI_TRAINING.md#robotwin-adjust_bottle-真实仿真与断点恢复)

The following Bash example uses physical GPU 6 for the actor and GPU 7 for rollout and simulation. Select an assigned pair on your host and adjust both the CUDA mask and the RLinf placement. The stock configs span GPUs 0–7. Set `ROBOTWIN_ASSETS_PATH` to the parent of `assets/` and choose a PyTorch checkpoint with the matching `assets/RoboTwin_lerobot_v30/norm_stats.json`.

```bash
export ROBOTWIN_ASSETS_PATH=/path/to/root-containing-assets
export POLICY_MODEL_PATH=/path/to/pi05-pytorch-checkpoint
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7
export RAY_ADDRESS=local RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export ROBOT_PLATFORM=ALOHA JAX_PLATFORMS=cpu

algo=ppo  # Set to grpo for the group-relative run.
config="robotwin_adjust_bottle_${algo}_xpolicylab_pi05"
out="$PWD/infra/RLinf/results/adjust_bottle_$(date +%Y%m%d-%H%M%S)"
if [[ "$algo" == grpo ]]; then
  envs=8; group=8; epochs=4; batch=32
else
  envs=2; group=1; epochs=1; batch=4
fi
args=('~cluster.component_placement' '+cluster.component_placement={actor:6,env:7,rollout:7}'
      "runner.logger.log_path=$out" "runner.logger.experiment_name=$algo"
      runner.val_check_interval=-1 runner.save_interval=1
      env.enable_offload=false actor.enable_offload=false rollout.enable_offload=false
      "env.train.total_num_envs=$envs" "env.train.rollout_epoch=$epochs"
      "algorithm.group_size=$group" algorithm.update_epoch=1 algorithm.recompute_logprobs=true
      actor.micro_batch_size=1 "actor.global_batch_size=$batch"
      +actor.fsdp_config.save_full_model_weights=false)

bash infra/RLinf/scripts/run.sh "$config" "${args[@]}" runner.max_steps=1
resume="$out/$algo/checkpoints/global_step_1"
bash infra/RLinf/scripts/run.sh "$config" "${args[@]}" runner.max_steps=2 "runner.resume_dir=$resume"
```

With `RAY_ADDRESS=local`, each invocation uses a private Ray instance. If you set `RAY_TMPDIR`, choose a short writable path for its Unix sockets. Keep `POLICY_MODEL_PATH` and the task statistics available for model construction. Pass the **`global_step_N` directory** to `runner.resume_dir`; the runner loads its `actor/dcp_checkpoint/` and treats `runner.max_steps` as the total target. Step 2 should produce a sibling `global_step_2` directory. To inspect either command without GPU work, append `--cfg job --resolve`.

For GRPO, check `loss_mask_fraction` and `actor/grad_norm`. In the single-group trial all four trajectories succeeded, so reward filtering removed the entire group and the gradient was zero. With four groups of eight, the verified run produced nonzero gradients before and after resume. The recorded success counts are from training rollouts. [Training results](TRAINING_RESULTS.md)

## Native PyTorch SFT

`infra/RLinf/scripts/train_pi05_sft.sh` adds full SFT using v0.4's native `Pi0` and
`SFTRunner`/`FSDPVlaSftWorker`. The worker only adapts the existing XPolicyLab
OpenPI data configuration and saves normalization/config metadata. It does not
route SFT through the RL sampler. Original `train.sh` remains unchanged.
See the [SFT workflow](PI05_SFT.md) for CPU stats generation,
dry runs, GPU training, checkpoint resume and SFT-to-RL loading.

## CPU verification

Use an isolated Python 3.11 environment. These dependencies exercise contracts,
not Isaac Sim or the complete pretrained VLM. Do not install them into Isaac's
Python environment.

```bash
# From XPolicyLab. On Linux install CPU torch/torchvision wheels first.
python -m pip install -r infra/RLinf/tests/requirements-cpu-test.txt
bash infra/RLinf/scripts/test_cpu.sh -q
```

`RL_PYTHON=/absolute/path/to/python` can select the test interpreter. Tests use
small differentiable networks with the actual Pi05/Abot samplers and recompute
kernels, temporary weight files, native RLinf data types, and a simulated env
boundary. IPC tests run real child Python processes and exercise crashes/timeouts.
They do not establish pretrained-model parity, Isaac stability, CUDA/FSDP weight
sync, checkpoint resume or training quality.

For config inspection, `bash infra/RLinf/scripts/run.sh CONFIG --cfg job` starts neither Ray nor a model.
For adapter validation, run `python infra/RLinf/scripts/check_config.py --config-name CONFIG` with the workspace parent and
`infra/RLinf/RLinf` on `PYTHONPATH`, and `RLINF_PATH` set. Resolve required path
environment variables to placeholders when only checking composition. This
checks adapter settings; full RLinf placement validation requires the target
cluster.

## RoboDojo stack_bowls

### OpenWAM NFT on either simulator

For copyable Bash commands on L20-3, including the host-specific weights, Luminis, the two-GPU placement, checkpoint resume and the contents of `results/`, see the [OpenWAM user guide (中文)](OPENWAM_QUICKSTART_zh.md).

Start from the simulator's own checkout. Its `XPolicyLab/` contains the policy and `infra/RLinf/`, including the pinned RLinf submodule. Run the same `infra/RLinf/scripts/run.sh` entry from that XPolicyLab directory. Both configurations compose `policy/OpenWAM/rlinf/config/recipe/openwam_nft.yaml`, which supplies the actor, rollout, optimizer and NFT loss.

- **RoboDojo** uses `robodojo_stack_bowls_nft_xpolicylab_openwam`, the RoboDojo EEF checkpoint, and `arx_x5` calibration. The env worker sends 16-D world-frame EE actions to RoboDojo's native IK controller. The native bridge provides the three RGB views, robot state, contact-based reward and episode boundary. Set `ROBODOJO_ROOT` to a checkout with `src/rlinf_bridge/runtime.py`, evaluator support for `external_control=True`, scene contact reporting and the stack-bowls layouts. Run Isaac Sim in Luminis using `ROBODOJO_SIM_PYTHON`. The policy and Ray use the OpenWAM/RLinf environment.
- **RoboTwin** uses `robotwin_adjust_bottle_nft_xpolicylab_openwam_split`, the RoboTwin EEF checkpoint, `aloha_agilex`, and the native quaternion order. Follow the [RoboTwin setup](OPENWAM_EXPERIMENTS.md) for the SAPIEN runtime and assets.

```bash
# From RoboDojo/XPolicyLab, with the OpenWAM/RLinf environment active.
export ROBODOJO_ROOT="$(cd .. && pwd)"
export ROBODOJO_SIM_PYTHON=/path/to/conda/envs/Luminis/bin/python
export POLICY_MODEL_PATH=/path/to/New_OpenWAM_RoboDojo_SFT_60k
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 RAY_ADDRESS=local
bash infra/RLinf/scripts/run.sh robodojo_stack_bowls_nft_xpolicylab_openwam \
  '~cluster.component_placement' '+cluster.component_placement={actor:6,env:7,rollout:7}' \
  env.train.render_gpu=7 env.eval.render_gpu=7 runner.max_steps=1
```

`render_gpu` is the Kit graphics-device index. On L20-3, index 7 matched the UUID of physical CUDA GPU 7. On a different host, use its Kit GPU table to select the index. If Luminis needs Conda activation hooks, `ROBODOJO_SIM_PYTHON` can point to a wrapper that activates the environment and executes its Python with all arguments. [Measured rollout, checkpoint and host paths](TRAINING_RESULTS.md#robodojo-stack_bowls--openwam-nft)

### Pi0.5 and Abot

For L20-3, select the host, policy and algorithm, then use the matching PyTorch environment, checkpoint step and normalization settings. [Pi0.5 and Abot launch, verification and resume (中文)](PI05_ABOT_QUICKSTART_zh.md)

Use the checkout **inside RoboDojo**, whose parent contains `env_cfg/arx_x5.yml`
and the `src/rlinf_bridge/runtime.py` companion. The env is registered under
`env_type: isaaclab`, ID `RoboDojo-Stack-Bowls-XPolicyLab-v0`; no upstream RLinf
source is patched. `RLINF_EXT_MODULE=XPolicyLab.infra.RLinf.register` is set by
`run.sh` for every Ray worker.

The training environment starts `ROBODOJO_SIM_PYTHON` as a separate process.
Only CPU arrays and metadata cross a private inherited pipe; there is no policy
WebSocket connection in training. AppLauncher precedes simulator imports. The
native evaluator supplies joint interpolation, gripper conversion, observations,
saved-layout reset and official success/score checks. Simulator exceptions abort
the rollout rather than fabricate observations or rewards.

Required later: Linux/NVIDIA/Isaac Sim 5.1 environment, RoboDojo assets,
`Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_*.json` for training and
`.../1/stack_bowls_*.json` for evaluation, and task-matched model weights plus
normalization statistics. Layout seeds are configurable. Training/evaluation
layouts must remain separate. Default robot/action dimensions come from shared
robot metadata. `cam_head` maps to policy `cam_high`; wrist order is left, right.

```bash
# Run in the separate RL policy environment; paths must be real for GPU runs.
export ROBODOJO_ROOT=/absolute/path/to/RoboDojo
export ROBODOJO_SIM_PYTHON=/absolute/path/to/isaac/python
export POLICY_MODEL_PATH=/absolute/path/to/task/checkpoint
export PI05_REPO_ID=your_training_dataset_id
export PI05_TRAIN_CONFIG=your_matching_openpi_config
cd "$ROBODOJO_ROOT/XPolicyLab"
bash infra/RLinf/scripts/run.sh robodojo_stack_bowls_xpolicylab_pi05_eval
bash infra/RLinf/scripts/run.sh robodojo_stack_bowls_ppo_xpolicylab_pi05
# Also provided: grpo_xpolicylab_pi05, ppo/grpo_xpolicylab_abot,
# and robodojo_stack_bowls_xpolicylab_abot_eval.
```

Abot additionally needs matching `checkpoint_num` and `unnorm_key`; do not assume
the example values describe a checkpoint you have. Configs start with one device,
PPO batch 1 / GRPO batch 4, chunk 50 and episode 800 actions. These are smoke-test
settings, not a measured memory or performance recommendation. Train/eval scenes
remain resident: simulator offload, video wrappers and partial/automatic resets
are rejected. Finished slots stop receiving actions and retain terminal
observations, while shared physics continues for the other slots.

The epoch length equals the episode limit and must divide into complete chunks.
The adapter bootstraps PPO timeouts from the actual terminal observation because
v0.4 only does this automatically for auto-reset environments. Successful or
failed terminations have no bootstrap bonus. Full-batch reset starts the next
epoch; GRPO members receive the same layout ID within each group.

## Reward and RoboValue scope

Accepted success (native success plus measured bowl support) earns one sparse
reward, once. Official 0–100 score remains a metric. The simulator returns `base_reward` (success once +1, otherwise 0), progress and
termination diagnostics, with no gamma. PPO, GRPO, evaluation and validation now
default to `enable_shaping=false`, `shaping_weight=0.0`. The RL algorithm alone
applies its return discount. Optional shaping belongs to the RLinf adapter and
requires **explicit** `enable_shaping=true`, a positive `shaping_weight`, and
`gamma: ${algorithm.gamma}`. At a **chunk boundary** it computes:

`r = success + beta * (gamma * Phi(next) - Phi(previous))`

`Phi` uses current native geometry **and upward pairwise PhysX contact force**:
0 without a supported pair, 0.15 for two supported bowls, 0.85 for a three-bowl
support chain. Version `stack_bowls_supported_v2` requires each support edge to
match the same ordered geometric pair (vertical force > 0.001 N). Gripper opening
and return-home state do not erase an existing stack's progress. Native success
without the complete support chain is rejected and its consumed checks rearmed;
the episode continues, or times out at its original step limit. Native success
and support forces are exposed separately for diagnostics.

This detects current contact support; release, dwell-time stability, tilt-boundary
hysteresis and finer within-stage progress are not implemented. Scene contact
reporting is opt-in, enabled before body creation for the CPU physics pipeline.
Frozen-pose rendering probes and pose-only offline comparisons explicitly retain
`stack_bowls_geometry_v1`: they cannot validate contact support from absent data.
Run the v2 physical counterexamples from RoboDojo using its simulator Python:

```bash
python -m src.rlinf_bridge.validate_support --output ../results/support_v2
```

True absorbing terminals use zero potential; truncations retain current
potential and bootstrap value. All shipped configurations default to sparse base rewards; shaping is opt-in.
The adapter exposes `reward_components.base_reward`, `shaping_reward` and
`returned_reward` separately. Videos label base reward and shaping status.
The adapter enforces the same gamma as the algorithm and chunk-level rewards.
Decreases remain negative; a dismantle/rebuild loop cannot accumulate unbounded
shaping return. Regressions/recoveries are recorded, without extra reward bonuses.

This borrows RoboValue's progress, failure/recovery and consistency principles.
It does not implement a RoboValue model, reproduce its evaluation suite, or claim
that the proposed reward improves policy success. Sparse-vs-shaped ablations and
real rollout diagnostics remain GPU validation work.

## Weight-free simulator and reward checks

Run `validate_simulator.py` with the **simulator Python**, from the simulator
checkout. It selects physical GPU 0 by UUID before importing Torch or a
simulator. Isaac multi-GPU rendering is disabled. RoboDojo retains the native
CPU physics/state pipeline; cameras and CuRobo still use GPU 0. No Ray server,
policy model, training job, or checkpoint is required.

```bash
# From RoboTwin:
/path/to/RoboTwin/python XPolicyLab/infra/RLinf/tests/tools/validate_simulator.py \
  robotwin --steps 8 --seed 0 --output /absolute/results/robotwin
# From RoboDojo:
/path/to/RoboDojo/python XPolicyLab/infra/RLinf/tests/tools/validate_simulator.py \
  robodojo --steps 8 --seed 0 --output /absolute/results/robodojo
/path/to/RoboDojo/python XPolicyLab/infra/RLinf/tests/tools/validate_simulator.py \
  robodojo --steps 8 --chunk-length 50 --layout-seed 1 --seed 0 \
  --output /absolute/results/robodojo_chunk50
/path/to/RoboDojo/python XPolicyLab/infra/RLinf/tests/tools/validate_simulator.py \
  robodojo --reward-probes --seed 0 --output /absolute/results/reward_probes
# Needs ffmpeg on PATH; use a new output directory for each attempt:
/path/to/RoboDojo/python XPolicyLab/infra/RLinf/tests/tools/validate_simulator.py \
  robodojo --reward-probes --record-video --seed 0 --layout-seed 0 \
  --output /absolute/results/reward_video
```

Each output directory contains `report.json` and unmodified RGB PNGs. Failed
assertions produce a failed report and nonzero exit. Joint hold targets come
from the initial observation, not a zero vector. The Dojo checks exercise the
production subprocess/RL environment bridge, repeated full reset, finite
observations, timeout masks inside chunks, terminal observation caching, and
zero reward on subsequent calls after termination.

`--reward-probes` selects a diagnostic backend in
`RoboDojo/src/rlinf_bridge/diagnostics.py`. It places actual bowl rigid bodies,
evaluates the native predicates, and passes their outputs through the real RL
reward code. Expected progress is `0.15 → 0 → 0.15 → 0.85`, with one regression
and recovery. For gamma 0.99 and beta 0.5, one-step chunk rewards are
`0.07425, -0.075, 0.07425, 0.925`; success is paid only once. These are controlled
state fixtures, not a learned policy trajectory, a settled-stack/contact test,
or a success-rate estimate. Training never selects this diagnostic backend.

`--record-video` records all three cameras for five stages: initial, progress,
regression, recovery, success. Eight newly rendered RGB triplets per stage at
3 fps produce a roughly 13-second `reward_validation.mp4`. The overlay shows
measured progress, reward, native historical score, and the progress history.
Frames within a stage hold the injected state without advancing physics; the
transitions are direct pose changes, not robot motion. Raw phase-boundary PNGs
and JSON remain available for inspection.

The bridge now supplies the native evaluator's zero-delay Kit startup settings.
The diagnostic backend publishes read-back PhysX local poses into the existing
USD translate/orient operations after direct rigid-body placement, preserving
scale and transform order. It checks world-position agreement within
0.1 mm, and rejects any physics advancement during capture. The video check is
specific to layout set 0 / index 0: white bowl silhouette groups must follow
`3 → 2 → 3 → 2 → 1` in the head-camera table region. It is a stale-image guard,
not a general detector or a replacement for task success predicates. Failure
produces a failed JSON report and nonzero exit; a partial MP4 is not a pass.

Instant-pose diagnostics use spatial FXAA and disable DLSS frame generation and
DL denoising: temporal DLAA retained old silhouettes when poses changed while
the physics clock was frozen. This override applies only to `RewardProbeBackend`;
ordinary environment rollouts retain their native rendering configuration.

**Verification status (2026-10-29):** real GPU-0 validation passed in 80.01 seconds,
including all five visual stages, four reward transitions, one regression and
recovery, zero repeated terminal reward, and repeat reset. PhysX/USD position
errors were zero. All 40 H.264 frames decoded successfully and their head-camera
silhouette counts matched the expected sequence. The video is 1024x800 at 3 fps
(13.334 seconds). GPUs 1–7 remained idle; GPU 0 returned to 1 MiB after shutdown.
The first attempt showed that `update_transformations` alone did not publish
tensor-API teleports; the second exposed temporal-rendering ghosts. Both failed
the guards before the final passing run. Local launch/capture contracts, image
checks and RGB encoding tests also passed.

The RoboTwin command exercises the current checkout's native evaluator and
XPolicyLab observation/action conversion, plus positive/negative native success
predicate fixtures. The separate Pi0.5 PPO and GRPO runs exercised the external
`robotwin.envs.vector_env.VectorEnv` from the RoboTwin `RLinf_support` checkout.
[Real rollout and checkpoint evidence](OPENPI_TRAINING.md#robotwin-adjust_bottle-真实仿真与断点恢复)

## Complete real-policy trajectory

`run_policy_trajectory.py` runs one physical, closed-loop `stack_bowls` episode
through the production subprocess bridge and RL reward implementation. It uses
the ordinary XPolicyLab WebSocket policy server in a separate policy Python.
The simulator always selects physical GPU 0. Policy GPU selection belongs to
the server launch; only select another card when the user authorizes it.

```bash
# With the policy server already listening, from the RoboDojo checkout:
/path/to/RoboDojo/python XPolicyLab/infra/RLinf/tests/tools/run_policy_trajectory.py \
  --policy-url ws://127.0.0.1:19605 --policy-config /absolute/server.yaml \
  --layout-seed 0 --layout-index 0 --max-steps 800 \
  --output /absolute/results/new_policy_episode
```

Use a fresh output directory. The recorder saves `initial_layout.json`, actual
post-reset `initial_rigid_poses.json`, a 25-fps three-camera `trajectory.mp4`,
`trace.jsonl`, and `episode.hdf5`. Each HDF5 row pairs the pre-action observation
and policy target with that transition's reward/metrics. Terminal observations
are saved as PNGs separately; the video includes the initial frame plus every
post-action observation. HDF5 also embeds the initial layout, instruction and
frequency; JPEG images use the shared RGB encoding helpers and XPL-RGB1 marker.
Policy chunks stop immediately on success/failure/timeout. Repeated terminal
calls are checked to emit zero reward and preserve the terminal observation.

On 2026-10-29, the supplied original OpenPI/JAX Pi05 checkpoint at step 59999
completed one episode successfully: **280 actions, 6 policy inferences, 11.2
simulated seconds**, recorded in a 281-frame, 11.24-second video. Simulation
used GPU 0, and policy inference temporarily used GPU 1 with explicit user
authorization. Both returned to 1 MiB after cleanup; GPUs 2–7 remained idle.
This validates original-checkpoint inference and the simulator/reward bridge;
it does not validate PyTorch conversion, SFT or RL optimizer updates.

The original trajectory exposed a baseline reward-semantics issue: at action 167,
all three rigid-body poses were unchanged from action 166, but closing the left
gripper crossed the native all-grippers-open threshold (0.802 to 0.763, threshold
0.8). The copied native score predicates therefore reduced progress from 0.15
to 0, emitted -0.075, and later counted recovery. This is a reward regression,
not evidence of physical stack destruction. This was fixed by
`RoboDojo/src/rlinf_bridge/progress.py`; the recorded baseline remains unchanged
for review. CPU re-evaluation of all 280 recorded states with the new production
predicates/accounting keeps action 167 at progress 0.15, changing its reward
from -0.075 to the ordinary discount term -0.00075. Official score and final
success are unchanged. This is an offline reward comparison, not a new policy
rollout. A new GPU-0 state-probe run also passed.

From RoboDojo, using its interpreter (no GPU is required for this command):

```bash
CUDA_VISIBLE_DEVICES='' /path/to/RoboDojo/python \
  -m src.rlinf_bridge.recompute_rewards \
  --episode-dir /absolute/results/original_policy_episode \
  --output /absolute/results/geometry_reward_comparison.json
```

Geometry remains instantaneous: the recorded episode still has three progress
regressions at native tilt thresholds (actions 101, 249 and 251), including
bottom-bowl tilt crossing 7 degrees. No hysteresis, contact/dwell test or maximum
vertical gap was added in this gate fix. Progress can describe a held/aligned
bowl; it is not itself a settled-stack or final-success label.
