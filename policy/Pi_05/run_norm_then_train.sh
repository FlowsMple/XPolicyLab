#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${POLICY_DIR}/openpi/.venv/bin/activate"
# shellcheck disable=SC1091
source "${VENV}"
export PATH="/vepfs-cnbje63de6fae220/yuechen/flywheel/bin:${PATH}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/vepfs-cnbje63de6fae220/zijian}"
export OPENPI_TRAIN_CONFIG_NAME="${OPENPI_TRAIN_CONFIG_NAME:-pi05_real_tianji_marvin_ee}"
export OPENPI_LEROBOT_REPO_ID="${OPENPI_LEROBOT_REPO_ID:-RoboDojo_real-pick_magnet-tianji_marvin-ee}"
export OPENPI_FSDP_DEVICES="${OPENPI_FSDP_DEVICES:-2}"
# train.sh derives the checkpoint dir from these; keep runs of different action spaces apart.
ACTION_TYPE="${OPENPI_ACTION_TYPE:-ee}"
CKPT_NAME="${OPENPI_CKPT_NAME:-pick_magnet}"
TRAIN_GPUS="${OPENPI_TRAIN_GPUS:-2,3,4,5,6,7}"

echo "[$(date -Is)] HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "[$(date -Is)] config=${OPENPI_TRAIN_CONFIG_NAME} repo=${OPENPI_LEROBOT_REPO_ID}"
echo "[$(date -Is)] action_type=${ACTION_TYPE} ckpt_name=${CKPT_NAME} gpus=${TRAIN_GPUS}"
if [[ -n "${OPENPI_NORM_MAX_FRAMES:-}" ]]; then
  echo "[$(date -Is)] norm_max_frames=${OPENPI_NORM_MAX_FRAMES}"
fi

cd "${POLICY_DIR}/openpi"

if [[ "${OPENPI_SKIP_NORM:-0}" == "1" ]]; then
  echo "[$(date -Is)] compute_norm_stats skipped (OPENPI_SKIP_NORM=1)"
else
  echo "[$(date -Is)] compute_norm_stats start"
  norm_args=(--config-name "${OPENPI_TRAIN_CONFIG_NAME}")
  if [[ -n "${OPENPI_NORM_MAX_FRAMES:-}" ]]; then
    norm_args+=(--max-frames "${OPENPI_NORM_MAX_FRAMES}")
  fi
  CUDA_VISIBLE_DEVICES="${OPENPI_NORM_GPU:-2}" \
    python scripts/compute_norm_stats.py "${norm_args[@]}"
  echo "[$(date -Is)] compute_norm_stats done"
fi

echo "[$(date -Is)] train start"
bash "${POLICY_DIR}/train.sh" \
  RoboDojo_real "${CKPT_NAME}" tianji_marvin "${ACTION_TYPE}" 0 "${TRAIN_GPUS}"
echo "[$(date -Is)] train exited $?"
