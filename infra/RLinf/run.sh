#!/usr/bin/env bash
# Launch RoboTwin RL training with the XPolicyLab adapter and the vendored RLinf checkout.
# Required environment variables when the selected config needs them:
#   ROBOTWIN_ASSETS_PATH  RoboTwin embodiment assets
#   POLICY_MODEL_PATH     Pi_05 or Abot_M0 checkpoint
# Optional:
#   ROBOTWIN_SIM_PYTHON   Python of a separate RoboTwin sim venv. When set, each env
#                         worker runs the simulator in a child process of that venv,
#                         so the policy venv needs no sapien/curobo.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XPL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ROBOTWIN_ROOT=$(cd "${XPL_ROOT}/.." && pwd)

export RLINF_PATH="${SCRIPT_DIR}/RLinf"
export REPO_PATH="${RLINF_PATH}"
export EMBODIED_PATH="${RLINF_PATH}/examples/embodiment"
export RLINF_EXT_MODULE=XPolicyLab.infra.RLinf.register
export PYTHONPATH="${ROBOTWIN_ROOT}:${RLINF_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_NAME="${1:-robotwin_adjust_bottle_grpo_xpolicylab_pi05}"
shift || true

if ! ray status >/dev/null 2>&1; then
  ray start --head --port="${RAY_PORT:-6379}" --disable-usage-stats
fi

cd "${SCRIPT_DIR}"
python train_rl.py --config-name "${CONFIG_NAME}" "$@"
