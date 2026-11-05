#!/usr/bin/env bash
# Run from the prepared OpenWAM + RLinf + RoboTwin simulator environment.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${POLICY_MODEL_PATH:?Set the RoboTwin OpenWAM EEF checkpoint directory}"
: "${ROBOTWIN_ASSETS_PATH:?Set the parent of the RoboTwin assets directory}"
: "${ROBOTWIN_RUNTIME_PATH:?Set the RoboTwin RLinf_support checkout}"
: "${ROBOTWIN_QUATERNION_ORDER:?Set xyzw or wxyz after verifying simulator poses}"
export ROBOT_PLATFORM=ALOHA
MODE="${1:?usage: run_openwam.sh train|eval [Hydra overrides]}"
shift
case "$MODE" in
  train) CONFIG=robotwin_adjust_bottle_nft_xpolicylab_openwam ;;
  eval) CONFIG=robotwin_adjust_bottle_xpolicylab_openwam_eval ;;
  *) echo "Expected train or eval" >&2; exit 2 ;;
esac
exec bash "${SCRIPT_DIR}/run.sh" "$CONFIG" "$@"
