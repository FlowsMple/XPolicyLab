#!/usr/bin/env bash
# Run from the prepared OpenWAM + RLinf + RoboTwin simulator environment.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XPL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
: "${POLICY_MODEL_PATH:?Set the RoboTwin OpenWAM EEF checkpoint directory}"
: "${ROBOTWIN_ASSETS_PATH:?Set the parent of the RoboTwin assets directory}"
: "${ROBOTWIN_RUNTIME_PATH:?Set the RoboTwin RLinf_support checkout}"
: "${ROBOTWIN_QUATERNION_ORDER:?Set xyzw or wxyz after verifying simulator poses}"
export RLINF_PATH="${SCRIPT_DIR}/RLinf"
export REPO_PATH="${RLINF_PATH}"
export EMBODIED_PATH="${RLINF_PATH}/examples/embodiment"
export RLINF_EXT_MODULE=XPolicyLab.infra.RLinf.register
export ROBOT_PLATFORM=ALOHA
# The simulator checkout is first so its `envs` package wins over the parent
# workspace. That checkout may also vendor an older XPolicyLab, which would
# hide this one. A directory that contains only this checkout's package goes
# ahead of the simulator tree.
XPL_IMPORT_SHIM="${XPL_ROOT}/.import_shim"
mkdir -p "${XPL_IMPORT_SHIM}"
ln -sfn "${XPL_ROOT}" "${XPL_IMPORT_SHIM}/XPolicyLab"
export PYTHONPATH="${XPL_IMPORT_SHIM}:${ROBOTWIN_RUNTIME_PATH}:${XPL_ROOT}/..:${RLINF_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
MODE="${1:?usage: run_openwam.sh train|eval [Hydra overrides]}"
shift
case "$MODE" in
  train) ENTRY=train_rl.py; CONFIG=robotwin_adjust_bottle_nft_xpolicylab_openwam ;;
  eval) ENTRY=eval_openwam.py; CONFIG=robotwin_adjust_bottle_xpolicylab_openwam_eval ;;
  *) echo "Expected train or eval" >&2; exit 2 ;;
esac
# Inspect configuration without starting Ray: append --cfg job --resolve.
if [[ " $* " != *" --cfg "* ]] && ! ray status >/dev/null 2>&1; then
  ray start --head --port="${RAY_PORT:-6379}" --disable-usage-stats
fi
# Ray workers put the driver cwd on sys.path ahead of PYTHONPATH. The simulator
# checkout vendors an older XPolicyLab, so the process must not start there.
cd "${XPL_ROOT}"
python "${SCRIPT_DIR}/${ENTRY}" --config-name "$CONFIG" "$@"
