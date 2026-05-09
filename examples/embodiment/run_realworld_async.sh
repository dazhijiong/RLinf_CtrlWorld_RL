#! /bin/bash

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
# This script is wired to an eval-only realworld config whose loss_type is
# actor_critic, so it must use the eval entrypoint instead of the async trainer.
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

if [ -f "${REPO_PATH}/.venv_pi05_ur5/bin/activate" ]; then
    # Use the local UR5/pi05 environment by default when it is available.
    source "${REPO_PATH}/.venv_pi05_ur5/bin/activate"
fi

CONFIG_NAME="${CONFIG_NAME:-${1:-realworld_ur5_open_book_openpi_pi05_eval}}"
if [ "$#" -gt 0 ]; then
    shift
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "$@"
)
printf 'Command: %q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
