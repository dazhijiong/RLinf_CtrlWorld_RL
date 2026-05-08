#!/usr/bin/env bash
#SBATCH -J rlinf_ctrl_world_pi05_eval_libero3d
#SBATCH -A naiss2025-22-1173
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/ctrl_world_pi05_eval_libero3d_out_%j.txt
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/ctrl_world_pi05_eval_libero3d_err_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH -p alvis

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run with sbatch." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/examples/embodiment/eval_embodied_agent.py" ]]; then
  REPO_PATH="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/RLinf/examples/embodiment/eval_embodied_agent.py" ]]; then
  REPO_PATH="${SLURM_SUBMIT_DIR}/RLinf"
else
  echo "Cannot locate RLinf repo from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_PATH}"
mkdir -p "${REPO_PATH}/logs"

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0
module load git-lfs/3.6.1

VENV_PATH="${VENV_PATH:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/.venv_pi05}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${VENV_PATH}/bin/python" ]]; then
    PYTHON_BIN="${VENV_PATH}/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python/python3 not found; load module/venv before sbatch." >&2
    exit 1
  fi
fi

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export CTRL_WORLD_PATH="${CTRL_WORLD_PATH:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/Ctrl-World}"
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-${VENV_PATH}/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH}:${PYTHONPATH:-}"

export WANDB_API_KEY='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'
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d "${LIBERO_REPO_PATH}" ]]; then
  echo "LIBERO_REPO_PATH does not exist: ${LIBERO_REPO_PATH}" >&2
  exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-ctrl_world_libero_spatial_grpo_openpi_pi05_eval_libero3d}"
DEFAULT_EVAL_CKPT_PATH=""
if [[ ! ${EVAL_CKPT_PATH+x} ]]; then
  EVAL_CKPT_PATH="${DEFAULT_EVAL_CKPT_PATH}"
elif [[ "${EVAL_CKPT_PATH}" == "none" || "${EVAL_CKPT_PATH}" == "null" ]]; then
  EVAL_CKPT_PATH=""
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ctrl_world_libero_spatial_grpo_openpi_pi05_eval_libero3d_RLinf_Pi05_LIBERO_SFT}"
TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${REPO_PATH}/logs/${TIMESTAMP}-${CONFIG_NAME}-job${SLURM_JOB_ID}"
LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"

if [[ -n "${EVAL_CKPT_PATH}" && ! -f "${EVAL_CKPT_PATH}" ]]; then
  echo "EVAL_CKPT_PATH does not exist: ${EVAL_CKPT_PATH}" >&2
  exit 1
fi

RAY_LOG_ARCHIVE_DIR="${LOG_DIR}/ray"
RAY_TMPDIR_RUNTIME="${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID}}"
export RAY_TMPDIR="${RAY_TMPDIR_RUNTIME}"
mkdir -p "${RAY_TMPDIR_RUNTIME}" "${RAY_LOG_ARCHIVE_DIR}"

mapfile -t NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#NODE_LIST[@]} -eq 0 ]]; then
  echo "Failed to parse SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}" >&2
  exit 1
fi
NUM_NODES="${#NODE_LIST[@]}"
HEAD_NODE="${NODE_LIST[0]}"
RAY_PORT="${RAY_PORT:-29500}"
HEAD_IP="${RAY_HEAD_IP:-$(
  srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" --quiet \
    bash -lc "hostname -I | awk '{print \$1}'" | tail -n1 | tr -d '[:space:]'
)}"
if [[ -z "${HEAD_IP}" ]]; then
  echo "Failed to determine Ray head IP. Set RAY_HEAD_IP manually." >&2
  exit 1
fi
HEAD_ADDR="${HEAD_IP}:${RAY_PORT}"
RAY_READY_TIMEOUT="${RAY_READY_TIMEOUT:-180}"
RAY_WORKER_RETRIES="${RAY_WORKER_RETRIES:-45}"
RAY_DONE_FILE="${LOG_DIR}/.ray_eval_done"
export HEAD_IP HEAD_ADDR RAY_PORT NUM_NODES RAY_READY_TIMEOUT RAY_WORKER_RETRIES RAY_DONE_FILE RAY_TMPDIR_RUNTIME PYTHON_BIN
rm -f "${RAY_DONE_FILE}"

stop_ray_cluster() {
  srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 --unbuffered \
    bash -lc "ray stop --force >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
}

sync_ray_logs() {
  local node
  for node in "${NODE_LIST[@]}"; do
    local node_archive_dir="${RAY_LOG_ARCHIVE_DIR}/${node}"
    mkdir -p "${node_archive_dir}"
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --unbuffered bash -lc "
      if [[ -d '${RAY_TMPDIR_RUNTIME}' ]]; then
        if command -v rsync >/dev/null 2>&1; then
          rsync -a '${RAY_TMPDIR_RUNTIME}/' '${node_archive_dir}/' || true
        else
          cp -a '${RAY_TMPDIR_RUNTIME}/.' '${node_archive_dir}/' 2>/dev/null || true
        fi
      fi
    " || true
  done
}

cleanup() {
  stop_ray_cluster
  sync_ray_logs
  rm -f "${RAY_DONE_FILE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

CMD=(
  "${PYTHON_BIN}" "${EMBODIED_PATH}/eval_embodied_agent.py"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${EXPERIMENT_NAME}"
  "cluster.num_nodes=${NUM_NODES}"
)

if [[ -n "${EVAL_CKPT_PATH}" ]]; then
  CMD+=("runner.ckpt_path=${EVAL_CKPT_PATH}")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf -v EVAL_CMD_STR '%q ' "${CMD[@]}"
export EVAL_CMD_STR

{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "REPO_PATH=${REPO_PATH}"
  echo "VENV_PATH=${VENV_PATH}"
  echo "Using Python at ${PYTHON_BIN}"
  echo "CONFIG_NAME=${CONFIG_NAME}"
  echo "EVAL_CKPT_PATH=${EVAL_CKPT_PATH}"
  echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  echo "CTRL_WORLD_PATH=${CTRL_WORLD_PATH}"
  echo "LIBERO_REPO_PATH=${LIBERO_REPO_PATH}"
  echo "ROBOT_PLATFORM=${ROBOT_PLATFORM}"
  echo "WANDB_MODE=${WANDB_MODE:-unset}"
  echo "WANDB_API_KEY_SET=$([[ -n "${WANDB_API_KEY:-}" ]] && echo true || echo false)"
  echo "WANDB_DIR=${WANDB_DIR}"
  echo "MUJOCO_GL=${MUJOCO_GL}"
  echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
  echo "RAY_TMPDIR(runtime)=${RAY_TMPDIR_RUNTIME}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
  echo "Node list: ${NODE_LIST[*]}"
  echo "Head node: ${HEAD_NODE} (${HEAD_IP})"
  echo "Ray address: ${HEAD_ADDR}"
  echo "Ray worker logs(runtime): ${RAY_TMPDIR_RUNTIME}/session_latest/logs"
  echo "Ray worker logs(archive): ${RAY_LOG_ARCHIVE_DIR}/<node>/session_latest/logs"
  nvidia-smi || true
  printf 'Command: %q ' "${CMD[@]}"
  echo
} | tee "${LOG_FILE}"

echo "Starting Ray cluster: head=${HEAD_NODE} ip=${HEAD_IP} port=${RAY_PORT}" | tee -a "${LOG_FILE}"

srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --unbuffered \
  bash -lc '
set -euo pipefail

if [[ "${SLURM_PROCID}" == "0" ]]; then
  trap "touch \"${RAY_DONE_FILE}\"" EXIT
  ray stop --force >/dev/null 2>&1 || true
  export RLINF_NODE_RANK=0
  ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --temp-dir="${RAY_TMPDIR_RUNTIME}" --disable-usage-stats

  "${PYTHON_BIN}" - <<'"'"'PY'"'"'
import os
import time
import ray

target_nodes = int(os.environ["NUM_NODES"])
timeout_s = int(os.environ.get("RAY_READY_TIMEOUT", "180"))
deadline = time.time() + timeout_s

ray.init(address="auto")
while True:
    alive = sum(1 for n in ray.nodes() if n.get("Alive"))
    print(f"Alive Ray nodes: {alive}/{target_nodes}", flush=True)
    if alive >= target_nodes:
        break
    if time.time() > deadline:
        raise SystemExit(
            f"Timed out waiting for Ray nodes: {alive}/{target_nodes} after {timeout_s}s."
        )
    time.sleep(2)
ray.shutdown()
PY

  echo "Launching evaluation on head with RAY_ADDRESS=${HEAD_ADDR}"
  export RAY_ADDRESS="${HEAD_ADDR}"
  eval "${EVAL_CMD_STR}"
else
  ray stop --force >/dev/null 2>&1 || true
  export RLINF_NODE_RANK="${SLURM_PROCID}"
  connected=0
  for i in $(seq 1 "${RAY_WORKER_RETRIES}"); do
    if ray start --address="${HEAD_ADDR}" --disable-usage-stats; then
      connected=1
      break
    fi
    sleep 2
  done

  if [[ "${connected}" != "1" ]]; then
    echo "Failed to connect Ray worker to ${HEAD_ADDR}" >&2
    exit 1
  fi

  while [[ ! -f "${RAY_DONE_FILE}" ]]; do
    sleep 2
  done
fi
' 2>&1 | tee -a "${LOG_FILE}"
