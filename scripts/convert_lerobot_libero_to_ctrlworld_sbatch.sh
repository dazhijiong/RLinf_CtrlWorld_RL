#!/usr/bin/env bash
#SBATCH -J rlinf_lerobot_to_ctrlworld
#SBATCH -A naiss2025-22-1173
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/convert_lerobot_to_ctrlworld_out_%j.txt
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/convert_lerobot_to_ctrlworld_err_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -p alvis

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run with sbatch." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/scripts/convert_lerobot_libero_to_ctrlworld.py" ]]; then
  REPO_PATH="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/RLinf/scripts/convert_lerobot_libero_to_ctrlworld.py" ]]; then
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

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SRC_ROOT="${SRC_ROOT:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/datasets/lerobot_libero_spatial_image}"
DST_ROOT="${DST_ROOT:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/datasets/ctrl_world_libero_spatial_from_lerobot}"
SVD_PATH="${SVD_PATH:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/Ctrl-World/pretrained/stable-video-diffusion-img2vid}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-192}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
STATE_REPEAT="${STATE_REPEAT:-3}"
FPS="${FPS:-10}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
GRIPPER_STRATEGY="${GRIPPER_STRATEGY:-mean_abs}"
MAX_EPISODES="${MAX_EPISODES:-}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "SRC_ROOT does not exist: ${SRC_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${SVD_PATH}" ]]; then
  echo "SVD_PATH does not exist: ${SVD_PATH}" >&2
  exit 1
fi

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${REPO_PATH}/logs/${TIMESTAMP}-convert_lerobot_to_ctrlworld-job${SLURM_JOB_ID}"
LOG_FILE="${LOG_DIR}/convert_lerobot_to_ctrlworld.log"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${REPO_PATH}/scripts/convert_lerobot_libero_to_ctrlworld.py"
  --src-root "${SRC_ROOT}"
  --dst-root "${DST_ROOT}"
  --svd-path "${SVD_PATH}"
  --image-height "${IMAGE_HEIGHT}"
  --image-width "${IMAGE_WIDTH}"
  --frame-stride "${FRAME_STRIDE}"
  --state-repeat "${STATE_REPEAT}"
  --fps "${FPS}"
  --val-ratio "${VAL_RATIO}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --gripper-strategy "${GRIPPER_STRATEGY}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
  CMD+=(--max-episodes "${MAX_EPISODES}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "REPO_PATH=${REPO_PATH}"
  echo "VENV_PATH=${VENV_PATH}"
  echo "Using Python at ${PYTHON_BIN}"
  echo "SRC_ROOT=${SRC_ROOT}"
  echo "DST_ROOT=${DST_ROOT}"
  echo "SVD_PATH=${SVD_PATH}"
  echo "IMAGE_HEIGHT=${IMAGE_HEIGHT}"
  echo "IMAGE_WIDTH=${IMAGE_WIDTH}"
  echo "FRAME_STRIDE=${FRAME_STRIDE}"
  echo "STATE_REPEAT=${STATE_REPEAT}"
  echo "FPS=${FPS}"
  echo "VAL_RATIO=${VAL_RATIO}"
  echo "SEED=${SEED}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "DEVICE=${DEVICE}"
  echo "GRIPPER_STRATEGY=${GRIPPER_STRATEGY}"
  echo "MAX_EPISODES=${MAX_EPISODES:-unset}"
  echo "OVERWRITE=${OVERWRITE}"
  nvidia-smi || true
  printf 'Command: %q ' "${CMD[@]}"
  echo
} | tee "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

