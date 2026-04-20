#!/usr/bin/env bash
#SBATCH -J rlinf_book_to_ctrlworld
#SBATCH -A naiss2025-22-1173
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/convert_lerobot_book_to_ctrlworld_out_%j.txt
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/convert_lerobot_book_to_ctrlworld_err_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -p alvis

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run with sbatch." >&2
  exit 1
fi

ROOT_PATH=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi
RLINF_PATH="${ROOT_PATH}/RLinf"
CTRL_WORLD_PATH="${ROOT_PATH}/Ctrl-World"

cd "${RLINF_PATH}"
mkdir -p "${RLINF_PATH}/logs"

if ! type module >/dev/null 2>&1; then
  # Some batch shells do not preload Lmod.
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load FFmpeg/7.0.2-GCCcore-13.3.0

VENV_PATH="${VENV_PATH:-${RLINF_PATH}/.venv_pi05}"
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

export PYTHONPATH="${RLINF_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SRC_ROOT="${SRC_ROOT:-${ROOT_PATH}/lerobot_book}"
DATASET_NAME="${DATASET_NAME:-ctrl_world_book_from_lerobot}"
DST_ROOT="${DST_ROOT:-${RLINF_PATH}/datasets/${DATASET_NAME}}"
SVD_PATH="${SVD_PATH:-${CTRL_WORLD_PATH}/pretrained/stable-video-diffusion-img2vid}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-192}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
STATE_REPEAT="${STATE_REPEAT:-3}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
MAX_EPISODES="${MAX_EPISODES:-}"
OVERWRITE="${OVERWRITE:-0}"

CAMERA_KEY_0="${CAMERA_KEY_0:-observation.images.d405_rgb}"
CAMERA_KEY_1="${CAMERA_KEY_1:-observation.images.d405_1_rgb}"
CAMERA_KEY_2="${CAMERA_KEY_2:-observation.images.d435_rgb}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "SRC_ROOT does not exist: ${SRC_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${SVD_PATH}" ]]; then
  echo "SVD_PATH does not exist: ${SVD_PATH}" >&2
  exit 1
fi

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${RLINF_PATH}/logs/${TIMESTAMP}-convert_lerobot_book_to_ctrlworld-job${SLURM_JOB_ID}"
LOG_FILE="${LOG_DIR}/convert_lerobot_book_to_ctrlworld.log"
mkdir -p "${LOG_DIR}"

CONVERT_CMD=(
  "${PYTHON_BIN}" "${RLINF_PATH}/scripts/convert_lerobot_video_to_ctrlworld.py"
  --src-root "${SRC_ROOT}"
  --dst-root "${DST_ROOT}"
  --svd-path "${SVD_PATH}"
  --camera-keys "${CAMERA_KEY_0}" "${CAMERA_KEY_1}" "${CAMERA_KEY_2}"
  --image-height "${IMAGE_HEIGHT}"
  --image-width "${IMAGE_WIDTH}"
  --frame-stride "${FRAME_STRIDE}"
  --state-repeat "${STATE_REPEAT}"
  --val-ratio "${VAL_RATIO}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
  CONVERT_CMD+=(--max-episodes "${MAX_EPISODES}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CONVERT_CMD+=(--overwrite)
fi

META_CMD=(
  "${PYTHON_BIN}" "${CTRL_WORLD_PATH}/dataset_meta_info/create_meta_info.py"
  --droid_output_path "${DST_ROOT}"
  --dataset_name "${DATASET_NAME}"
)

{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "ROOT_PATH=${ROOT_PATH}"
  echo "RLINF_PATH=${RLINF_PATH}"
  echo "CTRL_WORLD_PATH=${CTRL_WORLD_PATH}"
  echo "VENV_PATH=${VENV_PATH}"
  echo "Using Python at ${PYTHON_BIN}"
  echo "SRC_ROOT=${SRC_ROOT}"
  echo "DATASET_NAME=${DATASET_NAME}"
  echo "DST_ROOT=${DST_ROOT}"
  echo "SVD_PATH=${SVD_PATH}"
  echo "IMAGE_HEIGHT=${IMAGE_HEIGHT}"
  echo "IMAGE_WIDTH=${IMAGE_WIDTH}"
  echo "FRAME_STRIDE=${FRAME_STRIDE}"
  echo "STATE_REPEAT=${STATE_REPEAT}"
  echo "VAL_RATIO=${VAL_RATIO}"
  echo "SEED=${SEED}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "DEVICE=${DEVICE}"
  echo "MAX_EPISODES=${MAX_EPISODES}"
  echo "OVERWRITE=${OVERWRITE}"
  echo "CAMERA_KEY_0=${CAMERA_KEY_0}"
  echo "CAMERA_KEY_1=${CAMERA_KEY_1}"
  echo "CAMERA_KEY_2=${CAMERA_KEY_2}"
  module list 2>&1 || true
  nvidia-smi || true
  printf 'Convert command: %q ' "${CONVERT_CMD[@]}"
  echo
  printf 'Meta command: %q ' "${META_CMD[@]}"
  echo
} | tee "${LOG_FILE}"

"${CONVERT_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

cd "${CTRL_WORLD_PATH}"
"${META_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
