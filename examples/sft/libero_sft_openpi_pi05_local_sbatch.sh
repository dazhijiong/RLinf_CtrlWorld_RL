#!/usr/bin/env bash
#SBATCH -J rlinf_pi05_libero_sft_expert
#SBATCH -A naiss2025-22-1173
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/pi05_libero_sft_out_%j.txt
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/pi05_libero_sft_err_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -p alvis

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run with sbatch." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/examples/sft/train_vla_sft.py" ]]; then
  REPO_PATH="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/RLinf/examples/sft/train_vla_sft.py" ]]; then
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

VENV_PATH="${VENV_PATH:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/.venv_pio5_sft}"
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

export EMBODIED_PATH="${REPO_PATH}/examples/sft"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

CONFIG_NAME="${CONFIG_NAME:-libero_sft_openpi_pi05_local}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_sft_openpi_pi05_local_base_full_expert_lr1e5}"
MODEL_PATH="${MODEL_PATH:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/models/pi05_base_rlinf}"
LR="${LR:-1.0e-5}"
RESUME_DIR="${RESUME_DIR:-/mimer/NOBACKUP/groups/naiss2024-5-164/Hanzhi/RLinf/logs/20260401-193129-libero_sft_openpi_pi05_local-job6287746/libero_sft_openpi_pi05_local_base_full_expert_lr1e5/checkpoints/global_step_10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
if [[ -n "${RESUME_DIR}" && -z "${SAVE_INTERVAL}" ]]; then
  # Skip the immediate step-2000 checkpoint when resuming from step 1800.
  SAVE_INTERVAL="10000"
fi
TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${REPO_PATH}/logs/${TIMESTAMP}-${CONFIG_NAME}-job${SLURM_JOB_ID}"
LOG_FILE="${LOG_DIR}/run_vla_sft.log"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${EMBODIED_PATH}/train_vla_sft.py"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${EXPERIMENT_NAME}"
  "actor.model.model_path=${MODEL_PATH}"
  "actor.optim.lr=${LR}"
  "actor.model.is_lora=False"
  "actor.model.openpi.train_expert_only=True"
)

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=("runner.resume_dir=${RESUME_DIR}")
fi

if [[ -n "${SAVE_INTERVAL}" ]]; then
  CMD+=("runner.save_interval=${SAVE_INTERVAL}")
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
  echo "CONFIG_NAME=${CONFIG_NAME}"
  echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "LR=${LR}"
  echo "RESUME_DIR=${RESUME_DIR}"
  echo "SAVE_INTERVAL=${SAVE_INTERVAL}"
  echo "EMBODIED_PATH=${EMBODIED_PATH}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
  echo "MUJOCO_GL=${MUJOCO_GL}"
  echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
  nvidia-smi || true
  printf 'Command: %q ' "${CMD[@]}"
  echo
} | tee "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
