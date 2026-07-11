#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# NoiseAdaptive2Score: DIV2K Poisson training
#
# Usage:
#   ./train_exe.sh
#
# 다른 validation noise level 사용:
#   VAL_LEVEL=0.05 ./train_exe.sh
#
# GPU 변경:
#   GPU=1 ./train_exe.sh
# ---------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------
# Experiment configuration
# -------------------------
GPU="${GPU:-1}"
VAL_LEVEL="${VAL_LEVEL:-0.01}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_THREADS="${NUM_THREADS:-4}"

N_EPOCHS="${N_EPOCHS:-200}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-DIV2K_NA2S_Poisson_${VAL_LEVEL}}"

# -------------------------
# Dataset paths
# -------------------------
TRAIN_ROOT="${ROOT_DIR}/train_imges/DIV2K_train_HR"

VAL_ROOT="${ROOT_DIR}/test_images/DIV2K_valid_HR_poisson"
VAL_CLEAN_DIR="${VAL_ROOT}/clean"
VAL_NOISY_DIR="${VAL_ROOT}/poisson_${VAL_LEVEL}"

# test_dataset.py가 noisy_50을 하드코딩하므로
# 선택한 Poisson 폴더를 noisy_50이라는 이름으로 연결한다.
VAL_ALIAS="${VAL_ROOT}/noisy_50"

CHECKPOINTS_DIR="${ROOT_DIR}/checkpoints"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

# -------------------------
# Path validation
# -------------------------
require_nonempty_dir() {
    local dir="$1"

    if [[ ! -d "${dir}" ]]; then
        echo "[ERROR] Directory not found: ${dir}" >&2
        exit 1
    fi

    if ! find "${dir}" -type f -print -quit | grep -q .; then
        echo "[ERROR] Directory contains no files: ${dir}" >&2
        exit 1
    fi
}

require_nonempty_dir "${TRAIN_ROOT}"
require_nonempty_dir "${VAL_CLEAN_DIR}"
require_nonempty_dir "${VAL_NOISY_DIR}"

CLEAN_COUNT="$(
    find "${VAL_CLEAN_DIR}" -type f -name '*.npy' | wc -l
)"
NOISY_COUNT="$(
    find "${VAL_NOISY_DIR}" -type f -name '*.npy' | wc -l
)"

if [[ "${CLEAN_COUNT}" -eq 0 ]]; then
    echo "[ERROR] No .npy files found in ${VAL_CLEAN_DIR}" >&2
    exit 1
fi

if [[ "${NOISY_COUNT}" -eq 0 ]]; then
    echo "[ERROR] No .npy files found in ${VAL_NOISY_DIR}" >&2
    exit 1
fi

if [[ "${CLEAN_COUNT}" -ne "${NOISY_COUNT}" ]]; then
    echo "[ERROR] Validation pair count mismatch." >&2
    echo "        clean : ${CLEAN_COUNT}" >&2
    echo "        noisy : ${NOISY_COUNT}" >&2
    exit 1
fi

# 실제 noisy_50 디렉토리가 이미 있다면 덮어쓰지 않는다.
if [[ -e "${VAL_ALIAS}" && ! -L "${VAL_ALIAS}" ]]; then
    echo "[ERROR] ${VAL_ALIAS} already exists and is not a symbolic link." >&2
    echo "        Remove it or modify data/test_dataset.py directly." >&2
    exit 1
fi

ln -sfn "${VAL_NOISY_DIR}" "${VAL_ALIAS}"

mkdir -p "${CHECKPOINTS_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "NoiseAdaptive2Score Poisson training"
echo "============================================================"
echo "Root             : ${ROOT_DIR}"
echo "GPU              : ${GPU}"
echo "Training data    : ${TRAIN_ROOT}"
echo "Validation clean : ${VAL_CLEAN_DIR}"
echo "Validation noisy : ${VAL_NOISY_DIR}"
echo "Validation pairs : ${CLEAN_COUNT}"
echo "Batch size       : ${BATCH_SIZE}"
echo "Epochs           : ${N_EPOCHS} + ${N_EPOCHS_DECAY}"
echo "Learning rate    : ${LEARNING_RATE}"
echo "Experiment       : ${EXPERIMENT_NAME}"
echo "Checkpoints      : ${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}"
echo "Log              : ${LOG_FILE}"
echo "============================================================"

cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

python -u train.py \
    --model Poisson \
    --target_model Poisson \
    --dataset_mode aligned \
    --dataroot "${TRAIN_ROOT}" \
    --dataroot_valid "${VAL_ROOT}" \
    --name "${EXPERIMENT_NAME}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --gpu_ids 0 \
    --direction BtoA \
    --input_nc 3 \
    --output_nc 3 \
    --load_size 256 \
    --crop_size 256 \
    --batch_size "${BATCH_SIZE}" \
    --num_threads "${NUM_THREADS}" \
    --n_epochs "${N_EPOCHS}" \
    --n_epochs_decay "${N_EPOCHS_DECAY}" \
    --lr "${LEARNING_RATE}" \
    --beta1 0.9 \
    --lr_policy step \
    --lr_decay_iters 100 \
    --print_freq 100 \
    --display_freq 500 \
    --save_latest_freq 5000 \
    --save_epoch_freq 5 \
    --display_id -1 \
    --no_html \
    2>&1 | tee -a "${LOG_FILE}"