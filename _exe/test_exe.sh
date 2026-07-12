#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NoiseAdaptive2Score Poisson test
#
# 기본 실행:
#   ./test_exe.sh
#
# Poisson 0.05 테스트:
#   NOISE_LEVEL=0.05 ./test_exe.sh
#
# 특정 epoch 테스트:
#   EPOCH=200 ./test_exe.sh
#
# 다른 GPU 사용:
#   GPU=1 ./test_exe.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------
# User configuration
# ------------------------------------------------------------
GPU="${GPU:-1}"

# test_images/DIV2K_valid_HR_poisson/poisson_<NOISE_LEVEL>
NOISE_LEVEL="${NOISE_LEVEL:-0.01}"

# best, latest, 200 등 사용 가능
EPOCH="${EPOCH:-best}"

# 학습할 때 사용한 checkpoint 디렉토리 이름
CHECKPOINT_NAME="${CHECKPOINT_NAME:-DIV2K_NA2S_Poisson_0.01}"

# 처리할 최대 이미지 수
NUM_TEST="${NUM_TEST:-10000}"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
TEST_ROOT="${TEST_ROOT:-${ROOT_DIR}/test_images/DIV2K_valid_HR_poisson}"

CLEAN_DIR="${TEST_ROOT}/clean"
NOISY_SOURCE="${NOISY_SOURCE:-${TEST_ROOT}/poisson_${NOISE_LEVEL}}"

# test2_dataset.py가 요구하는 폴더 이름
NOISY_ALIAS="${TEST_ROOT}/noisy_${NOISE_LEVEL}"

CHECKPOINTS_DIR="${ROOT_DIR}/checkpoints"
CHECKPOINT_DIR="${CHECKPOINTS_DIR}/${CHECKPOINT_NAME}"
EMA_CHECKPOINT="${CHECKPOINT_DIR}/${EPOCH}_ema_f.pth"

RESULTS_DIR="${ROOT_DIR}/results"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/test_${CHECKPOINT_NAME}_${NOISE_LEVEL}_${EPOCH}.log"

# ------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------
require_dir() {
    local dir="$1"

    if [[ ! -d "${dir}" ]]; then
        echo "[ERROR] Directory not found: ${dir}" >&2
        exit 1
    fi
}

count_npy() {
    local dir="$1"
    find -L "${dir}" -type f -name '*.npy' | wc -l
}

require_dir "${CLEAN_DIR}"
require_dir "${NOISY_SOURCE}"
require_dir "${CHECKPOINT_DIR}"

if [[ ! -f "${EMA_CHECKPOINT}" ]]; then
    echo "[ERROR] EMA checkpoint not found:" >&2
    echo "        ${EMA_CHECKPOINT}" >&2
    echo >&2
    echo "Available EMA checkpoints:" >&2
    find "${CHECKPOINT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*_ema_f.pth' \
        -printf '  %f\n' \
        | sort -V >&2
    exit 1
fi

CLEAN_COUNT="$(count_npy "${CLEAN_DIR}")"
NOISY_COUNT="$(count_npy "${NOISY_SOURCE}")"

if [[ "${CLEAN_COUNT}" -eq 0 ]]; then
    echo "[ERROR] No .npy files found in: ${CLEAN_DIR}" >&2
    exit 1
fi

if [[ "${NOISY_COUNT}" -eq 0 ]]; then
    echo "[ERROR] No .npy files found in: ${NOISY_SOURCE}" >&2
    exit 1
fi

if [[ "${CLEAN_COUNT}" -ne "${NOISY_COUNT}" ]]; then
    echo "[ERROR] Clean/noisy file count mismatch." >&2
    echo "        clean : ${CLEAN_COUNT}" >&2
    echo "        noisy : ${NOISY_COUNT}" >&2
    exit 1
fi

# ------------------------------------------------------------
# test2_dataset.py expects:
#
#   TEST_ROOT/
#   ├── clean/
#   └── noisy_<NOISE_LEVEL>/
#
# 현재 실제 폴더 poisson_<NOISE_LEVEL>을 심볼릭 링크로 연결한다.
# ------------------------------------------------------------
if [[ -e "${NOISY_ALIAS}" && ! -L "${NOISY_ALIAS}" ]]; then
    echo "[ERROR] Path already exists and is not a symbolic link:" >&2
    echo "        ${NOISY_ALIAS}" >&2
    echo "Remove it manually or modify data/test2_dataset.py." >&2
    exit 1
fi

ln -sfn "${NOISY_SOURCE}" "${NOISY_ALIAS}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# test.py가 실제로 만드는 결과 경로
EXPECTED_RESULT_DIR="${RESULTS_DIR}/${CHECKPOINT_NAME}/test_${EPOCH}"

echo "============================================================"
echo "NoiseAdaptive2Score Poisson test"
echo "============================================================"
echo "Project root      : ${ROOT_DIR}"
echo "GPU               : ${GPU}"
echo "Noise level       : ${NOISE_LEVEL}"
echo "Clean data        : ${CLEAN_DIR}"
echo "Noisy data        : ${NOISY_SOURCE}"
echo "Dataset alias     : ${NOISY_ALIAS}"
echo "Number of pairs   : ${CLEAN_COUNT}"
echo "Checkpoint        : ${EMA_CHECKPOINT}"
echo "Maximum test data : ${NUM_TEST}"
echo "Results           : ${EXPECTED_RESULT_DIR}"
echo "Log               : ${LOG_FILE}"
echo "============================================================"

cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

# CUDA_VISIBLE_DEVICES로 물리 GPU를 선택했으므로,
# Python 프로세스 내부에서는 해당 GPU가 cuda:0으로 보인다.
python -u test.py \
    --model Poisson \
    --target_model Poisson \
    --dataset_mode test2 \
    --dataroot "${TEST_ROOT}" \
    --dataroot_valid "${TEST_ROOT}" \
    --noise_level "${NOISE_LEVEL}" \
    --name "${CHECKPOINT_NAME}" \
    --pretrain_name "${CHECKPOINT_NAME}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --epoch "${EPOCH}" \
    --results_dir "${RESULTS_DIR}" \
    --num_test "${NUM_TEST}" \
    --gpu_ids 0 \
    --direction BtoA \
    --input_nc 3 \
    --output_nc 3 \
    --ngf 64 \
    --load_size 256 \
    --crop_size 256 \
    --phase test \
    2>&1 | tee "${LOG_FILE}"

echo
echo "============================================================"
echo "Test completed."
echo "Result directory:"
echo "  ${EXPECTED_RESULT_DIR}"
echo
echo "Open the HTML result:"
echo "  ${EXPECTED_RESULT_DIR}/index.html"
echo "============================================================"