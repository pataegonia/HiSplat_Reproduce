#!/usr/bin/bash
#SBATCH -J msh-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 0-12:00:00
#SBATCH -o slurm-msh-test-%A.out
#SBATCH -e slurm-msh-test-%A.err

# ============================================================
# MSH Compression Evaluation on the Re10K test set.
#
# Loads a Phase 2 checkpoint with the exact same compression
# config it was trained with, then runs mode=test. The model_wrapper
# automatically switches msh_compress_mode to "actual" during
# test_step so that per-sample bitstream bytes are measured with
# compress()/decompress() (not STE estimates).
#
# Required env vars (or set defaults via cli args / env):
#   TARGET    : "pre_refine", "refine_out", or "raw_gaussians"
#   LAMBDA    : rate-distortion lambda (used to locate the ckpt)
#   SH_DEGREE : must match the training run
#
# Either provide CKPT_PATH explicitly, or PHASE2_OUTPUT_DIR to
# auto-resolve the highest-step checkpoint under
#   <PHASE2_OUTPUT_DIR>/checkpoints/*.ckpt
# ============================================================

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SUBMIT_DIR}}"

# ---- User settings ----
TARGET="${TARGET:-refine_out}"
LAMBDA="${LAMBDA:-0.001}"
SH_DEGREE="${SH_DEGREE:-4}"
EXPERIMENT="${EXPERIMENT:-re10k}"
CKPT_PATH="${CKPT_PATH:-}"
PHASE2_OUTPUT_DIR="${PHASE2_OUTPUT_DIR:-outputs/msh_${TARGET}_phase2_lmd${LAMBDA}}"
OUTPUT_NAME="${OUTPUT_NAME:-test_msh_${TARGET}_lmd${LAMBDA}}"

TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
COMPUTE_SCORES="${COMPUTE_SCORES:-true}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
MSH_COMPRESS_MODE="${MSH_COMPRESS_MODE:-actual}"  # actual, training(STE), or bypass
if [[ "${TARGET}" == "pre_refine" ]]; then
    MSH_N="${MSH_N:-192}"
    MSH_M="${MSH_M:-320}"
    MSH_DOWNSAMPLE="${MSH_DOWNSAMPLE:-[2,2,2]}"
else
    MSH_N="${MSH_N:-128}"
    MSH_M="${MSH_M:-192}"
    MSH_DOWNSAMPLE="${MSH_DOWNSAMPLE:-[2,2,3]}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
DATASET_BASE="${DATASET_BASE:-/data3/local_datasets}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${EXPERIMENT}}"
DATASET_ROOT="${DATASET_ROOT:-${DATASET_BASE}/${DATASET_SUBDIR}}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_${EXPERIMENT}.json}"

# ---- Auto-resolve Phase 2 ckpt if not given ----
if [[ -z "${CKPT_PATH}" ]]; then
    CKPT_DIR="${PHASE2_OUTPUT_DIR}/checkpoints"
    if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "ERROR: Phase 2 checkpoint dir not found: ${CKPT_DIR}"
        echo "Either set CKPT_PATH= directly, or ensure PHASE2_OUTPUT_DIR exists."
        exit 1
    fi
    CKPT_PATH=$(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null \
        | sed 's/.*step_\([0-9]*\)\.ckpt/\1 &/' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "WARNING: Dataset directory is not visible from this node: ${DATASET_ROOT}"
    echo "         If you submit with sbatch to ariel-v10, it may still be visible there."
fi

if [[ ! -f "${REPO_DIR}/${INDEX_PATH}" ]]; then
    echo "ERROR: Evaluation index not found: ${REPO_DIR}/${INDEX_PATH}"
    exit 1
fi

# ---- Conda ----
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "Conda not found"; exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_DIR}"

echo "============================================"
echo "MSH Compression Evaluation"
echo "Target: ${TARGET}"
echo "Lambda: ${LAMBDA}"
echo "SH degree: ${SH_DEGREE}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output: outputs/${OUTPUT_NAME}"
echo "MSH compress mode: ${MSH_COMPRESS_MODE}"
echo "MSH N/M: ${MSH_N}/${MSH_M}"
echo "MSH downsample: ${MSH_DOWNSAMPLE}"
echo "Extra hydra args: ${EXTRA_HYDRA_ARGS:-<none>}"
echo "============================================"

read -ra EXTRA_ARGS_ARR <<< "${EXTRA_HYDRA_ARGS:-}"

python -m src.main \
    "+experiment=${EXPERIMENT}" \
    "mode=test" \
    "device=1" \
    "dataset.roots=[${DATASET_ROOT}]" \
    "dataset/view_sampler=evaluation" \
    "dataset.view_sampler.index_path=${INDEX_PATH}" \
    "data_loader.test.batch_size=${TEST_BATCH_SIZE}" \
    "data_loader.test.num_workers=${TEST_NUM_WORKERS}" \
    "test.compute_scores=${COMPUTE_SCORES}" \
    "test.save_image=${SAVE_IMAGE}" \
    "test.save_video=${SAVE_VIDEO}" \
    "test.msh_compress_mode=${MSH_COMPRESS_MODE}" \
    "output_dir=${OUTPUT_NAME}" \
    "checkpointing.load=${CKPT_PATH}" \
    "model.encoder.gaussian_adapter.sh_degree=${SH_DEGREE}" \
    "model.encoder.compression.enabled=true" \
    "model.encoder.compression.N=${MSH_N}" \
    "model.encoder.compression.M=${MSH_M}" \
    "model.encoder.compression.n_downsample_per_stage=${MSH_DOWNSAMPLE}" \
    "model.encoder.compression.lmbda=${LAMBDA}" \
    "model.encoder.compression.target=${TARGET}" \
    "model.encoder.compression.compress_sh_degree=${SH_DEGREE}" \
    "wandb.mode=disabled" \
    "wandb.name=test_msh_${TARGET}_lmd${LAMBDA}" \
    "${EXTRA_ARGS_ARR[@]}"
