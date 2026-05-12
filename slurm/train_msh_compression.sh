#!/usr/bin/bash
#SBATCH -J msh-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-msh-%A.out
#SBATCH -e slurm-msh-%A.err

# ============================================================
# MSH Compression Training for HiSplat
#
# Phase 1 (freeze_hisplat=true):
#   HiSplat encoder+decoder frozen, only MSH codec trains.
#   The codec compresses refine_out (UNet features) before the
#   Gaussian/depth heads, so those heads learn to compensate
#   for quantization noise.
#
# Phase 2 (unfreeze_step > 0):
#   After unfreeze_step, HiSplat encoder also unfreezes with
#   reduced lr, so it adapts features to be compression-friendly.
# ============================================================

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SUBMIT_DIR}}"

# ---- User settings ----
SH_DEGREE="${SH_DEGREE:-4}"              # Must match pretrained ckpt (SH4=84ch)
LAMBDA="${LAMBDA:-0.001}"
PHASE="${PHASE:-1}"                 # 1 = freeze only, 2 = freeze + unfreeze
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/hisplat_re10k.ckpt}"
CODEC_INIT_CKPT="${CODEC_INIT_CKPT:-null}"
MAX_STEPS="${MAX_STEPS:-50000}"
UNFREEZE_STEP="${UNFREEZE_STEP:-30000}"  # only used if PHASE=2
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
TARGET="${TARGET:-refine_out}"           # "pre_refine", "refine_out", or "raw_gaussians"
RENDER_LOSS_WEIGHT="${RENDER_LOSS_WEIGHT:-1.0}"
if [[ "${TARGET}" == "pre_refine" ]]; then
    MSH_N="${MSH_N:-192}"
    MSH_M="${MSH_M:-320}"
    MSH_DOWNSAMPLE="${MSH_DOWNSAMPLE:-[2,2,2]}"
else
    MSH_N="${MSH_N:-128}"
    MSH_M="${MSH_M:-192}"
    MSH_DOWNSAMPLE="${MSH_DOWNSAMPLE:-[2,2,3]}"
fi
OUTPUT_NAME="${OUTPUT_NAME:-msh_${TARGET}_phase${PHASE}_lmd${LAMBDA}_$(date +%Y%m%d_%H%M%S)}"

# ---- Auto-resolve Phase 1 checkpoint for Phase 2 ----
if [[ "${PRETRAINED_CKPT}" == "__AUTO_FROM_PHASE1__" ]]; then
    PHASE1_OUTPUT_DIR="${PHASE1_OUTPUT_DIR:?PHASE1_OUTPUT_DIR must be set when using __AUTO_FROM_PHASE1__}"
    CKPT_DIR="${PHASE1_OUTPUT_DIR}/checkpoints"
    # Find the checkpoint with the highest step number (epoch_N-step_M.ckpt)
    PRETRAINED_CKPT=$(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null \
        | sed 's/.*step_\([0-9]*\)\.ckpt/\1 &/' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')
    if [[ -z "${PRETRAINED_CKPT}" ]]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
    echo "Auto-resolved Phase 1 checkpoint: ${PRETRAINED_CKPT}"
fi

# ---- Derived settings ----
if [[ "${PHASE}" == "1" ]]; then
    FREEZE_HISPLAT=true
    UNFREEZE_STEP_VAL=0
elif [[ "${PHASE}" == "2" ]]; then
    FREEZE_HISPLAT=true
    UNFREEZE_STEP_VAL=1  # unfreeze at step 1 (immediately after freeze init)
else
    echo "PHASE must be 1 or 2"
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

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_DIR}"

echo "============================================"
echo "MSH Compression Training"
echo "Phase: ${PHASE}"
echo "Lambda: ${LAMBDA}"
echo "SH Degree (model): ${SH_DEGREE}"
echo "Pretrained ckpt: ${PRETRAINED_CKPT}"
echo "Codec init ckpt: ${CODEC_INIT_CKPT}"
echo "Max steps: ${MAX_STEPS}"
echo "Unfreeze step: ${UNFREEZE_STEP_VAL}"
echo "Batch size: ${BATCH_SIZE_PER_GPU}"
echo "MSH N/M: ${MSH_N}/${MSH_M}"
echo "MSH downsample: ${MSH_DOWNSAMPLE}"
echo "Render loss weight: ${RENDER_LOSS_WEIGHT}"
echo "Output: ${OUTPUT_NAME}"
echo "Extra hydra args: ${EXTRA_HYDRA_ARGS:-<none>}"
echo "============================================"

# Optional extra Hydra overrides (e.g. for stage-weighted lambda).
# Pass via env: EXTRA_HYDRA_ARGS="model.encoder.compression.lambda_stage_weights=[4,2,1]"
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_HYDRA_ARGS:-}"

python -m src.main \
    "+experiment=re10k" \
    "data_loader.train.batch_size=${BATCH_SIZE_PER_GPU}" \
    "device=1" \
    "dataset.roots=[${DATASET_ROOT}]" \
    "output_dir=${OUTPUT_NAME}" \
    "trainer.max_steps=${MAX_STEPS}" \
    "trainer.val_check_interval=1000" \
    "checkpointing.load=${PRETRAINED_CKPT}" \
    "checkpointing.every_n_train_steps=5000" \
    "model.encoder.gaussian_adapter.sh_degree=${SH_DEGREE}" \
    "model.encoder.compression.enabled=true" \
    "model.encoder.compression.N=${MSH_N}" \
    "model.encoder.compression.M=${MSH_M}" \
    "model.encoder.compression.n_downsample_per_stage=${MSH_DOWNSAMPLE}" \
    "model.encoder.compression.lmbda=${LAMBDA}" \
    "model.encoder.compression.target=${TARGET}" \
    "model.encoder.compression.compress_sh_degree=${SH_DEGREE}" \
    "model.encoder.compression.codec_init_ckpt=${CODEC_INIT_CKPT}" \
    "train.freeze_hisplat=${FREEZE_HISPLAT}" \
    "train.unfreeze_step=${UNFREEZE_STEP_VAL}" \
    "train.unfreeze_lr_scale=0.1" \
    "train.render_loss_weight=${RENDER_LOSS_WEIGHT}" \
    "wandb.mode=offline" \
    "wandb.name=msh_phase${PHASE}_lmd${LAMBDA}" \
    "${EXTRA_ARGS_ARR[@]}"
