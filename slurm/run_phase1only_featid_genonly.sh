#!/usr/bin/bash
set -euo pipefail

# Phase 1 from the gen-only HiSplat checkpoint with feature identity loss.
# This avoids inheriting a Phase 1 MSH checkpoint where stage 2 may already
# have drifted into a non-identity feature coordinate system.

LOG_DIR="${LOG_DIR:-MSH_log}"
mkdir -p "${LOG_DIR}"

BASE_CKPT="${BASE_CKPT:-checkpoints/hisplat_re10k.ckpt}"
OUTPUT_NAME="${OUTPUT_NAME:-msh_refine_out_phase1only_genonly_extremerate_lmd1e-7_featid_a1}"

FEAT_LOSS_ALPHA="${FEAT_LOSS_ALPHA:-1.0}"
FEAT_LOSS_START_STEP="${FEAT_LOSS_START_STEP:-5000}"
FEAT_LOSS_WARMUP_STEPS="${FEAT_LOSS_WARMUP_STEPS:-10000}"
FEAT_LOSS_STAGE_WEIGHTS="${FEAT_LOSS_STAGE_WEIGHTS:-[1.0,1.0,1.0]}"

EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-} \
+model.encoder.compression.feat_loss_alpha=${FEAT_LOSS_ALPHA} \
+model.encoder.compression.feat_loss_start_step=${FEAT_LOSS_START_STEP} \
+model.encoder.compression.feat_loss_warmup_steps=${FEAT_LOSS_WARMUP_STEPS} \
+model.encoder.compression.feat_loss_stage_weights=${FEAT_LOSS_STAGE_WEIGHTS}"

echo "Submitting gen-only Phase 1 feature-identity run"
echo "  Base ckpt: ${BASE_CKPT}"
echo "  Output: ${OUTPUT_NAME}"
echo "  Feature loss alpha: ${FEAT_LOSS_ALPHA}"
echo "  Feature loss start: ${FEAT_LOSS_START_STEP}"
echo "  Feature loss warmup: ${FEAT_LOSS_WARMUP_STEPS}"
echo "  Feature loss stage weights: ${FEAT_LOSS_STAGE_WEIGHTS}"

PHASE=1 \
LAMBDA=0.0000001 \
TARGET=refine_out \
MAX_STEPS=60000 \
PRETRAINED_CKPT="${BASE_CKPT}" \
OUTPUT_NAME="${OUTPUT_NAME}" \
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS}" \
sbatch \
    -o "${LOG_DIR}/refine_out_phase1only_genonly_featid_%j.out" \
    -e "${LOG_DIR}/refine_out_phase1only_genonly_featid_%j.err" \
    slurm/train_msh_compression.sh
