#!/usr/bin/bash
set -euo pipefail

# Evaluation wrapper for the gen-only-start Phase 1 MSH feature-identity run.
# Usage:
#   bash slurm/test_phase1only_featid_genonly.sh
#   CKPT_KIND=best bash slurm/test_phase1only_featid_genonly.sh

LOG_DIR="${LOG_DIR:-MSH_log}"
mkdir -p "${LOG_DIR}"

RUN_DIR="${RUN_DIR:-outputs/msh_refine_out_phase1only_genonly_extremerate_lmd1e-7_featid_a1}"
CKPT_KIND="${CKPT_KIND:-latest}"  # latest or best
MSH_COMPRESS_MODE="${MSH_COMPRESS_MODE:-actual}"  # actual, training(STE), or bypass
OUTPUT_NAME="${OUTPUT_NAME:-test_msh_refine_out_phase1only_genonly_extremerate_lmd1e-7_featid_a1_${CKPT_KIND}_${MSH_COMPRESS_MODE}}"

FEAT_LOSS_ALPHA="${FEAT_LOSS_ALPHA:-1.0}"
FEAT_LOSS_START_STEP="${FEAT_LOSS_START_STEP:-5000}"
FEAT_LOSS_WARMUP_STEPS="${FEAT_LOSS_WARMUP_STEPS:-10000}"
FEAT_LOSS_STAGE_WEIGHTS="${FEAT_LOSS_STAGE_WEIGHTS:-[1.0,1.0,1.0]}"

CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "${CKPT_PATH}" && "${CKPT_KIND}" == "latest" ]]; then
    CKPT_PATH=$(ls -1 "${RUN_DIR}/checkpoints"/*.ckpt 2>/dev/null \
        | sed 's/.*step_\([0-9]*\)\.ckpt/\1 &/' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "ERROR: no checkpoint found under ${RUN_DIR}/checkpoints"
        exit 1
    fi
elif [[ -z "${CKPT_PATH}" && "${CKPT_KIND}" == "best" ]]; then
    CKPT_PATH=$(ls -1 "${RUN_DIR}/checkpoints"/best-val*.ckpt 2>/dev/null \
        | sed -E 's/.*best-val-([0-9]+)\.ckpt/\1 &/' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}')
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "ERROR: no best-val checkpoint found under ${RUN_DIR}/checkpoints"
        exit 1
    fi
elif [[ "${CKPT_KIND}" != "latest" && -z "${CKPT_PATH}" ]]; then
    echo "ERROR: CKPT_KIND must be latest or best, or set CKPT_PATH directly"
    exit 1
fi

EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-} \
+model.encoder.compression.feat_loss_alpha=${FEAT_LOSS_ALPHA} \
+model.encoder.compression.feat_loss_start_step=${FEAT_LOSS_START_STEP} \
+model.encoder.compression.feat_loss_warmup_steps=${FEAT_LOSS_WARMUP_STEPS} \
+model.encoder.compression.feat_loss_stage_weights=${FEAT_LOSS_STAGE_WEIGHTS}"

echo "Submitting gen-only Phase 1 feature-identity evaluation"
echo "  Run dir: ${RUN_DIR}"
echo "  CKPT kind: ${CKPT_KIND}"
echo "  CKPT path: ${CKPT_PATH}"
echo "  MSH compress mode: ${MSH_COMPRESS_MODE}"
echo "  Output: ${OUTPUT_NAME}"

TARGET=refine_out \
LAMBDA=0.0000001 \
PHASE2_OUTPUT_DIR="${RUN_DIR}" \
CKPT_PATH="${CKPT_PATH}" \
OUTPUT_NAME="${OUTPUT_NAME}" \
MSH_COMPRESS_MODE="${MSH_COMPRESS_MODE}" \
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS}" \
sbatch \
    -o "${LOG_DIR}/test_phase1only_genonly_featid_${CKPT_KIND}_%j.out" \
    -e "${LOG_DIR}/test_phase1only_genonly_featid_${CKPT_KIND}_%j.err" \
    slurm/test_msh_compression.sh
