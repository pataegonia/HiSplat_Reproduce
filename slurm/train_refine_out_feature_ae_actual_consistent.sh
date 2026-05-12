#!/usr/bin/env bash
set -euo pipefail

# Resume the refine_out feature-AE run after enabling actual-consistent
# quantization and entropy auxiliary loss. This is the Gate-2 repair stage:
# keep rendering off, preserve feature identity, and calibrate the entropy
# bottleneck so STE/training mode better matches real bitstream decoding.

RUN_DIR="${RUN_DIR:-outputs/msh_refine_out_feature_ae_genonly_codecinit_lmd1e-7}"
CKPT_DIR="${RUN_DIR}/checkpoints"

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: checkpoint dir not found: ${CKPT_DIR}"
    exit 1
fi

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
if [[ -z "${PRETRAINED_CKPT}" ]]; then
    PRETRAINED_CKPT=$(ls -1 "${CKPT_DIR}"/best-val-*.ckpt 2>/dev/null | sort -V | tail -1 || true)
fi
if [[ -z "${PRETRAINED_CKPT}" ]]; then
    PRETRAINED_CKPT=$(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null | sort -V | tail -1 || true)
fi
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "ERROR: no checkpoint found under ${CKPT_DIR}"
    exit 1
fi

echo "Resume from: ${PRETRAINED_CKPT}"

PHASE="${PHASE:-1}" \
LAMBDA="${LAMBDA:-0.0000001}" \
TARGET="${TARGET:-refine_out}" \
MAX_STEPS="${MAX_STEPS:-15000}" \
PRETRAINED_CKPT="${PRETRAINED_CKPT}" \
RENDER_LOSS_WEIGHT="${RENDER_LOSS_WEIGHT:-0.0}" \
OUTPUT_NAME="${OUTPUT_NAME:-msh_refine_out_feature_ae_actual_consistent_lmd1e-7}" \
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-model.encoder.compression.entropy_aux_weight=1.0 model.encoder.compression.actual_consistent_quant=true +model.encoder.compression.feat_loss_alpha=1.0 +model.encoder.compression.feat_loss_start_step=0 +model.encoder.compression.feat_loss_warmup_steps=0 trainer.val_check_interval=1000 checkpointing.every_n_train_steps=1000 optimizer.lr=1e-5 trainer.gradient_clip_val=0.05}" \
sbatch \
    -o "MSH_log/refine_out_feature_ae_actual_consistent_%j.out" \
    -e "MSH_log/refine_out_feature_ae_actual_consistent_%j.err" \
    slurm/train_msh_compression.sh
