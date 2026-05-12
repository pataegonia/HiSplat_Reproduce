#!/usr/bin/env bash
# Feature-AE pretraining for the refine_out MSH codec.
#
# This keeps the gen-only HiSplat generator fixed, optionally transplants a
# pretrained MSH codec, and trains the codec with feature identity + tiny rate
# loss only. Rendering loss is disabled via RENDER_LOSS_WEIGHT=0.

set -euo pipefail

mkdir -p MSH_log

: "${CODEC_INIT_CKPT:?Set CODEC_INIT_CKPT to the step3/best MSH checkpoint to transplant}"

PHASE="${PHASE:-1}" \
TARGET="${TARGET:-refine_out}" \
LAMBDA="${LAMBDA:-0.0000001}" \
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/hisplat_re10k.ckpt}" \
CODEC_INIT_CKPT="${CODEC_INIT_CKPT}" \
RENDER_LOSS_WEIGHT="${RENDER_LOSS_WEIGHT:-0.0}" \
MAX_STEPS="${MAX_STEPS:-30000}" \
MSH_N="${MSH_N:-128}" \
MSH_M="${MSH_M:-192}" \
MSH_DOWNSAMPLE="${MSH_DOWNSAMPLE:-[2,2,3]}" \
OUTPUT_NAME="${OUTPUT_NAME:-msh_refine_out_feature_ae_genonly_codecinit_lmd1e-7}" \
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-+model.encoder.compression.feat_loss_alpha=1.0 +model.encoder.compression.feat_loss_start_step=0 +model.encoder.compression.feat_loss_warmup_steps=0 checkpointing.every_n_train_steps=1000 trainer.val_check_interval=1000 optimizer.lr=2e-5 trainer.gradient_clip_val=0.05}" \
sbatch \
    -o "MSH_log/refine_out_feature_ae_%j.out" \
    -e "MSH_log/refine_out_feature_ae_%j.err" \
    slurm/train_msh_compression.sh
