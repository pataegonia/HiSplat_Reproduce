#!/usr/bin/env bash
# Submit the first MVSplat-style pre-refine MSH experiment.
#
# This keeps the HiSplat 3-stage pyramid unchanged, starts from the gen-only
# HiSplat checkpoint, freezes the generator, and trains only the MSH codec on
# the pre-refine feature bundle.

set -euo pipefail

mkdir -p MSH_log

PHASE="${PHASE:-1}" \
TARGET="${TARGET:-pre_refine}" \
LAMBDA="${LAMBDA:-0.000001}" \
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints/hisplat_re10k.ckpt}" \
MAX_STEPS="${MAX_STEPS:-30000}" \
MSH_N="${MSH_N:-192}" \
MSH_M="${MSH_M:-320}" \
MSH_DOWNSAMPLE="${MSH_DOWNSAMPLE:-[2,2,2]}" \
OUTPUT_NAME="${OUTPUT_NAME:-msh_pre_refine_phase1_genonly_lmd1e-6_bypass_featid}" \
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-+model.encoder.compression.feat_loss_alpha=1.0 +model.encoder.compression.feat_loss_start_step=0 +model.encoder.compression.feat_loss_warmup_steps=10000 +model.encoder.compression.pre_refine_bypass_start_step=0 +model.encoder.compression.pre_refine_bypass_warmup_steps=15000 optimizer.lr=2e-5 trainer.gradient_clip_val=0.05 checkpointing.every_n_train_steps=500 trainer.val_check_interval=500}" \
sbatch \
    -o "MSH_log/pre_refine_msh_%j.out" \
    -e "MSH_log/pre_refine_msh_%j.err" \
    slurm/train_msh_compression.sh
