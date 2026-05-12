# Refine-Out Feature-AE Plan

## Goal

This track targets the immediate experiment goal:

```text
HiSplat + MSH, without progressive/scalable coding, should approach gen-only
HiSplat quality at high bitrate.
```

The current best independent MSH result is still about 1 dB below gen-only. The
next question is not whether more bits help, but whether the codec can behave as
an identity map on the gen-only `refine_out` features.

## Strategy

Use a clean gen-only HiSplat generator and transplant only a pretrained MSH
codec:

```text
generator weights: checkpoints/hisplat_re10k.ckpt
codec weights:     step3/best MSH checkpoint
target:            refine_out
trainable:         MSH codec only
loss:              per-channel relative feature L1 + tiny rate
render loss:       disabled at first
```

This removes the main confounders from earlier experiments:

```text
Phase 2 backbone drift
random codec initialization
heads/codec noise co-adaptation
render-loss local minima
pre_refine heterogeneous input collapse
```

## New Controls

`test.msh_compress_mode=bypass`
: Diagnostic oracle. The codec is skipped and `x_hat = x`, giving the upper
  bound for the current generator/checkpoint.

`model.encoder.compression.codec_init_ckpt`
: Optional checkpoint path used to initialize only `encoder.msh_codec.*`
  tensors after the main generator checkpoint has been loaded.

`train.render_loss_weight`
: Set to `0.0` for feature-AE pretraining. The encoder still runs and feature
  losses are computed, but decoder rendering losses are not added.

## Submit

```bash
CODEC_INIT_CKPT=outputs/<step3-run>/checkpoints/<best>.ckpt \
bash slurm/train_refine_out_feature_ae.sh
```

Recommended first run:

```text
PRETRAINED_CKPT=checkpoints/hisplat_re10k.ckpt
CODEC_INIT_CKPT=<step3 best checkpoint>
TARGET=refine_out
PHASE=1
LAMBDA=1e-7
RENDER_LOSS_WEIGHT=0
feat_loss_alpha=1.0
feat_loss_warmup_steps=0
MAX_STEPS=30000
```

## Gates

Gate 1: feature identity

```text
train/val feat_stage0, feat_stage1, feat_stage2 should decrease.
```

Gate 2: STE vs actual coding

```text
Test the same checkpoint with MSH_COMPRESS_MODE=training and actual.
The PSNR and feature gap should be small.
```

Gate 3: rendering quality

```text
After Gate 1/2 pass, enable render loss for a short fine-tune:
RENDER_LOSS_WEIGHT=1
feat_loss_alpha=0.1~0.3
MAX_STEPS=5k~10k
```
