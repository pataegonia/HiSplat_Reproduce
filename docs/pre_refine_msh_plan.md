# Pre-Refine MSH Compression Plan

## Goal

The current project is a preliminary experiment for a scalable/progressive
feed-forward 3DGS codec. Before designing the full scalable codec, we want to
verify whether an FF generator such as HiSplat can preserve rendering quality
when an internal representation is compressed with a Mean-Scale Hyperprior
codec.

The previous `refine_out` MSH setup showed that simply spending more bits does
not recover gen-only HiSplat quality. High-rate runs either saturated around the
step3 result or became worse, and STE test quality was much higher than actual
entropy-coded quality.

## Why Change The Compression Point

The current HiSplat MSH path compresses after the refinement U-Net:

```text
cost/depth/cascade features
    -> refine_unet
    -> refine_out
    -> MSH
    -> to_gaussians / to_disparity / lim_surf
```

This is a late compression point. After the codec, only the small Gaussian and
depth heads remain, so compression artifacts have little downstream capacity to
be corrected. In Phase 1, these heads were also trainable, which allowed them to
co-adapt to the STE quantization pattern used during training.

MVSplat's MSH implementation uses an earlier compression point:

```text
projected feature + RGB + coarse depth + confidence
    -> MSH
    -> refine_unet
    -> heads
```

That leaves the frozen generator tail after the codec. The codec is asked to
preserve a pre-refine feature bundle, while the original refinement network and
heads remain responsible for producing Gaussians.

## New Target

Add a new compression target:

```text
model.encoder.compression.target=pre_refine
```

HiSplat's three-stage pyramid remains unchanged. The only change is where the
codec is inserted inside each stage.

New per-stage flow:

```text
Stage i
cost/depth/cascade features
    -> pre-refine bundle
    -> MSH
    -> reconstructed pre-refine bundle
    -> refine_unet
    -> to_gaussians / to_disparity / lim_surf
    -> stage i Gaussians
```

The recommended compressed bundle follows the MVSplat pattern and excludes
`proj_feature`, because `proj_feature` can be recomputed from the decoded
`proj_feat_in_fullres`.

```text
stage0: extra_img + proj_feat_in_fullres + coarse_disps + pdf_max
stage1: extra_img + proj_feat_in_fullres + coarse_disps + pdf_max + pre_stage_residual
stage2: extra_img + proj_feat_in_fullres + coarse_disps + pdf_max + pre_stage_residual
```

Expected channels:

```text
stage0: 3 + 128 + 1 + 1 = 133
stage1: 3 + 64  + 1 + 1 + 3 = 72
stage2: 3 + 32  + 1 + 1 + 3 = 40
```

After decoding, `proj_feature` is recomputed:

```text
proj_feature = self.proj_feature(proj_feat_in_fullres_hat)
```

## Training Policy

For this validation experiment, Phase 1 should train the codec only:

```text
trainable: MSH codec
frozen: backbone, cost/refine U-Net, heads, decoder
```

This matches the question we want to answer:

```text
Can the existing HiSplat generator tolerate compressed pre-refine features?
```

It avoids turning the experiment into a new generator whose heads specialize to
the training-time STE noise pattern.

## Stabilized Experiment Settings

The first finite pre-refine runs still collapsed: NaNs were avoided, but the
codec learned a degenerate high-bpp / low-PSNR mapping. The pre-refine tensor is
heterogeneous, so render loss alone is not enough to keep the random codec on an
identity-like path.

The current recommended run uses three stabilizers:

```text
train-time bypass warmup:
    x_used = x_orig + beta_t * (x_hat - x_orig)
    beta_t ramps from 0 to 1

feature identity loss:
    per-channel relative L1 between x_hat and stopgrad(x_orig)

low fixed lambda:
    keep rate pressure weak until the codec learns a usable identity mapping
```

Recommended settings:

```text
PHASE=1
TARGET=pre_refine
N=192
M=320
n_downsample_per_stage=[2,2,2]
LAMBDA=1e-6
optimizer.lr=2e-5
trainer.gradient_clip_val=0.05
feat_loss_alpha=1.0
feat_loss_start_step=0
feat_loss_warmup_steps=10000
pre_refine_bypass_start_step=0
pre_refine_bypass_warmup_steps=15000
PRETRAINED_CKPT=checkpoints/hisplat_re10k.ckpt
```

Convenience submit script:

```bash
bash slurm/train_pre_refine_msh.sh
```

Feature identity loss should now stay enabled. The pre-refine bundle mixes RGB,
depth, confidence, residuals, and learned features, so the loss is normalized
per channel instead of using one stage-wide scale. The old render-only setting
is now treated as a negative result; disable the feature loss only when the goal
is specifically to reproduce that failure mode.

## NaN Stability Guard

Early pre-refine runs showed that the random MSH codec can drive the entropy
likelihoods to zero or non-finite values. Once `-log2(likelihood)` becomes NaN,
one optimizer step can poison the codec weights and all later checkpoints become
invalid.

The codec now clamps likelihoods before bit estimation, converts hyperprior
scales to bounded positive values, sanitizes non-finite latent activations, and
skips a training update if the final loss is still non-finite. This does not make
an unstable run good, but it prevents one bad batch from corrupting the rest of
the experiment.

## Evaluation Requirements

Keep both test modes:

```text
MSH_COMPRESS_MODE=training  # STE path
MSH_COMPRESS_MODE=actual    # entropy-coded path
```

Before trusting any new result, compare STE and actual on the same checkpoint.
The feature reconstruction from `forward()` and `compress()->decompress()` should
be nearly identical. If they differ, fix the actual coding path before running
longer experiments.

## Expected Outcome

If this works, it supports the key project hypothesis:

```text
HiSplat's pre-refine representation is compressible and can serve as a base
layer for a scalable feed-forward 3DGS codec.
```

If it fails, the next likely direction is an enhancement/residual layer rather
than more low-lambda training of the same single MSH codec.
