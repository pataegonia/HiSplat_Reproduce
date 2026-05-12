"""Adaptive Mean-Scale Hyperprior codec for HiSplat multi-stage compression.

Each stage can use a different number of downsampling layers to handle
varying spatial resolutions in HiSplat's pyramid architecture:
  - Stage 0 (~32x44):  n_downsample=2  (4x spatial reduction)
  - Stage 1 (~64x88):  n_downsample=2  (4x spatial reduction)
  - Stage 2 (~128x176): n_downsample=3  (8x spatial reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers  # noqa: F401

from .ops import GDN, quantize_ste


class IdentityResBlock(nn.Module):
    """ResBlock that initializes as identity (zero-init last conv).

    Adding a stack of these after g_s boosts synthesis capacity at high
    bitrate without changing any bits. On load from an old checkpoint
    (before this block existed), missing params default to this init,
    so forward output is unchanged; subsequent fine-tuning can only improve.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


def _get_default_scale_table(min_val: float = 0.11, max_val: float = 256.0,
                             levels: int = 64) -> Tensor:
    """CompressAI's default logarithmic scale table for GaussianConditional."""
    return torch.exp(torch.linspace(
        torch.log(torch.tensor(min_val)),
        torch.log(torch.tensor(max_val)),
        levels,
    ))


def calc_bits(likelihood: Tensor, eps: float = 1e-9) -> Tensor:
    """Numerically safe entropy estimate.

    A single zero or NaN likelihood poisons the rate loss and then the codec
    weights. Clamp only the probability mass used for logging/training the
    entropy model; the actual tensors still flow through the codec normally.
    """
    likelihood = torch.nan_to_num(
        likelihood,
        nan=eps,
        posinf=1.0,
        neginf=eps,
    )
    likelihood = likelihood.clamp(min=eps, max=1.0)
    return -torch.log2(likelihood).sum()


class AdaptiveMSH(nn.Module):
    """Mean-Scale Hyperprior with configurable encoder/decoder depth.

    Args:
        in_channels: Number of input feature channels (e.g. 84 for HiSplat raw_gaussians).
        N: Latent bottleneck channels.
        M: Hyperprior expansion channels.
        n_downsample: Number of stride-2 conv layers in g_a / g_s.
    """

    def __init__(self, in_channels: int, N: int = 128, M: int = 192,
                 n_downsample: int = 2,
                 n_pre_blocks: int = 2, n_post_blocks: int = 4,
                 actual_consistent_quant: bool = True):
        super().__init__()
        self.N = N
        self.M = M
        self.n_downsample = n_downsample
        self.actual_consistent_quant = actual_consistent_quant

        # --- Encoder g_a: n_downsample layers of stride-2 conv ---
        g_a_layers = []
        for i in range(n_downsample):
            ch_in = in_channels if i == 0 else N
            ch_out = M if i == n_downsample - 1 else N
            g_a_layers.append(nn.Conv2d(ch_in, ch_out, 5, stride=2, padding=2))
            if i < n_downsample - 1:
                g_a_layers.append(GDN(ch_out))
        self.g_a = nn.Sequential(*g_a_layers)

        # --- Decoder g_s: n_downsample layers of stride-2 deconv ---
        g_s_layers = []
        for i in range(n_downsample):
            ch_in = M if i == 0 else N
            ch_out = in_channels if i == n_downsample - 1 else N
            g_s_layers.append(
                nn.ConvTranspose2d(ch_in, ch_out, 5, stride=2, padding=2,
                                   output_padding=1)
            )
            if i < n_downsample - 1:
                g_s_layers.append(GDN(ch_out, inverse=True))
        self.g_s = nn.Sequential(*g_s_layers)

        # Identity-initialized refinement stacks. pre_g_s acts on the latent
        # (M channels) before g_s; post acts on the reconstructed feature
        # (in_channels) after g_s. Both start as no-ops so existing
        # checkpoints load with unchanged outputs; subsequent fine-tuning can
        # only reduce reconstruction error. When n_pre_blocks/n_post_blocks
        # exceeds the count saved in an old ckpt, the extra blocks fall back
        # to identity init.
        self.pre_g_s = nn.Sequential(
            *[IdentityResBlock(M) for _ in range(n_pre_blocks)]
        )
        self.post = nn.Sequential(
            *[IdentityResBlock(in_channels) for _ in range(n_post_blocks)]
        )

        # --- Hyper encoder h_a (fixed: 2 stride-2 layers) ---
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
        )

        # --- Hyper decoder h_s → produces means and scales ---
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, M, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(M, M * 3 // 2, 5, stride=2, padding=2,
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 3 // 2, M * 2, 3, padding=1),
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    @staticmethod
    def _sanitize_tensor(x: Tensor, bound: float = 1e4) -> Tensor:
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=bound,
            neginf=-bound,
        ).clamp(min=-bound, max=bound)

    @staticmethod
    def _sanitize_scales(scales: Tensor, min_scale: float = 0.11,
                         max_scale: float = 256.0) -> Tensor:
        scales = torch.nan_to_num(
            scales,
            nan=min_scale,
            posinf=max_scale,
            neginf=-max_scale,
        )
        scales = F.softplus(scales)
        return scales.clamp(min=min_scale, max=max_scale)

    def _pad_to_divisible(self, x):
        """Pad input so spatial dims are divisible by 2^(n_downsample + 2).
        g_a downsamples 2^n_downsample, h_a downsamples another 4x.
        """
        h, w = x.size(2), x.size(3)
        p = 2 ** (self.n_downsample + 2)
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, h, w

    def forward(self, x):
        x_padded, orig_h, orig_w = self._pad_to_divisible(x)

        # Encode
        y = self._sanitize_tensor(self.g_a(x_padded))
        z = self._sanitize_tensor(self.h_a(y))

        # Quantize z with STE. Using CompressAI's dequantized value for the
        # forward pass keeps training closer to real compress/decompress.
        offset = self.entropy_bottleneck._get_medians()
        if self.actual_consistent_quant:
            z_deq = self.entropy_bottleneck.quantize(
                z, "dequantize", offset
            )
            z_hat = self._sanitize_tensor(z + (z_deq - z).detach())
        else:
            z_hat = self._sanitize_tensor(quantize_ste(z - offset) + offset)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_bits = calc_bits(z_likelihoods)

        # Get conditional parameters for y
        scales, means = self.h_s(z_hat).chunk(2, 1)
        scales = self._sanitize_scales(scales)
        means = self._sanitize_tensor(means)

        # Quantize y with the same dequantization rule used by arithmetic
        # coding, while keeping a straight-through gradient for g_a.
        if self.actual_consistent_quant:
            y_deq = self.gaussian_conditional.quantize(
                y, "dequantize", means
            )
            y_hat = self._sanitize_tensor(y + (y_deq - y).detach())
        else:
            y_hat = self._sanitize_tensor(quantize_ste(y - means) + means)
        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        y_bits = calc_bits(y_likelihoods)

        # Decode
        y_hat = self.pre_g_s(y_hat)
        x_hat = self._sanitize_tensor(self.g_s(y_hat))
        x_hat = self._sanitize_tensor(self.post(x_hat))

        # Crop to original size
        x_hat = x_hat[:, :, :orig_h, :orig_w]

        return {
            "x_hat": x_hat,
            "estimated_bits": {"y": y_bits, "z": z_bits},
        }

    def compress(self, x):
        """Actual entropy coding for evaluation."""
        x_padded, orig_h, orig_w = self._pad_to_divisible(x)
        y = self._sanitize_tensor(self.g_a(x_padded))
        z = self._sanitize_tensor(self.h_a(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self._sanitize_tensor(z_hat)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat = self._sanitize_scales(scales_hat)
        means_hat = self._sanitize_tensor(means_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes,
                                                       means=means_hat)
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "orig_size": (orig_h, orig_w),
        }

    def decompress(self, strings, shape, orig_size):
        """Actual entropy decoding for evaluation."""
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self._sanitize_tensor(z_hat)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat = self._sanitize_scales(scales_hat)
        means_hat = self._sanitize_tensor(means_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        y_hat = self._sanitize_tensor(y_hat)
        y_hat = self.pre_g_s(y_hat)
        x_hat = self._sanitize_tensor(self.g_s(y_hat))
        x_hat = self._sanitize_tensor(self.post(x_hat))
        x_hat = x_hat[:, :, :orig_size[0], :orig_size[1]]
        return {"x_hat": x_hat}


class HiSplatMSHCodec(nn.Module):
    """Manages per-stage MSH codecs for HiSplat's 3-stage pyramid.

    Compresses the refine_out feature maps (UNet output) before the
    Gaussian head and depth head, so both heads can learn to compensate
    for quantization noise.

    Args:
        in_channels_per_stage: Feature channels per stage (e.g. [128, 64, 32]).
        N: Latent channels for MSH.
        M: Hyperprior expansion channels for MSH.
        n_downsample_per_stage: List of downsampling depths per stage.
    """

    def __init__(
        self,
        in_channels_per_stage: list[int],
        N: int = 128,
        M: int = 192,
        n_downsample_per_stage: list[int] | None = None,
        n_pre_blocks_per_stage: list[int] | None = None,
        n_post_blocks_per_stage: list[int] | None = None,
        actual_consistent_quant: bool = True,
    ):
        super().__init__()
        if n_downsample_per_stage is None:
            n_downsample_per_stage = [2, 2, 3]
        if n_pre_blocks_per_stage is None:
            n_pre_blocks_per_stage = [2, 2, 2]
        if n_post_blocks_per_stage is None:
            n_post_blocks_per_stage = [4, 4, 4]

        self.codecs = nn.ModuleList([
            AdaptiveMSH(
                in_ch, N, M, n_ds, n_pre, n_post,
                actual_consistent_quant=actual_consistent_quant,
            )
            for in_ch, n_ds, n_pre, n_post in zip(
                in_channels_per_stage,
                n_downsample_per_stage,
                n_pre_blocks_per_stage,
                n_post_blocks_per_stage,
            )
        ])

    def forward(self, x: Tensor, stage_id: int) -> dict:
        """Run MSH codec for the given stage.

        Args:
            x: [B*V, C, H, W] refine_out feature map from UNet.
            stage_id: 0, 1, or 2.
        """
        return self.codecs[stage_id](x)

    def compress(self, x: Tensor, stage_id: int) -> dict:
        return self.codecs[stage_id].compress(x)

    def decompress(self, strings, shape, orig_size, stage_id: int) -> dict:
        return self.codecs[stage_id].decompress(strings, shape, orig_size)

    def update(self, force: bool = False) -> bool:
        """Build CDF tables for every stage's EntropyBottleneck and
        GaussianConditional. Must be called once before real compress()/decompress().

        CompressAI trains with STE estimates and does not keep CDFs in sync — this
        propagates the medians / scale table into the quantized CDFs used for
        arithmetic coding.
        """
        scale_table = _get_default_scale_table()
        updated = False
        for codec in self.codecs:
            updated |= codec.entropy_bottleneck.update(force=force)
            updated |= codec.gaussian_conditional.update_scale_table(
                scale_table, force=force
            )
        return updated

    def entropy_aux_loss(self) -> Tensor:
        aux_losses = [
            codec.entropy_bottleneck.loss()
            for codec in self.codecs
        ]
        return sum(aux_losses)
