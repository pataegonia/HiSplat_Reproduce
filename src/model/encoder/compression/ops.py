# Adapted from CompressAI (InterDigital Communications, Inc)
# https://github.com/InterDigitalInc/CompressAI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def quantize_ste(x: Tensor) -> Tensor:
    """Rounding with straight-through estimator gradient."""
    return (torch.round(x) - x).detach() + x


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding."""
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)
    return pad, unpad


class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x >= bound) | (grad_output < 0)
        return pass_through_if * grad_output, None


class LowerBound(nn.Module):
    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class NonNegativeParametrizer(nn.Module):
    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


class GDN(nn.Module):
    """Generalized Divisive Normalization layer."""

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        return x * norm
