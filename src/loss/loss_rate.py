"""Rate loss for MSH compression: computes bpp from estimated bits."""

import torch
from torch import Tensor


def compute_rate_loss(
    estimated_bits: dict[str, Tensor],
    num_pixels: int,
) -> Tensor:
    """Compute bits-per-pixel from estimated bits.

    Args:
        estimated_bits: {"y": y_bits, "z": z_bits} from MSH codec.
        num_pixels: Total number of target view pixels (b * v_target * h * w).

    Returns:
        bpp: Scalar tensor.
    """
    total_bits = sum(estimated_bits.values())
    return total_bits / num_pixels
