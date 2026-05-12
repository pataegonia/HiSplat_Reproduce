from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    # FCGS 압축 파이프라인용 export: 뷰어용 좌표 변환 없이 world space 그대로 저장.
    # (centering/scaling/rotation 제거 — eval 시 원본 카메라 extrinsics와 좌표계 일치 필요)

    # quaternion (w, x, y, z) 포맷으로 변환 (HiSplat 내부: x, y, z, w 순서)
    rotations_np = rotations.detach().cpu().numpy()
    x, y, z, w = rearrange(rotations_np, "g xyzw -> xyzw g")
    rotations_wxyz = np.stack((w, x, y, z), axis=-1)

    # Export SH coefficients up to degree 3 (16 coefficients per channel).
    # HiSplat may use sh_degree=4 (25 coeffs), but FCGS only supports degree 3 (16 coeffs).
    # We truncate to the first 16 coefficients: DC (1) + degree1~3 rest (15).
    SH_DEGREE_MAX = 3
    SH_COEFFS = (SH_DEGREE_MAX + 1) ** 2  # 16
    harmonics_truncated = harmonics[..., :SH_COEFFS]  # [N, 3, 16]
    f_dc = harmonics_truncated[..., 0]         # [N, 3] — DC coefficients (R, G, B)
    f_rest = harmonics_truncated[..., 1:]      # [N, 3, 15] — higher-order coefficients
    # Reshape to channel-major [N, 45]: R0..R14, G0..G14, B0..B14
    # This matches the GaussianModel PLY format expected by FCGS.
    num_rest = f_rest.shape[-2] * f_rest.shape[-1]  # 3 * 15 = 45
    f_rest_flat = f_rest.reshape(f_rest.shape[0], -1)  # [N, 45]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(num_rest)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest_flat.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations_wxyz,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
