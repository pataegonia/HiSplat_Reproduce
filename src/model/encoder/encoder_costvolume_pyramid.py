from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ...global_cfg import get_cfg
from ..types import Gaussians
from .backbone import BackbonePyramid
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .costvolume.depth_predictor_multiview import DepthPredictorMultiViewPyramid
from .encoder import Encoder
from .compression import HiSplatMSHCodec
from .visualization.encoder_visualizer_costvolume_cfg import (
    EncoderVisualizerCostVolumeCfg,
)


@dataclass
class CompressionCfg:
    enabled: bool
    N: int  # latent channels
    M: int  # hyperprior expansion channels
    n_downsample_per_stage: List[int]  # downsampling depth per stage
    lmbda: float  # rate-distortion tradeoff weight
    target: str = "refine_out"  # "pre_refine", "refine_out", or "raw_gaussians"
    compress_sh_degree: int = 4  # only used if target=="raw_gaussians"
    # Optional checkpoint used only to initialize the MSH codec weights after
    # the main generator checkpoint has been loaded. This lets us keep a clean
    # gen-only HiSplat generator while transplanting a pretrained codec.
    codec_init_ckpt: str | None = None
    # Per-stage rate weights. Final rate loss = lmbda * sum_i (w_i * bpp_i).
    # Larger w_i → stronger penalty on stage_i bits → fewer bits there.
    # Test PSNR is measured on stage 2 only, so [4, 2, 1] shifts bits to
    # the final stage. None means uniform [1, 1, 1] (legacy behavior).
    lambda_stage_weights: List[float | int] | None = None
    # Per-stage IRB depth (zero-init, identity at init). None falls back to
    # uniform [2,2,2] pre and [4,4,4] post (matches phase2step3+ defaults).
    # To boost stage 2 capacity specifically, e.g. [2,2,4] / [4,4,8].
    n_pre_blocks_per_stage: List[int] | None = None
    n_post_blocks_per_stage: List[int] | None = None
    # Use CompressAI's own dequantization rule in the training forward pass
    # so STE reconstruction is closer to real compress/decompress.
    actual_consistent_quant: bool = True
    # Optional auxiliary loss for EntropyBottleneck quantile/CDF calibration.
    # Kept off by default to preserve legacy training unless explicitly enabled.
    entropy_aux_weight: float = 0.0
    # Optional identity loss on the compressed tensor for feature targets.
    feat_loss_alpha: float = 0.0
    feat_loss_warmup_steps: int = 5000
    feat_loss_start_step: int = 0
    feat_loss_stage_weights: List[float | int] | None = None
    # Train-time warmup for target="pre_refine". During training only, the
    # frozen refine_unet sees x_orig + beta * (x_hat - x_orig), with beta
    # ramped to 1.0. Defaults preserve the legacy direct-codec behavior.
    pre_refine_bypass_start_step: int = 0
    pre_refine_bypass_warmup_steps: int = 0


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfgPyramid:
    name: str
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    compression: CompressionCfg | None = None


class EncoderCostVolumePyramid(Encoder):
    backbone: BackbonePyramid
    depth_predictor: DepthPredictorMultiViewPyramid
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        self.backbone = BackbonePyramid(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
        )

        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == "train":
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = {}
                for k, v in unimatch_pretrained_model.items():
                    if k in self.backbone.state_dict():
                        updated_state_dict[k] = v
                    else:
                        possible_k = "backbone.encoder." + ".".join(k.split(".")[1:])
                        if possible_k in self.backbone.state_dict():
                            updated_state_dict["backbone.encoder." + possible_k] = v
                updated_state_dict = OrderedDict(updated_state_dict)
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                self.backbone.load_state_dict(updated_state_dict, strict=False)
        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        gaussian_raw_channels = cfg.num_surfaces * (self.gaussian_adapter.d_in + 2)
        self.depth_predictor = DepthPredictorMultiViewPyramid(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=gaussian_raw_channels,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )

        # MSH compression codec (optional)
        # Compression targets (selectable via compression.target):
        #   - "pre_refine": compresses the feature bundle before refine_unet,
        #     matching the MVSplat-MSH placement more closely. The frozen
        #     refine_unet/heads remain after the codec.
        #   - "refine_out" (default, new): compresses UNet features before the
        #     Gaussian and depth heads so heads can compensate for quant noise.
        #   - "raw_gaussians" (old): compresses the final to_gaussians output.
        comp_cfg = getattr(cfg, "compression", None)
        if comp_cfg is not None and comp_cfg.enabled:
            target = getattr(comp_cfg, "target", "refine_out")
            if target == "pre_refine":
                # Bundle:
                # stage0: image(3) + proj_feat + coarse_disp(1) + pdf_max(1)
                # stage1/2 add pre_stage_residual(3).
                proj_channels = [cfg.d_feature // (2 ** i) for i in range(3)]
                in_channels_per_stage = [
                    proj_channels[0] + 5,
                    proj_channels[1] + 8,
                    proj_channels[2] + 8,
                ]
            elif target == "refine_out":
                base_feat_dim = cfg.depth_unet_feat_dim
                in_channels_per_stage = [
                    base_feat_dim * (2 ** (2 - i)) for i in range(3)
                ]  # e.g. [128, 64, 32] for base=32
            elif target == "raw_gaussians":
                # All three stages share the same raw_gaussians channel count
                in_channels_per_stage = [gaussian_raw_channels] * 3
            else:
                raise ValueError(f"Unknown compression target: {target}")
            self.msh_codec = HiSplatMSHCodec(
                in_channels_per_stage=in_channels_per_stage,
                N=comp_cfg.N,
                M=comp_cfg.M,
                n_downsample_per_stage=comp_cfg.n_downsample_per_stage,
                n_pre_blocks_per_stage=getattr(
                    comp_cfg, "n_pre_blocks_per_stage", None
                ),
                n_post_blocks_per_stage=getattr(
                    comp_cfg, "n_post_blocks_per_stage", None
                ),
                actual_consistent_quant=getattr(
                    comp_cfg, "actual_consistent_quant", True
                ),
            )
            self.compression_lmbda = comp_cfg.lmbda
            self.compression_entropy_aux_weight = getattr(
                comp_cfg, "entropy_aux_weight", 0.0
            )
            self.compression_lambda_stage_weights = [
                float(w)
                for w in (getattr(comp_cfg, "lambda_stage_weights", None) or [1.0, 1.0, 1.0])
            ]
            self.compression_feat_loss_alpha = getattr(comp_cfg, "feat_loss_alpha", 0.0)
            self.compression_feat_loss_warmup_steps = getattr(
                comp_cfg, "feat_loss_warmup_steps", 5000
            )
            self.compression_feat_loss_start_step = getattr(comp_cfg, "feat_loss_start_step", 0)
            self.compression_feat_loss_stage_weights = [
                float(w)
                for w in (getattr(comp_cfg, "feat_loss_stage_weights", None) or [1.0, 1.0, 1.0])
            ]
            self.compression_pre_refine_bypass_start_step = getattr(
                comp_cfg, "pre_refine_bypass_start_step", 0
            )
            self.compression_pre_refine_bypass_warmup_steps = getattr(
                comp_cfg, "pre_refine_bypass_warmup_steps", 0
            )
            self.msh_target = target
            self.msh_compress_mode = "training"
        else:
            self.msh_codec = None
            self.compression_lmbda = 0.0
            self.compression_entropy_aux_weight = 0.0
            self.compression_lambda_stage_weights = [1.0, 1.0, 1.0]
            self.compression_feat_loss_alpha = 0.0
            self.compression_feat_loss_warmup_steps = 5000
            self.compression_feat_loss_start_step = 0
            self.compression_feat_loss_stage_weights = [1.0, 1.0, 1.0]
            self.compression_pre_refine_bypass_start_step = 0
            self.compression_pre_refine_bypass_warmup_steps = 0
            self.msh_target = "none"
            self.msh_compress_mode = "training"

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity. default is pdf
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        # Encode the context images.
        epipolar_kwargs = None
        features_list = self.backbone(
            context,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        in_feats = features_list
        extra_info = {}
        extra_info["images"] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        extra_info["global_step"] = global_step
        gpp = self.cfg.gaussians_per_pixel
        gaussian_dict, result_dict = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            encoder=self,
        )
        return gaussian_dict, result_dict

    def convert_to_gaussians(self, result_dict, context, features_list, global_step, visualization_dump):
        stage_num = len(result_dict)
        gaussian_dict = {k: {} for k in result_dict.keys()}
        device = context["image"].device
        for i in range(stage_num):
            raw_gaussians = result_dict[f"stage{i}"]["raw_gaussians"]
            densities = result_dict[f"stage{i}"]["densities"]
            depths = result_dict[f"stage{i}"]["depths"]
            h, w = features_list[0][i].shape[-2:]
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()  # [offset: 2, scales: 3, rotation: 4, sh: 3*25 ]
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size  # maximum change 0.5 pixel, normed xy ray
            gpp = self.cfg.gaussians_per_pixel
            gaussians, scales = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),  # 1 2 4096 1 2
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )

            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
                visualization_dump["scales"] = rearrange(gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
                visualization_dump["rotations"] = rearrange(
                    gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                )

            # Optionally apply a per-pixel opacity.
            opacity_multiplier = 1
            scales = rearrange(scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
            rotations = rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
            gaussian_dict[f"stage{i}"]["gaussians"] = Gaussians(
                rearrange(
                    gaussians.means.float(),
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances.float(),
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics.float(),
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    (opacity_multiplier * gaussians.opacities).float(),
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
            gaussian_dict[f"stage{i}"]["depths"] = depths
            gaussian_dict[f"stage{i}"]["scales"] = scales
            gaussian_dict[f"stage{i}"]["rotations"] = rotations
        return gaussian_dict

    def convert_to_gaussians_single_stge(
        self,
        raw_gaussians,
        densities,
        depths,
        image_size,
        extrinsics,
        intrinsics,
        global_step,
        opacity_multiplier=1.0,
        stage_id=0,
    ):
        device = raw_gaussians.device
        h, w = image_size[0], image_size[1]
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()  # [offset: 2, scales: 3, rotation: 4, sh: 3*25 ]
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size  # maximum change 0.5 pixel, normed xy ray
        gpp = self.cfg.gaussians_per_pixel
        gaussians, scales = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),  # 1 2 4096 1 2
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            stage_id=stage_id,
        )

        # Optionally apply a per-pixel opacity.
        scales = rearrange(scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
        rotations = rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
        return_gaussians = Gaussians(
            rearrange(
                gaussians.means.float(),
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances.float(),
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics.float(),
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                (opacity_multiplier * gaussians.opacities).float(),
                "b v r srf spp -> b (v r srf spp)",
            ),
        )
        return return_gaussians, scales, rotations

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
