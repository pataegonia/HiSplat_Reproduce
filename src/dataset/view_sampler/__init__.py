from pathlib import Path
from typing import Any

from ...global_cfg import get_cfg
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import (
    ViewSamplerBounded,
    ViewSamplerBoundedCfg,
    ViewSamplerBoundedDTU,
)
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "bounded_dtu": ViewSamplerBoundedDTU,
}

ViewSamplerCfg = ViewSamplerArbitraryCfg | ViewSamplerBoundedCfg | ViewSamplerEvaluationCfg | ViewSamplerAllCfg


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    # TODO: only a temporary fix, need to support cfg input
    if not stage == "train":
        dataset_root = str(get_cfg().dataset.roots[0]).lower()
        dataset_name = str(get_cfg().dataset.name).lower()
        if dataset_name == "dtu" or "dtu" in dataset_root:
            index_path = f'assets/evaluation_index_dtu_nctx{get_cfg().dataset.view_sampler.num_context_views}.json'
        elif dataset_name == "re10k" or "re10k" in dataset_root or "realestate10k" in dataset_root:
            index_path = "assets/evaluation_index_re10k.json"
        elif dataset_name == "acid" or "acid" in dataset_root:
            index_path = "assets/evaluation_index_acid.json"
        elif dataset_name == "replica" or "replica" in dataset_root:
            index_path = f"assets/evaluation_index_replica_nctx{get_cfg().dataset.view_sampler.num_context_views}.json"
        else:
            index_path = None
        if index_path is None:
            raise ValueError(f"Unsupported dataset root for evaluation sampler: {dataset_root}")
        cfg = ViewSamplerEvaluationCfg(
            name="evaluation",
            index_path=Path(index_path),
            num_context_views=get_cfg().dataset.view_sampler.num_context_views,
        )

    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
