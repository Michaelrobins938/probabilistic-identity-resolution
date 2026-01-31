"""Validation framework for identity resolution."""

from .synthetic_households import (
    generate_synthetic_household_data,
    SyntheticConfig,
    GroundTruth,
)
from .resolution_metrics import (
    ResolutionMetrics,
    evaluate_resolution,
    compare_to_baseline,
)

__all__ = [
    "generate_synthetic_household_data",
    "SyntheticConfig",
    "GroundTruth",
    "ResolutionMetrics",
    "evaluate_resolution",
    "compare_to_baseline",
]
