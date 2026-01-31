"""
Probabilistic Identity Resolution Engine

A multi-platform identity graph system for:
- Household Attribution (Netflix co-viewing problem)
- Cross-Device Stitching (same user, multiple devices)
- Behavioral Clustering (pattern-based identity inference)

Designed for marketing attribution with privacy-first principles.
"""

__version__ = "1.0.0"
__author__ = "Michael Robins"

from .models import (
    IdentityEntity,
    DeviceProfile,
    HouseholdProfile,
    StreamingEvent,
    Session,
)

from .core import (
    HouseholdInferenceEngine,
    CrossDeviceLinker,
    ProbabilisticIdentityResolver,
    IdentityGraph,
)

__all__ = [
    # Models
    "IdentityEntity",
    "DeviceProfile",
    "HouseholdProfile",
    "StreamingEvent",
    "Session",
    # Core Engines
    "HouseholdInferenceEngine",
    "CrossDeviceLinker",
    "ProbabilisticIdentityResolver",
    "IdentityGraph",
]
