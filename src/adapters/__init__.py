"""Adapters for data ingestion and integration."""

from .attribution_adapter import (
    AttributionAdapter,
    AttributionEvent,
    convert_to_attribution_format,
)

__all__ = [
    "AttributionAdapter",
    "AttributionEvent",
    "convert_to_attribution_format",
]
