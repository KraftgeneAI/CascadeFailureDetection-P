"""
Preprocessing Package
=====================
Data preprocessing utilities for cascade prediction.

This package contains:
- Normalization: Per-unit normalization of pressure, flow and temperature
- Truncation: Sliding window truncation for temporal sequences
- Edge Masking: Dynamic topology masking for line failures
"""

from .normalization import (
    normalize_pressure,
    denormalize_pressure,
    normalize_flow,
    denormalize_flow,
    normalize_temperature,
    denormalize_temperature,
    BASE_TEMPERATURE_C,
)

from .truncation import (
    calculate_truncation_window,
    apply_truncation,
)

from .edge_masking import (
    create_edge_mask_from_failures,
    create_edge_mask_sequence,
    to_tensor,
)

__all__ = [
    # Normalization
    'normalize_pressure',
    'denormalize_pressure',
    'normalize_flow',
    'denormalize_flow',
    'normalize_temperature',
    'denormalize_temperature',
    'BASE_TEMPERATURE_C',
    # Truncation
    'calculate_truncation_window',
    'apply_truncation',
    # Edge Masking
    'create_edge_mask_from_failures',
    'create_edge_mask_sequence',
    'to_tensor',
]
