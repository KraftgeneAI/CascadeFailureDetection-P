"""
Normalization Module
====================
Physics-based normalization for liquid-pipeline SCADA data.

Values are converted to per-unit against the system bases declared in
`Settings.PipelineSystem`, so the network sees dimensionless quantities of
comparable magnitude regardless of the units the simulator emitted.

  pressure     psi     -> per-unit of REFERENCE_PRESSURE_PSI (1000 psi)
  flow         bbl/hr  -> per-unit of BASE_FLOW_BBL_HR       (1000 bbl/hr)
  temperature  degC    -> per-unit of BASE_TEMPERATURE_C     (100 degC)

This module previously held power-system helpers (MW per-unit on a base MVA,
and Hz per-unit on a base frequency), inherited from the power-grid codebase
this project was ported from. Neither unit occurs anywhere in a liquid
pipeline, so they have been replaced with the equivalents above.
"""

import numpy as np
import torch
from typing import Union

from pipeline_cascade_prediction.data.generator.config import Settings

Numeric = Union[torch.Tensor, np.ndarray, float]

#: Reference temperature (degC) bringing temperatures into a ~1.0 range.
#: Matches the /100.0 scaling PhysicsInformedLoss already applies to temp targets.
BASE_TEMPERATURE_C = 100.0


def normalize_pressure(
    pressure_psi: Numeric,
    base_pressure_psi: float = Settings.PipelineSystem.REFERENCE_PRESSURE_PSI,
) -> Numeric:
    """Normalize pressure to per-unit.

    Args:
        pressure_psi: Pressure values in psi
        base_pressure_psi: Base pressure (default: the pipeline reference pressure)

    Returns:
        Pressure in per-unit
    """
    return pressure_psi / base_pressure_psi


def denormalize_pressure(
    pressure_pu: Numeric,
    base_pressure_psi: float = Settings.PipelineSystem.REFERENCE_PRESSURE_PSI,
) -> Numeric:
    """Convert per-unit pressure back to psi."""
    return pressure_pu * base_pressure_psi


def normalize_flow(
    flow_bph: Numeric,
    base_flow_bph: float = Settings.PipelineSystem.BASE_FLOW_BBL_HR,
) -> Numeric:
    """Normalize volumetric flow to per-unit.

    Args:
        flow_bph: Flow rate in bbl/hr
        base_flow_bph: Base flow rate (default: the pipeline base flow)

    Returns:
        Flow in per-unit
    """
    return flow_bph / base_flow_bph


def denormalize_flow(
    flow_pu: Numeric,
    base_flow_bph: float = Settings.PipelineSystem.BASE_FLOW_BBL_HR,
) -> Numeric:
    """Convert per-unit flow back to bbl/hr."""
    return flow_pu * base_flow_bph


def normalize_temperature(
    temperature_c: Numeric,
    base_temperature_c: float = BASE_TEMPERATURE_C,
) -> Numeric:
    """Normalize temperature to per-unit.

    Args:
        temperature_c: Temperature in degrees Celsius
        base_temperature_c: Base temperature (default: 100 degC)

    Returns:
        Temperature in per-unit
    """
    return temperature_c / base_temperature_c


def denormalize_temperature(
    temperature_pu: Numeric,
    base_temperature_c: float = BASE_TEMPERATURE_C,
) -> Numeric:
    """Convert per-unit temperature back to degrees Celsius."""
    return temperature_pu * base_temperature_c
