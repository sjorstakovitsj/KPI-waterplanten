from __future__ import annotations

"""Compatibiliteitslaag voor bestaande imports naar `ecology_service.py`.

Deze module leest nu expliciet uit `config` en `core`.
"""

from waterplanten_app.config.settings import (
    COORD_CACHE_PARQUET,
    FILE_PATH,
    FINAL_PARQUET,
    LOOKUP_PARQUET,
    MEAS_PARQUET,
    PIPELINE_VERSION,
    SPECIES_LOOKUP_PATH,
)
from waterplanten_app.config.mappings import (
    EXCLUDED_SPECIES_CODES,
    GROWTH_FORM_MAPPING,
    KRW_WATERTYPE_BY_WATERLICHAAM,
    PROJECT_MAPPING,
    RWS_GROEIVORM_CODES,
    WATERBODY_MAPPING,
)
from waterplanten_app.core.data_access import *
from waterplanten_app.core.diagnostics import calculate_kpi, categorize_slope_trend, interpret_soil_state
from waterplanten_app.core.taxonomy import *

__all__ = [
    'COORD_CACHE_PARQUET',
    'FILE_PATH',
    'FINAL_PARQUET',
    'LOOKUP_PARQUET',
    'MEAS_PARQUET',
    'PIPELINE_VERSION',
    'SPECIES_LOOKUP_PATH',
    'EXCLUDED_SPECIES_CODES',
    'GROWTH_FORM_MAPPING',
    'KRW_WATERTYPE_BY_WATERLICHAAM',
    'PROJECT_MAPPING',
    'RWS_GROEIVORM_CODES',
    'WATERBODY_MAPPING',
    'calculate_kpi',
    'categorize_slope_trend',
    'interpret_soil_state',
]
