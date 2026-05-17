from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class DashboardFilters:
    year: Optional[int] = None
    projects: Tuple[str, ...] = field(default_factory=tuple)
    waterbodies: Tuple[str, ...] = field(default_factory=tuple)
    analysis_level: Optional[str] = None
    coverage_type: Optional[str] = None
    layer_mode: Optional[str] = None


ANALYSIS_LEVEL_GROUPS_AGGREGATIONS = 'groepen & aggregaties'
ANALYSIS_LEVEL_INDIVIDUAL_SPECIES = 'individuele soorten'
ANALYSIS_LEVEL_OPTIONS: Tuple[str, ...] = (
    ANALYSIS_LEVEL_GROUPS_AGGREGATIONS,
    ANALYSIS_LEVEL_INDIVIDUAL_SPECIES,
)

COVERAGE_TYPE_TOTAL_BEDEKKING = 'totale bedekking'
COVERAGE_TYPE_GROEIVORMEN = 'Groeivormen'
COVERAGE_TYPE_TROFIENIVEAU = 'Trofieniveau'
COVERAGE_TYPE_KRW_SCORE = 'KRW score'
COVERAGE_TYPE_SOORTGROEPEN = 'soortgroepen'
COVERAGE_TYPE_OPTIONS: Tuple[str, ...] = (
    COVERAGE_TYPE_TOTAL_BEDEKKING,
    COVERAGE_TYPE_GROEIVORMEN,
    COVERAGE_TYPE_TROFIENIVEAU,
    COVERAGE_TYPE_KRW_SCORE,
    COVERAGE_TYPE_SOORTGROEPEN,
)

PIE_TYPES: Tuple[str, ...] = (
    COVERAGE_TYPE_GROEIVORMEN,
    COVERAGE_TYPE_TROFIENIVEAU,
    COVERAGE_TYPE_KRW_SCORE,
    COVERAGE_TYPE_SOORTGROEPEN,
)

LAYER_MODE_VEGETATIE = 'Vegetatie'
LAYER_MODE_DIEPTE = 'Diepte'
LAYER_MODE_DOORZICHT = 'Doorzicht'
LAYER_MODE_OPTIONS: Tuple[str, ...] = (
    LAYER_MODE_VEGETATIE,
    LAYER_MODE_DIEPTE,
    LAYER_MODE_DOORZICHT,
)

@dataclass(frozen=True)
class LegendItem:
    label: str
    color: str

@dataclass(frozen=True)
class PieMapInput:
    map_points: pd.DataFrame
    counts_by_location: Mapping[str, Any]
    label: str
    order: Tuple[str, ...] = field(default_factory=tuple)
    color_map: Mapping[str, str] = field(default_factory=dict)

@dataclass
class SpatialPageState:
    result: SpatialResult
    chem_points: pd.DataFrame
    selected_filters: DashboardFilters
    filtered_base_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    species_options: Tuple[str, ...] = field(default_factory=tuple)
    legend_title: Optional[str] = None
    legend_items: Tuple[LegendItem, ...] = field(default_factory=tuple)
    legend_note: Optional[str] = None
    pie_input: Optional[PieMapInput] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OverviewKpi:
    label: str
    value: Union[float, int, str]
    delta: Union[float, int, str]
    unit: Optional[str] = None
    precision: Optional[int] = None


@dataclass
class OverviewResult:
    filters: DashboardFilters
    kpis: Mapping[str, OverviewKpi]
    overview_table: pd.DataFrame
    krw_pie: pd.DataFrame
    trophic_pie: pd.DataFrame
    species_group_pie: pd.DataFrame
    n2000_pie: pd.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SpatialResult:
    filters: DashboardFilters
    map_points: pd.DataFrame
    location_table: pd.DataFrame
    layer_mode: str
    analysis_level: str
    coverage_type: str
    legend_key: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TimeseriesResult:
    filters: DashboardFilters
    ecology_yearly: pd.DataFrame
    chemistry_yearly: pd.DataFrame
    common_years: Tuple[int, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
