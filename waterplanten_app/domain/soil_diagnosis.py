"""Domeinmodellen voor locatie- en waterlichaamdiagnoses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from waterplanten_app.domain.contracts import DashboardFilters


@dataclass(frozen=True)
class ParameterDiagnosis:
    parameter_key: str
    parameter_label: str
    value: float | str | None
    category: str
    evidence_strength: str
    species_count: int
    observed_species_count: int
    profile_coverage: float
    total_indicator_strength: float
    spread: float | None = None
    minimum_indication: float | None = None
    maximum_indication: float | None = None
    unit: str = ""
    explanation: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "parameter_key": self.parameter_key,
            "Karakteristiek": self.parameter_label,
            "Waarde": self.value,
            "Categorie": self.category,
            "Bewijskracht": self.evidence_strength,
            "Aantal indicatieve soorten": self.species_count,
            "Aantal aangetroffen soorten": self.observed_species_count,
            "Profieldekking": self.profile_coverage,
            "Totale indicatorkracht": self.total_indicator_strength,
            "Spreiding": self.spread,
            "Minimum indicatie": self.minimum_indication,
            "Maximum indicatie": self.maximum_indication,
            "Eenheid": self.unit,
            "Toelichting": self.explanation,
        }


@dataclass
class SoilDiagnosisResult:
    filters: DashboardFilters
    location_id: str
    weighting: str
    summary: str
    parameter_diagnoses: tuple[ParameterDiagnosis, ...] = field(default_factory=tuple)
    species_evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched_species: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def diagnoses_frame(self) -> pd.DataFrame:
        return pd.DataFrame([item.as_record() for item in self.parameter_diagnoses])


@dataclass(frozen=True)
class WaterbodyParameterDiagnosis:
    parameter_key: str
    parameter_label: str
    central_value: float | str | None
    category: str
    lower_quartile: float | None
    upper_quartile: float | None
    minimum_value: float | None
    maximum_value: float | None
    location_count: int
    sufficient_location_count: int
    location_coverage: float
    evidence_strength: str
    unit: str = ""
    explanation: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "parameter_key": self.parameter_key,
            "Karakteristiek": self.parameter_label,
            "Diagnose waterlichaam": self.central_value,
            "Categorie": self.category,
            "P25": self.lower_quartile,
            "P75": self.upper_quartile,
            "Minimum": self.minimum_value,
            "Maximum": self.maximum_value,
            "Aantal locaties": self.location_count,
            "Locaties met voldoende informatie": self.sufficient_location_count,
            "Locatiedekking": self.location_coverage,
            "Bewijskracht": self.evidence_strength,
            "Eenheid": self.unit,
            "Toelichting": self.explanation,
        }


@dataclass
class WaterbodyDiagnosisResult:
    filters: DashboardFilters
    waterbody: str
    weighting: str
    summary: str
    parameter_diagnoses: tuple[WaterbodyParameterDiagnosis, ...] = field(default_factory=tuple)
    location_diagnoses: pd.DataFrame = field(default_factory=pd.DataFrame)
    location_overview: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def diagnoses_frame(self) -> pd.DataFrame:
        return pd.DataFrame([item.as_record() for item in self.parameter_diagnoses])


@dataclass(frozen=True)
class TrendEstimate:
    slope_per_year: float|None
    lower_interval: float|None
    upper_interval: float|None
    confidence_level: float
    classification: str
    method: str
    year_count: int
    bootstrap_iterations: int

@dataclass(frozen=True)
class TrendQualityAssessment:
    status: str
    score: int
    year_count: int
    minimum_locations_per_year: int
    median_locations_per_year: float
    minimum_profile_coverage: float
    recurring_location_share: float
    warnings: tuple[str,...]=()

@dataclass(frozen=True)
class TrendScenarioComparison:
    scenarios: pd.DataFrame=field(default_factory=pd.DataFrame)
    conclusion_stable: bool=False
    explanation: str=""


@dataclass(frozen=True)
class TrendInterpretation:
    """Gestructureerde duiding van tijd, ruimte, betrouwbaarheid en betekenis."""
    headline: str
    temporal_pattern: str
    spatial_pattern: str
    sampling_context: str
    uncertainty: str
    ecological_meaning: str
    limitations: tuple[str, ...] = ()

@dataclass
class SoilDiagnosisTrendResult:
    """Jaarlijkse waterlichaamdiagnoses voor een geselecteerde parameter."""

    filters: DashboardFilters
    waterbody: str
    parameter_key: str
    parameter_label: str
    location_mode: str
    minimum_years: int
    annual_diagnoses: pd.DataFrame = field(default_factory=pd.DataFrame)
    location_year_diagnoses: pd.DataFrame = field(default_factory=pd.DataFrame)
    recurring_locations: tuple[str, ...] = field(default_factory=tuple)
    formal_trend: TrendEstimate | None = None
    quality: TrendQualityAssessment | None = None
    sensitivity: TrendScenarioComparison | None = None
    interpretation: TrendInterpretation | None = None
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
