from __future__ import annotations
from typing import Any
import pandas as pd
from waterplanten_app.config.settings import (
    SOIL_BOOTSTRAP_DEFAULT_MODE,
    SOIL_BOOTSTRAP_ITERATIONS_EXTENDED,
    SOIL_BOOTSTRAP_ITERATIONS_FAST,
    SOIL_BOOTSTRAP_ITERATIONS_STANDARD,
)
from waterplanten_app.config.soil_diagnosis_config import (
    ANION_LABELS,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RANDOM_SEED,
    MIN_ANNUAL_PROFILE_COVERAGE,
    MIN_FORMAL_TREND_YEARS,
    MIN_LOCATIONS_PER_YEAR,
    MIN_RECURRING_LOCATION_SHARE,
    NUMERIC_PARAMETER_SPECS,
    TREND_STABLE_SLOPE_TOLERANCE,
    WEIGHTING_COVER,
    WEIGHTING_OPTIONS,
    WEIGHTING_PRESENCE,
)
from waterplanten_app.core.soil_diagnosis_calculations import (
    aggregate_annual_waterbody_diagnoses,
    aggregate_waterbody_diagnoses,
    assess_trend_quality,
    bootstrap_location_trend,
    calculate_all_diagnoses,
    calculate_descriptive_trend,
    calculate_location_diagnosis_records,
    calculate_theil_sen_slope,
    classify_formal_trend,
    compare_trend_scenarios,
    interpret_diagnosis_value,
    prepare_annual_boxplot_data,
)
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.domain.soil_diagnosis import (
    SoilDiagnosisResult,
    SoilDiagnosisTrendResult,
    TrendEstimate,
    TrendQualityAssessment,
    TrendScenarioComparison,
    WaterbodyDiagnosisResult,
)
from waterplanten_app.repositories.soil_diagnosis_repository import (
    get_available_locations as repository_get_available_locations,
    get_species_profile_matches,
    get_waterbody_species_profile_matches,
    get_waterbody_trend_species_profile_matches,
)

TREND_ALL_LOCATIONS = "all"
TREND_RECURRING_LOCATIONS = "recurring"
BOOTSTRAP_ITERATIONS_BY_MODE = {
    "fast": SOIL_BOOTSTRAP_ITERATIONS_FAST,
    "standard": SOIL_BOOTSTRAP_ITERATIONS_STANDARD,
    "extended": SOIL_BOOTSTRAP_ITERATIONS_EXTENDED,
}


def get_available_locations(filters: DashboardFilters) -> list[str]:
    return repository_get_available_locations(filters)


def get_available_trend_parameters(filters: DashboardFilters | None = None) -> list[tuple[str, str]]:
    """Statische lijst: geen DuckDB-query of diagnoseberekening nodig."""
    del filters
    numeric = [(spec.key, spec.label) for spec in NUMERIC_PARAMETER_SPECS]
    anions = list(ANION_LABELS.items())
    return numeric + anions


def _validate_weighting(weighting: str) -> None:
    if weighting not in WEIGHTING_OPTIONS:
        raise ValueError(f"Onbekende berekeningswijze: {weighting}")


def _split_matches(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or "soortnaam" not in frame:
        return frame.iloc[0:0].copy(), frame.copy()
    matched = frame.loc[frame["soortnaam"].notna()].reset_index(drop=True)
    columns = [c for c in ("locatie_id", "soort", "bedekking_pct", "aantal_records") if c in frame]
    unmatched = frame.loc[frame["soortnaam"].isna(), columns].reset_index(drop=True)
    return matched, unmatched


def get_soil_diagnosis(filters: DashboardFilters, location_id: str, weighting: str = WEIGHTING_PRESENCE) -> SoilDiagnosisResult:
    _validate_weighting(weighting)
    frame = get_species_profile_matches(filters, location_id)
    matched, unmatched = _split_matches(frame)
    observed = int(frame["soort"].nunique()) if "soort" in frame else len(frame)
    matched_count = int(matched["soort"].nunique()) if "soort" in matched else len(matched)
    diagnoses = calculate_all_diagnoses(matched, observed, weighting) if not matched.empty else tuple()
    coverage = matched_count / observed if observed else 0.0
    return SoilDiagnosisResult(
        filters=filters,
        location_id=str(location_id),
        weighting=weighting,
        summary=(f"Voor {matched_count} van de {observed} soorten ({coverage:.0%}) is een profiel beschikbaar."),
        parameter_diagnoses=tuple(diagnoses),
        species_evidence=matched,
        unmatched_species=unmatched,
        metadata={"observed_species_count": observed, "matched_species_count": matched_count, "unmatched_species_count": len(unmatched), "profile_coverage": coverage, "parameter_count": len(diagnoses)},
    )


def get_waterbody_diagnosis(filters: DashboardFilters, weighting: str = WEIGHTING_PRESENCE) -> WaterbodyDiagnosisResult:
    _validate_weighting(weighting)
    if len(filters.waterbodies) != 1:
        raise ValueError("Selecteer precies één waterlichaam.")
    matches = get_waterbody_species_profile_matches(filters)
    location_records, overview = calculate_location_diagnosis_records(matches, weighting)
    # Fase 1 gebruikt één geselecteerd jaar; verwijder jaar uit weergave-overzicht.
    location_count = int(matches["locatie_id"].nunique()) if "locatie_id" in matches else 0
    parameters = aggregate_waterbody_diagnoses(location_records, location_count)
    display_overview = overview.rename(columns={
        "location_id": "Locatie", "observed_species_count": "Aangetroffen soorten",
        "matched_species_count": "Soorten met profiel", "unmatched_species_count": "Soorten zonder profiel",
        "profile_coverage": "Profieldekking",
    }).drop(columns=["jaar"], errors="ignore")
    strong = [p for p in parameters if p.evidence_strength == "Sterke aanwijzing" and p.central_value is not None]
    selected = strong[:4] or [p for p in parameters if p.central_value is not None][:4]
    parameter_map = {item.parameter_key: item for item in parameters}
    parts = []
    for parameter in selected:
        companion = None
        if parameter.parameter_key == "organic_matter":
            companion_parameter = parameter_map.get("os_bodem")
            if companion_parameter is not None:
                companion = companion_parameter.category
        interpreted = interpret_diagnosis_value(
            parameter_key=parameter.parameter_key,
            value=parameter.central_value,
            category=parameter.category,
            companion_category=companion,
        )
        parts.append(f"{parameter.parameter_label.lower()}: {interpreted}")
    summary = (
        f"De soortensamenstelling op {location_count} locaties geeft aanwijzingen voor "
        + "; ".join(parts)
        + "."
    )
    return WaterbodyDiagnosisResult(
        filters=filters, waterbody=str(filters.waterbodies[0]), weighting=weighting,
        summary=summary, parameter_diagnoses=tuple(parameters),
        location_diagnoses=location_records, location_overview=display_overview,
        metadata={"location_count": location_count, "median_profile_coverage": float(overview["profile_coverage"].median()) if not overview.empty else 0.0, "parameter_count": len(parameters), "strong_parameter_count": len(strong)},
    )


def _recurring_locations(matches: pd.DataFrame, minimum_years: int) -> tuple[str, ...]:
    if matches.empty:
        return tuple()
    counts = matches.groupby("locatie_id")["jaar"].nunique()
    return tuple(sorted(counts[counts >= minimum_years].index.astype(str)))


def _prepare_trend_collections(matches: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Bereken locatiediagnoses slechts tweemaal: aanwezigheid en bedekking."""
    return {
        WEIGHTING_PRESENCE: calculate_location_diagnosis_records(matches, WEIGHTING_PRESENCE),
        WEIGHTING_COVER: calculate_location_diagnosis_records(matches, WEIGHTING_COVER),
    }


def _scenario_from_collection(
    collections: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    parameter_key: str,
    weighting: str,
    location_mode: str,
    recurring_locations: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    location_records = collections[weighting][0]
    if location_mode == TREND_RECURRING_LOCATIONS:
        location_records = location_records[location_records["location_id"].isin(recurring_locations)].copy()
    selected_locations = location_records[location_records["parameter_key"].eq(parameter_key)].copy()
    annual = aggregate_annual_waterbody_diagnoses(location_records)
    selected_annual = annual[annual["parameter_key"].eq(parameter_key)].copy() if not annual.empty else annual
    return selected_annual.sort_values("jaar") if not selected_annual.empty else selected_annual, selected_locations


def _formal_analysis(annual: pd.DataFrame, detail: pd.DataFrame, recurring_count: int, total_locations: int, iterations: int) -> tuple[TrendEstimate, TrendQualityAssessment]:
    quality = TrendQualityAssessment(**assess_trend_quality(
        annual, recurring_count, total_locations, MIN_FORMAL_TREND_YEARS,
        MIN_LOCATIONS_PER_YEAR, MIN_ANNUAL_PROFILE_COVERAGE, MIN_RECURRING_LOCATION_SHARE,
    ))
    slope = calculate_theil_sen_slope(annual)
    lower, upper, completed = bootstrap_location_trend(detail, iterations, BOOTSTRAP_CONFIDENCE_LEVEL, BOOTSTRAP_RANDOM_SEED)
    estimate = TrendEstimate(
        slope, lower, upper, BOOTSTRAP_CONFIDENCE_LEVEL,
        classify_formal_trend(slope, lower, upper, quality.status, TREND_STABLE_SLOPE_TOLERANCE),
        "Theil-Sen met bootstrap op locaties binnen jaren", quality.year_count, completed,
    )
    return estimate, quality



def _attach_boxplot_metadata(result: SoilDiagnosisTrendResult) -> SoilDiagnosisTrendResult:
    values,stats,available,missing=prepare_annual_boxplot_data(result.location_year_diagnoses)
    result.metadata={**dict(result.metadata),"boxplot_values":values,"boxplot_statistics":stats,"available_years":available,"missing_years":missing}
    return result


def _attach_interpretation(result):
    from waterplanten_app.core.soil_diagnosis_calculations import interpret_temporal_pattern,interpret_spatial_variability,interpret_sampling_context,interpret_uncertainty
    from waterplanten_app.domain.soil_diagnosis import TrendInterpretation
    t=interpret_temporal_pattern(result.annual_diagnoses); sp=interpret_spatial_variability(result.annual_diagnoses); sa=interpret_sampling_context(result.annual_diagnoses,len(result.recurring_locations),int(result.metadata.get("available_location_count",0))); u=interpret_uncertainty(result.formal_trend,result.sensitivity); direction=str(t.get("direction","onvoldoende")); classification=result.formal_trend.classification if result.formal_trend else "Beschrijvende duiding"
    meaning=f"De soortensamenstelling wijst op een {direction} patroon in {result.parameter_label.lower()}. Dit is een bio-indicatieve associatie en geen bewijs voor causaliteit."
    limits=["Geen directe water- of bodemmeting.","Verschillen in locaties, soorten en profieldekking kunnen jaaruitkomsten beïnvloeden."]
    if t.get("missing_years"): limits.append("De tijdreeks bevat meetonderbrekingen.")
    result.interpretation=TrendInterpretation(f"{classification}: {direction} tijdspatroon",str(t["text"]),str(sp["text"]),str(sa["text"]),str(u["text"]),meaning,tuple(limits)); return result

def get_waterbody_trends(
    filters: DashboardFilters,
    parameter_key: str,
    weighting: str = WEIGHTING_PRESENCE,
    location_mode: str = TREND_ALL_LOCATIONS,
    minimum_years: int = 3,
    include_formal_analysis: bool = True,
    bootstrap_mode: str = SOIL_BOOTSTRAP_DEFAULT_MODE,
) -> SoilDiagnosisTrendResult:
    """Eén repositoryquery, twee locatieberekeningen en hergebruik voor vier scenario's."""
    _validate_weighting(weighting)
    if len(filters.waterbodies) != 1:
        raise ValueError("Selecteer precies één waterlichaam.")
    if bootstrap_mode not in BOOTSTRAP_ITERATIONS_BY_MODE:
        raise ValueError(f"Onbekende bootstrapmodus: {bootstrap_mode}")

    matches = get_waterbody_trend_species_profile_matches(filters)
    recurring = _recurring_locations(matches, minimum_years)
    total_locations = int(matches["locatie_id"].nunique()) if not matches.empty else 0
    collections = _prepare_trend_collections(matches)
    annual, detail = _scenario_from_collection(collections, parameter_key, weighting, location_mode, recurring)
    label = str(annual["parameter_label"].iloc[0]) if not annual.empty else parameter_key
    descriptive = calculate_descriptive_trend(annual)

    result = SoilDiagnosisTrendResult(
        filters=filters, waterbody=str(filters.waterbodies[0]), parameter_key=parameter_key,
        parameter_label=label, location_mode=location_mode, minimum_years=minimum_years,
        annual_diagnoses=annual, location_year_diagnoses=detail, recurring_locations=recurring,
        summary=f"Beschrijvende ontwikkeling: {str(descriptive['direction']).lower()}.",
        metadata={**descriptive, "available_location_count": total_locations, "recurring_location_count": len(recurring)},
    )
    if not include_formal_analysis:
        return _attach_interpretation(_attach_boxplot_metadata(result))

    iterations = BOOTSTRAP_ITERATIONS_BY_MODE[bootstrap_mode]
    estimate, quality = _formal_analysis(annual, detail, len(recurring), total_locations, iterations)
    scenario_rows: list[dict[str, Any]] = []
    for name, mode, weight in (
        ("all_presence", TREND_ALL_LOCATIONS, WEIGHTING_PRESENCE),
        ("all_cover", TREND_ALL_LOCATIONS, WEIGHTING_COVER),
        ("recurring_presence", TREND_RECURRING_LOCATIONS, WEIGHTING_PRESENCE),
        ("recurring_cover", TREND_RECURRING_LOCATIONS, WEIGHTING_COVER),
    ):
        scenario_annual, scenario_detail = _scenario_from_collection(collections, parameter_key, weight, mode, recurring)
        scenario_estimate, scenario_quality = _formal_analysis(scenario_annual, scenario_detail, len(recurring), total_locations, iterations)
        scenario_rows.append({"scenario": name, "location_mode": mode, "weighting": weight, "slope_per_year": scenario_estimate.slope_per_year, "lower_interval": scenario_estimate.lower_interval, "upper_interval": scenario_estimate.upper_interval, "classification": scenario_estimate.classification, "quality_status": scenario_quality.status, "year_count": scenario_quality.year_count, "minimum_locations_per_year": scenario_quality.minimum_locations_per_year})
    scenarios = pd.DataFrame(scenario_rows)
    comparison = compare_trend_scenarios(scenarios)
    sensitivity = TrendScenarioComparison(scenarios, bool(comparison["conclusion_stable"]), str(comparison["explanation"]))
    result.formal_trend, result.quality, result.sensitivity = estimate, quality, sensitivity
    result.summary = f"{estimate.classification}. Robuuste verandering: {estimate.slope_per_year:.4g} per jaar. Datakwaliteit: {quality.status.lower()}. {sensitivity.explanation}" if estimate.slope_per_year is not None else f"Onvoldoende gegevens voor een formele trenduitspraak. Datakwaliteit: {quality.status.lower()}."
    result.metadata = {**dict(result.metadata), "formal_classification": estimate.classification, "quality_status": quality.status, "quality_score": quality.score, "sensitivity_stable": sensitivity.conclusion_stable, "bootstrap_mode": bootstrap_mode, "bootstrap_iterations": iterations}
    return _attach_interpretation(_attach_boxplot_metadata(result))
