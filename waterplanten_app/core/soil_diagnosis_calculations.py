"""Pure berekeningen voor de soortgebaseerde bodemdiagnose."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from waterplanten_app.config.soil_diagnosis_config import (
    ANION_CLASS_BREAKS,
    ANION_LABELS,
    CATEGORY_PARAMETER_LABELS,
    DIAGNOSIS_VALUE_CLASS_BREAKS,
    DIAGNOSIS_USE_EXISTING_CATEGORY,
    MAX_WEIGHTED_CV_STRONG,
    MIN_PROFILE_COVERAGE_STRONG,
    MIN_SPECIES_FOR_INDICATION,
    MIN_SPECIES_FOR_STRONG_INDICATION,
    NUMERIC_PARAMETER_SPECS,
    WEIGHTING_COVER,
)
from waterplanten_app.domain.soil_diagnosis import ParameterDiagnosis


def _cover_weight(series: pd.Series) -> pd.Series:
    cover = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    return 1.0 + np.log1p(cover)


def _weights(frame: pd.DataFrame, indicator_column: str, weighting: str) -> pd.Series:
    indicator = pd.to_numeric(frame[indicator_column], errors="coerce")
    if weighting == WEIGHTING_COVER and "bedekking_pct" in frame.columns:
        return indicator * _cover_weight(frame["bedekking_pct"])
    return indicator


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask].astype(float), weights=weights[mask].astype(float)))


def _weighted_std(values: pd.Series, weights: pd.Series, mean: float) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any() or math.isnan(mean):
        return float("nan")
    variance = np.average((values[mask].astype(float) - mean) ** 2, weights=weights[mask])
    return float(math.sqrt(variance))


def determine_evidence_strength(
    species_count: int,
    profile_coverage: float,
    coefficient_of_variation: float | None = None,
) -> str:
    if species_count == 0:
        return "Geen informatie"
    if species_count < MIN_SPECIES_FOR_INDICATION:
        return "Onvoldoende informatie"
    if (
        species_count >= MIN_SPECIES_FOR_STRONG_INDICATION
        and profile_coverage >= MIN_PROFILE_COVERAGE_STRONG
        and (coefficient_of_variation is None or coefficient_of_variation <= MAX_WEIGHTED_CV_STRONG)
    ):
        return "Sterke aanwijzing"
    return "Matige aanwijzing"


def classify_anion_association(value: float) -> str:
    for lower, upper, label in ANION_CLASS_BREAKS:
        if lower <= value < upper:
            return label
    return "Geen geldige associatiescore"


def calculate_numeric_parameter(
    evidence: pd.DataFrame,
    observed_species_count: int,
    spec,
    weighting: str,
) -> ParameterDiagnosis:
    usable = evidence[
        evidence[spec.mean_column].notna() & evidence[spec.indicator_column].notna()
    ].copy()
    species_count = int(usable["soort"].nunique()) if "soort" in usable else len(usable)
    coverage = species_count / observed_species_count if observed_species_count else 0.0
    weights = _weights(usable, spec.indicator_column, weighting) if not usable.empty else pd.Series(dtype=float)
    values = pd.to_numeric(usable[spec.mean_column], errors="coerce") if not usable.empty else pd.Series(dtype=float)
    value = _weighted_mean(values, weights)
    spread = _weighted_std(values, weights, value)
    cv = abs(spread / value) if value not in (0, np.nan) and not math.isnan(value) else None
    strength = determine_evidence_strength(species_count, coverage, cv)

    minimum = None
    maximum = None
    if spec.minimum_column and not usable.empty and usable[spec.minimum_column].notna().any():
        minimum = float(pd.to_numeric(usable[spec.minimum_column], errors="coerce").min())
    if spec.maximum_column and not usable.empty and usable[spec.maximum_column].notna().any():
        maximum = float(pd.to_numeric(usable[spec.maximum_column], errors="coerce").max())

    value_out = None if math.isnan(value) else round(value, 3)
    spread_out = None if math.isnan(spread) else round(spread, 3)
    explanation = (
        f"Gewogen gemiddelde responswaarde op basis van {species_count} unieke soort(en). "
        "Dit is een bio-indicatie en geen direct gemeten waarde."
        if species_count else "Geen soorten met zowel een responswaarde als indicatorkracht."
    )
    return ParameterDiagnosis(
        parameter_key=spec.key,
        parameter_label=spec.label,
        value=value_out,
        category="Numerieke bio-indicatie" if species_count else "Geen informatie",
        evidence_strength=strength,
        species_count=species_count,
        observed_species_count=observed_species_count,
        profile_coverage=coverage,
        total_indicator_strength=float(weights.sum()) if not weights.empty else 0.0,
        spread=spread_out,
        minimum_indication=minimum,
        maximum_indication=maximum,
        unit=spec.unit,
        explanation=explanation,
    )


def calculate_categorical_parameter(
    evidence: pd.DataFrame,
    observed_species_count: int,
    category_column: str,
    indicator_column: str,
    weighting: str,
) -> ParameterDiagnosis:
    usable = evidence[
        evidence[category_column].notna() & evidence[indicator_column].notna()
    ].copy()
    species_count = int(usable["soort"].nunique()) if "soort" in usable else len(usable)
    coverage = species_count / observed_species_count if observed_species_count else 0.0
    if usable.empty:
        dominant, share, total = None, 0.0, 0.0
    else:
        usable["_weight"] = _weights(usable, indicator_column, weighting)
        scores = usable.groupby(category_column, dropna=True)["_weight"].sum().sort_values(ascending=False)
        total = float(scores.sum())
        dominant = str(scores.index[0]) if not scores.empty else None
        share = float(scores.iloc[0] / total) if total else 0.0
    strength = determine_evidence_strength(species_count, coverage)
    if strength == "Sterke aanwijzing" and share < 0.60:
        strength = "Matige aanwijzing"
    label = CATEGORY_PARAMETER_LABELS[category_column]
    explanation = (
        f"Dominante klasse heeft {share:.0%} van het gewogen indicatoraandeel. "
        "De uitkomst is een ecologische aanwijzing."
        if dominant else "Geen bruikbare categorische soortindicaties."
    )
    return ParameterDiagnosis(
        parameter_key=category_column,
        parameter_label=label,
        value=dominant,
        category=dominant or "Geen informatie",
        evidence_strength=strength,
        species_count=species_count,
        observed_species_count=observed_species_count,
        profile_coverage=coverage,
        total_indicator_strength=total,
        spread=None,
        explanation=explanation,
    )


def calculate_anion_parameters(
    evidence: pd.DataFrame,
    observed_species_count: int,
    weighting: str,
) -> tuple[ParameterDiagnosis, ...]:
    diagnoses: list[ParameterDiagnosis] = []
    for column, label in ANION_LABELS.items():
        usable = evidence[evidence[column].notna()].copy()
        species_count = int(usable["soort"].nunique()) if "soort" in usable else len(usable)
        coverage = species_count / observed_species_count if observed_species_count else 0.0
        values = pd.to_numeric(usable[column], errors="coerce")
        if weighting == WEIGHTING_COVER and "bedekking_pct" in usable:
            weights = _cover_weight(usable["bedekking_pct"])
            value = _weighted_mean(values, weights)
            total_weight = float(weights.sum())
        else:
            value = float(values.mean()) if not values.empty else float("nan")
            total_weight = float(species_count)
        value_out = None if math.isnan(value) else round(value, 3)
        category = classify_anion_association(value) if value_out is not None else "Geen informatie"
        diagnoses.append(ParameterDiagnosis(
            parameter_key=column,
            parameter_label=label,
            value=value_out,
            category=category,
            evidence_strength=determine_evidence_strength(species_count, coverage),
            species_count=species_count,
            observed_species_count=observed_species_count,
            profile_coverage=coverage,
            total_indicator_strength=total_weight,
            explanation=(
                "Gemiddelde soortassociatie op een schaal van -3 tot 3. "
                "Een positieve score duidt op goed gedijen bij hoge gehalten; "
                "een negatieve score op slecht gedijen."
            ),
        ))
    return tuple(diagnoses)


def calculate_all_diagnoses(
    evidence: pd.DataFrame,
    observed_species_count: int,
    weighting: str = "presence",
) -> tuple[ParameterDiagnosis, ...]:
    results: list[ParameterDiagnosis] = [
        calculate_numeric_parameter(evidence, observed_species_count, spec, weighting)
        for spec in NUMERIC_PARAMETER_SPECS
    ]
    results.extend(calculate_anion_parameters(evidence, observed_species_count, weighting))
    results.extend([
        calculate_categorical_parameter(evidence, observed_species_count, "bodemtype", "ind_bodem", weighting),
        calculate_categorical_parameter(evidence, observed_species_count, "os_bodem", "ind_os_bodem", weighting),
        calculate_categorical_parameter(evidence, observed_species_count, "redox", "ind_redox", weighting),
        calculate_categorical_parameter(evidence, observed_species_count, "carb_bodem", "ind_carb", weighting),
    ])
    return tuple(results)


def aggregate_waterbody_diagnoses(
    location_diagnoses: pd.DataFrame,
    total_location_count: int,
):
    """Aggregeer locatiediagnoses naar één diagnose per waterlichaamparameter.

    Iedere locatie telt eenmaal mee. Numerieke waarden worden samengevat met de
    mediaan en kwartielen. Categorische waarden worden samengevat met de meest
    voorkomende locatiecategorie.
    """
    from waterplanten_app.domain.soil_diagnosis import WaterbodyParameterDiagnosis

    if location_diagnoses.empty:
        return tuple()

    results = []
    for parameter_key, group in location_diagnoses.groupby("parameter_key", sort=False):
        label = str(group["parameter_label"].iloc[0])
        unit_values = group["unit"].dropna().astype(str) if "unit" in group else pd.Series(dtype=str)
        unit = unit_values.iloc[0] if not unit_values.empty else ""
        sufficient = group[
            group["evidence_strength"].isin(["Matige aanwijzing", "Sterke aanwijzing"])
        ].copy()
        base = sufficient if not sufficient.empty else group[group["value"].notna()].copy()
        location_count = int(base["location_id"].nunique()) if not base.empty else 0
        coverage = location_count / total_location_count if total_location_count else 0.0

        numeric_values = pd.to_numeric(base["value"], errors="coerce")
        numeric = numeric_values[numeric_values.notna()]
        if not numeric.empty:
            central = float(numeric.median())
            q25 = float(numeric.quantile(0.25))
            q75 = float(numeric.quantile(0.75))
            minimum = float(numeric.min())
            maximum = float(numeric.max())
            category = "Numerieke bio-indicatie"
            explanation = (
                f"Mediaan van {location_count} locatiediagnoses; de middelste 50% ligt "
                f"tussen {q25:.2f} en {q75:.2f}. Iedere locatie telt eenmaal mee."
            )
        else:
            categories = base["category"].dropna().astype(str)
            categories = categories[
                ~categories.isin(["Geen informatie", "Numerieke bio-indicatie"])
            ]
            if categories.empty:
                central = None
                category = "Geen informatie"
                dominant_share = 0.0
            else:
                counts = categories.value_counts()
                category = str(counts.index[0])
                central = category
                dominant_share = float(counts.iloc[0] / counts.sum())
            q25 = q75 = minimum = maximum = None
            explanation = (
                f"Dominante locatiecategorie op basis van {location_count} locaties"
                + (f" ({dominant_share:.0%} van de locaties)." if not categories.empty else ".")
            )

        if location_count == 0:
            evidence = "Geen informatie"
        elif location_count < 3:
            evidence = "Onvoldoende informatie"
        elif location_count >= 5 and coverage >= 0.70:
            evidence = "Sterke aanwijzing"
        else:
            evidence = "Matige aanwijzing"

        results.append(
            WaterbodyParameterDiagnosis(
                parameter_key=str(parameter_key),
                parameter_label=label,
                central_value=round(central, 3) if isinstance(central, float) else central,
                category=category,
                lower_quartile=round(q25, 3) if q25 is not None else None,
                upper_quartile=round(q75, 3) if q75 is not None else None,
                minimum_value=round(minimum, 3) if minimum is not None else None,
                maximum_value=round(maximum, 3) if maximum is not None else None,
                location_count=location_count,
                sufficient_location_count=(
                    int(sufficient["location_id"].nunique()) if not sufficient.empty else 0
                ),
                location_coverage=coverage,
                evidence_strength=evidence,
                unit=unit,
                explanation=explanation,
            )
        )
    return tuple(results)


def calculate_descriptive_trend(annual_diagnoses: pd.DataFrame) -> dict[str, object]:
    """Beschrijf een jaarlijkse numerieke reeks zonder formele trendtoets."""
    if annual_diagnoses.empty:
        return {
            "direction": "Onvoldoende informatie",
            "first_value": None,
            "last_value": None,
            "absolute_change": None,
            "annual_change": None,
            "year_count": 0,
        }

    series = annual_diagnoses.copy()
    series["jaar"] = pd.to_numeric(series["jaar"], errors="coerce")
    series["central_value"] = pd.to_numeric(series["central_value"], errors="coerce")
    series = series.dropna(subset=["jaar", "central_value"]).sort_values("jaar")

    if len(series) < 2:
        value = float(series["central_value"].iloc[0]) if len(series) else None
        return {
            "direction": "Onvoldoende informatie",
            "first_value": value,
            "last_value": value,
            "absolute_change": None,
            "annual_change": None,
            "year_count": len(series),
        }

    first_year = float(series["jaar"].iloc[0])
    last_year = float(series["jaar"].iloc[-1])
    first_value = float(series["central_value"].iloc[0])
    last_value = float(series["central_value"].iloc[-1])
    change = last_value - first_value
    year_span = last_year - first_year
    annual_change = change / year_span if year_span > 0 else None
    tolerance = max(abs(first_value) * 0.02, 0.05)

    if change > tolerance:
        direction = "Stijgende indicatie"
    elif change < -tolerance:
        direction = "Dalende indicatie"
    else:
        direction = "Overwegend stabiel"

    return {
        "direction": direction,
        "first_value": first_value,
        "last_value": last_value,
        "absolute_change": change,
        "annual_change": annual_change,
        "year_count": len(series),
    }


def calculate_theil_sen_slope(annual_diagnoses: pd.DataFrame, year_column: str = "jaar", value_column: str = "central_value") -> float | None:
    if annual_diagnoses.empty or not {year_column, value_column}.issubset(annual_diagnoses.columns): return None
    f=annual_diagnoses[[year_column,value_column]].copy(); f[year_column]=pd.to_numeric(f[year_column],errors="coerce"); f[value_column]=pd.to_numeric(f[value_column],errors="coerce"); f=f.dropna().groupby(year_column,as_index=False)[value_column].median()
    if len(f)<2: return None
    y=f[year_column].to_numpy(float); v=f[value_column].to_numpy(float)
    slopes=[float((v[j]-v[i])/(y[j]-y[i])) for i in range(len(f)-1) for j in range(i+1,len(f)) if y[j]!=y[i]]
    return float(np.median(slopes)) if slopes else None


def bootstrap_location_trend(location_year_diagnoses: pd.DataFrame, iterations: int, confidence_level: float, random_seed: int) -> tuple[float|None,float|None,int]:
    req={"jaar","location_id","value"}
    if location_year_diagnoses.empty or not req.issubset(location_year_diagnoses.columns): return None,None,0
    f=location_year_diagnoses[list(req)].copy(); f["jaar"]=pd.to_numeric(f["jaar"],errors="coerce"); f["value"]=pd.to_numeric(f["value"],errors="coerce"); f=f.dropna()
    if f["jaar"].nunique()<2: return None,None,0
    groups=[(int(y),g["value"].to_numpy(float)) for y,g in f.groupby("jaar")]; rng=np.random.default_rng(random_seed); slopes=[]
    for _ in range(iterations):
        annual=pd.DataFrame([{"jaar":y,"central_value":float(np.median(rng.choice(vals,size=len(vals),replace=True)))} for y,vals in groups if len(vals)])
        slope=calculate_theil_sen_slope(annual)
        if slope is not None and math.isfinite(slope): slopes.append(slope)
    if not slopes: return None,None,0
    a=1-confidence_level
    return float(np.quantile(slopes,a/2)),float(np.quantile(slopes,1-a/2)),len(slopes)


def classify_formal_trend(slope, lower_interval, upper_interval, quality_status: str, tolerance: float=1e-9) -> str:
    if slope is None or quality_status=="Onvoldoende": return "Onvoldoende gegevens"
    if lower_interval is None or upper_interval is None: return "Beschrijvende trend"
    if lower_interval>tolerance: return "Overtuigend stijgend"
    if upper_interval < -tolerance: return "Overtuigend dalend"
    if slope>tolerance: return "Mogelijk stijgend"
    if slope < -tolerance: return "Mogelijk dalend"
    return "Stabiel / geen duidelijke richting"


def assess_trend_quality(annual_diagnoses, recurring_location_count, total_location_count, min_years, min_locations_per_year, min_profile_coverage, min_recurring_share):
    if annual_diagnoses.empty: return {"status":"Onvoldoende","score":0,"year_count":0,"minimum_locations_per_year":0,"median_locations_per_year":0.0,"minimum_profile_coverage":0.0,"recurring_location_share":0.0,"warnings":("Geen jaarlijkse diagnoses beschikbaar.",)}
    years=int(annual_diagnoses["jaar"].nunique()); loc=pd.to_numeric(annual_diagnoses["location_count"],errors="coerce"); cov=pd.to_numeric(annual_diagnoses["location_coverage"],errors="coerce")
    minloc=int(loc.min()) if loc.notna().any() else 0; medloc=float(loc.median()) if loc.notna().any() else 0.0; mincov=float(cov.min()) if cov.notna().any() else 0.0; share=recurring_location_count/total_location_count if total_location_count else 0.0; warnings=[]; score=4
    if years<min_years: warnings.append(f"Minder dan {min_years} jaren beschikbaar."); score-=2
    if minloc<min_locations_per_year: warnings.append(f"In minimaal een jaar zijn minder dan {min_locations_per_year} locaties beschikbaar."); score-=1
    if mincov<min_profile_coverage: warnings.append("De locatiedekking is in minimaal een jaar te laag."); score-=1
    if share<min_recurring_share: warnings.append("Het aandeel terugkerende locaties is laag."); score-=1
    score=max(score,0); status="Hoog" if score>=4 else "Matig" if score>=2 else "Onvoldoende"
    return {"status":status,"score":score,"year_count":years,"minimum_locations_per_year":minloc,"median_locations_per_year":medloc,"minimum_profile_coverage":mincov,"recurring_location_share":share,"warnings":tuple(warnings)}


def compare_trend_scenarios(scenarios: pd.DataFrame):
    if scenarios.empty or "classification" not in scenarios: return {"conclusion_stable":False,"explanation":"Geen scenarioresultaten beschikbaar."}
    u=scenarios[~scenarios["classification"].isin(["Onvoldoende gegevens","Beschrijvende trend"])]
    if u.empty: return {"conclusion_stable":False,"explanation":"Geen scenario heeft voldoende kwaliteit."}
    directions=u["classification"].astype(str).map(lambda x:"up" if "stijgend" in x else "down" if "dalend" in x else "stable"); stable=directions.nunique()==1
    return {"conclusion_stable":stable,"explanation":"De richting is consistent tussen de bruikbare scenario's." if stable else "De trendrichting is gevoelig voor locatiekeuze of bedekkingsweging."}



def interpret_diagnosis_value(
    parameter_key: str,
    value: object,
    category: str | None = None,
    companion_category: str | None = None,
) -> str | object:
    """Vertaal een diagnosewaarde naar dezelfde klasse als in de diagnosetabellen."""
    if companion_category and companion_category not in {
        "Geen informatie",
        "Numerieke bio-indicatie",
    }:
        return companion_category

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return category or value
    numeric = float(numeric)

    if parameter_key == "salinity":
        if numeric <= 2:
            label = "Zeer ionenarm water"
        elif numeric <= 4:
            label = "Ionenarm water"
        elif numeric <= 9:
            label = "Matig ionenrijk water"
        elif numeric <= 15:
            label = "Ionenrijk water"
        else:
            label = "Zeer ionenrijk water"

        if numeric > 850:
            salt_class = "zout water"
        elif numeric > 16:
            salt_class = "brak water"
        elif numeric < 16:
            salt_class = "zoet water"
        else:
            salt_class = "grenswaarde zoet/brak"
        return f"{label} ({salt_class})"

    if parameter_key == "alkalinity":
        if numeric < 0.1:
            return "Ongebufferd (zuur) water"
        if numeric < 0.5:
            return "Zeer zacht water"
        if numeric < 1:
            return "Zacht water"
        if numeric < 2:
            return "Zacht tot matig hard water"
        if numeric <= 4:
            return "Hard water"
        return "Zeer hard water"

    if parameter_key == "redox":
        if numeric < -175:
            return "Zeer reductief"
        if numeric < -100:
            return "Reductief"
        if numeric < 0:
            return "Matig reductief"
        if numeric <= 100:
            return "Matig oxiderend"
        return "Oxiderend"

    if parameter_key == "carbonate":
        if numeric < 0.1:
            return "Zeer laag; geen afzonderlijke klasse"
        if numeric < 1:
            return "Zeer carbonaatarm"
        if numeric < 20:
            return "Carbonaatarm"
        if numeric < 200:
            return "Matig carbonaatrijk"
        if numeric <= 360:
            return "Carbonaatrijk"
        return "Zeer carbonaatrijk"

    if parameter_key in DIAGNOSIS_USE_EXISTING_CATEGORY:
        return category or value

    for upper_bound, label in DIAGNOSIS_VALUE_CLASS_BREAKS.get(parameter_key, ()):
        if numeric < upper_bound:
            return label
    return value


def calculate_location_diagnosis_records(profile_matches: pd.DataFrame, weighting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bereken alle locatie-jaarresultaten eenmaal voor hergebruik."""
    records=[]; overview=[]
    if profile_matches.empty or not {'jaar','locatie_id'}.issubset(profile_matches.columns): return pd.DataFrame(),pd.DataFrame()
    for (year,location_id),frame in profile_matches.groupby(['jaar','locatie_id'],sort=True):
        observed=int(frame['soort'].nunique()) if 'soort' in frame else len(frame)
        matched=frame.loc[frame['soortnaam'].notna()].copy() if 'soortnaam' in frame else frame.iloc[0:0].copy()
        matched_count=int(matched['soort'].nunique()) if 'soort' in matched else len(matched)
        diagnoses=calculate_all_diagnoses(matched,observed,weighting) if not matched.empty else tuple()
        for item in diagnoses:
            records.append({'jaar':int(year),'location_id':str(location_id),'parameter_key':item.parameter_key,'parameter_label':item.parameter_label,'value':item.value,'category':item.category,'evidence_strength':item.evidence_strength,'species_count':item.species_count,'profile_coverage':item.profile_coverage,'unit':item.unit})
        overview.append({'jaar':int(year),'location_id':str(location_id),'observed_species_count':observed,'matched_species_count':matched_count,'unmatched_species_count':max(observed-matched_count,0),'profile_coverage':matched_count/observed if observed else 0.0})
    return pd.DataFrame(records),pd.DataFrame(overview)


def aggregate_annual_waterbody_diagnoses(location_year_diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Aggregeer een vooraf berekende locatie-jaarcollectie naar jaarlijkse resultaten."""
    if location_year_diagnoses.empty: return pd.DataFrame()
    records=[]
    for year,frame in location_year_diagnoses.groupby('jaar',sort=True):
        for p in aggregate_waterbody_diagnoses(frame,int(frame['location_id'].nunique())):
            records.append({'jaar':int(year),'parameter_key':p.parameter_key,'parameter_label':p.parameter_label,'central_value':p.central_value,'category':p.category,'p25':p.lower_quartile,'p75':p.upper_quartile,'minimum':p.minimum_value,'maximum':p.maximum_value,'location_count':p.location_count,'sufficient_location_count':p.sufficient_location_count,'location_coverage':p.location_coverage,'evidence_strength':p.evidence_strength,'unit':p.unit})
    return pd.DataFrame(records)


def prepare_annual_boxplot_data(location_year_diagnoses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...], tuple[int, ...]]:
    """Maak locatiewaarden, jaarlijkse boxstatistieken en ontbrekende jaren."""
    required={"jaar","location_id","value"}
    empty_values=pd.DataFrame(columns=["jaar","jaar_label","location_id","value"])
    empty_stats=pd.DataFrame(columns=["jaar","jaar_label","minimum","p25","median","p75","maximum","location_count"])
    if location_year_diagnoses.empty or not required.issubset(location_year_diagnoses.columns): return empty_values,empty_stats,tuple(),tuple()
    values=location_year_diagnoses[["jaar","location_id","value"]].copy()
    values["jaar"]=pd.to_numeric(values["jaar"],errors="coerce"); values["value"]=pd.to_numeric(values["value"],errors="coerce")
    values=values.dropna(subset=["jaar","location_id","value"])
    if values.empty: return empty_values,empty_stats,tuple(),tuple()
    values["jaar"]=values["jaar"].astype(int); values["jaar_label"]=values["jaar"].astype(str)
    stats=values.groupby("jaar",as_index=False).agg(minimum=("value","min"),p25=("value",lambda x:x.quantile(.25)),median=("value","median"),p75=("value",lambda x:x.quantile(.75)),maximum=("value","max"),location_count=("location_id","nunique")).sort_values("jaar")
    stats["jaar_label"]=stats["jaar"].astype(str)
    available=tuple(int(x) for x in stats["jaar"]); all_years=range(min(available),max(available)+1); aset=set(available)
    missing=tuple(y for y in all_years if y not in aset)
    return values.reset_index(drop=True),stats.reset_index(drop=True),available,missing


def interpret_temporal_pattern(annual_diagnoses: pd.DataFrame) -> dict[str, object]:
    if annual_diagnoses.empty: return {"direction":"onvoldoende","text":"Geen jaarlijkse resultaten beschikbaar.","missing_years":()}
    f=annual_diagnoses.copy(); f["jaar"]=pd.to_numeric(f["jaar"],errors="coerce"); f["central_value"]=pd.to_numeric(f["central_value"],errors="coerce"); f=f.dropna(subset=["jaar","central_value"]).sort_values("jaar")
    if f.empty: return {"direction":"onvoldoende","text":"Jaaruitkomsten zijn niet numeriek te duiden.","missing_years":()}
    years=f["jaar"].astype(int).tolist(); missing=tuple(y for y in range(min(years),max(years)+1) if y not in set(years)); first=float(f["central_value"].iloc[0]); last=float(f["central_value"].iloc[-1]); change=last-first; relative=change/max(abs(first),abs(last),1.0)
    direction="stijgend" if relative>.10 else "dalend" if relative<-.10 else "stabiel"
    text=f"De mediane indicatie is tussen {years[0]} en {years[-1]} {'hoger' if direction=='stijgend' else 'lager' if direction=='dalend' else 'beperkt veranderd'} ({change:+.3g})."
    if missing: text+=f" De reeks bevat {len(missing)} kalenderja(a)r(en) zonder bruikbare meetdata."
    return {"direction":direction,"text":text,"missing_years":missing}


def interpret_spatial_variability(annual_diagnoses: pd.DataFrame) -> dict[str, object]:
    if annual_diagnoses.empty: return {"level":"onvoldoende","text":"Ruimtelijke spreiding kan niet worden beoordeeld."}
    f=annual_diagnoses.copy()
    for c in ("central_value","p25","p75"): f[c]=pd.to_numeric(f[c],errors="coerce")
    f=f.dropna(subset=["central_value","p25","p75"])
    if f.empty: return {"level":"onvoldoende","text":"Geen numerieke kwartielen beschikbaar."}
    r=((f["p75"]-f["p25"]).abs()/f["central_value"].abs().clip(lower=1e-9)).replace([np.inf,-np.inf],np.nan).dropna(); v=float(r.median()) if not r.empty else None
    if v is None: return {"level":"onvoldoende","text":"Relatieve spreiding kan niet betrouwbaar worden berekend."}
    if v<.15: return {"level":"klein","text":"De middelste helft van de locaties ligt dicht bij elkaar; het ruimtelijke beeld is relatief homogeen."}
    if v<.40: return {"level":"matig","text":"De locaties verschillen merkbaar, maar de middelste helft blijft binnen een matige bandbreedte."}
    return {"level":"groot","text":"De locaties lopen sterk uiteen; een waterlichaammediaan maskeert ruimtelijke heterogeniteit."}


def interpret_sampling_context(annual_diagnoses: pd.DataFrame, recurring_location_count: int, total_location_count: int) -> dict[str, object]:
    if annual_diagnoses.empty: return {"level":"onvoldoende","text":"Geen bemonsteringscontext beschikbaar."}
    loc=pd.to_numeric(annual_diagnoses["location_count"],errors="coerce"); cov=pd.to_numeric(annual_diagnoses["location_coverage"],errors="coerce"); minimum=int(loc.min()) if loc.notna().any() else 0; median=float(loc.median()) if loc.notna().any() else 0.; mincov=float(cov.min()) if cov.notna().any() else 0.; share=recurring_location_count/total_location_count if total_location_count else 0.; issues=[]
    if minimum<3: issues.append("minder dan drie locaties in minimaal een meetjaar")
    if mincov<.50: issues.append("locatiedekking onder 50% in minimaal een meetjaar")
    if share<.40: issues.append("laag aandeel terugkerende locaties")
    text=f"Per jaar zijn mediaan {median:.1f} locaties beschikbaar; het minimum is {minimum}. Het aandeel terugkerende locaties is {share:.0%}."
    if issues: text+=" Aandachtspunten: "+"; ".join(issues)+"."
    return {"level":"beperkt" if issues else "voldoende","text":text}


def interpret_uncertainty(formal_trend, sensitivity) -> dict[str, object]:
    if formal_trend is None: return {"level":"beschrijvend","text":"Geen formele analyse uitgevoerd; de ontwikkeling is uitsluitend beschrijvend."}
    lo,hi=formal_trend.lower_interval,formal_trend.upper_interval
    if lo is None or hi is None: level="onvoldoende"; text="Het onzekerheidsinterval kon niet worden bepaald."
    elif lo<=0<=hi: level="onzeker"; text="Het bootstrapinterval omvat nul; de richting is niet overtuigend onderscheiden van geen verandering."
    else: level="ondersteund"; text="Het bootstrapinterval ligt volledig aan een kant van nul; de richting wordt consistent ondersteund."
    if sensitivity is not None: text+=" "+sensitivity.explanation
    return {"level":level,"text":text}
