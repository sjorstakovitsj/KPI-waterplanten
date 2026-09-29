"""Validatie en normalisatie van soortspecifieke standplaatsprofielen."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from waterplanten_app.config.soil_diagnosis_config import (
    ANION_SCORE_COLUMNS,
    INDICATOR_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
)
from waterplanten_app.core.taxonomy import normalize_species_name


class SoilProfileValidationError(ValueError):
    """Het bronbestand kan niet veilig naar het domeincontract worden omgezet."""


@dataclass(frozen=True)
class SoilProfileValidationReport:
    row_count: int
    profile_count: int
    warning_count: int
    warnings: tuple[str, ...] = ()


def _read_source(path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, sep=";", decimal=",", encoding="latin-1")
    if len(frame.columns) == 1:
        frame = pd.read_csv(path, sep=None, engine="python", decimal=",", encoding="utf-8-sig")
    return frame


def load_and_validate_soil_profiles(path) -> tuple[pd.DataFrame, SoilProfileValidationReport]:
    raw = _read_source(path)
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in raw.columns]
    if missing:
        raise SoilProfileValidationError(
            "Ontbrekende verplichte kolommen: " + ", ".join(missing)
        )

    frame = raw.loc[:, list(REQUIRED_SOURCE_COLUMNS)].copy()
    frame = frame.dropna(how="all")
    frame["soortnaam"] = frame["soortnaam"].astype("string").str.strip()
    frame["species_match_key"] = frame["soortnaam"].map(normalize_species_name)

    if frame["species_match_key"].eq("").any():
        rows = (frame.index[frame["species_match_key"].eq("")] + 2).tolist()
        raise SoilProfileValidationError(f"Lege soortnaam op CSV-regel(s): {rows}")

    duplicates = frame.loc[
        frame["species_match_key"].duplicated(keep=False), "soortnaam"
    ].astype(str).tolist()
    if duplicates:
        raise SoilProfileValidationError(
            "Dubbele genormaliseerde soortnamen: " + ", ".join(sorted(set(duplicates)))
        )

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    issues: list[str] = []
    for column in INDICATOR_COLUMNS:
        invalid = frame[column].notna() & ~frame[column].isin([1, 2, 3])
        if invalid.any():
            values = sorted(frame.loc[invalid, column].unique().tolist())
            issues.append(f"{column} bevat ongeldige waarden {values}; toegestaan: 1, 2, 3")

    for column in ANION_SCORE_COLUMNS:
        invalid = frame[column].notna() & ~frame[column].between(-3, 3, inclusive="both")
        if invalid.any():
            values = sorted(frame.loc[invalid, column].unique().tolist())
            issues.append(f"{column} bevat waarden buiten -3 tot en met 3: {values}")

    for prefix in ("pH", "sal", "alk"):
        indicator, mean, minimum, maximum = (
            f"ind_{prefix}", f"gg_{prefix}", f"min_{prefix}", f"max_{prefix}"
        )
        invalid_order = (
            frame[minimum].notna() & frame[mean].notna() & frame[maximum].notna()
            & ((frame[minimum] > frame[mean]) | (frame[mean] > frame[maximum]))
        )
        if invalid_order.any():
            names = frame.loc[invalid_order, "soortnaam"].astype(str).tolist()
            issues.append(
                f"Voor {prefix} geldt niet minimum <= gemiddelde <= maximum bij: "
                + ", ".join(names)
            )
        incomplete = frame[[indicator, mean, minimum, maximum]].notna().any(axis=1) & frame[
            [indicator, mean, minimum, maximum]
        ].isna().any(axis=1)
        if incomplete.any():
            issues.append(
                f"{int(incomplete.sum())} soort(en) hebben een onvolledig {prefix}-profiel; "
                "alleen complete, bruikbare velden tellen per berekening mee."
            )

    if any("ongeldige" in issue or "buiten" in issue or "geldt niet" in issue for issue in issues):
        raise SoilProfileValidationError("; ".join(issues))

    text_columns = ("bodemtype", "os_bodem", "redox", "carb_bodem")
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()
        frame.loc[frame[column].eq(""), column] = pd.NA

    frame = frame.sort_values("species_match_key").reset_index(drop=True)
    report = SoilProfileValidationReport(
        row_count=len(raw),
        profile_count=len(frame),
        warning_count=len(issues),
        warnings=tuple(issues),
    )
    return frame, report
