"""Gecachte DuckDB-repository voor bodem-, waterlichaam- en trenddiagnoses."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from waterplanten_app.config.settings import (
    FINAL_PARQUET,
    SOIL_DIAGNOSIS_CACHE_MAX_ENTRIES,
    SOIL_DIAGNOSIS_CACHE_TTL_SECONDS,
    SOIL_PROFILES_PARQUET,
)
from waterplanten_app.config.soil_diagnosis_config import SOIL_PROFILES_VIEW
from waterplanten_app.core.duckdb_runtime import _paths_fingerprint
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.pipelines.gold_views import ensure_gold_views

_LATITUDE_CANDIDATES = (
    "latitude", "lat", "locatie_latitude", "meetpunt_latitude", "wgs84_lat", "y_wgs84",
)
_LONGITUDE_CANDIDATES = (
    "longitude", "lon", "lng", "locatie_longitude", "meetpunt_longitude", "wgs84_lon", "x_wgs84",
)
_RD_X_CANDIDATES = ("x", "x_coord", "x_coordinate", "rd_x", "x_coordinaat")
_RD_Y_CANDIDATES = ("y", "y_coord", "y_coordinate", "rd_y", "y_coordinaat")


def _quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_in(values: Iterable[str]) -> str:
    cleaned = [str(value) for value in values if str(value).strip()]
    return "(" + ", ".join(_quote(value) for value in cleaned) + ")" if cleaned else "('')"


def _normalize_sql(expression: str) -> str:
    return f"LOWER(REGEXP_REPLACE(TRIM(CAST({expression} AS VARCHAR)), '\\s+', ' ', 'g'))"


def _filter_key(filters: DashboardFilters, include_year: bool = True) -> tuple:
    return (
        int(filters.year) if include_year and filters.year is not None else None,
        tuple(map(str, filters.projects)),
        tuple(map(str, filters.waterbodies)),
    )


def _where_from_key(key: tuple, alias: str = "e") -> str:
    year, projects, waterbodies = key
    clauses = [
        f"COALESCE({alias}.type, '') = 'Soort'",
        f"NULLIF(TRIM(CAST({alias}.soort AS VARCHAR)), '') IS NOT NULL",
    ]
    if year is not None:
        clauses.append(f"CAST({alias}.jaar AS INTEGER) = {int(year)}")
    if projects:
        clauses.append(f"{alias}.Project IN {_sql_in(projects)}")
    if waterbodies:
        clauses.append(f"{alias}.Waterlichaam IN {_sql_in(waterbodies)}")
    return "WHERE " + " AND ".join(clauses)


def _relation_columns(con, relation: str) -> tuple[str, ...]:
    frame = con.execute(f"DESCRIBE {relation}").fetch_df()
    name_column = "column_name" if "column_name" in frame.columns else frame.columns[0]
    return tuple(str(value) for value in frame[name_column].tolist())


def _first_available(columns: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def _coordinate_definition(con) -> tuple[str, dict[str, str | None]]:
    columns = _relation_columns(con, "gold_ecology_base")
    metadata = {
        "latitude_column": _first_available(columns, _LATITUDE_CANDIDATES),
        "longitude_column": _first_available(columns, _LONGITUDE_CANDIDATES),
        "rd_x_column": _first_available(columns, _RD_X_CANDIDATES),
        "rd_y_column": _first_available(columns, _RD_Y_CANDIDATES),
    }

    def average(source: str | None, alias: str) -> str:
        if source is None:
            return f"CAST(NULL AS DOUBLE) AS {alias}"
        return f"AVG(TRY_CAST(e.{_quote_identifier(source)} AS DOUBLE)) AS {alias}"

    sql = ", ".join(
        (
            average(metadata["latitude_column"], "latitude"),
            average(metadata["longitude_column"], "longitude"),
            average(metadata["rd_x_column"], "rd_x"),
            average(metadata["rd_y_column"], "rd_y"),
        )
    )
    return sql, metadata


def _data_fingerprint() -> tuple:
    return _paths_fingerprint(FINAL_PARQUET, SOIL_PROFILES_PARQUET)


@st.cache_data(
    show_spinner=False,
    ttl=SOIL_DIAGNOSIS_CACHE_TTL_SECONDS,
    max_entries=SOIL_DIAGNOSIS_CACHE_MAX_ENTRIES,
)
def _cached_waterbody_matches(filter_key: tuple, data_fingerprint: tuple) -> pd.DataFrame:
    del data_fingerprint
    con = ensure_gold_views()
    coordinate_sql, _ = _coordinate_definition(con)
    return con.execute(
        f"""
        WITH observed AS (
            SELECT
                CAST(e.jaar AS INTEGER) AS jaar,
                CAST(e.Project AS VARCHAR) AS Project,
                CAST(e.Waterlichaam AS VARCHAR) AS Waterlichaam,
                CAST(e.locatie_id AS VARCHAR) AS locatie_id,
                CAST(e.soort AS VARCHAR) AS soort,
                {_normalize_sql('e.soort')} AS species_match_key,
                AVG(TRY_CAST(e.bedekking_pct AS DOUBLE)) AS bedekking_pct,
                COUNT(*) AS aantal_records,
                {coordinate_sql}
            FROM gold_ecology_base e
            {_where_from_key(filter_key)}
            GROUP BY 1, 2, 3, 4, 5, 6
        )
        SELECT o.*, p.* EXCLUDE (species_match_key)
        FROM observed o
        LEFT JOIN {SOIL_PROFILES_VIEW} p
          ON o.species_match_key = p.species_match_key
        ORDER BY o.Waterlichaam, o.jaar, o.locatie_id, o.soort
        """
    ).fetch_df()


def clear_soil_diagnosis_repository_cache() -> None:
    _cached_waterbody_matches.clear()


def get_waterbody_species_profile_matches(filters: DashboardFilters) -> pd.DataFrame:
    return _cached_waterbody_matches(_filter_key(filters, True), _data_fingerprint()).copy()


def get_waterbody_trend_species_profile_matches(filters: DashboardFilters) -> pd.DataFrame:
    return _cached_waterbody_matches(_filter_key(filters, False), _data_fingerprint()).copy()


def get_available_locations(filters: DashboardFilters) -> list[str]:
    frame = get_waterbody_species_profile_matches(filters)
    if frame.empty or "locatie_id" not in frame:
        return []
    return sorted(frame["locatie_id"].dropna().astype(str).unique().tolist())


def get_species_profile_matches(filters: DashboardFilters, location_id: str) -> pd.DataFrame:
    frame = get_waterbody_species_profile_matches(filters)
    if frame.empty or "locatie_id" not in frame:
        return frame
    return frame.loc[frame["locatie_id"].astype(str).eq(str(location_id))].reset_index(drop=True)


def get_observed_species(filters: DashboardFilters, location_id: str) -> pd.DataFrame:
    frame = get_species_profile_matches(filters, location_id)
    columns = [column for column in ("soort", "species_match_key", "bedekking_pct", "aantal_records") if column in frame]
    return frame.loc[:, columns].copy()


def get_matched_species_profiles(filters: DashboardFilters, location_id: str) -> pd.DataFrame:
    frame = get_species_profile_matches(filters, location_id)
    if "soortnaam" not in frame:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["soortnaam"].notna()].reset_index(drop=True)


def get_unmatched_species(filters: DashboardFilters, location_id: str) -> pd.DataFrame:
    frame = get_species_profile_matches(filters, location_id)
    if "soortnaam" not in frame:
        return frame.copy()
    columns = [column for column in ("locatie_id", "soort", "bedekking_pct", "aantal_records") if column in frame]
    return frame.loc[frame["soortnaam"].isna(), columns].reset_index(drop=True)


def get_coordinate_metadata() -> dict[str, str | None]:
    """Geef terug welke WGS84- of RD-kolommen in de gold-view zijn gevonden."""
    con = ensure_gold_views()
    _, metadata = _coordinate_definition(con)
    return metadata
