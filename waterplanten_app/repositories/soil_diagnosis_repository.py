"""Gecachte DuckDB-repository voor bodem-, waterlichaam- en trenddiagnoses."""
from __future__ import annotations
from typing import Iterable
import pandas as pd
import streamlit as st
from waterplanten_app.config.settings import FINAL_PARQUET, SOIL_DIAGNOSIS_CACHE_MAX_ENTRIES, SOIL_DIAGNOSIS_CACHE_TTL_SECONDS, SOIL_PROFILES_PARQUET
from waterplanten_app.config.soil_diagnosis_config import SOIL_PROFILES_VIEW
from waterplanten_app.core.duckdb_runtime import _paths_fingerprint
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.pipelines.gold_views import ensure_gold_views


def _quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values: Iterable[str]) -> str:
    cleaned=[str(v) for v in values if str(v).strip()]
    return "("+", ".join(_quote(v) for v in cleaned)+")" if cleaned else "('')"


def _normalize_sql(expression: str) -> str:
    return f"LOWER(REGEXP_REPLACE(TRIM(CAST({expression} AS VARCHAR)), '\\s+', ' ', 'g'))"


def _filter_key(filters: DashboardFilters, include_year: bool=True) -> tuple:
    return (int(filters.year) if include_year and filters.year is not None else None, tuple(map(str,filters.projects)), tuple(map(str,filters.waterbodies)))


def _where_from_key(key: tuple, alias: str='e') -> str:
    year,projects,waterbodies=key
    clauses=[f"COALESCE({alias}.type, '') = 'Soort'",f"NULLIF(TRIM(CAST({alias}.soort AS VARCHAR)), '') IS NOT NULL"]
    if year is not None: clauses.append(f"CAST({alias}.jaar AS INTEGER) = {int(year)}")
    if projects: clauses.append(f"{alias}.Project IN {_sql_in(projects)}")
    if waterbodies: clauses.append(f"{alias}.Waterlichaam IN {_sql_in(waterbodies)}")
    return "WHERE "+" AND ".join(clauses)


def _data_fingerprint() -> tuple:
    return _paths_fingerprint(FINAL_PARQUET,SOIL_PROFILES_PARQUET)


@st.cache_data(show_spinner=False,ttl=SOIL_DIAGNOSIS_CACHE_TTL_SECONDS,max_entries=SOIL_DIAGNOSIS_CACHE_MAX_ENTRIES)
def _cached_waterbody_matches(filter_key: tuple,data_fingerprint: tuple) -> pd.DataFrame:
    del data_fingerprint
    con=ensure_gold_views()
    return con.execute(f"""
    WITH observed AS (
      SELECT CAST(e.jaar AS INTEGER) AS jaar, CAST(e.Project AS VARCHAR) AS Project,
             CAST(e.Waterlichaam AS VARCHAR) AS Waterlichaam, CAST(e.locatie_id AS VARCHAR) AS locatie_id,
             CAST(e.soort AS VARCHAR) AS soort, {_normalize_sql('e.soort')} AS species_match_key,
             AVG(TRY_CAST(e.bedekking_pct AS DOUBLE)) AS bedekking_pct, COUNT(*) AS aantal_records
      FROM gold_ecology_base e {_where_from_key(filter_key)} GROUP BY 1,2,3,4,5,6)
    SELECT o.*, p.* EXCLUDE (species_match_key) FROM observed o
    LEFT JOIN {SOIL_PROFILES_VIEW} p ON o.species_match_key=p.species_match_key
    ORDER BY o.Waterlichaam,o.jaar,o.locatie_id,o.soort
    """).fetch_df()


def clear_soil_diagnosis_repository_cache() -> None:
    _cached_waterbody_matches.clear()


def get_waterbody_species_profile_matches(filters: DashboardFilters) -> pd.DataFrame:
    return _cached_waterbody_matches(_filter_key(filters,True),_data_fingerprint()).copy()


def get_waterbody_trend_species_profile_matches(filters: DashboardFilters) -> pd.DataFrame:
    return _cached_waterbody_matches(_filter_key(filters,False),_data_fingerprint()).copy()


def get_available_locations(filters: DashboardFilters) -> list[str]:
    frame=get_waterbody_species_profile_matches(filters)
    return sorted(frame['locatie_id'].dropna().astype(str).unique().tolist()) if not frame.empty and 'locatie_id' in frame else []


def get_species_profile_matches(filters: DashboardFilters,location_id: str) -> pd.DataFrame:
    frame=get_waterbody_species_profile_matches(filters)
    return frame.loc[frame['locatie_id'].astype(str).eq(str(location_id))].reset_index(drop=True) if not frame.empty and 'locatie_id' in frame else frame


def get_observed_species(filters: DashboardFilters,location_id: str) -> pd.DataFrame:
    frame=get_species_profile_matches(filters,location_id); cols=[c for c in ('soort','species_match_key','bedekking_pct','aantal_records') if c in frame]
    return frame.loc[:,cols].copy()


def get_matched_species_profiles(filters: DashboardFilters,location_id: str) -> pd.DataFrame:
    frame=get_species_profile_matches(filters,location_id)
    return frame.loc[frame['soortnaam'].notna()].reset_index(drop=True) if 'soortnaam' in frame else frame.iloc[0:0].copy()


def get_unmatched_species(filters: DashboardFilters,location_id: str) -> pd.DataFrame:
    frame=get_species_profile_matches(filters,location_id)
    if 'soortnaam' not in frame: return frame.copy()
    cols=[c for c in ('locatie_id','soort','bedekking_pct','aantal_records') if c in frame]
    return frame.loc[frame['soortnaam'].isna(),cols].reset_index(drop=True)
