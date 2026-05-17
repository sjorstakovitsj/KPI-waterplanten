from __future__ import annotations

from typing import Mapping

import pandas as pd

from waterplanten_app.config.mappings import RWS_GROEIVORM_CODES
from waterplanten_app.core.diagnostics import calculate_kpi
from waterplanten_app.domain.contracts import DashboardFilters, OverviewKpi, OverviewResult
from waterplanten_app.repositories.overview_repository import get_pie_counts, get_previous_year_match, get_waterbody_summary


def _species_only(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=getattr(df, 'columns', []))
    out = df.copy()
    if 'type' in out.columns:
        out = out[out['type'].astype(str) == 'Soort'].copy()
    if 'soort' in out.columns:
        out = out[~out['soort'].isin(RWS_GROEIVORM_CODES)].copy()
    return out


def build_overview_kpis(filters: DashboardFilters) -> Mapping[str, OverviewKpi]:
    year_df, prev_df = get_previous_year_match(filters)
    avg_bedekking, d_bedekking = calculate_kpi(year_df, prev_df, 'totaal_bedekking_locatie', is_loc_metric=True)
    avg_doorzicht, d_doorzicht = calculate_kpi(year_df, prev_df, 'doorzicht_m', is_loc_metric=True)
    avg_diepte, d_diepte = calculate_kpi(year_df, prev_df, 'diepte_m', is_loc_metric=True)
    year_species = _species_only(year_df)
    prev_species = _species_only(prev_df)
    n_soorten = int(year_species['soort'].nunique()) if 'soort' in year_species.columns else 0
    prev_soorten = int(prev_species['soort'].nunique()) if not prev_species.empty and 'soort' in prev_species.columns else n_soorten
    d_soorten = n_soorten - prev_soorten
    return {
        'bedekking': OverviewKpi('gem. totale bedekking', float(avg_bedekking), float(d_bedekking), unit='%', precision=1),
        'doorzicht': OverviewKpi('gem. doorzicht', float(avg_doorzicht), float(d_doorzicht), unit='m', precision=2),
        'diepte': OverviewKpi('gem. diepte', float(avg_diepte), float(d_diepte), unit='m', precision=2),
        'soortenrijkdom': OverviewKpi('gem. soortenrijkdom', int(n_soorten), int(d_soorten), unit='soorten', precision=0),
    }


def build_overview_result(filters: DashboardFilters) -> OverviewResult:
    kpis = build_overview_kpis(filters)
    overview_table = get_waterbody_summary(filters)
    krw_pie = get_pie_counts(filters, 'krw')
    trophic_pie = get_pie_counts(filters, 'trofie')
    species_group_pie = get_pie_counts(filters, 'soortgroep')
    n2000_pie = get_pie_counts(filters, 'n2000')
    metadata = {'selected_year': filters.year, 'project_count': len(filters.projects), 'waterbody_count': len(filters.waterbodies)}
    return OverviewResult(filters=filters, kpis=kpis, overview_table=overview_table, krw_pie=krw_pie, trophic_pie=trophic_pie, species_group_pie=species_group_pie, n2000_pie=n2000_pie, metadata=metadata)
