from __future__ import annotations

import pandas as pd

from waterplanten_app.domain.contracts import DashboardFilters, SpatialResult
from waterplanten_app.repositories.spatial_repository import (
    build_location_table,
    get_distribution_by_location,
    get_dominant_trophic_by_location,
    get_location_base,
    get_species_value_by_location,
    get_total_cover_by_location,
    get_weighted_krw_by_location,
)


PIE_DIMENSIONS = {'groeivormen', 'trofieniveau', 'trofie', 'krw score', 'krw', 'soortgroepen', 'soortgroep'}


def _normalize(value: str | None) -> str:
    return str(value or '').strip().lower()


def _counts_by_location(distribution: pd.DataFrame) -> dict:
    if distribution is None or distribution.empty:
        return {}
    grouped = {}
    for row in distribution.itertuples(index=False):
        loc = getattr(row, 'locatie_id')
        cat = getattr(row, 'categorie')
        val = getattr(row, 'waarde')
        grouped.setdefault(loc, {})[cat] = val
    return grouped


def _merge_metric(base: pd.DataFrame, metric: pd.DataFrame, column_name: str = 'waarde_veg') -> pd.DataFrame:
    if base is None or base.empty:
        cols = list(getattr(base, 'columns', []))
        if column_name not in cols:
            cols.append(column_name)
        return pd.DataFrame(columns=cols)
    out = base.copy()
    if metric is None or metric.empty:
        if column_name not in out.columns:
            out[column_name] = 0.0
        return out
    cols = list(metric.columns)
    if len(cols) >= 2 and cols[1] != column_name:
        metric = metric.rename(columns={cols[1]: column_name})
    out = out.merge(metric, on='locatie_id', how='left')
    if column_name in out.columns:
        out[column_name] = pd.to_numeric(out[column_name], errors='coerce').fillna(0.0)
    return out


def build_spatial_result(
    filters: DashboardFilters,
    analysis_level: str,
    coverage_type: str,
    layer_mode: str,
) -> SpatialResult:
    base = get_location_base(filters)
    location_table = build_location_table(filters)
    analysis_level_norm = _normalize(analysis_level)
    coverage_norm = _normalize(coverage_type)
    layer_mode_norm = _normalize(layer_mode)

    metadata: dict = {}
    legend_key: str | None = None
    map_points = base.copy()

    if layer_mode_norm == 'diepte':
        legend_key = 'diepte'
    elif layer_mode_norm == 'doorzicht':
        legend_key = 'doorzicht'
    else:
        if analysis_level_norm.startswith('individuele'):
            metric = get_species_value_by_location(filters, coverage_type)
            map_points = _merge_metric(base, metric, 'waarde_veg')
            location_table = _merge_metric(location_table, metric, 'waarde_veg')
            legend_key = 'vegetatie'
        else:
            if coverage_norm == 'totale bedekking':
                metric = get_total_cover_by_location(filters)
                map_points = _merge_metric(base, metric, 'waarde_veg')
                location_table = _merge_metric(location_table, metric, 'waarde_veg')
                legend_key = 'total_bedekking'
            elif coverage_norm in {'krw score', 'krw'}:
                metric = get_weighted_krw_by_location(filters).rename(columns={'krw_score_loc': 'waarde_veg'})
                map_points = _merge_metric(base, metric, 'waarde_veg')
                location_table = _merge_metric(location_table, metric, 'waarde_veg')
                legend_key = 'krw'
            elif coverage_norm in {'trofieniveau', 'trofie'}:
                metric = get_dominant_trophic_by_location(filters)
                map_points = base.merge(metric, on='locatie_id', how='left')
                location_table = location_table.merge(metric, on='locatie_id', how='left', suffixes=('', '_selected'))
                legend_key = 'trofieniveau'
            elif coverage_norm in PIE_DIMENSIONS:
                distribution = get_distribution_by_location(filters, coverage_type)
                metadata['distribution_by_location'] = distribution
                metadata['counts_by_location'] = _counts_by_location(distribution)
                legend_key = coverage_norm
            else:
                legend_key = 'vegetatie'


    if (
        layer_mode_norm not in {'diepte', 'doorzicht'}
        and not analysis_level_norm.startswith('individuele')
        and coverage_norm in PIE_DIMENSIONS
    ):
        distribution = get_distribution_by_location(filters, coverage_type)
        metadata['distribution_by_location'] = distribution
        metadata['counts_by_location'] = _counts_by_location(distribution)

    metadata.update({
        'selected_year': filters.year,
        'analysis_level': analysis_level,
        'coverage_type': coverage_type,
        'layer_mode': layer_mode,
        'n_locations': int(len(base)) if base is not None else 0,
    })

    return SpatialResult(
        filters=filters,
        map_points=map_points,
        location_table=location_table,
        layer_mode=layer_mode,
        analysis_level=analysis_level,
        coverage_type=coverage_type,
        legend_key=legend_key,
        metadata=metadata,
    )
