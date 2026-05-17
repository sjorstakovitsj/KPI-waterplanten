from __future__ import annotations

from typing import Iterable

import pandas as pd

from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.pipelines.gold_views import ensure_gold_views
from waterplanten_app.core.taxonomy import add_species_group_columns


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values: Iterable[str]) -> str:
    vals = [str(v) for v in values if str(v) != '']
    if not vals:
        return "('')"
    return '(' + ', '.join(_quote(v) for v in vals) + ')'


def _col(name: str, alias: str | None = None) -> str:
    return f"{alias}.{name}" if alias else name


def _where(filters: DashboardFilters, alias: str | None = None) -> str:
    clauses: list[str] = []
    if filters.year is not None:
        clauses.append(f"{_col('jaar', alias)} = {int(filters.year)}")
    if filters.projects:
        clauses.append(f"{_col('Project', alias)} IN {_sql_in(filters.projects)}")
    if filters.waterbodies:
        clauses.append(f"{_col('Waterlichaam', alias)} IN {_sql_in(filters.waterbodies)}")
    return ('WHERE ' + ' AND '.join(clauses)) if clauses else ''


def get_location_base(filters: DashboardFilters) -> pd.DataFrame:
    con = ensure_gold_views()
    sql = f"SELECT g.* FROM gold_spatial_location_base g {_where(filters, alias='g')} ORDER BY g.Waterlichaam, g.locatie_id"
    return con.execute(sql).fetch_df()


def get_distribution_by_location(filters: DashboardFilters, dimension: str) -> pd.DataFrame:
    con = ensure_gold_views()
    dim = str(dimension).lower().strip()
    if dim in {'groeivorm', 'groeivormen'}:
        sql = f"SELECT g.locatie_id, g.categorie, g.waarde FROM gold_spatial_group_distribution g {_where(filters, alias='g')} ORDER BY g.locatie_id, g.categorie"
        return con.execute(sql).fetch_df()
    if dim in {'trofie', 'trofieniveau'}:
        sql = f"SELECT g.locatie_id, g.categorie, g.waarde FROM gold_spatial_trophic_counts g {_where(filters, alias='g')} ORDER BY g.locatie_id, g.categorie"
        return con.execute(sql).fetch_df()
    if dim in {'krw', 'krw score'}:
        sql = f"SELECT g.locatie_id, g.categorie, g.waarde FROM gold_spatial_krw_counts g {_where(filters, alias='g')} ORDER BY g.locatie_id, g.categorie"
        return con.execute(sql).fetch_df()
    if dim in {'soortgroep', 'soortgroepen'}:
        df = con.execute(f"SELECT r.* FROM gold_overview_species_records r {_where(filters, alias='r')}").fetch_df()
        if df.empty:
            return pd.DataFrame(columns=['locatie_id', 'categorie', 'waarde'])
        mapped = add_species_group_columns(df)
        if 'is_kenmerkende_soort_n2000' in mapped.columns:
            mapped = mapped[~mapped['is_kenmerkende_soort_n2000'].fillna(False)].copy()
        category_col = 'soortgroep_weergave' if 'soortgroep_weergave' in mapped.columns else 'soortgroep'
        if category_col not in mapped.columns:
            return pd.DataFrame(columns=['locatie_id', 'categorie', 'waarde'])
        value_col = 'bedekkingsgraad_proc' if 'bedekkingsgraad_proc' in mapped.columns else 'bedekking_pct'
        return (
            mapped.groupby(['locatie_id', category_col], as_index=False)[value_col]
            .sum()
            .rename(columns={category_col: 'categorie', value_col: 'waarde'})
            .sort_values(['locatie_id', 'categorie'])
        )
    raise ValueError(f'Onbekende ruimtelijke dimensie: {dimension}')


def get_total_cover_by_location(filters: DashboardFilters) -> pd.DataFrame:
    con = ensure_gold_views()
    sql = f'''
    SELECT
        e.locatie_id,
        AVG(COALESCE(TRY_CAST(e.totaal_bedekking_locatie AS DOUBLE), 0.0)) AS waarde_veg
    FROM gold_ecology_base e
    {_where(filters, alias='e')}
    GROUP BY 1
    ORDER BY e.locatie_id
    '''
    return con.execute(sql).fetch_df()


def get_weighted_krw_by_location(filters: DashboardFilters) -> pd.DataFrame:
    con = ensure_gold_views()
    sql = f"SELECT g.locatie_id, g.krw_score_loc FROM gold_spatial_weighted_krw g {_where(filters, alias='g')} ORDER BY g.locatie_id"
    return con.execute(sql).fetch_df()


def get_dominant_trophic_by_location(filters: DashboardFilters) -> pd.DataFrame:
    con = ensure_gold_views()
    sql = f"SELECT g.locatie_id, g.trofieniveau_loc FROM gold_spatial_dominant_trophic g {_where(filters, alias='g')} ORDER BY g.locatie_id"
    return con.execute(sql).fetch_df()


def get_species_value_by_location(filters: DashboardFilters, species_name: str) -> pd.DataFrame:
    con = ensure_gold_views()
    safe_species = str(species_name).replace("'", "''")
    where = _where(filters, alias='e')
    where_species = (where + f" AND e.soort = '{safe_species}'") if where else f"WHERE e.soort = '{safe_species}'"
    sql = f'''
    SELECT
        e.locatie_id,
        AVG(COALESCE(TRY_CAST(e.bedekking_pct AS DOUBLE), 0.0)) AS waarde_veg
    FROM gold_ecology_base e
    {where_species}
    GROUP BY 1
    ORDER BY e.locatie_id
    '''
    return con.execute(sql).fetch_df()


def build_location_table(filters: DashboardFilters) -> pd.DataFrame:
    base = get_location_base(filters)
    if base.empty:
        return base
    krw = get_weighted_krw_by_location(filters)
    trof = get_dominant_trophic_by_location(filters)
    return base.merge(krw, on='locatie_id', how='left').merge(trof, on='locatie_id', how='left')





def get_chemistry_location_points(df_chem: pd.DataFrame | None) -> pd.DataFrame:
    """Leid unieke chemische meetpunten af uit ruwe chemiedata.

    De output sluit aan op add_chemistry_locations_to_map() en bevat minimaal:
    locatie_code, meetlocatie_naam, chem_lat, chem_lon, n_records, n_parameters.
    De functie werkt heuristisch op veelvoorkomende kolomnamen zodat bestaande functionaliteit
    behouden blijft zonder utils-afhankelijkheid.
    """
    if df_chem is None or df_chem.empty:
        return pd.DataFrame(columns=['locatie_code', 'meetlocatie_naam', 'chem_lat', 'chem_lon', 'n_records', 'n_parameters'])

    df = df_chem.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    def _pick(options: list[str]) -> str | None:
        for opt in options:
            col = cols_lower.get(opt.lower())
            if col:
                return col
        return None

    loc_code_col = _pick(['locatie_code', 'meetobject', 'meetpunt', 'meetlocatie_code', 'locatie', 'code'])
    loc_name_col = _pick(['meetlocatie_naam', 'meetlocatie', 'meetpunt_naam', 'locatienaam', 'naam'])
    lat_col = _pick(['chem_lat', 'lat', 'latitude', 'breedtegraad'])
    lon_col = _pick(['chem_lon', 'lon', 'lng', 'longitude', 'lengtegraad'])
    param_col = _pick(['parameter', 'grootheid', 'stof', 'component'])

    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=['locatie_code', 'meetlocatie_naam', 'chem_lat', 'chem_lon', 'n_records', 'n_parameters'])

    tmp = pd.DataFrame(index=df.index)
    tmp['locatie_code'] = df[loc_code_col].astype(str).str.strip() if loc_code_col else ''
    tmp['meetlocatie_naam'] = df[loc_name_col].astype(str).str.strip() if loc_name_col else tmp['locatie_code']
    tmp['chem_lat'] = pd.to_numeric(df[lat_col], errors='coerce')
    tmp['chem_lon'] = pd.to_numeric(df[lon_col], errors='coerce')
    tmp['parameter'] = df[param_col].astype(str).str.strip() if param_col else ''

    tmp = tmp.dropna(subset=['chem_lat', 'chem_lon'])
    if tmp.empty:
        return pd.DataFrame(columns=['locatie_code', 'meetlocatie_naam', 'chem_lat', 'chem_lon', 'n_records', 'n_parameters'])

    grouped = (
        tmp.groupby(['locatie_code', 'meetlocatie_naam', 'chem_lat', 'chem_lon'], dropna=False, as_index=False)
        .agg(n_records=('locatie_code', 'size'), n_parameters=('parameter', lambda s: s.replace('', pd.NA).dropna().nunique()))
    )
    grouped['n_parameters'] = grouped['n_parameters'].fillna(0).astype(int)
    return grouped.sort_values(['meetlocatie_naam', 'locatie_code']).reset_index(drop=True)

__all__ = [
    'get_location_base',
    'get_distribution_by_location',
    'get_total_cover_by_location',
    'get_weighted_krw_by_location',
    'get_dominant_trophic_by_location',
    'get_species_value_by_location',
    'build_location_table',
    'get_chemistry_location_points',
]
