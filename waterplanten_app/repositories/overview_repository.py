from __future__ import annotations

from typing import Iterable

import pandas as pd

from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.pipelines.gold_views import ensure_gold_views
from waterplanten_app.core.taxonomy import add_species_group_columns


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values: Iterable[str]) -> str:
    vals = [str(v) for v in values if str(v) != ""]
    if not vals:
        return "('')"
    return "(" + ", ".join(_quote(v) for v in vals) + ")"


def _col(name: str, alias: str | None = None) -> str:
    return f"{alias}.{name}" if alias else name


def _where(filters: DashboardFilters, include_year: bool = True, alias: str | None = None) -> str:
    clauses: list[str] = []
    if include_year and filters.year is not None:
        clauses.append(f"{_col('jaar', alias)} = {int(filters.year)}")
    if filters.projects:
        clauses.append(f"{_col('Project', alias)} IN {_sql_in(filters.projects)}")
    if filters.waterbodies:
        clauses.append(f"{_col('Waterlichaam', alias)} IN {_sql_in(filters.waterbodies)}")
    return ('WHERE ' + ' AND '.join(clauses)) if clauses else ''


def get_filtered_overview_frame(filters: DashboardFilters) -> pd.DataFrame:
    con = ensure_gold_views()
    return con.execute(f"SELECT g.* FROM gold_ecology_base g {_where(filters, alias='g')}").fetch_df()


def get_previous_year_match(filters: DashboardFilters) -> tuple[pd.DataFrame, pd.DataFrame]:
    if filters.year is None:
        return pd.DataFrame(), pd.DataFrame()

    con = ensure_gold_views()
    scope_e = _where(filters, include_year=False, alias='e')
    scope_prev = _where(filters, include_year=False, alias='pbase')
    scope_curr = _where(filters, include_year=False, alias='c')

    prev_sql = f"""
    WITH current_wb AS (
        SELECT DISTINCT c.Waterlichaam
        FROM gold_ecology_base c
        WHERE c.jaar = {int(filters.year)}
        {'AND ' + scope_curr.replace('WHERE ', '') if scope_curr else ''}
    ),
    prev_years AS (
        SELECT
            pbase.Waterlichaam,
            MAX(CAST(pbase.jaar AS INTEGER)) AS prev_jaar
        FROM gold_ecology_base pbase
        WHERE CAST(pbase.jaar AS INTEGER) < {int(filters.year)}
        {'AND ' + scope_prev.replace('WHERE ', '') if scope_prev else ''}
        AND pbase.Waterlichaam IN (SELECT Waterlichaam FROM current_wb)
        GROUP BY 1
    )
    SELECT e.*
    FROM gold_ecology_base e
    INNER JOIN prev_years p
      ON e.Waterlichaam = p.Waterlichaam
     AND CAST(e.jaar AS INTEGER) = p.prev_jaar
    {scope_e}
    """
    prev_df = con.execute(prev_sql).fetch_df()

    current_sql = f"""
    SELECT e.*
    FROM gold_ecology_base e
    WHERE e.jaar = {int(filters.year)}
    {'AND ' + scope_e.replace('WHERE ', '') if scope_e else ''}
    AND e.Waterlichaam IN (SELECT DISTINCT Waterlichaam FROM ({prev_sql}))
    """
    year_df = con.execute(current_sql).fetch_df()
    return year_df, prev_df


def get_waterbody_summary(filters: DashboardFilters) -> pd.DataFrame:
    con = ensure_gold_views()
    sql = f"""
    WITH richness AS (
        SELECT
            r.jaar,
            r.Project,
            r.Waterlichaam,
            COUNT(DISTINCT r.soort) AS soortenrijkdom
        FROM gold_overview_species_records r
        {_where(filters, alias='r')}
        GROUP BY 1,2,3
    )
    SELECT
        b.Waterlichaam,
        b.bedekking_totaal_pct AS \"Bedekking (Totaal %)\",
        b.gem_diepte_m AS \"Gem. Diepte (m)\",
        b.gem_doorzicht_m AS \"Gem. Doorzicht (m)\",
        COALESCE(r.soortenrijkdom, 0) AS \"Soortenrijkdom\"
    FROM gold_overview_waterbody_summary_base b
    LEFT JOIN richness r
      ON b.jaar = r.jaar
     AND b.Project = r.Project
     AND b.Waterlichaam = r.Waterlichaam
    {_where(filters, alias='b')}
    ORDER BY b.Waterlichaam
    """
    return con.execute(sql).fetch_df()


def get_pie_counts(filters: DashboardFilters, dimension: str) -> pd.DataFrame:
    con = ensure_gold_views()
    dim = dimension.lower().strip()

    if dim == 'krw':
        return con.execute(f"""
        SELECT COALESCE(r.krw_class_weergave, 'Geen match') AS categorie, COUNT(*) AS aantal
        FROM gold_overview_species_records r
        {_where(filters, alias='r')}
        GROUP BY 1
        ORDER BY 2 DESC, 1
        """).fetch_df()

    if dim in {'trofie', 'trofieniveau'}:
        return con.execute(f"""
        SELECT COALESCE(r.trofisch_niveau_weergave, 'Geen match') AS categorie, COUNT(*) AS aantal
        FROM gold_overview_species_records r
        {_where(filters, alias='r')}
        GROUP BY 1
        ORDER BY 2 DESC, 1
        """).fetch_df()

    df = con.execute(f"SELECT r.* FROM gold_overview_species_records r {_where(filters, alias='r')}").fetch_df()
    if df.empty:
        if dim in {'n2000', 'kenmerkende_soorten', 'kenmerkend'}:
            return pd.DataFrame(columns=['categorie', 'label_kort', 'aantal'])
        return pd.DataFrame(columns=['categorie', 'aantal'])

    mapped = add_species_group_columns(df)

    if dim in {'soortgroep', 'soortgroepen'}:
        category_col = 'soortgroep_weergave' if 'soortgroep_weergave' in mapped.columns else 'soortgroep'
        if category_col not in mapped.columns:
            return pd.DataFrame(columns=['categorie', 'aantal'])
        return (
            mapped[category_col]
            .fillna('Geen match')
            .value_counts(dropna=False)
            .rename_axis('categorie')
            .reset_index(name='aantal')
        )

    if dim in {'n2000', 'kenmerkende_soorten', 'kenmerkend'}:
        if 'is_kenmerkende_soort_n2000' not in mapped.columns:
            return pd.DataFrame(columns=['categorie', 'label_kort', 'aantal'])

        n2000 = mapped[mapped['is_kenmerkende_soort_n2000'].fillna(False)].copy()
        if n2000.empty:
            return pd.DataFrame(columns=['categorie', 'label_kort', 'aantal'])

        full_label = (
            n2000.get('kenmerkende_soort_n2000_weergave', n2000.get('soort_display', n2000['soort']))
            .fillna('Geen match')
            .astype(str)
            .str.strip()
        )
        short_label = (
            n2000.get('soort_triviaal', pd.Series('', index=n2000.index, dtype='object'))
            .fillna('')
            .astype(str)
            .str.strip()
        )
        short_label = short_label.where(
            short_label != '',
            full_label.str.replace(r'\s*\([^)]+\)\s*$', '', regex=True).str.strip(),
        )

        out = pd.DataFrame({'categorie': full_label, 'label_kort': short_label})
        out = out[(out['categorie'] != 'Geen match') & (out['categorie'] != '') & (out['label_kort'] != '')]
        return (
            out.groupby(['categorie', 'label_kort'], as_index=False)
            .size()
            .rename(columns={'size': 'aantal'})
            .sort_values('aantal', ascending=False)
        )

    raise ValueError(f'Onbekende pie-dimensie: {dimension}')
