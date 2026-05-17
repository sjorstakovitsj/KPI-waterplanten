from __future__ import annotations

from pathlib import Path
from typing import Iterable

from waterplanten_app.config.settings import FINAL_PARQUET
from waterplanten_app.config.mappings import RWS_GROEIVORM_CODES
from waterplanten_app.core.duckdb_runtime import _get_duckdb


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values: Iterable[str]) -> str:
    vals = [str(v) for v in values if str(v) != '']
    if not vals:
        return "('')"
    return '(' + ', '.join(_quote(v) for v in vals) + ')'


def _relation_columns(con, relation_sql: str) -> list[str]:
    try:
        df = con.execute(f"DESCRIBE SELECT * FROM {relation_sql}").fetch_df()
        col_name_col = 'column_name' if 'column_name' in df.columns else df.columns[0]
        return [str(x) for x in df[col_name_col].tolist()]
    except Exception:
        return []


def _display_expr(cols: list[str], display_col: str, raw_col: str, default: str = 'Geen match') -> str:
    if display_col in cols:
        return f"COALESCE(NULLIF(TRIM(CAST(g.{display_col} AS VARCHAR)), ''), '{default}')"
    if raw_col in cols:
        return f"COALESCE(NULLIF(TRIM(CAST(g.{raw_col} AS VARCHAR)), ''), '{default}')"
    return f"'{default}'"


def ensure_gold_views(con=None):
    con = con or _get_duckdb()
    if con is None:
        raise RuntimeError('DuckDB is niet beschikbaar.')

    parquet_path = Path(FINAL_PARQUET)
    if not parquet_path.exists():
        raise FileNotFoundError(f'FINAL_PARQUET niet gevonden: {parquet_path}')

    excl = _sql_in(tuple(RWS_GROEIVORM_CODES))
    p = parquet_path.as_posix().replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW gold_ecology_base AS SELECT * FROM read_parquet('{p}')")

    cols = _relation_columns(con, 'gold_ecology_base')
    trophic_display_expr = _display_expr(cols, 'trofisch_niveau_weergave', 'trofisch_niveau')
    krw_display_expr = _display_expr(cols, 'krw_class_weergave', 'krw_class')
    select_cols = [f'g."{c}"' for c in cols if c not in {'trofisch_niveau_weergave', 'krw_class_weergave'}]
    select_cols.append(f"{trophic_display_expr} AS trofisch_niveau_weergave")
    select_cols.append(f"{krw_display_expr} AS krw_class_weergave")
    select_list = ',\n        '.join(select_cols)

    con.execute("""
    CREATE OR REPLACE VIEW gold_overview_sample_metrics AS
    SELECT
        CAST(jaar AS INTEGER) AS jaar,
        Project,
        Waterlichaam,
        CollectieReferentie,
        AVG(TRY_CAST(totaal_bedekking_locatie AS DOUBLE)) AS totaal_bedekking_locatie,
        AVG(TRY_CAST(diepte_m AS DOUBLE)) AS diepte_m,
        AVG(TRY_CAST(doorzicht_m AS DOUBLE)) AS doorzicht_m
    FROM gold_ecology_base
    GROUP BY 1,2,3,4
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW gold_overview_species_records AS
    SELECT
        {select_list}
    FROM gold_ecology_base g
    WHERE g.type = 'Soort'
      AND g.soort NOT IN {excl}
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_overview_waterbody_summary_base AS
    SELECT
        s.jaar,
        s.Project,
        s.Waterlichaam,
        AVG(s.totaal_bedekking_locatie) AS bedekking_totaal_pct,
        AVG(s.diepte_m) AS gem_diepte_m,
        AVG(s.doorzicht_m) AS gem_doorzicht_m
    FROM gold_overview_sample_metrics s
    GROUP BY 1,2,3
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_spatial_location_base AS
    SELECT
        CAST(jaar AS INTEGER) AS jaar,
        Project,
        locatie_id,
        Waterlichaam,
        ANY_VALUE(lat) AS lat,
        ANY_VALUE(lon) AS lon,
        AVG(TRY_CAST(diepte_m AS DOUBLE)) AS diepte_m,
        AVG(TRY_CAST(doorzicht_m AS DOUBLE)) AS doorzicht_m
    FROM gold_ecology_base
    GROUP BY 1,2,3,4
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_spatial_group_distribution AS
    SELECT
        CAST(jaar AS INTEGER) AS jaar,
        Project,
        locatie_id,
        groeivorm AS categorie,
        SUM(COALESCE(TRY_CAST(bedekking_pct AS DOUBLE), 0.0)) AS waarde
    FROM gold_ecology_base
    WHERE type = 'Groep'
    GROUP BY 1,2,3,4
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_spatial_trophic_counts AS
    SELECT
        CAST(jaar AS INTEGER) AS jaar,
        Project,
        locatie_id,
        COALESCE(trofisch_niveau_weergave, 'Geen match') AS categorie,
        COUNT(*) AS waarde
    FROM gold_overview_species_records
    GROUP BY 1,2,3,4
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_spatial_krw_counts AS
    SELECT
        CAST(jaar AS INTEGER) AS jaar,
        Project,
        locatie_id,
        COALESCE(krw_class_weergave, 'Geen match') AS categorie,
        COUNT(*) AS waarde
    FROM gold_overview_species_records
    GROUP BY 1,2,3,4
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_spatial_weighted_krw AS
    SELECT
        CAST(jaar AS INTEGER) AS jaar,
        Project,
        locatie_id,
        SUM(COALESCE(TRY_CAST(krw_score AS DOUBLE), 0.0) * GREATEST(COALESCE(TRY_CAST(bedekking_pct AS DOUBLE), 0.0), 0.0))
        / NULLIF(SUM(GREATEST(COALESCE(TRY_CAST(bedekking_pct AS DOUBLE), 0.0), 0.0)), 0.0) AS krw_score_loc
    FROM gold_overview_species_records
    WHERE krw_score IS NOT NULL
    GROUP BY 1,2,3
    """)

    con.execute("""
    CREATE OR REPLACE VIEW gold_spatial_dominant_trophic AS
    WITH ranked AS (
        SELECT
            CAST(jaar AS INTEGER) AS jaar,
            Project,
            locatie_id,
            COALESCE(trofisch_niveau_weergave, 'Geen match') AS trofieniveau_loc,
            SUM(GREATEST(COALESCE(TRY_CAST(bedekking_pct AS DOUBLE), 0.0), 0.0)) AS gewicht,
            ROW_NUMBER() OVER (
                PARTITION BY CAST(jaar AS INTEGER), Project, locatie_id
                ORDER BY SUM(GREATEST(COALESCE(TRY_CAST(bedekking_pct AS DOUBLE), 0.0), 0.0)) DESC,
                         COALESCE(trofisch_niveau_weergave, 'Geen match')
            ) AS rn
        FROM gold_overview_species_records
        GROUP BY 1,2,3,4
    )
    SELECT jaar, Project, locatie_id, trofieniveau_loc
    FROM ranked
    WHERE rn = 1
    """)

    return con


def list_gold_views() -> list[str]:
    return [
        'gold_ecology_base',
        'gold_overview_sample_metrics',
        'gold_overview_species_records',
        'gold_overview_waterbody_summary_base',
        'gold_spatial_location_base',
        'gold_spatial_group_distribution',
        'gold_spatial_trophic_counts',
        'gold_spatial_krw_counts',
        'gold_spatial_weighted_krw',
        'gold_spatial_dominant_trophic',
    ]
