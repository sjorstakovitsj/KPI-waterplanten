from __future__ import annotations

"""Definitieve data-accesslaag.

Deze module hangt uitsluitend af van:
- settings.py
- mappings.py
- duckdb_runtime.py
- taxonomy.py

Er zijn bewust geen imports uit pipelines of services, zodat er geen teruglussen
bij circular imports ontstaan.
"""

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from waterplanten_app.config.settings import (
    CHEMISTRY_FILE_PATH,
    CHEMISTRY_PARQUET,
    CHEMISTRY_PIPELINE_VERSION,
    COORD_CACHE_PARQUET,
    DATA_SOURCE,
    DATA_SOURCE_FALLBACK,
    DD_API_MEAS_PARQUET,
    DD_API_PIPELINE_VERSION,
    DD_API_REFERENCES_PARQUET,
    FILE_PATH,
    FINAL_PARQUET,
    LOOKUP_PARQUET,
    MEAS_PARQUET,
    PIPELINE_VERSION,
    SPECIES_LOOKUP_PATH,
)
from waterplanten_app.config.mappings import (
    EXCLUDED_SPECIES_CODES,
    GROWTH_FORM_MAPPING,
    KRW_WATERTYPE_BY_WATERLICHAAM,
    PROJECT_MAPPING,
    WATERBODY_MAPPING,
)
from waterplanten_app.core.duckdb_runtime import (
    _ensure_measurements_parquet,
    _get_duckdb,
    _mtime_or_zero,
    _read_parquet_to_pandas,
    _write_parquet_from_pandas,
)
from waterplanten_app.core.taxonomy import add_species_group_columns



def to_positive_number(value) -> float:
    """Zet veelvoorkomende numerieke formats om naar een positieve float.

    Ondersteunt o.a. 60, 60.0, '60,0', '60%', '<1', '>5'. Niet-parsebare of niet-positieve
    waarden worden 0.0. Dit voorkomt dat geldige pie-verdelingen onterecht leeg raken.
    """
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except Exception:
        pass
    if isinstance(value, (int, float)):
        try:
            num = float(value)
        except Exception:
            return 0.0
        return num if num > 0 else 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    text = text.replace('%', '').replace('<', '').replace('>', '').replace('≤', '').replace('≥', '').strip()
    if ',' in text and '.' in text:
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    else:
        text = text.replace(',', '.')
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if not m:
        return 0.0
    try:
        num = float(m.group(0))
    except Exception:
        return 0.0
    return num if num > 0 else 0.0


def normalize_locatie_key(value) -> str:
    """Normaliseer locatie-ids voor robuuste matching tussen metadata en kaartpunten."""
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    if text.endswith('.0') and text[:-2].lstrip('+-').isdigit():
        text = text[:-2]
    return text


# ============================================================================
# COORDINATEN / WATERLICHAAM
# ============================================================================

def rd_to_wgs84(x: float, y: float) -> Tuple[float | None, float | None]:
    """Converteert Rijksdriehoek (RD) coördinaten naar WGS84 (Lat/Lon)."""
    try:
        x0 = 155000
        y0 = 463000
        phi0 = 52.15517440
        lam0 = 5.38720621
        dx = (x - x0) * 10 ** -5
        dy = (y - y0) * 10 ** -5
        sum_phi = (
            (3235.65389 * dy)
            + (-32.58297 * dx ** 2)
            + (-0.24750 * dy ** 2)
            + (-0.84978 * dx ** 2 * dy)
            + (-0.06550 * dy ** 3)
            + (1.70776 * dx ** 2 * dy ** 2)
            + (-0.10715 * dy ** 4)
            + (0.009 * dy ** 5)
        )
        sum_lam = (
            (5260.52916 * dx)
            + (105.94684 * dx * dy)
            + (2.45656 * dx * dy ** 2)
            + (-0.81885 * dx ** 3)
            + (0.05594 * dx * dy ** 3)
            + (-0.05607 * dx ** 3 * dy)
            + (0.01199 * dy * dx ** 4)
            + (-0.00256 * dx ** 3 * dy ** 2)
        )
        lat = phi0 + sum_phi / 3600
        lon = lam0 + sum_lam / 3600
        return lat, lon
    except Exception:
        return None, None


_num_re = re.compile(r"(-?\d+(?:[.,]\d+)?)")


def _parse_wkt_point_numbers(wkt: str) -> Tuple[float | None, float | None]:
    if wkt is None:
        return None, None
    s = str(wkt).strip()
    if not s:
        return None, None
    nums = _num_re.findall(s)
    if len(nums) < 2:
        return None, None
    try:
        a = float(nums[0].replace(',', '.'))
        b = float(nums[1].replace(',', '.'))
        return a, b
    except Exception:
        return None, None


def _parse_coordinates_epsg(epsg: str, wkt: str) -> Tuple[float | None, float | None, float | None, float | None]:
    a, b = _parse_wkt_point_numbers(wkt)
    if a is None or b is None:
        return None, None, None, None

    epsg = (epsg or '').strip()
    if epsg == 'EPSG:4258':
        lon = a
        lat = b
        return None, None, lat, lon
    if epsg == 'EPSG:28992':
        x = a
        y = b
        lat, lon = rd_to_wgs84(x, y)
        return x, y, lat, lon
    if 3.0 <= a <= 8.2 and 50.0 <= b <= 54.8:
        return None, None, b, a
    if 0 <= a <= 300000 and 250000 <= b <= 700000:
        lat, lon = rd_to_wgs84(a, b)
        return a, b, lat, lon
    return a, b, None, None


def _load_coord_cache() -> pd.DataFrame:
    if COORD_CACHE_PARQUET.exists():
        try:
            df = _read_parquet_to_pandas(COORD_CACHE_PARQUET)
            for c in ['epsg', 'wkt', 'x_rd', 'y_rd', 'lat', 'lon']:
                if c not in df.columns:
                    df[c] = np.nan
            return df[['epsg', 'wkt', 'x_rd', 'y_rd', 'lat', 'lon']]
        except Exception:
            pass
    return pd.DataFrame(columns=['epsg', 'wkt', 'x_rd', 'y_rd', 'lat', 'lon'])


def _save_coord_cache(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.drop_duplicates(subset=['epsg', 'wkt'], keep='last')
    try:
        _write_parquet_from_pandas(df, COORD_CACHE_PARQUET)
    except Exception:
        pass


def _apply_coordinates_cached(df: pd.DataFrame, epsg_col: str = 'GeografieDatum', wkt_col: str = 'GeografieVorm') -> pd.DataFrame:
    df = df.copy()
    if epsg_col not in df.columns or wkt_col not in df.columns:
        df['x_rd'] = np.nan
        df['y_rd'] = np.nan
        df['lat'] = np.nan
        df['lon'] = np.nan
        return df

    epsg_series = df[epsg_col].fillna('').astype(str)
    wkt_series = df[wkt_col].fillna('').astype(str)
    uniq = pd.DataFrame({'epsg': epsg_series, 'wkt': wkt_series}).drop_duplicates()
    cache = _load_coord_cache()
    if not cache.empty:
        cache_idx = cache.set_index(['epsg', 'wkt'])
    else:
        cache_idx = pd.DataFrame(columns=['x_rd', 'y_rd', 'lat', 'lon']).set_index(
            pd.MultiIndex.from_arrays([[], []], names=['epsg', 'wkt'])
        )

    uniq_idx = uniq.set_index(['epsg', 'wkt'])
    missing_idx = uniq_idx.index.difference(cache_idx.index)
    if len(missing_idx) > 0:
        missing_pairs = list(missing_idx)
        parsed = [_parse_coordinates_epsg(epsg, wkt) for (epsg, wkt) in missing_pairs]
        add = pd.DataFrame({
            'epsg': [p[0] for p in missing_pairs],
            'wkt': [p[1] for p in missing_pairs],
            'x_rd': [t[0] for t in parsed],
            'y_rd': [t[1] for t in parsed],
            'lat': [t[2] for t in parsed],
            'lon': [t[3] for t in parsed],
        })
        cache = pd.concat([cache, add], ignore_index=True)
        _save_coord_cache(cache)
        cache_idx = cache.set_index(['epsg', 'wkt'])

    mapped = pd.DataFrame({'epsg': epsg_series, 'wkt': wkt_series})
    mapped = mapped.join(cache_idx, on=['epsg', 'wkt'])
    df['x_rd'] = mapped['x_rd'].astype(float)
    df['y_rd'] = mapped['y_rd'].astype(float)
    df['lat'] = mapped['lat'].astype(float)
    df['lon'] = mapped['lon'].astype(float)
    return df


def determine_waterbody(meetobject_code: str) -> str:
    for code, name in WATERBODY_MAPPING.items():
        if code in str(meetobject_code):
            return name
    return str(meetobject_code)


# ============================================================================
# LOOKUP / MATCH DISPLAY
# ============================================================================

@st.cache_data
def load_species_lookup() -> pd.DataFrame:
    """Laad koppeltabel met NL naam, trofie (Watertype) en KRW-scores (M14/M21)."""
    csv_path = Path(SPECIES_LOOKUP_PATH)
    if LOOKUP_PARQUET.exists() and _mtime_or_zero(LOOKUP_PARQUET) >= _mtime_or_zero(csv_path):
        df_cached = _read_parquet_to_pandas(LOOKUP_PARQUET)
        want = ['soort_norm', 'NL naam', 'Watertype', 'M14', 'M21']
        for c in want:
            if c not in df_cached.columns:
                df_cached[c] = np.nan
        return df_cached[want]

    try:
        df_lu = pd.read_csv(SPECIES_LOOKUP_PATH, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        st.warning(f'⚠️ Koppeltabel kon niet worden ingelezen ({SPECIES_LOOKUP_PATH}): {e}')
        return pd.DataFrame(columns=['soort_norm', 'NL naam', 'Watertype', 'M14', 'M21'])

    df_lu.columns = df_lu.columns.str.strip()
    rename_map = {
        'NL_naam': 'NL naam',
        'NLnaam': 'NL naam',
        'Trofisch niveau': 'Watertype',
        'Trofie': 'Watertype',
        'Trofieniveau': 'Watertype',
        'Wetenschappelijke_naam': 'Wetenschappelijke naam',
        'WetenschappelijkeNaam': 'Wetenschappelijke naam',
    }
    df_lu = df_lu.rename(columns={k: v for k, v in rename_map.items() if k in df_lu.columns})

    if 'Wetenschappelijke naam' not in df_lu.columns:
        st.warning("⚠️ Koppeltabel mist kolom 'Wetenschappelijke naam'. Verrijking wordt overgeslagen.")
        return pd.DataFrame(columns=['soort_norm', 'NL naam', 'Watertype', 'M14', 'M21'])

    if 'NL naam' not in df_lu.columns:
        df_lu['NL naam'] = np.nan
    if 'Watertype' not in df_lu.columns:
        df_lu['Watertype'] = np.nan
    for c in ['M14', 'M21']:
        if c not in df_lu.columns:
            df_lu[c] = np.nan
        df_lu[c] = pd.to_numeric(df_lu[c], errors='coerce')

    df_lu['soort_norm'] = (
        df_lu['Wetenschappelijke naam']
        .fillna('')
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    sort_cols = [c for c in ['NL naam', 'Watertype', 'M14', 'M21'] if c in df_lu.columns]
    if sort_cols:
        df_lu = df_lu.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    df_lu = df_lu.drop_duplicates(subset=['soort_norm'], keep='first')
    out = df_lu[['soort_norm', 'NL naam', 'Watertype', 'M14', 'M21']].copy()
    try:
        _write_parquet_from_pandas(out, LOOKUP_PARQUET)
    except Exception:
        pass
    return out


def _ensure_match_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Zorg dat match-/weergavekolommen voor KRW, trofieniveau en N2000 altijd aanwezig zijn."""
    if df is None or df.empty:
        return df
    df = df.copy()

    if 'trofisch_niveau' not in df.columns:
        df['trofisch_niveau'] = np.nan
    if 'trofisch_niveau_match_status' not in df.columns:
        df['trofisch_niveau_match_status'] = np.where(
            df['trofisch_niveau'].notna() & (df['trofisch_niveau'].astype(str).str.strip() != ''),
            'Match',
            'Geen match',
        )
    if 'trofisch_niveau_weergave' not in df.columns:
        df['trofisch_niveau_weergave'] = np.where(
            df['trofisch_niveau_match_status'] == 'Match',
            df['trofisch_niveau'].astype(str),
            'Geen match',
        )

    if 'krw_score' not in df.columns:
        df['krw_score'] = np.nan
    if 'krw_class' not in df.columns:
        df['krw_class'] = pd.cut(
            pd.to_numeric(df['krw_score'], errors='coerce'),
            bins=[0, 2, 4, 5],
            labels=['Gunstig (1-2)', 'Neutraal (3-4)', 'Ongewenst (5)'],
            include_lowest=True,
        )
    if 'krw_match_status' not in df.columns:
        df['krw_match_status'] = np.where(pd.to_numeric(df['krw_score'], errors='coerce').notna(), 'Match', 'Geen match')
    if 'krw_class_weergave' not in df.columns:
        df['krw_class_weergave'] = df['krw_class'].astype(object)
        df.loc[df['krw_class_weergave'].isna(), 'krw_class_weergave'] = 'Geen match'

    if 'is_kenmerkende_soort_n2000' not in df.columns:
        if 'Grootheid' in df.columns:
            df['is_kenmerkende_soort_n2000'] = df['Grootheid'].astype(str).eq('AANWZHD')
        else:
            df['is_kenmerkende_soort_n2000'] = False
    if 'kenmerkende_soort_n2000_match_status' not in df.columns:
        df['kenmerkende_soort_n2000_match_status'] = np.where(
            df['is_kenmerkende_soort_n2000'].fillna(False),
            'Match',
            'Geen match',
        )
    if 'kenmerkende_soort_n2000_weergave' not in df.columns:
        display_source = df['soort_display'] if 'soort_display' in df.columns else (df['soort'] if 'soort' in df.columns else pd.Series('', index=df.index))
        df['kenmerkende_soort_n2000_weergave'] = np.where(
            df['is_kenmerkende_soort_n2000'].fillna(False),
            display_source.astype(str),
            'Geen match',
        )
    return df


# ============================================================================
# CORE LOAD DATA
# ============================================================================

def _file_signature() -> Tuple[str, float, float, str]:
    source = str(DATA_SOURCE).strip().lower()
    if source == 'dd_eco_api_v3':
        return (
            'dd_eco_api_v3',
            _mtime_or_zero(DD_API_MEAS_PARQUET),
            _mtime_or_zero(DD_API_REFERENCES_PARQUET),
            DD_API_PIPELINE_VERSION,
        )
    csv_path = Path(FILE_PATH)
    lookup_path = Path(SPECIES_LOOKUP_PATH)
    return (
        str(csv_path.resolve()) if csv_path.exists() else str(csv_path),
        _mtime_or_zero(csv_path),
        _mtime_or_zero(lookup_path),
        PIPELINE_VERSION,
    )


@st.cache_data
def _load_data_cached(sig: Tuple[str, float, float, str]) -> pd.DataFrame:
    csv_path = Path(FILE_PATH)
    lookup_csv = Path(SPECIES_LOOKUP_PATH)

    source_is_api = str(DATA_SOURCE).strip().lower() == 'dd_eco_api_v3'

    if FINAL_PARQUET.exists():
        if source_is_api:
            ok = (
                _mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(DD_API_MEAS_PARQUET)
                and _mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(DD_API_REFERENCES_PARQUET)
                and _mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(lookup_csv)
            )
        else:
            ok = (
                _mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(csv_path)
                and _mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(lookup_csv)
            )
        if ok:
            df_final = _read_parquet_to_pandas(FINAL_PARQUET)
            if not df_final.empty:
                repaired = _ensure_match_display_columns(df_final)

                # schrijf terug als schema is uitgebreid
                missing_before = set(repaired.columns) - set(df_final.columns)
                if missing_before:
                    try:
                        _write_parquet_from_pandas(repaired, FINAL_PARQUET)
                    except Exception:
                        pass

                return repaired

    con = _get_duckdb()
    if con is not None:
        _ensure_measurements_parquet(csv_path if not source_is_api else None)
        if MEAS_PARQUET.exists():
            gf_keys = list(GROWTH_FORM_MAPPING.keys())
            gf_case = ' '.join([f"WHEN Parameter='{k}' THEN '{GROWTH_FORM_MAPPING[k]}'" for k in gf_keys])
            gf_in = ','.join([f"'{k}'" for k in gf_keys])
            sql = f"""
            WITH base AS (
                SELECT
                    *,
                    COALESCE(
                        TRY_CAST(MetingDatumTijd AS TIMESTAMP),
                        TRY_STRPTIME(CAST(MetingDatumTijd AS VARCHAR), '%d-%m-%Y %H:%M:%S'),
                        TRY_STRPTIME(CAST(MetingDatumTijd AS VARCHAR), '%d-%m-%Y'),
                        TRY_STRPTIME(CAST(MetingDatumTijd AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
                        TRY_STRPTIME(CAST(MetingDatumTijd AS VARCHAR), '%Y-%m-%d')
                    ) AS meting_ts
                FROM read_parquet('{MEAS_PARQUET.as_posix()}')
            ),
            env AS (
                SELECT
                    CollectieReferentie,
                    AVG(CASE WHEN Grootheid='DIEPTE' THEN CAST(WaardeGemeten AS DOUBLE) END) AS DIEPTE,
                    AVG(CASE WHEN Grootheid='ZICHT' THEN CAST(WaardeGemeten AS DOUBLE) END) AS ZICHT
                FROM base
                WHERE Grootheid IN ('DIEPTE','ZICHT')
                GROUP BY 1
            ),
            total AS (
                SELECT
                    CollectieReferentie,
                    AVG(CAST(WaardeGemeten AS DOUBLE)) AS totaal_bedekking_locatie
                FROM base
                WHERE Parameter='WATPTN'
                GROUP BY 1
            ),
            bedkg AS (
                SELECT
                    meting_ts,
                    DATE_TRUNC('day', meting_ts) AS datum,
                    EXTRACT(year FROM meting_ts) AS jaar,
                    MeetObject,
                    Projecten,
                    GeografieDatum,
                    GeografieVorm,
                    CollectieReferentie,
                    Parameter,
                    Grootheid,
                    WaardeGemeten,
                    EenheidGemeten,
                    CASE
                        WHEN Grootheid='AANWZHD' THEN 'Kenmerkende soort (N2000)'
                        {gf_case}
                        ELSE 'Individuele soort'
                    END AS groeivorm,
                    CASE
                        WHEN Grootheid='AANWZHD' THEN 'Soort'
                        WHEN Parameter IN ({gf_in}) THEN 'Groep'
                        ELSE 'Soort'
                    END AS type
                FROM base
                WHERE Grootheid IN ('BEDKG','AANWZHD') AND Parameter <> 'WATPTN'
            )
            SELECT
                b.datum,
                b.jaar,
                b.MeetObject,
                b.Projecten,
                b.GeografieDatum,
                b.GeografieVorm,
                b.CollectieReferentie,
                b.Parameter,
                b.Grootheid,
                b.WaardeGemeten,
                b.EenheidGemeten,
                b.groeivorm,
                b.type,
                e.DIEPTE,
                e.ZICHT,
                t.totaal_bedekking_locatie
            FROM bedkg b
            LEFT JOIN env e USING (CollectieReferentie)
            LEFT JOIN total t USING (CollectieReferentie)
            """
            try:
                df_merged = con.execute(sql).fetch_df()
            except Exception as e:
                df_merged = pd.DataFrame()
                st.warning(f'⚠️ DuckDB pad faalde, val terug op pandas: {e}')

            if not df_merged.empty:
                df_merged['Project'] = df_merged['Projecten'].map(PROJECT_MAPPING).fillna(df_merged['Projecten'])
                df_merged['Waterlichaam'] = df_merged['MeetObject'].apply(determine_waterbody)
                df_merged['diepte_m'] = pd.to_numeric(df_merged.get('DIEPTE'), errors='coerce') / 100.0
                df_merged['doorzicht_m'] = pd.to_numeric(df_merged.get('ZICHT'), errors='coerce') / 10.0
                final_df = df_merged.rename(columns={
                    'MeetObject': 'locatie_id',
                    'Parameter': 'soort',
                    'WaardeGemeten': 'waarde_bedekking',
                    'EenheidGemeten': 'eenheid',
                })
                final_df['bedekking_pct'] = final_df['waarde_bedekking']
                final_df = _apply_coordinates_cached(final_df, epsg_col='GeografieDatum', wkt_col='GeografieVorm')

                lookup = load_species_lookup()
                final_df['soort_norm'] = final_df['soort'].fillna('').astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                final_df = final_df.merge(lookup, on='soort_norm', how='left')
                final_df = final_df.rename(columns={'NL naam': 'soort_triviaal', 'Watertype': 'trofisch_niveau'})
                final_df['trofisch_niveau_match_status'] = np.where(
                    final_df['trofisch_niveau'].notna() & (final_df['trofisch_niveau'].astype(str).str.strip() != ''),
                    'Match', 'Geen match'
                )
                final_df['trofisch_niveau_weergave'] = np.where(
                    final_df['trofisch_niveau_match_status'] == 'Match',
                    final_df['trofisch_niveau'].astype(str),
                    'Geen match',
                )
                final_df['krw_watertype'] = final_df['Waterlichaam'].map(KRW_WATERTYPE_BY_WATERLICHAAM)
                final_df['krw_score'] = np.nan
                mask_m14 = final_df['krw_watertype'] == 'M14'
                mask_m21 = final_df['krw_watertype'] == 'M21'
                if 'M14' in final_df.columns:
                    final_df.loc[mask_m14, 'krw_score'] = final_df.loc[mask_m14, 'M14']
                if 'M21' in final_df.columns:
                    final_df.loc[mask_m21, 'krw_score'] = final_df.loc[mask_m21, 'M21']
                final_df['krw_class'] = pd.cut(
                    final_df['krw_score'],
                    bins=[0, 2, 4, 5],
                    labels=['Gunstig (1-2)', 'Neutraal (3-4)', 'Ongewenst (5)'],
                    include_lowest=True,
                )
                final_df['krw_match_status'] = np.where(final_df['krw_score'].notna(), 'Match', 'Geen match')
                final_df['krw_class_weergave'] = final_df['krw_class'].astype(object)
                final_df.loc[final_df['krw_class_weergave'].isna(), 'krw_class_weergave'] = 'Geen match'
                final_df['soort_display'] = np.where(
                    final_df['soort_triviaal'].notna() & (final_df['soort_triviaal'].astype(str).str.len() > 0),
                    final_df['soort_triviaal'] + ' (' + final_df['soort'] + ')',
                    final_df['soort'],
                )
                cols_to_keep = [
                    'datum', 'jaar', 'locatie_id', 'Waterlichaam', 'Project', 'CollectieReferentie',
                    'soort', 'bedekking_pct', 'waarde_bedekking', 'totaal_bedekking_locatie',
                    'diepte_m', 'doorzicht_m', 'lat', 'lon', 'x_rd', 'y_rd',
                    'groeivorm', 'type', 'Grootheid', 'soort_triviaal', 'trofisch_niveau',
                    'trofisch_niveau_weergave', 'trofisch_niveau_match_status',
                    'krw_watertype', 'krw_score', 'krw_class', 'krw_class_weergave', 'krw_match_status', 'soort_display',
                ]
                for col in cols_to_keep:
                    if col not in final_df.columns:
                        final_df[col] = np.nan
                final_df = final_df[cols_to_keep]
                final_df = _ensure_match_display_columns(final_df)
                try:
                    _write_parquet_from_pandas(final_df, FINAL_PARQUET)
                except Exception:
                    pass
                return final_df

    if source_is_api:
        if FINAL_PARQUET.exists():
            try:
                return _ensure_match_display_columns(_read_parquet_to_pandas(FINAL_PARQUET))
            except Exception:
                pass
        st.warning('⚠️ DD-ECO API bron kon niet worden omgezet naar parquet; geen CSV-fallback uitgevoerd binnen data_access.py.')
        return pd.DataFrame()

    # pandas fallback
    try:
        df_raw = pd.read_csv(FILE_PATH, sep=';', engine='python', encoding='utf-8-sig')
    except Exception:
        df_raw = pd.read_csv(FILE_PATH, sep=None, engine='python', encoding='utf-8-sig')

    df_raw.columns = df_raw.columns.str.strip()
    if 'MetingDatumTijd' in df_raw.columns:
        df_raw['MetingDatumTijd'] = pd.to_datetime(df_raw['MetingDatumTijd'], dayfirst=True, errors='coerce')
        df_raw['datum'] = df_raw['MetingDatumTijd'].dt.floor('D')
        df_raw['jaar'] = df_raw['datum'].dt.year
    else:
        st.error("Kolom 'MetingDatumTijd' mist.")
        return pd.DataFrame()

    df_raw['Project'] = df_raw['Projecten'].map(PROJECT_MAPPING).fillna(df_raw['Projecten'])
    df_raw['Waterlichaam'] = df_raw['MeetObject'].apply(determine_waterbody)

    df_abiotic = df_raw[df_raw['Grootheid'].isin(['DIEPTE', 'ZICHT'])].copy()
    if not df_abiotic.empty:
        df_env = df_abiotic.pivot_table(index='CollectieReferentie', columns='Grootheid', values='WaardeGemeten', aggfunc='mean').reset_index()
    else:
        df_env = pd.DataFrame(columns=['CollectieReferentie'])
    df_env['diepte_m'] = pd.to_numeric(df_env.get('DIEPTE'), errors='coerce') / 100.0
    df_env['doorzicht_m'] = pd.to_numeric(df_env.get('ZICHT'), errors='coerce') / 10.0

    df_total = df_raw[df_raw['Parameter'] == 'WATPTN'].copy()
    df_total = df_total[['CollectieReferentie', 'WaardeGemeten']].rename(columns={'WaardeGemeten': 'totaal_bedekking_locatie'})
    df_total['totaal_bedekking_locatie'] = pd.to_numeric(df_total['totaal_bedekking_locatie'], errors='coerce')
    df_total = df_total.groupby('CollectieReferentie', as_index=False).mean()

    df_bedkg = df_raw[(df_raw['Grootheid'].isin(['BEDKG', 'AANWZHD'])) & (df_raw['Parameter'] != 'WATPTN')].copy()
    def classify_row(row):
        param = row['Parameter']
        grootheid = row['Grootheid']
        if grootheid == 'AANWZHD':
            return 'Kenmerkende soort (N2000)', 'Soort'
        if param in GROWTH_FORM_MAPPING:
            return GROWTH_FORM_MAPPING[param], 'Groep'
        return 'Individuele soort', 'Soort'

    classificatie = df_bedkg.apply(classify_row, axis=1)
    df_bedkg['groeivorm'] = [x[0] for x in classificatie]
    df_bedkg['type'] = [x[1] for x in classificatie]
    df_merged = pd.merge(df_bedkg, df_env[['CollectieReferentie', 'diepte_m', 'doorzicht_m']], on='CollectieReferentie', how='left')
    df_merged = pd.merge(df_merged, df_total, on='CollectieReferentie', how='left')

    final_df = df_merged.rename(columns={
        'MeetObject': 'locatie_id',
        'Parameter': 'soort',
        'WaardeGemeten': 'waarde_bedekking',
        'EenheidGemeten': 'eenheid',
    })
    final_df['bedekking_pct'] = final_df['waarde_bedekking']
    final_df = _apply_coordinates_cached(final_df, epsg_col='GeografieDatum', wkt_col='GeografieVorm')
    lookup = load_species_lookup()
    final_df['soort_norm'] = final_df['soort'].fillna('').astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    final_df = final_df.merge(lookup, on='soort_norm', how='left')
    final_df = final_df.rename(columns={'NL naam': 'soort_triviaal', 'Watertype': 'trofisch_niveau'})
    final_df['trofisch_niveau_match_status'] = np.where(
        final_df['trofisch_niveau'].notna() & (final_df['trofisch_niveau'].astype(str).str.strip() != ''),
        'Match', 'Geen match'
    )
    final_df['trofisch_niveau_weergave'] = np.where(
        final_df['trofisch_niveau_match_status'] == 'Match',
        final_df['trofisch_niveau'].astype(str),
        'Geen match',
    )
    final_df['krw_watertype'] = final_df['Waterlichaam'].map(KRW_WATERTYPE_BY_WATERLICHAAM)
    final_df['krw_score'] = np.nan
    mask_m14 = final_df['krw_watertype'] == 'M14'
    mask_m21 = final_df['krw_watertype'] == 'M21'
    final_df.loc[mask_m14, 'krw_score'] = final_df.loc[mask_m14, 'M14']
    final_df.loc[mask_m21, 'krw_score'] = final_df.loc[mask_m21, 'M21']
    final_df['krw_class'] = pd.cut(
        final_df['krw_score'],
        bins=[0, 2, 4, 5],
        labels=['Gunstig (1-2)', 'Neutraal (3-4)', 'Ongewenst (5)'],
        include_lowest=True,
    )
    final_df['krw_match_status'] = np.where(final_df['krw_score'].notna(), 'Match', 'Geen match')
    final_df['krw_class_weergave'] = final_df['krw_class'].astype(object)
    final_df.loc[final_df['krw_class_weergave'].isna(), 'krw_class_weergave'] = 'Geen match'
    final_df['soort_display'] = np.where(
        final_df['soort_triviaal'].notna() & (final_df['soort_triviaal'].astype(str).str.len() > 0),
        final_df['soort_triviaal'] + ' (' + final_df['soort'] + ')',
        final_df['soort'],
    )
    cols_to_keep = [
        'datum', 'jaar', 'locatie_id', 'Waterlichaam', 'Project', 'CollectieReferentie',
        'soort', 'bedekking_pct', 'waarde_bedekking', 'totaal_bedekking_locatie',
        'diepte_m', 'doorzicht_m', 'lat', 'lon', 'x_rd', 'y_rd',
        'groeivorm', 'type', 'Grootheid', 'soort_triviaal', 'trofisch_niveau',
        'trofisch_niveau_weergave', 'trofisch_niveau_match_status',
        'krw_watertype', 'krw_score', 'krw_class', 'krw_class_weergave', 'krw_match_status', 'soort_display',
    ]
    for col in cols_to_keep:
        if col not in final_df.columns:
            final_df[col] = np.nan
    final_df = final_df[cols_to_keep]
    final_df = _ensure_match_display_columns(final_df)
    try:
        _write_parquet_from_pandas(final_df, FINAL_PARQUET)
    except Exception:
        pass
    return final_df


@st.cache_data
def load_data() -> pd.DataFrame:
    sig = _file_signature()
    return _load_data_cached(sig)


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in_clause(values: tuple[str, ...] | list[str]) -> str:
    vals = [v for v in values if str(v) != '']
    if not vals:
        return "('')"
    return '(' + ', '.join(_sql_quote(v) for v in vals) + ')'


@st.cache_data(show_spinner=False)
def load_filtered_ecology_base(projects: tuple[str, ...] = tuple(), bodies: tuple[str, ...] = tuple()) -> pd.DataFrame:
    """Laad direct de gefilterde ecologische basisset, bij voorkeur via DuckDB op parquet."""
    con = _get_duckdb()
    if FINAL_PARQUET.exists() and con is not None:
        where_parts = []
        if projects:
            where_parts.append(f"Project IN {_sql_in_clause(projects)}")
        if bodies:
            where_parts.append(f"Waterlichaam IN {_sql_in_clause(bodies)}")
        where_sql = ('WHERE ' + ' AND '.join(where_parts)) if where_parts else ''
        sql = f"SELECT * FROM read_parquet('{FINAL_PARQUET.as_posix()}') {where_sql}"
        try:
            df = con.execute(sql).fetch_df()
            if df is not None:
                return _ensure_match_display_columns(df)
        except Exception:
            pass

    df = load_data()
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if projects:
        out = out[out['Project'].isin(projects)].copy()
    if bodies:
        out = out[out['Waterlichaam'].isin(bodies)].copy()
    return out


@st.cache_data(show_spinner=False)
def get_bubble_yearly_filtered(projects: tuple[str, ...] = tuple(), bodies: tuple[str, ...] = tuple()) -> pd.DataFrame:
    """Geaggregeerde bubble-brondata per soort x jaar, bij voorkeur direct via DuckDB."""
    con = _get_duckdb()
    if FINAL_PARQUET.exists() and con is not None:
        where_parts = ["type = 'Soort'"]
        if projects:
            where_parts.append(f"Project IN {_sql_in_clause(projects)}")
        if bodies:
            where_parts.append(f"Waterlichaam IN {_sql_in_clause(bodies)}")
        where_sql = ' AND '.join(where_parts)
        sql = f"""
        SELECT
            soort,
            CAST(jaar AS INTEGER) AS jaar,
            AVG(TRY_CAST(doorzicht_m AS DOUBLE)) AS doorzicht_m,
            AVG(TRY_CAST(bedekking_pct AS DOUBLE)) AS bedekking_pct,
            AVG(TRY_CAST(diepte_m AS DOUBLE)) AS diepte_m
        FROM read_parquet('{FINAL_PARQUET.as_posix()}')
        WHERE {where_sql}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
        try:
            df = con.execute(sql).fetch_df()
            if df is not None:
                return df
        except Exception:
            pass

    df_base = load_filtered_ecology_base(projects, bodies)
    if df_base.empty:
        return pd.DataFrame(columns=['soort', 'jaar', 'doorzicht_m', 'bedekking_pct', 'diepte_m'])
    df_s = df_base[df_base['type'] == 'Soort'].copy()
    for col in ['doorzicht_m', 'diepte_m', 'bedekking_pct']:
        if col in df_s.columns:
            df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
    if df_s.empty:
        return pd.DataFrame(columns=['soort', 'jaar', 'doorzicht_m', 'bedekking_pct', 'diepte_m'])
    return (
        df_s.groupby(['soort', 'jaar'], as_index=False)
        .agg(
            doorzicht_m=('doorzicht_m', 'mean'),
            bedekking_pct=('bedekking_pct', 'mean'),
            diepte_m=('diepte_m', 'mean'),
        )
    )


def _add_time_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        out[date_col] = pd.NaT
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    out['maand'] = out[date_col].dt.month
    out['jaar'] = pd.to_numeric(out.get('jaar'), errors='coerce')
    out.loc[out['jaar'].isna(), 'jaar'] = out.loc[out['jaar'].isna(), date_col].dt.year
    return out


@st.cache_data(show_spinner=False)
def load_ecology_timeseries_data_filtered(project_sel: tuple[str, ...] = tuple(), body_sel: tuple[str, ...] = tuple()) -> pd.DataFrame:
    """Laad ecologische tijdreeksdata, maar filter zo vroeg mogelijk via DuckDB/parquet."""
    base = load_filtered_ecology_base(project_sel, body_sel)
    if base is None or base.empty:
        return pd.DataFrame()
    out = _ensure_match_display_columns(base.copy())
    out = _add_time_columns(out, 'datum')
    out['bedekking_pct'] = pd.to_numeric(out.get('bedekking_pct'), errors='coerce')
    out['totaal_bedekking_locatie'] = pd.to_numeric(out.get('totaal_bedekking_locatie'), errors='coerce')
    out['krw_score'] = pd.to_numeric(out.get('krw_score'), errors='coerce')
    out['__row_id__'] = np.arange(len(out))
    mask_species = out.get('type', pd.Series(index=out.index, dtype='object')).astype(str).eq('Soort')
    species = out.loc[mask_species].copy()
    if not species.empty:
        species = add_species_group_columns(species)
        add_cols = [
            c for c in [
                '__row_id__', 'soortgroep', 'soortgroep_weergave', 'soortgroep_match_status',
                'is_kenmerkende_soort_n2000', 'kenmerkende_soort_n2000_weergave',
                'kenmerkende_soort_n2000_match_status', 'bedekkingsgraad_proc',
            ] if c in species.columns
        ]
        if add_cols:
            out = out.merge(species[add_cols], on='__row_id__', how='left', suffixes=('', '_new'))
            for col in [c for c in add_cols if c != '__row_id__']:
                new_col = f'{col}_new'
                if new_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].where(out[col].notna(), out[new_col])
                        out = out.drop(columns=[new_col])
                    else:
                        out = out.rename(columns={new_col: col})
    defaults = {
        'soortgroep': 'Overig / Individueel',
        'soortgroep_weergave': 'Geen match',
        'soortgroep_match_status': 'Geen match',
        'kenmerkende_soort_n2000_weergave': 'Geen match',
        'kenmerkende_soort_n2000_match_status': 'Geen match',
        'bedekkingsgraad_proc': 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    if 'is_kenmerkende_soort_n2000' not in out.columns:
        out['is_kenmerkende_soort_n2000'] = False
    out['is_kenmerkende_soort_n2000'] = out['is_kenmerkende_soort_n2000'].fillna(False).astype(bool)
    out = out.drop(columns=['__row_id__'], errors='ignore')
    return out


@st.cache_data(show_spinner=False)
def load_ecology_timeseries_data() -> pd.DataFrame:
    """Laad ecologische data voor tijdreekskoppelingen zonder groeivormen te verliezen."""
    df = load_data()
    if df is None or df.empty:
        return pd.DataFrame()
    out = _ensure_match_display_columns(df.copy())
    out = _add_time_columns(out, 'datum')
    out['bedekking_pct'] = pd.to_numeric(out.get('bedekking_pct'), errors='coerce')
    out['totaal_bedekking_locatie'] = pd.to_numeric(out.get('totaal_bedekking_locatie'), errors='coerce')
    out['krw_score'] = pd.to_numeric(out.get('krw_score'), errors='coerce')
    out['__row_id__'] = np.arange(len(out))
    mask_species = out.get('type', pd.Series(index=out.index, dtype='object')).astype(str).eq('Soort')
    species = out.loc[mask_species].copy()
    if not species.empty:
        species = add_species_group_columns(species)
        add_cols = [
            c for c in [
                '__row_id__', 'soortgroep', 'soortgroep_weergave', 'soortgroep_match_status',
                'is_kenmerkende_soort_n2000', 'kenmerkende_soort_n2000_weergave',
                'kenmerkende_soort_n2000_match_status', 'bedekkingsgraad_proc',
            ] if c in species.columns
        ]
        if add_cols:
            out = out.merge(species[add_cols], on='__row_id__', how='left', suffixes=('', '_new'))
            for col in [c for c in add_cols if c != '__row_id__']:
                new_col = f'{col}_new'
                if new_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].where(out[col].notna(), out[new_col])
                        out = out.drop(columns=[new_col])
                    else:
                        out = out.rename(columns={new_col: col})
    defaults = {
        'soortgroep': 'Overig / Individueel',
        'soortgroep_weergave': 'Geen match',
        'soortgroep_match_status': 'Geen match',
        'kenmerkende_soort_n2000_weergave': 'Geen match',
        'kenmerkende_soort_n2000_match_status': 'Geen match',
        'bedekkingsgraad_proc': 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    if 'is_kenmerkende_soort_n2000' not in out.columns:
        out['is_kenmerkende_soort_n2000'] = False
    out['is_kenmerkende_soort_n2000'] = out['is_kenmerkende_soort_n2000'].fillna(False).astype(bool)
    out = out.drop(columns=['__row_id__'], errors='ignore')
    return out




@st.cache_data(show_spinner=False)
def load_chemistry_data() -> pd.DataFrame:
    """Laad chemiedata uit cache/parquet of CSV-bronbestand."""
    csv_path = Path(CHEMISTRY_FILE_PATH)
    if CHEMISTRY_PARQUET.exists() and _mtime_or_zero(CHEMISTRY_PARQUET) >= _mtime_or_zero(csv_path):
        try:
            return _read_parquet_to_pandas(CHEMISTRY_PARQUET)
        except Exception:
            pass
    if not csv_path.exists():
        return pd.DataFrame()
    attempts = [
        {'sep': ';', 'engine': 'python', 'encoding': 'utf-8-sig'},
        {'sep': ',', 'engine': 'python', 'encoding': 'utf-8-sig'},
        {'sep': None, 'engine': 'python', 'encoding': 'utf-8-sig'},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            break
        except Exception as exc:
            last_exc = exc
    else:
        st.warning(f'⚠️ Chemiebestand kon niet worden ingelezen ({CHEMISTRY_FILE_PATH}): {last_exc}')
        return pd.DataFrame()
    df.columns = df.columns.str.strip()
    for col in ['datum', 'meetdatum', 'MetingDatumTijd', 'sample_date', 'date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    try:
        _write_parquet_from_pandas(df, CHEMISTRY_PARQUET)
    except Exception:
        pass
    return df

__all__ = [
    'COORD_CACHE_PARQUET',
    'FILE_PATH',
    'FINAL_PARQUET',
    'LOOKUP_PARQUET',
    'MEAS_PARQUET',
    'PIPELINE_VERSION',
    'SPECIES_LOOKUP_PATH',
    'to_positive_number',
    'normalize_locatie_key',
    'load_species_lookup',
    '_ensure_match_display_columns',
    '_file_signature',
    '_load_data_cached',
    'load_data',
    'load_chemistry_data',
    'load_filtered_ecology_base',
    'get_bubble_yearly_filtered',
    'load_ecology_timeseries_data_filtered',
    'load_ecology_timeseries_data',
]
