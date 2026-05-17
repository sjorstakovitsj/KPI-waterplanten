from __future__ import annotations

"""Definitieve chemie-/ecologie-helperlaag.

Deze module is de enige bron voor:
- load_chemistry_data()
- _load_chemistry_data_cached()
- _read_chemistry_raw() en de csv/duckdb parsinghelpers
- get_chemistry_location_points()
- get_preferred_chemistry_locations()
- get_available_chemistry_locations()
- get_available_chemistry_parameter_labels()
- aggregate_chemistry_yearly()
- summarize_chemistry_period_average()
- aggregate_ecology_yearly()
- get_chem_ecology_timeseries()

De module importeert bewust niet uit services of pipelines.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from waterplanten_app.config.settings import CHEMISTRY_FILE_PATH, CHEMISTRY_PARQUET, CHEMISTRY_PIPELINE_VERSION
from waterplanten_app.config.mappings import (
    CHEM_LOCATION_PREFERENCES,
    CHEM_MARKER_COLOR,
    CHEM_PARAM_SUGGESTIONS,
    SEASON_MONTH_MAP,
    SEASON_ORDER,
)
from waterplanten_app.core.duckdb_runtime import _get_duckdb, _mtime_or_zero, _read_parquet_to_pandas, _write_parquet_from_pandas


def _normalize_season_value(value: str) -> str:
    lookup = {
        'voorjaar': 'Voorjaar',
        'lente': 'Voorjaar',
        'zomer': 'Zomer',
        'najaar': 'Najaar',
        'herfst': 'Najaar',
        'winter': 'Winter',
    }
    key = str(value or '').strip().lower()
    return lookup.get(key, str(value).strip())


def _normalize_seasons(seasons: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not seasons:
        return tuple()
    requested = {_normalize_season_value(x) for x in seasons if str(x).strip()}
    return tuple([s for s in SEASON_ORDER if s in requested])


def _add_time_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        out[date_col] = pd.NaT
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    out['maand'] = out[date_col].dt.month
    out['jaar'] = pd.to_numeric(out.get('jaar'), errors='coerce')
    out.loc[out['jaar'].isna(), 'jaar'] = out.loc[out['jaar'].isna(), date_col].dt.year
    out['seizoen'] = out['maand'].map(SEASON_MONTH_MAP)
    return out


def _filter_seasons(df: pd.DataFrame, seasons: tuple[str, ...] | list[str] | None) -> pd.DataFrame:
    normalized = _normalize_seasons(seasons)
    if not normalized:
        return df.copy()
    if 'seizoen' not in df.columns:
        return df.iloc[0:0].copy()
    return df[df['seizoen'].isin(normalized)].copy()


def _chemistry_file_signature(path: str = CHEMISTRY_FILE_PATH) -> tuple[str, float, str]:
    csv_path = Path(path)
    return (
        str(csv_path.resolve()) if csv_path.exists() else str(csv_path),
        _mtime_or_zero(csv_path),
        CHEMISTRY_PIPELINE_VERSION,
    )


def _detect_csv_delimiter(path: Path, default: str = ',') -> str:
    if not path.exists():
        return default
    try:
        with path.open('r', encoding='utf-8-sig', errors='ignore', newline='') as f:
            sample = f.read(65536)
            if not sample:
                return default
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;	|')
                return getattr(dialect, 'delimiter', default) or default
            except Exception:
                counts = {sep: sample.count(sep) for sep in [';', ',', '	', '|']}
                return max(counts, key=counts.get) if counts else default
    except Exception:
        return default


def _read_chemistry_csv_pandas(path: Path, usecols_func, sep: str, chunksize: int | None = None) -> pd.DataFrame:
    dtype_map = {
        'locatie_code': 'string',
        'locatie_lat_etrs89': 'string',
        'locatie_lon_etrs89': 'string',
        'parameter_code': 'string',
        'parameter_omschrijving': 'string',
        'hoedanigheid_code': 'string',
        'eenheid_code': 'string',
        'eenheid_omschrijving': 'string',
        'status_waarde': 'string',
        'eventdatum': 'string',
        'event_datum': 'string',
        'event_waarde': 'string',
        'event_waarde_tekst': 'string',
        'event_waarde_limietsymbool': 'string',
    }
    kwargs = dict(
        sep=sep,
        encoding='utf-8-sig',
        low_memory=True,
        usecols=usecols_func,
        dtype=dtype_map,
        on_bad_lines='skip',
    )
    if chunksize is not None:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _read_chemistry_csv_duckdb(path: Path, preferred_cols: list[str]) -> pd.DataFrame:
    con = _get_duckdb()
    if con is None or not path.exists():
        return pd.DataFrame()
    select_parts = [f'"{c}"' for c in preferred_cols]
    select_sql = ', '.join(select_parts)
    safe_path = path.as_posix().replace("'", "''")
    for delim in [None, ';', ',', '	', '|']:
        if delim is None:
            sql = f"""
            SELECT {select_sql}
            FROM read_csv_auto(
                '{safe_path}',
                SAMPLE_SIZE=-1,
                IGNORE_ERRORS=TRUE,
                ALL_VARCHAR=TRUE
            )
            """
        else:
            safe_delim = delim.replace("'", "''")
            sql = f"""
            SELECT {select_sql}
            FROM read_csv_auto(
                '{safe_path}',
                delim='{safe_delim}',
                SAMPLE_SIZE=-1,
                IGNORE_ERRORS=TRUE,
                ALL_VARCHAR=TRUE
            )
            """
        try:
            df = con.execute(sql).fetch_df()
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _read_chemistry_raw(path: str, required: list[str]) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    preferred_cols = list(dict.fromkeys(required))
    if 'hoedanigheid_code' not in preferred_cols:
        preferred_cols.append('hoedanigheid_code')
    usecols_func = lambda c: str(c).strip() in set(preferred_cols)

    delimiters = []
    detected = _detect_csv_delimiter(csv_path, default=',')
    for sep in [detected, ';', ',', '	', '|']:
        if sep not in delimiters:
            delimiters.append(sep)

    for sep in delimiters:
        try:
            df = _read_chemistry_csv_pandas(csv_path, usecols_func=usecols_func, sep=sep, chunksize=None)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    for sep in delimiters:
        try:
            df = _read_chemistry_csv_pandas(csv_path, usecols_func=usecols_func, sep=sep, chunksize=250_000)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    try:
        df = _read_chemistry_csv_duckdb(csv_path, preferred_cols=preferred_cols)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    for sep in delimiters:
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding='utf-8-sig', low_memory=False, on_bad_lines='skip')
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _clean_chem_numeric_strings(series: pd.Series) -> pd.Series:
    return (
        series.fillna('')
        .astype(str)
        .str.strip()
        .str.replace(',', '.', regex=False)
        .str.replace('<', '', regex=False)
        .str.replace('>', '', regex=False)
        .str.replace(r'^\+', '', regex=True)
        .str.strip()
    )


@st.cache_data(show_spinner=False)
def _load_chemistry_data_cached(sig: tuple[str, float, str], path: str = CHEMISTRY_FILE_PATH) -> pd.DataFrame:
    csv_path = Path(path)
    if CHEMISTRY_PARQUET.exists() and _mtime_or_zero(CHEMISTRY_PARQUET) >= _mtime_or_zero(csv_path):
        try:
            cached = _read_parquet_to_pandas(CHEMISTRY_PARQUET)
            if cached is not None and not cached.empty:
                return cached
        except Exception:
            pass

    required = [
        'locatie_code', 'locatie_lat_etrs89', 'locatie_lon_etrs89',
        'parameter_code', 'parameter_omschrijving', 'hoedanigheid_code',
        'eenheid_code', 'eenheid_omschrijving', 'status_waarde',
        'eventdatum', 'event_datum', 'event_waarde', 'event_waarde_tekst',
        'event_waarde_limietsymbool',
    ]
    df = _read_chemistry_raw(path, required=required)
    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    if 'event_datum' in df.columns and 'eventdatum' not in df.columns:
        df = df.rename(columns={'event_datum': 'eventdatum'})

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = _add_time_columns(df, 'eventdatum')

    raw_num = pd.to_numeric(_clean_chem_numeric_strings(df.get('event_waarde', pd.Series(np.nan, index=df.index))), errors='coerce')
    text_num = pd.to_numeric(_clean_chem_numeric_strings(df.get('event_waarde_tekst', pd.Series(np.nan, index=df.index))), errors='coerce')
    df['event_waarde_num'] = text_num.combine_first(raw_num)
    df['event_waarde_conflict'] = raw_num.notna() & text_num.notna() & ((raw_num - text_num).abs() > (text_num.abs() * 0.01 + 1e-12))
    df['event_waarde_bron'] = np.where(text_num.notna(), 'event_waarde_tekst', 'event_waarde')
    df.loc[df['event_waarde_num'] >= 1e10, 'event_waarde_num'] = np.nan

    df['meetlocatie_naam'] = df['locatie_code'].fillna('').astype(str).str.strip()
    df['x_coord'] = pd.to_numeric(df['locatie_lat_etrs89'], errors='coerce')
    df['y_coord'] = pd.to_numeric(df['locatie_lon_etrs89'], errors='coerce')
    df['resultaat'] = df['event_waarde_num']
    df['datum_bemonstering'] = df['eventdatum']
    df['eenheid'] = df['eenheid_code'].fillna('').astype(str).str.strip()

    parameter_code = df['parameter_code'].fillna('').astype(str).str.strip()
    parameter_omschrijving = df['parameter_omschrijving'].fillna('').astype(str).str.strip()
    hoedanigheid_code = df['hoedanigheid_code'].fillna('').astype(str).str.strip()
    eenheid_code = df['eenheid_code'].fillna('').astype(str).str.strip()

    stofnaam = np.where(parameter_omschrijving != '', parameter_omschrijving, parameter_code)
    mask_nvt = parameter_code.str.upper().eq('NVT')
    stofnaam = np.where(mask_nvt & eenheid_code.eq('mS/m'), 'geleidbaarheid', stofnaam)
    stofnaam = np.where(mask_nvt & hoedanigheid_code.eq('CaCO3'), 'hardheid', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('dm'), 'doorzicht', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('oC'), 'temperatuur', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('/m'), 'extinctie', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('DIMSLS') & df['event_waarde_num'].lt(3), 'saliniteit', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('DIMSLS') & df['event_waarde_num'].gt(5), 'zuurgraad', stofnaam)

    df['stofnaam'] = pd.Series(stofnaam, index=df.index).astype(str).str.strip()
    df.loc[df['stofnaam'].eq('') & parameter_code.ne(''), 'stofnaam'] = parameter_code[parameter_code.ne('')]
    df.loc[df['stofnaam'].eq(''), 'stofnaam'] = 'Onbekend'

    df['parameter_omschrijving'] = df['stofnaam']
    df['parameter_code'] = parameter_code
    df['eenheid_code'] = df['eenheid_code'].fillna('').astype(str).str.strip()
    df['eenheid_omschrijving'] = df['eenheid_omschrijving'].fillna('').astype(str).str.strip()
    df['eenheid_label'] = np.where(df['eenheid_omschrijving'] != '', df['eenheid_omschrijving'], df['eenheid_code'])

    base_label = np.where(
        df['stofnaam'].astype(str).str.strip() != '',
        df['parameter_code'].astype(str).str.strip() + ' — ' + df['stofnaam'].astype(str).str.strip(),
        df['parameter_code'].astype(str).str.strip(),
    )
    df['chem_label'] = np.where(
        df['eenheid_label'].astype(str).str.strip() != '',
        pd.Series(base_label, index=df.index).astype(str) + ' (' + df['eenheid_label'].astype(str).str.strip() + ')',
        pd.Series(base_label, index=df.index).astype(str),
    )

    mask_nvt_label = df['parameter_code'].str.upper().eq('NVT')
    df.loc[mask_nvt_label, 'chem_label'] = np.where(
        df.loc[mask_nvt_label, 'eenheid_label'].astype(str).str.strip() != '',
        df.loc[mask_nvt_label, 'stofnaam'].astype(str).str.strip() + ' (' + df.loc[mask_nvt_label, 'eenheid_label'].astype(str).str.strip() + ')',
        df.loc[mask_nvt_label, 'stofnaam'].astype(str).str.strip(),
    )

    keep_cols = [
        'locatie_code', 'locatie_lat_etrs89', 'locatie_lon_etrs89', 'meetlocatie_naam',
        'x_coord', 'y_coord', 'parameter_code', 'parameter_omschrijving', 'stofnaam',
        'hoedanigheid_code', 'eenheid_code', 'eenheid_omschrijving', 'eenheid_label', 'eenheid',
        'status_waarde', 'eventdatum', 'datum_bemonstering', 'event_waarde',
        'event_waarde_limietsymbool', 'event_waarde_num', 'event_waarde_conflict',
        'event_waarde_bron', 'resultaat', 'jaar', 'maand', 'seizoen', 'chem_label',
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep_cols].copy()

    try:
        _write_parquet_from_pandas(df, CHEMISTRY_PARQUET)
    except Exception:
        pass
    return df


@st.cache_data(show_spinner=False)
def load_chemistry_data(path: str = CHEMISTRY_FILE_PATH) -> pd.DataFrame:
    sig = _chemistry_file_signature(path)
    return _load_chemistry_data_cached(sig, path)


@st.cache_data(show_spinner=False)
def get_chemistry_location_points(
    body_sel: tuple[str, ...] | list[str] | None = None,
    preferred_only: bool = False,
    df_chem: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df_chem is None or df_chem.empty:
        df_chem = load_chemistry_data()
    if df_chem is None or df_chem.empty:
        return pd.DataFrame(columns=['locatie_code', 'meetlocatie_naam', 'chem_lat', 'chem_lon', 'n_records', 'n_parameters'])

    d = df_chem.copy()
    for col in ['locatie_code', 'meetlocatie_naam', 'x_coord', 'y_coord', 'parameter_code']:
        if col not in d.columns:
            d[col] = np.nan
    d['locatie_code'] = d['locatie_code'].fillna('').astype(str).str.strip()
    d['meetlocatie_naam'] = d['meetlocatie_naam'].fillna(d['locatie_code']).astype(str).str.strip()
    d['chem_lat'] = pd.to_numeric(d['x_coord'], errors='coerce')
    d['chem_lon'] = pd.to_numeric(d['y_coord'], errors='coerce')
    d = d[(d['locatie_code'] != '') & d['chem_lat'].notna() & d['chem_lon'].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=['locatie_code', 'meetlocatie_naam', 'chem_lat', 'chem_lon', 'n_records', 'n_parameters'])

    selected_bodies = [str(x).strip() for x in (body_sel or []) if str(x).strip()]
    if preferred_only and selected_bodies:
        preferred = []
        for body in selected_bodies:
            preferred.extend(CHEM_LOCATION_PREFERENCES.get(body, []))
        preferred = [x for x in preferred if str(x).strip()]
        if preferred:
            d = d[d['locatie_code'].isin(preferred)].copy()

    stats = (
        d.groupby('locatie_code', as_index=False)
        .agg(
            meetlocatie_naam=('meetlocatie_naam', 'first'),
            chem_lat=('chem_lat', 'first'),
            chem_lon=('chem_lon', 'first'),
            n_records=('locatie_code', 'size'),
            n_parameters=('parameter_code', lambda s: s.dropna().astype(str).str.strip().replace('', np.nan).dropna().nunique()),
        )
        .sort_values('locatie_code')
        .reset_index(drop=True)
    )
    return stats


def get_preferred_chemistry_locations(
    body_sel: tuple[str, ...] | list[str] | None,
    available_locations: list[str] | None,
) -> tuple[list[str], str | None, bool]:
    available = [str(x) for x in (available_locations or []) if str(x).strip()]
    if not available:
        return [], None, True

    selected_bodies = [str(x).strip() for x in (body_sel or []) if str(x).strip()]
    if not selected_bodies:
        default_loc = 'veluwemeer.midden' if 'veluwemeer.midden' in available else (available[0] if available else None)
        return available, default_loc, False

    if 'Randmeren' in selected_bodies:
        return available, None, True

    preferred: list[str] = []
    for body in selected_bodies:
        preferred.extend(CHEM_LOCATION_PREFERENCES.get(body, []))

    preferred_available: list[str] = []
    for loc in preferred:
        if loc in available and loc not in preferred_available:
            preferred_available.append(loc)

    if not preferred_available:
        default_loc = 'veluwemeer.midden' if 'veluwemeer.midden' in available else (available[0] if available else None)
        return available, default_loc, False

    if len(selected_bodies) == 1:
        return preferred_available, preferred_available[0], False
    return preferred_available, None, True


@st.cache_data(show_spinner=False)
def get_available_chemistry_locations(df_chem: pd.DataFrame | None = None) -> list[str]:
    if df_chem is None or df_chem.empty:
        df_chem = load_chemistry_data()
    if df_chem.empty or 'locatie_code' not in df_chem.columns:
        return []
    return sorted(df_chem['locatie_code'].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_available_chemistry_parameter_labels(
    df_chem: pd.DataFrame | None = None,
    suggestions: tuple[str, ...] = tuple(CHEM_PARAM_SUGGESTIONS),
) -> list[str]:
    if df_chem is None or df_chem.empty:
        df_chem = load_chemistry_data()
    if df_chem.empty:
        return []
    labels = (
        df_chem[['chem_label', 'parameter_code']]
        .dropna(subset=['chem_label'])
        .drop_duplicates()
        .sort_values(['parameter_code', 'chem_label'])
    )
    ordered: list[str] = []
    for code in suggestions:
        ordered.extend(labels.loc[labels['parameter_code'] == code, 'chem_label'].tolist())
    ordered.extend([x for x in labels['chem_label'].tolist() if x not in ordered])
    return ordered


@st.cache_data(show_spinner=False)
def aggregate_chemistry_yearly(
    df_chem: pd.DataFrame,
    chemistry_labels: tuple[str, ...],
    location: str | None = None,
    definitive_only: bool = False,
    seasons: tuple[str, ...] = tuple(),
) -> pd.DataFrame:
    if df_chem is None or df_chem.empty or not chemistry_labels:
        return pd.DataFrame(columns=['jaar', 'serie', 'chem_value', 'eenheid_code', 'eenheid_omschrijving', 'parameter_code'])
    d = df_chem.copy()
    d = d[d['chem_label'].isin(list(chemistry_labels))].copy()
    if location:
        d = d[d['locatie_code'].astype(str) == str(location)].copy()
    if definitive_only and 'status_waarde' in d.columns:
        d = d[d['status_waarde'].astype(str).str.lower() == 'definitief'].copy()
    d = _filter_seasons(d, seasons)
    d = d.dropna(subset=['jaar', 'event_waarde_num'])
    if d.empty:
        return pd.DataFrame(columns=['jaar', 'serie', 'chem_value', 'eenheid_code', 'eenheid_omschrijving', 'parameter_code'])

    group_cols = ['jaar', 'chem_label', 'parameter_code', 'eenheid_code', 'eenheid_omschrijving']
    for col in group_cols:
        if col not in d.columns:
            d[col] = ''
    out = (
        d.groupby(group_cols, as_index=False)
        .agg(chem_value=('event_waarde_num', 'mean'))
        .rename(columns={'chem_label': 'serie'})
    )
    out['jaar'] = pd.to_numeric(out['jaar'], errors='coerce').astype(int)
    return out[['jaar', 'serie', 'chem_value', 'eenheid_code', 'eenheid_omschrijving', 'parameter_code']]


def aggregate_ecology_yearly(
    df_eco: pd.DataFrame,
    project_sel: tuple[str, ...],
    body_sel: tuple[str, ...],
    metric: str,
    mode: str = 'default',
    seasons: tuple[str, ...] = tuple(),
    top_n: int | None = None,
) -> pd.DataFrame:
    if df_eco is None or df_eco.empty:
        return pd.DataFrame(columns=['jaar', 'serie', 'waarde'])
    d = df_eco.copy()
    if project_sel:
        d = d[d['Project'].isin(project_sel)].copy()
    if body_sel:
        d = d[d['Waterlichaam'].isin(body_sel)].copy()
    d = _filter_seasons(d, seasons)
    d = d.dropna(subset=['jaar'])
    if d.empty:
        return pd.DataFrame(columns=['jaar', 'serie', 'waarde'])
    d['jaar'] = pd.to_numeric(d['jaar'], errors='coerce')
    d = d.dropna(subset=['jaar'])
    d['jaar'] = d['jaar'].astype(int)

    if metric == 'Totale bedekking':
        x = d.dropna(subset=['totaal_bedekking_locatie']).copy()
        if x.empty:
            return pd.DataFrame(columns=['jaar', 'serie', 'waarde'])
        x = x.groupby(['jaar', 'CollectieReferentie'], as_index=False)['totaal_bedekking_locatie'].first()
        out = x.groupby('jaar', as_index=False)['totaal_bedekking_locatie'].mean()
        out = out.rename(columns={'totaal_bedekking_locatie': 'waarde'})
        out['serie'] = 'Totale bedekking'
        return out[['jaar', 'serie', 'waarde']]

    if metric == 'Groeivormen':
        x = d[d.get('type', '').astype(str) == 'Groep'].copy()
        x = x.dropna(subset=['groeivorm', 'bedekking_pct'])
        out = x.groupby(['jaar', 'groeivorm'], as_index=False)['bedekking_pct'].mean()
        out = out.rename(columns={'groeivorm': 'serie', 'bedekking_pct': 'waarde'})
    elif metric == 'Soortgroep':
        x = d[d.get('type', '').astype(str) == 'Soort'].copy()
        x['serie'] = x.get('soortgroep_weergave', pd.Series('Geen match', index=x.index)).fillna('Geen match').astype(str)
        x = x.dropna(subset=['bedekking_pct'])
        out = x.groupby(['jaar', 'serie'], as_index=False)['bedekking_pct'].mean().rename(columns={'bedekking_pct': 'waarde'})
    elif metric == 'Trofieniveau':
        x = d[d.get('type', '').astype(str) == 'Soort'].copy()
        serie = x.get('trofisch_niveau_weergave')
        if serie is None:
            serie = x.get('trofisch_niveau', pd.Series('Geen match', index=x.index))
        x['serie'] = serie.fillna('Geen match').astype(str)
        x = x.dropna(subset=['bedekking_pct'])
        out = x.groupby(['jaar', 'serie'], as_index=False)['bedekking_pct'].mean().rename(columns={'bedekking_pct': 'waarde'})
    elif metric == 'KRW score':
        x = d[d.get('type', '').astype(str) == 'Soort'].copy()
        if mode == 'index':
            out = x.groupby('jaar', as_index=False)['krw_score'].mean().rename(columns={'krw_score': 'waarde'})
            out['serie'] = 'Gemiddelde KRW-score'
        else:
            serie = x.get('krw_class_weergave')
            if serie is None:
                serie = x.get('krw_class', pd.Series('Geen match', index=x.index))
            x['serie'] = serie.fillna('Geen match').astype(str)
            out = x.groupby(['jaar', 'serie'], as_index=False)['bedekking_pct'].mean().rename(columns={'bedekking_pct': 'waarde'})
    elif metric == 'Kenmerkende soort (N2000)':
        x = d[d.get('is_kenmerkende_soort_n2000', pd.Series(False, index=d.index)).fillna(False)].copy()
        if x.empty:
            return pd.DataFrame(columns=['jaar', 'serie', 'waarde'])
        if mode == 'records':
            out = x.groupby('jaar', as_index=False).size().rename(columns={'size': 'waarde'})
            out['serie'] = 'N2000 aanwezigheidsrecords'
            return out[['jaar', 'serie', 'waarde']]
        serie = x.get('kenmerkende_soort_n2000_weergave')
        if serie is None:
            serie = x.get('soort_display', x.get('soort', pd.Series('N2000', index=x.index)))
        x['serie'] = serie.fillna('Geen match').astype(str)
        out = x.groupby(['jaar', 'serie'], as_index=False).size().rename(columns={'size': 'waarde'})
    else:
        return pd.DataFrame(columns=['jaar', 'serie', 'waarde'])

    if out.empty:
        return pd.DataFrame(columns=['jaar', 'serie', 'waarde'])
    if top_n is not None and 'serie' in out.columns and metric in {'Groeivormen', 'Soortgroep', 'Trofieniveau', 'Kenmerkende soort (N2000)'}:
        totals = out.groupby('serie', as_index=False)['waarde'].sum().sort_values('waarde', ascending=False)
        keep = totals.head(int(top_n))['serie'].tolist()
        out = out[out['serie'].isin(keep)].copy()
    return out[['jaar', 'serie', 'waarde']]


def summarize_chemistry_period_average(
    df_chem_year: pd.DataFrame,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    if df_chem_year is None or df_chem_year.empty:
        return pd.DataFrame(columns=['stof', 'gemiddelde_geselecteerde_periode', 'eenheid_omschrijving', 'aantal_jaren'])
    d = df_chem_year.copy()
    d['jaar'] = pd.to_numeric(d.get('jaar'), errors='coerce')
    d = d.dropna(subset=['jaar', 'chem_value'])
    if year_min is not None:
        d = d[d['jaar'] >= int(year_min)]
    if year_max is not None:
        d = d[d['jaar'] <= int(year_max)]
    if d.empty:
        return pd.DataFrame(columns=['stof', 'gemiddelde_geselecteerde_periode', 'eenheid_omschrijving', 'aantal_jaren'])
    out = (
        d.groupby('serie', as_index=False)
        .agg(
            gemiddelde_geselecteerde_periode=('chem_value', 'mean'),
            eenheid_omschrijving=('eenheid_omschrijving', lambda s: s.dropna().iloc[0] if len(s.dropna()) else ''),
            aantal_jaren=('jaar', 'nunique'),
        )
        .rename(columns={'serie': 'stof'})
        .sort_values('stof')
    )
    return out


def get_chem_ecology_timeseries(
    df_eco: pd.DataFrame,
    df_chem: pd.DataFrame,
    project_sel: tuple[str, ...],
    body_sel: tuple[str, ...],
    ecology_metric: str,
    chemistry_labels: tuple[str, ...],
    chemistry_location: str | None = None,
    ecology_mode: str = 'default',
    definitive_only: bool = False,
    seasons: tuple[str, ...] = tuple(),
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    eco_year = aggregate_ecology_yearly(
        df_eco=df_eco,
        project_sel=project_sel,
        body_sel=body_sel,
        metric=ecology_metric,
        mode=ecology_mode,
        seasons=tuple(),
        top_n=top_n,
    )
    chem_year = aggregate_chemistry_yearly(
        df_chem=df_chem,
        chemistry_labels=chemistry_labels,
        location=chemistry_location,
        definitive_only=definitive_only,
        seasons=seasons,
    )
    if eco_year.empty or chem_year.empty:
        return eco_year, chem_year, []
    common_years = sorted(
        set(pd.to_numeric(eco_year['jaar'], errors='coerce').dropna().astype(int))
        & set(pd.to_numeric(chem_year['jaar'], errors='coerce').dropna().astype(int))
    )
    if not common_years:
        return eco_year.iloc[0:0].copy(), chem_year.iloc[0:0].copy(), []
    eco_year = eco_year[eco_year['jaar'].isin(common_years)].copy()
    chem_year = chem_year[chem_year['jaar'].isin(common_years)].copy()
    return eco_year, chem_year, common_years


__all__ = [
    'CHEMISTRY_FILE_PATH',
    'CHEMISTRY_PARQUET',
    'CHEMISTRY_PIPELINE_VERSION',
    'CHEM_PARAM_SUGGESTIONS',
    'CHEM_LOCATION_PREFERENCES',
    'CHEM_MARKER_COLOR',
    'SEASON_ORDER',
    'SEASON_MONTH_MAP',
    'load_chemistry_data',
    '_load_chemistry_data_cached',
    '_read_chemistry_raw',
    'get_chemistry_location_points',
    'get_preferred_chemistry_locations',
    'get_available_chemistry_locations',
    'get_available_chemistry_parameter_labels',
    'aggregate_chemistry_yearly',
    'summarize_chemistry_period_average',
    'aggregate_ecology_yearly',
    'get_chem_ecology_timeseries',
]
