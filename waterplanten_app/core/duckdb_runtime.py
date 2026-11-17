from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from waterplanten_app.config import settings as app_settings
from waterplanten_app.config.settings import (
    DATA_SOURCE,
    DATA_SOURCE_FALLBACK,
    DD_API_MEAS_PARQUET,
    DD_API_REFERENCES_PARQUET,
    FILE_PATH,
    MEAS_PARQUET,
)
from waterplanten_app.core.dd_eco_api_ingest import ensure_dd_eco_measurements_parquet

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


@st.cache_resource
def _get_duckdb() -> Optional['duckdb.DuckDBPyConnection']:
    """Eén DuckDB-connectie per Streamlit-proces."""
    if duckdb is None:
        return None
    con = duckdb.connect(database=':memory:')
    try:
        con.execute('PRAGMA threads=4;')
    except Exception:
        pass
    return con


def preloaded_parquets_enabled() -> bool:
    """Geef aan of de applicatie uitsluitend vooraf gebouwde Parquets mag lezen.

    Voorkeursinstelling: PRELOADED_PARQUETS in config/settings.py. Voor deployment
    kan dezelfde instelling ook als omgevingsvariabele worden gezet.
    """
    value = os.getenv(
        'PRELOADED_PARQUETS',
        str(getattr(app_settings, 'PRELOADED_PARQUETS', False)),
    )
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def require_preloaded_parquet(path: Path, name: str | None = None) -> Path:
    """Controleer dat een verplicht preloadbestand bestaat en niet leeg is."""
    path = Path(path)
    label = name or path.name
    if not path.exists():
        raise FileNotFoundError(
            f'Preloadmodus is actief, maar {label} ontbreekt: {path.resolve()}. '
            'Bouw en commit de Parquets voordat de Streamlit-app wordt gestart.'
        )
    if path.stat().st_size <= 0:
        raise RuntimeError(f'Preloadbestand is leeg: {path.resolve()}')
    return path


def _mtime_or_zero(path: Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0


def _path_fingerprint(path: Path) -> tuple[str, int, int]:
    """Lichte cachesleutel: absoluut pad, wijzigingstijd in ns en grootte."""
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
        return (resolved.as_posix(), int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        return (resolved.as_posix(), 0, 0)


def _paths_fingerprint(*paths: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(_path_fingerprint(Path(path)) for path in paths)


def _write_parquet_from_pandas(df: pd.DataFrame, path: Path) -> None:
    """Schrijf Parquet atomair, zodat een mislukte build het oude bestand niet beschadigt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.tmp')
    tmp_path.unlink(missing_ok=True)
    try:
        if pq is not None and pa is not None:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, tmp_path.as_posix(), compression='zstd')
        else:
            df.to_parquet(tmp_path.as_posix(), index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_parquet_to_pandas(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if pq is not None:
        return pq.read_table(path.as_posix()).to_pandas()
    return pd.read_parquet(path.as_posix())


def _sql_path(path: Path) -> str:
    """Escape een bestandspad voor gebruik als DuckDB SQL-string."""
    return Path(path).as_posix().replace("'", "''")


def parquet_row_count(path: Path) -> int:
    """Lees alleen metadata/COUNT(*) voor een lichte buildvalidatie."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if duckdb is not None:
        return int(duckdb.sql(
            f"SELECT COUNT(*) FROM read_parquet('{_sql_path(path)}')"
        ).fetchone()[0])
    if pq is not None:
        return int(pq.ParquetFile(path.as_posix()).metadata.num_rows)
    return len(pd.read_parquet(path.as_posix()))


def build_measurements_parquet(
    csv_path: Path | None = None,
    parquet_path: Path | None = None,
    *,
    force: bool = False,
    ignore_errors: bool = False,
) -> Path:
    """Bouw measurements.parquet rechtstreeks uit CSV.

    Dit is de gedeelde, Streamlit-onafhankelijke bouwfunctie voor zowel het
    offline buildscript als de runtime-fallback. Er wordt eerst naar een
    tijdelijk bestand geschreven en pas na validatie atomair gepubliceerd.
    """
    if duckdb is None:
        raise RuntimeError('DuckDB is niet geïnstalleerd; measurements.parquet kan niet worden gebouwd.')

    source = Path(csv_path or FILE_PATH)
    target = Path(parquet_path or MEAS_PARQUET)
    if not source.exists():
        raise FileNotFoundError(f'CSV-bronbestand ontbreekt: {source.resolve()}')
    if not force and target.exists() and _mtime_or_zero(target) >= _mtime_or_zero(source):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f'.{target.name}.tmp')
    tmp_path.unlink(missing_ok=True)
    con = duckdb.connect(database=':memory:')
    try:
        con.execute('PRAGMA threads=4;')
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_csv_auto(
                    '{_sql_path(source)}',
                    delim=';',
                    SAMPLE_SIZE=-1,
                    IGNORE_ERRORS={'TRUE' if ignore_errors else 'FALSE'}
                )
            ) TO '{_sql_path(tmp_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        rows = int(con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{_sql_path(tmp_path)}')"
        ).fetchone()[0])
        if rows <= 0:
            raise RuntimeError(f'Het gebouwde Parquet bevat geen rijen: {tmp_path}')
        tmp_path.replace(target)
        return target
    finally:
        con.close()
        tmp_path.unlink(missing_ok=True)


def ensure_measurements_parquet(
    csv_path: Path | None = None,
    *,
    force: bool = False,
    ignore_errors: bool = True,
) -> Path:
    """Zorg dat het geconfigureerde measurements.parquet beschikbaar en actueel is.

    De runtime behoudt de bestaande API- en CSV-fallback. Het offline
    buildscript kan dezelfde functie met ``force=True`` aanroepen.
    """
    source = str(DATA_SOURCE).strip().lower()
    csv_path = Path(csv_path or FILE_PATH)

    # In preloadmodus mag de online app nooit CSV/API-data omzetten.
    # force=True blijft beschikbaar voor het expliciete offline buildscript.
    if preloaded_parquets_enabled() and not force:
        return require_preloaded_parquet(MEAS_PARQUET, 'measurements.parquet')

    if source == 'dd_eco_api_v3':
        try:
            needs_build = (
                force
                or not MEAS_PARQUET.exists()
                or _mtime_or_zero(DD_API_MEAS_PARQUET) > _mtime_or_zero(MEAS_PARQUET)
                or _mtime_or_zero(DD_API_REFERENCES_PARQUET) > _mtime_or_zero(MEAS_PARQUET)
            )
            if needs_build:
                return ensure_dd_eco_measurements_parquet(
                    out_path=MEAS_PARQUET,
                    refs_path=DD_API_REFERENCES_PARQUET,
                )
            return MEAS_PARQUET
        except Exception:
            if str(DATA_SOURCE_FALLBACK or '').strip().lower() != 'csv':
                raise

    return build_measurements_parquet(
        csv_path=csv_path,
        parquet_path=MEAS_PARQUET,
        force=force,
        ignore_errors=ignore_errors,
    )


# Achterwaartse compatibiliteit: bestaande imports in data_access.py blijven werken.
def _ensure_measurements_parquet(csv_path: Path | None = None) -> Path:
    return ensure_measurements_parquet(csv_path=csv_path)


__all__ = [
    'build_measurements_parquet',
    'ensure_measurements_parquet',
    'parquet_row_count',
    'preloaded_parquets_enabled',
    'require_preloaded_parquet',
    '_ensure_measurements_parquet',
    '_get_duckdb',
    '_mtime_or_zero',
    '_path_fingerprint',
    '_paths_fingerprint',
    '_read_parquet_to_pandas',
    '_write_parquet_from_pandas',
]
