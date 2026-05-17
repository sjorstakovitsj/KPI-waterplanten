from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

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
    """Eén DuckDB connectie per Streamlit sessie."""
    if duckdb is None:
        return None
    con = duckdb.connect(database=':memory:')
    try:
        con.execute('PRAGMA threads=4;')
    except Exception:
        pass
    return con


def _mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _write_parquet_from_pandas(df: pd.DataFrame, path: Path) -> None:
    """Schrijf Parquet met pyarrow indien beschikbaar, anders pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if pq is not None and pa is not None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path.as_posix(), compression='zstd')
    else:
        df.to_parquet(path.as_posix(), index=False)


def _read_parquet_to_pandas(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if pq is not None:
        return pq.read_table(path.as_posix()).to_pandas()
    return pd.read_parquet(path.as_posix())


def _ensure_measurements_parquet(csv_path: Path | None = None) -> Path:
    """Bouw measurements.parquet vanuit de geconfigureerde bron.

    - csv: bestaand gedrag via DuckDB read_csv_auto(...)
    - dd_eco_api_v3: observations/references ophalen en naar raw-contract parquet schrijven
    - fallback: als API faalt en DATA_SOURCE_FALLBACK == 'csv', gebruik het bestaande CSV-pad
    """
    csv_path = csv_path or Path(FILE_PATH)

    def _ensure_from_csv(path: Path) -> Path:
        if not path.exists() or duckdb is None:
            return MEAS_PARQUET
        need_build = (not MEAS_PARQUET.exists()) or (_mtime_or_zero(path) > _mtime_or_zero(MEAS_PARQUET))
        if need_build:
            con = _get_duckdb()
            if con is not None:
                con.execute(f"""
                COPY (
                    SELECT * FROM read_csv_auto(
                        '{path.as_posix()}',
                        delim=';',
                        SAMPLE_SIZE=-1,
                        IGNORE_ERRORS=TRUE
                    )
                ) TO '{MEAS_PARQUET.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
        return MEAS_PARQUET

    if str(DATA_SOURCE).strip().lower() == 'dd_eco_api_v3':
        try:
            need_build = (
                (not MEAS_PARQUET.exists())
                or (_mtime_or_zero(DD_API_MEAS_PARQUET) > _mtime_or_zero(MEAS_PARQUET))
                or (_mtime_or_zero(DD_API_REFERENCES_PARQUET) > _mtime_or_zero(MEAS_PARQUET))
            )
            if need_build:
                return ensure_dd_eco_measurements_parquet(out_path=MEAS_PARQUET, refs_path=DD_API_REFERENCES_PARQUET)
            return MEAS_PARQUET
        except Exception:
            if str(DATA_SOURCE_FALLBACK or '').strip().lower() == 'csv':
                return _ensure_from_csv(csv_path)
            raise

    return _ensure_from_csv(csv_path)
