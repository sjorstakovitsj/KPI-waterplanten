# utils.py
from __future__ import annotations

import re
import json
import math
from html import escape
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium

# Optionele high-performance dependencies
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


# =============================================================================
# CONFIGURATIE
# =============================================================================
FILE_PATH = "AquaDeskMeasurementExport_RWS_20260129204403.csv"
SPECIES_LOOKUP_PATH = "Koppeltabel_score_namen.csv"

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache_waterplanten"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MEAS_PARQUET = CACHE_DIR / "measurements.parquet"
FINAL_PARQUET = CACHE_DIR / "final_df.parquet"
LOOKUP_PARQUET = CACHE_DIR / "species_lookup.parquet"

# Nieuw: coord-cache op basis van (GeografieDatum, GeografieVorm)
COORD_CACHE_PARQUET = CACHE_DIR / "coord_cache.parquet"

# Verhoog dit als je pipeline-logica wijzigt (force rebuild via cache key)
PIPELINE_VERSION = "2026-02-20_duckdb_parquet_coords_v2"


# =============================================================================
# MAPPINGS (ongewijzigd)
# =============================================================================
PROJECT_MAPPING = {
    "MWTL_WOP": "KRW",
    "GRID_WOP": "N2000",
}

WATERBODY_MAPPING = {
    "DRNMR": "Drontermeer",
    "EEMMR": "Eemmeer",
    "GOOIM": "Gooimeer",
    "GOUWZ": "Gouwzee",
    "IJMR": "IJmeer",
    "IJSMR": "IJsselmeer",
    "KETMR": "Ketelmeer",
    "MRKMR": "Markermeer",
    "NIJKN": "Nijkerkernauw",
    "NULDN": "Nuldernauw",
    "RNDMRN": "Randmeren",
    "RNDMR": "Randmeren",
    "VELWM": "Veluwemeer",
    "VOSSM": "Vossemeer",
    "WOLDW": "Wolderwijd",
    "ZWTMR": "Zwartemeer",
}

GROWTH_FORM_MAPPING = {
    "DRAADAGN": "Draadalgen",
    "DRIJFBPTN": "Drijvend",
    "EMSPTN": "Emergent",
    "SUBMSPTN": "Ondergedoken",
    "FLAB": "FLAB",
    "KROOS": "Kroos",
}

EXCLUDED_SPECIES_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]
RWS_GROEIVORM_CODES = EXCLUDED_SPECIES_CODES

KRW_WATERTYPE_BY_WATERLICHAAM = {
    # M21
    "Markermeer": "M21",
    "Gouwzee": "M21",
    "IJmeer": "M21",
    "IJsselmeer": "M21",
    # M14
    "Drontermeer": "M14",
    "Eemmeer": "M14",
    "Gooimeer": "M14",
    "Ketelmeer": "M14",
    "Nijkerkernauw": "M14",
    "Nuldernauw": "M14",
    "Randmeren": "M14",
    "Veluwemeer": "M14",
    "Vossemeer": "M14",
    "Wolderwijd": "M14",
    "Zwartemeer": "M14",
}


# =============================================================================
# DUCKDB / PARQUET HELPERS
# =============================================================================
@st.cache_resource
def _get_duckdb() -> Optional["duckdb.DuckDBPyConnection"]:
    """Één DuckDB connectie per Streamlit sessie."""
    if duckdb is None:
        return None
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass
    return con


def _mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _write_parquet_from_pandas(df: pd.DataFrame, path: Path) -> None:
    """Schrijf Parquet met pyarrow indien beschikbaar (sneller), anders pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if pq is not None and pa is not None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path.as_posix(), compression="zstd")
    else:
        df.to_parquet(path.as_posix(), index=False)


def _read_parquet_to_pandas(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if pq is not None:
        return pq.read_table(path.as_posix()).to_pandas()
    return pd.read_parquet(path.as_posix())


def _ensure_measurements_parquet(csv_path: Path) -> Path:
    """
    Bouw measurements.parquet als die ontbreekt of als CSV nieuwer is.
    Gebruikt DuckDB COPY voor performance.
    """
    if not csv_path.exists():
        return MEAS_PARQUET

    if duckdb is None:
        return MEAS_PARQUET

    need_build = (not MEAS_PARQUET.exists()) or (_mtime_or_zero(csv_path) > _mtime_or_zero(MEAS_PARQUET))
    if need_build:
        con = _get_duckdb()
        if con is None:
            return MEAS_PARQUET

        # CSV is semicolon-delimited (zoals in je extract)
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    delim=';',
                    SAMPLE_SIZE=-1,
                    IGNORE_ERRORS=TRUE
                )
            )
            TO '{MEAS_PARQUET.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    return MEAS_PARQUET


# =============================================================================
# COORDINATEN
# =============================================================================
def rd_to_wgs84(x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
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


def _parse_wkt_point_numbers(wkt: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extraheer de eerste 2 getallen uit WKT string, bijv. 'POINT (144090 482193)'.
    """
    if wkt is None:
        return None, None
    s = str(wkt).strip()
    if not s:
        return None, None
    nums = _num_re.findall(s)
    if len(nums) < 2:
        return None, None
    try:
        a = float(nums[0].replace(",", "."))
        b = float(nums[1].replace(",", "."))
        return a, b
    except Exception:
        return None, None


def parse_coordinates(geo_str: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Backwards compatible wrapper (oude signature).
    Probeert RD te interpreteren.
    """
    a, b = _parse_wkt_point_numbers(geo_str)
    if a is None or b is None:
        return None, None, None, None
    lat, lon = rd_to_wgs84(a, b)
    return a, b, lat, lon


def _parse_coordinates_epsg(epsg: str, wkt: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    CRS-aware parsing:
      - EPSG:28992: RD meters -> RD->WGS84
      - EPSG:4258 : lon/lat degrees (ETRS89) -> lat/lon direct
      - anders: heuristiek (range-check) of None
    """
    a, b = _parse_wkt_point_numbers(wkt)
    if a is None or b is None:
        return None, None, None, None

    epsg = (epsg or "").strip()

    # EPSG:4258 = lon lat (graden) (in je volledige dataset aanwezig)
    if epsg == "EPSG:4258":
        lon = a
        lat = b
        return None, None, lat, lon

    # EPSG:28992 = RD x y (meters)
    if epsg == "EPSG:28992":
        x = a
        y = b
        lat, lon = rd_to_wgs84(x, y)
        return x, y, lat, lon

    # Fallback: range-based heuristiek
    # Lijkt op lon/lat (NL)
    if 3.0 <= a <= 8.2 and 50.0 <= b <= 54.8:
        return None, None, b, a

    # Lijkt op RD
    if 0 <= a <= 300000 and 250000 <= b <= 700000:
        lat, lon = rd_to_wgs84(a, b)
        return a, b, lat, lon

    return a, b, None, None


def _load_coord_cache() -> pd.DataFrame:
    """
    Parquet cache met kolommen:
      epsg, wkt, x_rd, y_rd, lat, lon
    """
    if COORD_CACHE_PARQUET.exists():
        try:
            df = _read_parquet_to_pandas(COORD_CACHE_PARQUET)
            # schema hardening
            for c in ["epsg", "wkt", "x_rd", "y_rd", "lat", "lon"]:
                if c not in df.columns:
                    df[c] = np.nan
            return df[["epsg", "wkt", "x_rd", "y_rd", "lat", "lon"]]
        except Exception:
            pass
    return pd.DataFrame(columns=["epsg", "wkt", "x_rd", "y_rd", "lat", "lon"])


def _save_coord_cache(df: pd.DataFrame) -> None:
    """
    Sla coord-cache op, dedupe op (epsg,wkt).
    """
    if df.empty:
        return
    df = df.drop_duplicates(subset=["epsg", "wkt"], keep="last")
    try:
        _write_parquet_from_pandas(df, COORD_CACHE_PARQUET)
    except Exception:
        pass


def _apply_coordinates_cached(df: pd.DataFrame, epsg_col: str = "GeografieDatum", wkt_col: str = "GeografieVorm") -> pd.DataFrame:
    """
    PERFORMANCE HOTSPOT FIX:
    - Parse coördinaten per unieke (GeografieDatum, GeografieVorm)
    - Cache dit persistent (Parquet) zodat volgende runs vrijwel niets parsen
    - Cache key is volledige geometriestring + EPSG (dus niet locatie_id)
    """
    df = df.copy()

    if epsg_col not in df.columns or wkt_col not in df.columns:
        df["x_rd"] = np.nan
        df["y_rd"] = np.nan
        df["lat"] = np.nan
        df["lon"] = np.nan
        return df

    epsg_series = df[epsg_col].fillna("").astype(str)
    wkt_series = df[wkt_col].fillna("").astype(str)

    # Unieke combinaties
    uniq = pd.DataFrame({"epsg": epsg_series, "wkt": wkt_series}).drop_duplicates()

    cache = _load_coord_cache()
    if not cache.empty:
        cache_idx = cache.set_index(["epsg", "wkt"])
    else:
        cache_idx = pd.DataFrame(columns=["x_rd", "y_rd", "lat", "lon"]).set_index(pd.MultiIndex.from_arrays([[], []], names=["epsg", "wkt"]))

    # Welke combos missen?
    uniq_idx = uniq.set_index(["epsg", "wkt"])
    missing_idx = uniq_idx.index.difference(cache_idx.index)

    if len(missing_idx) > 0:
        # Parse alleen missenden
        missing_pairs = list(missing_idx)
        parsed = [ _parse_coordinates_epsg(epsg, wkt) for (epsg, wkt) in missing_pairs ]
        add = pd.DataFrame({
            "epsg": [p[0] for p in missing_pairs],
            "wkt":  [p[1] for p in missing_pairs],
            "x_rd": [t[0] for t in parsed],
            "y_rd": [t[1] for t in parsed],
            "lat":  [t[2] for t in parsed],
            "lon":  [t[3] for t in parsed],
        })
        cache = pd.concat([cache, add], ignore_index=True)
        _save_coord_cache(cache)
        cache_idx = cache.set_index(["epsg", "wkt"])

    # Map terug (vectorized join)
    mapped = pd.DataFrame({"epsg": epsg_series, "wkt": wkt_series})
    mapped = mapped.join(cache_idx, on=["epsg", "wkt"])

    df["x_rd"] = mapped["x_rd"].astype(float)
    df["y_rd"] = mapped["y_rd"].astype(float)
    df["lat"] = mapped["lat"].astype(float)
    df["lon"] = mapped["lon"].astype(float)

    return df


# =============================================================================
# WATERLICHAAM
# =============================================================================
def determine_waterbody(meetobject_code: str) -> str:
    for code, name in WATERBODY_MAPPING.items():
        if code in str(meetobject_code):
            return name
    return str(meetobject_code)


# =============================================================================
# SOORTGROEPEN MAPPING
# =============================================================================
def get_species_group_mapping() -> Dict[str, str]:
    """
    Retourneert een dictionary die Latijnse namen mapt naar de gevraagde soortgroepen.
    (inhoud identiek aan je bestaande mapping)
    """
    return {
        # --- CHARIDEN ---
        "Chara": "chariden",
        "Chara aspera": "chariden",
        "Chara canescens": "chariden",
        "Chara connivens": "chariden",
        "Chara contraria": "chariden",
        "Chara globularis": "chariden",
        "Chara hispida": "chariden",
        "Chara major": "chariden",
        "Chara vulgaris": "chariden",
        "Chara virgata": "chariden",
        "Nitella": "chariden",
        "Nitella flexilis": "chariden",
        "Nitella hyalina": "chariden",
        "Nitella mucronata": "chariden",
        "Nitella opaca": "chariden",
        "Nitella translucens": "chariden",
        "Nitellopsis obtusa": "chariden",
        "Tolypella": "chariden",
        "Tolypella intricata": "chariden",
        "Tolypella prolifera": "chariden",
        # --- ISOETIDEN ---
        "Isoetes": "iseotiden",
        "Isoetes lacustris": "iseotiden",
        "Isoetes echinospora": "iseotiden",
        "Littorella uniflora": "iseotiden",
        "Lobelia dortmanna": "iseotiden",
        "Subularia aquatica": "iseotiden",
        "Pilularia globulifera": "iseotiden",
        # --- PARVOPOTAMIDEN ---
        "Potamogeton berchtoldii": "parvopotamiden",
        "Potamogeton compressus": "parvopotamiden",
        "Potamogeton acutifolius": "parvopotamiden",
        "Potamogeton friesii": "parvopotamiden",
        "Potamogeton pusillus": "parvopotamiden",
        "Potamogeton trichoides": "parvopotamiden",
        "Potamogeton obtusifolius": "parvopotamiden",
        "Potamogeton pectinatus": "parvopotamiden",
        "Stuckenia pectinata": "parvopotamiden",
        "Zannichellia palustris": "parvopotamiden",
        "Zannichellia palustris ssp. palustris": "parvopotamiden",
        "Zannichellia palustris ssp. pedicellata": "parvopotamiden",
        "Zannichellia": "parvopotamiden",
        "Ruppia": "parvopotamiden",
        "Ruppia cirrhosa": "parvopotamiden",
        "Ruppia maritima": "parvopotamiden",
        "Najas": "parvopotamiden",
        "Najas marina": "parvopotamiden",
        "Najas minor": "parvopotamiden",
        # --- MAGNOPOTAMIDEN ---
        "Potamogeton lucens": "magnopotamiden",
        "Potamogeton perfoliatus": "magnopotamiden",
        "Potamogeton alpinus": "magnopotamiden",
        "Potamogeton praelongus": "magnopotamiden",
        "Potamogeton gramineus": "magnopotamiden",
        "Potamogeton coloratus": "magnopotamiden",
        "Potamogeton nodosus": "magnopotamiden",
        "Potamogeton crispus": "magnopotamiden",
        "Groenlandia densa": "magnopotamiden",
        # --- MYRIOPHYLLIDEN ---
        "Myriophyllum": "myriophylliden",
        "Myriophyllum spicatum": "myriophylliden",
        "Myriophyllum verticillatum": "myriophylliden",
        "Myriophyllum alterniflorum": "myriophylliden",
        "Myriophyllum heterophyllum": "myriophylliden",
        "Hottonia palustris": "myriophylliden",
        # --- VALLISNERIIDEN ---
        "Vallisneria": "vallisneriiden",
        "Vallisneria spiralis": "vallisneriiden",
        # --- ELODEIDEN ---
        "Elodea": "elodeiden",
        "Elodea canadensis": "elodeiden",
        "Elodea nuttallii": "elodeiden",
        "Elodea callitrichoides": "elodeiden",
        "Egeria densa": "elodeiden",
        "Lagarosiphon major": "elodeiden",
        "Hydrilla verticillata": "elodeiden",
        "Ceratophyllum": "elodeiden",
        "Ceratophyllum demersum": "elodeiden",
        "Ceratophyllum submersum": "elodeiden",
        # --- STRATIOTIDEN ---
        "Stratiotes aloides": "stratiotiden",
        # --- PEPLIDEN ---
        "Peplis portula": "pepliden",
        "Lythrum portula": "pepliden",
        # --- BATRACHIIDEN ---
        "Ranunculus": "batrachiiden",
        "Ranunculus aquatilis": "batrachiiden",
        "Ranunculus circinatus": "batrachiiden",
        "Ranunculus fluitans": "batrachiiden",
        "Ranunculus peltatus": "batrachiiden",
        "Ranunculus penicillatus": "batrachiiden",
        "Ranunculus trichophyllus": "batrachiiden",
        "Ranunculus baudotii": "batrachiiden",
        "Callitriche": "batrachiiden",
        "Callitriche stagnalis": "batrachiiden",
        "Callitriche platycarpa": "batrachiiden",
        "Callitriche obtusangula": "batrachiiden",
        "Callitriche cophocarpa": "batrachiiden",
        "Callitriche hamulata": "batrachiiden",
        "Callitriche truncata": "batrachiiden",
        # --- NYMPHAEIDEN ---
        "Nuphar lutea": "nymphaeiden",
        "Nymphaea alba": "nymphaeiden",
        "Nymphaea candida": "nymphaeiden",
        "Nymphoides peltata": "nymphaeiden",
        "Potamogeton natans": "nymphaeiden",
        "Persicaria amphibia": "nymphaeiden",
        "Sparganium emersum": "nymphaeiden",
        "Sagittaria sagittifolia": "nymphaeiden",
        # --- HAPTOFYTEN ---
        "Fontinalis antipyretica": "haptofyten",
        "Enteromorpha": "haptofyten",
        "Enteromorpha intestinalis": "haptofyten",
        "Ulva intestinalis": "haptofyten",
        "Hydrodictyon reticulatum": "haptofyten",
        "Cladophora": "haptofyten",
        "Vaucheria": "haptofyten",
        "Amblystegium varium": "haptofyten",
        "Amblystegium fluviatile": "haptofyten",
        "Leptodictyum riparium": "haptofyten",
    }


def add_species_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt 'soortgroep' toe aan de dataset, exclusief de algemene groeivorm-codes.
    PERFORMANCE: vectorized i.p.v. df.apply(axis=1).
    """
    mapping = get_species_group_mapping()
    df = df.copy()

    if "soort" not in df.columns:
        return df

    # Filter verzamelcodes en type Groep
    df = df[~df["soort"].isin(EXCLUDED_SPECIES_CODES)]
    if "type" in df.columns:
        df = df[df["type"] != "Groep"]

    if df.empty:
        df["soortgroep"] = pd.Series(dtype="object")
        df["bedekkingsgraad_proc"] = pd.Series(dtype="float")
        return df

    soort = df["soort"].fillna("").astype(str).str.strip()
    genus = soort.str.split().str[0].fillna("")

    soortgroep = pd.Series("Overig / Individueel", index=df.index, dtype="object")

    # Kenmerkende soorten o.b.v. Grootheid
    if "Grootheid" in df.columns:
        mask_aanw = df["Grootheid"].astype(str) == "AANWZHD"
        soortgroep.loc[mask_aanw] = "Kenmerkende soort (N2000)"
    else:
        mask_aanw = pd.Series(False, index=df.index)

    # Directe mapping
    direct = soort.map(mapping)
    mask_direct = direct.notna() & (~mask_aanw)
    soortgroep.loc[mask_direct] = direct.loc[mask_direct].astype(str)

    # Genus mapping (behalve Potamogeton)
    mask_need = (soortgroep == "Overig / Individueel") & (~mask_aanw)
    genus_map = genus.map(mapping)
    mask_genus = mask_need & (genus != "Potamogeton") & genus_map.notna()
    soortgroep.loc[mask_genus] = genus_map.loc[mask_genus].astype(str)

    df["soortgroep"] = soortgroep

    # Bedekkingsgraad numeriek
    target_col = "bedekking_pct"
    if target_col not in df.columns:
        target_col = "waarde_bedekking" if "waarde_bedekking" in df.columns else ("WaardeGemeten" if "WaardeGemeten" in df.columns else None)

    if target_col is None:
        df["bedekkingsgraad_proc"] = 0.0
    else:
        s = df[target_col].fillna(0).astype(str).str.replace(",", ".", regex=False)
        s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False).str.strip()
        df["bedekkingsgraad_proc"] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)

    return df


def get_sorted_species_list(df: pd.DataFrame) -> list:
    """Gesorteerde lijst met individuele soorten (excl. verzamelcodes)."""
    if "soort" not in df.columns:
        return []
    mask_not_excluded = ~df["soort"].isin(EXCLUDED_SPECIES_CODES)
    mask_type = (df["type"] != "Groep") if ("type" in df.columns) else True
    species_df = df[mask_not_excluded & mask_type]
    return sorted(species_df["soort"].dropna().unique())


# =============================================================================
# LOOKUP: CSV -> genormaliseerde DF (+ Parquet cache)
# =============================================================================
@st.cache_data
def load_species_lookup() -> pd.DataFrame:
    """
    Laad koppeltabel met NL naam, trofie (Watertype) en KRW-scores (M14/M21).
    Bouwt (en gebruikt) LOOKUP_PARQUET cache.
    """
    csv_path = Path(SPECIES_LOOKUP_PATH)

    if LOOKUP_PARQUET.exists() and _mtime_or_zero(LOOKUP_PARQUET) >= _mtime_or_zero(csv_path):
        df_cached = _read_parquet_to_pandas(LOOKUP_PARQUET)
        want = ["soort_norm", "NL naam", "Watertype", "M14", "M21"]
        for c in want:
            if c not in df_cached.columns:
                df_cached[c] = np.nan
        return df_cached[want]

    try:
        df_lu = pd.read_csv(SPECIES_LOOKUP_PATH, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"⚠️ Koppeltabel kon niet worden ingelezen ({SPECIES_LOOKUP_PATH}): {e}")
        return pd.DataFrame(columns=["soort_norm", "NL naam", "Watertype", "M14", "M21"])

    df_lu.columns = df_lu.columns.str.strip()

    rename_map = {
        "NL_naam": "NL naam",
        "NLnaam": "NL naam",
        "Trofisch niveau": "Watertype",
        "Trofie": "Watertype",
        "Trofieniveau": "Watertype",
        "Wetenschappelijke_naam": "Wetenschappelijke naam",
        "WetenschappelijkeNaam": "Wetenschappelijke naam",
    }
    df_lu = df_lu.rename(columns={k: v for k, v in rename_map.items() if k in df_lu.columns})

    if "Wetenschappelijke naam" not in df_lu.columns:
        st.warning("⚠️ Koppeltabel mist kolom 'Wetenschappelijke naam'. Verrijking wordt overgeslagen.")
        return pd.DataFrame(columns=["soort_norm", "NL naam", "Watertype", "M14", "M21"])

    if "NL naam" not in df_lu.columns:
        df_lu["NL naam"] = np.nan
    if "Watertype" not in df_lu.columns:
        df_lu["Watertype"] = np.nan

    for c in ["M14", "M21"]:
        if c not in df_lu.columns:
            df_lu[c] = np.nan
        df_lu[c] = pd.to_numeric(df_lu[c], errors="coerce")

    df_lu["soort_norm"] = (
        df_lu["Wetenschappelijke naam"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    sort_cols = [c for c in ["NL naam", "Watertype", "M14", "M21"] if c in df_lu.columns]
    if sort_cols:
        df_lu = df_lu.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    df_lu = df_lu.drop_duplicates(subset=["soort_norm"], keep="first")
    out = df_lu[["soort_norm", "NL naam", "Watertype", "M14", "M21"]].copy()

    try:
        _write_parquet_from_pandas(out, LOOKUP_PARQUET)
    except Exception:
        pass

    return out


# =============================================================================
# CORE LOAD_DATA (DuckDB + Parquet) met fallback
# =============================================================================
def _file_signature() -> Tuple[str, float, float, str]:
    """Cache key: pipeline versie + mtimes."""
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
    """
    Interne cached loader. 'sig' zorgt voor invalidatie wanneer CSV/lookup verandert
    of wanneer PIPELINE_VERSION wordt verhoogd.
    """
    csv_path = Path(FILE_PATH)
    lookup_csv = Path(SPECIES_LOOKUP_PATH)

    # 0) Als FINAL_PARQUET up-to-date is: direct lezen
    if FINAL_PARQUET.exists():
        ok = (_mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(csv_path)) and (_mtime_or_zero(FINAL_PARQUET) >= _mtime_or_zero(lookup_csv))
        if ok:
            df_final = _read_parquet_to_pandas(FINAL_PARQUET)
            if not df_final.empty:
                return df_final

    # 1) DuckDB pad
    con = _get_duckdb()
    if con is not None and csv_path.exists():
        _ensure_measurements_parquet(csv_path)

        if MEAS_PARQUET.exists():
            # SQL: abiotiek + total WATPTN + BEDKG/AANWZHD selectie
            gf_keys = list(GROWTH_FORM_MAPPING.keys())
            gf_case = " ".join([f"WHEN Parameter='{k}' THEN '{GROWTH_FORM_MAPPING[k]}'" for k in gf_keys])
            gf_in = ",".join([f"'{k}'" for k in gf_keys])

            sql = f"""
            WITH base AS (
                SELECT
                    *,
                    COALESCE(
                        TRY_CAST(MetingDatumTijd AS TIMESTAMP),
                        TRY_STRPTIME(MetingDatumTijd, '%d-%m-%Y %H:%M:%S'),
                        TRY_STRPTIME(MetingDatumTijd, '%d-%m-%Y'),
                        TRY_STRPTIME(MetingDatumTijd, '%Y-%m-%d %H:%M:%S'),
                        TRY_STRPTIME(MetingDatumTijd, '%Y-%m-%d')
                    ) AS meting_ts
                FROM read_parquet('{MEAS_PARQUET.as_posix()}')
            ),
            env AS (
                SELECT
                    CollectieReferentie,
                    AVG(CASE WHEN Grootheid='DIEPTE' THEN CAST(WaardeGemeten AS DOUBLE) END) AS DIEPTE,
                    AVG(CASE WHEN Grootheid='ZICHT' THEN CAST(WaardeGemeten AS DOUBLE) END)  AS ZICHT
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
                st.warning(f"⚠️ DuckDB pad faalde, val terug op pandas: {e}")

            if not df_merged.empty:
                # Project
                df_merged["Project"] = df_merged["Projecten"].map(PROJECT_MAPPING).fillna(df_merged["Projecten"])

                # Waterlichaam
                df_merged["Waterlichaam"] = df_merged["MeetObject"].apply(determine_waterbody)

                # Abiotic conversions
                df_merged["diepte_m"] = pd.to_numeric(df_merged.get("DIEPTE"), errors="coerce") / 100.0
                df_merged["doorzicht_m"] = pd.to_numeric(df_merged.get("ZICHT"), errors="coerce") / 10.0

                # Rename naar final schema
                final_df = df_merged.rename(
                    columns={
                        "MeetObject": "locatie_id",
                        "Parameter": "soort",
                        "WaardeGemeten": "waarde_bedekking",
                        "EenheidGemeten": "eenheid",
                    }
                )

                # bedekking_pct consistent
                final_df["bedekking_pct"] = final_df["waarde_bedekking"]

                # Coördinaten: parse per unieke (GeografieDatum, GeografieVorm) en cache persistent
                final_df = _apply_coordinates_cached(final_df, epsg_col="GeografieDatum", wkt_col="GeografieVorm")

                # Verrijking lookup
                lookup = load_species_lookup()
                final_df["soort_norm"] = (
                    final_df["soort"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                )
                final_df = final_df.merge(lookup, on="soort_norm", how="left")
                final_df = final_df.rename(columns={"NL naam": "soort_triviaal", "Watertype": "trofisch_niveau"})

                # KRW score/class
                final_df["krw_watertype"] = final_df["Waterlichaam"].map(KRW_WATERTYPE_BY_WATERLICHAAM)
                final_df["krw_score"] = np.nan
                mask_m14 = final_df["krw_watertype"] == "M14"
                mask_m21 = final_df["krw_watertype"] == "M21"
                if "M14" in final_df.columns:
                    final_df.loc[mask_m14, "krw_score"] = final_df.loc[mask_m14, "M14"]
                if "M21" in final_df.columns:
                    final_df.loc[mask_m21, "krw_score"] = final_df.loc[mask_m21, "M21"]

                final_df["krw_class"] = pd.cut(
                    final_df["krw_score"],
                    bins=[0, 2, 4, 5],
                    labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                    include_lowest=True,
                )

                final_df["soort_display"] = np.where(
                    final_df["soort_triviaal"].notna() & (final_df["soort_triviaal"].astype(str).str.len() > 0),
                    final_df["soort_triviaal"] + " (" + final_df["soort"] + ")",
                    final_df["soort"],
                )

                cols_to_keep = [
                    "datum", "jaar", "locatie_id", "Waterlichaam", "Project", "CollectieReferentie",
                    "soort", "bedekking_pct", "waarde_bedekking", "totaal_bedekking_locatie",
                    "diepte_m", "doorzicht_m", "lat", "lon", "x_rd", "y_rd",
                    "groeivorm", "type", "Grootheid", "soort_triviaal", "trofisch_niveau",
                    "krw_watertype", "krw_score", "krw_class", "soort_display"
                ]

                for col in cols_to_keep:
                    if col not in final_df.columns:
                        final_df[col] = np.nan

                final_df = final_df[cols_to_keep]

                try:
                    _write_parquet_from_pandas(final_df, FINAL_PARQUET)
                except Exception:
                    pass

                return final_df

    # 2) Fallback: pandas pad (origineel gedrag, semicolon eerst)
    try:
        df_raw = pd.read_csv(FILE_PATH, sep=";", engine="python", encoding="utf-8-sig")
    except Exception:
        df_raw = pd.read_csv(FILE_PATH, sep=None, engine="python", encoding="utf-8-sig")

    df_raw.columns = df_raw.columns.str.strip()

    if "MetingDatumTijd" in df_raw.columns:
        df_raw["MetingDatumTijd"] = pd.to_datetime(df_raw["MetingDatumTijd"], dayfirst=True, errors="coerce")
        df_raw["datum"] = df_raw["MetingDatumTijd"].dt.floor("D")
        df_raw["jaar"] = df_raw["datum"].dt.year
    else:
        st.error("Kolom 'MetingDatumTijd' mist.")
        return pd.DataFrame()

    df_raw["Project"] = df_raw["Projecten"].map(PROJECT_MAPPING).fillna(df_raw["Projecten"])
    df_raw["Waterlichaam"] = df_raw["MeetObject"].apply(determine_waterbody)

    # ABIOTIEK
    df_abiotic = df_raw[df_raw["Grootheid"].isin(["DIEPTE", "ZICHT"])].copy()
    if not df_abiotic.empty:
        df_env = df_abiotic.pivot_table(
            index="CollectieReferentie",
            columns="Grootheid",
            values="WaardeGemeten",
            aggfunc="mean",
        ).reset_index()
    else:
        df_env = pd.DataFrame(columns=["CollectieReferentie"])

    df_env["diepte_m"] = pd.to_numeric(df_env.get("DIEPTE"), errors="coerce") / 100.0
    df_env["doorzicht_m"] = pd.to_numeric(df_env.get("ZICHT"), errors="coerce") / 10.0

    # WATPTN
    df_total = df_raw[df_raw["Parameter"] == "WATPTN"].copy()
    df_total = df_total[["CollectieReferentie", "WaardeGemeten"]].rename(columns={"WaardeGemeten": "totaal_bedekking_locatie"})
    df_total["totaal_bedekking_locatie"] = pd.to_numeric(df_total["totaal_bedekking_locatie"], errors="coerce")
    df_total = df_total.groupby("CollectieReferentie", as_index=False).mean()

    # BEDKG/AANWZHD
    df_bedkg = df_raw[(df_raw["Grootheid"].isin(["BEDKG", "AANWZHD"])) & (df_raw["Parameter"] != "WATPTN")].copy()

    def classify_row(row):
        param = row["Parameter"]
        grootheid = row["Grootheid"]
        if grootheid == "AANWZHD":
            return "Kenmerkende soort (N2000)", "Soort"
        if param in GROWTH_FORM_MAPPING:
            return GROWTH_FORM_MAPPING[param], "Groep"
        return "Individuele soort", "Soort"

    classificatie = df_bedkg.apply(classify_row, axis=1)
    df_bedkg["groeivorm"] = [x[0] for x in classificatie]
    df_bedkg["type"] = [x[1] for x in classificatie]

    df_merged = pd.merge(df_bedkg, df_env[["CollectieReferentie", "diepte_m", "doorzicht_m"]], on="CollectieReferentie", how="left")
    df_merged = pd.merge(df_merged, df_total, on="CollectieReferentie", how="left")

    final_df = df_merged.rename(
        columns={
            "MeetObject": "locatie_id",
            "Parameter": "soort",
            "WaardeGemeten": "waarde_bedekking",
            "EenheidGemeten": "eenheid",
        }
    )
    final_df["bedekking_pct"] = final_df["waarde_bedekking"]

    # Coördinaten via cache (EPSG + WKT)
    final_df = _apply_coordinates_cached(final_df, epsg_col="GeografieDatum", wkt_col="GeografieVorm")

    lookup = load_species_lookup()
    final_df["soort_norm"] = (
        final_df["soort"].fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    final_df = final_df.merge(lookup, on="soort_norm", how="left")
    final_df = final_df.rename(columns={"NL naam": "soort_triviaal", "Watertype": "trofisch_niveau"})

    final_df["krw_watertype"] = final_df["Waterlichaam"].map(KRW_WATERTYPE_BY_WATERLICHAAM)
    final_df["krw_score"] = np.nan
    mask_m14 = final_df["krw_watertype"] == "M14"
    mask_m21 = final_df["krw_watertype"] == "M21"
    final_df.loc[mask_m14, "krw_score"] = final_df.loc[mask_m14, "M14"]
    final_df.loc[mask_m21, "krw_score"] = final_df.loc[mask_m21, "M21"]
    final_df["krw_class"] = pd.cut(
        final_df["krw_score"],
        bins=[0, 2, 4, 5],
        labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
        include_lowest=True,
    )
    final_df["soort_display"] = np.where(
        final_df["soort_triviaal"].notna() & (final_df["soort_triviaal"].astype(str).str.len() > 0),
        final_df["soort_triviaal"] + " (" + final_df["soort"] + ")",
        final_df["soort"],
    )

    cols_to_keep = [
        "datum", "jaar", "locatie_id", "Waterlichaam", "Project", "CollectieReferentie",
        "soort", "bedekking_pct", "waarde_bedekking", "totaal_bedekking_locatie",
        "diepte_m", "doorzicht_m", "lat", "lon", "x_rd", "y_rd",
        "groeivorm", "type", "Grootheid", "soort_triviaal", "trofisch_niveau",
        "krw_watertype", "krw_score", "krw_class", "soort_display"
    ]
    for col in cols_to_keep:
        if col not in final_df.columns:
            final_df[col] = np.nan

    final_df = final_df[cols_to_keep]

    try:
        _write_parquet_from_pandas(final_df, FINAL_PARQUET)
    except Exception:
        pass

    return final_df


@st.cache_data
def load_data() -> pd.DataFrame:
    """Public API: laadt verrijkte dataset (DuckDB+Parquet waar mogelijk)."""
    sig = _file_signature()
    return _load_data_cached(sig)


# =============================================================================
# PLOT FUNCTIES
# =============================================================================
def plot_trend_line(df, x_col, y_col, color=None, title="Trend"):
    """Genereert een standaard trendlijn plot."""
    fig = px.line(df, x=x_col, y=y_col, color=color, markers=True, title=title)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def interpret_soil_state(df_loc):
    """Genereert automatische tekstinterpretatie van bodemconditie."""
    if df_loc.empty:
        return "Geen data beschikbaar."
    total_cover = df_loc["totaal_bedekking_locatie"].mean()
    modes = df_loc["groeivorm"].mode()
    dom_type = modes[0] if not modes.empty else "Onbekend"
    text = "**Automatische Interpretatie:**\n"
    if pd.isna(total_cover):
        text += "Geen bedekkingsgegevens beschikbaar.\n"
    elif total_cover < 5:
        text += "⚠️ **Zeer kale bodem** (<5% bedekking).\n"
    elif dom_type == "Ondergedoken":
        text += f"✅ Goede ontwikkeling (**{total_cover:.0f}%**).\n"
    elif dom_type == "Drijvend":
        text += f"⚠️ Veel drijfbladplanten (**{total_cover:.0f}%**).\n"
    elif dom_type == "Draadalgen":
        text += "❌ Dominantie van draadalgen wijst op verstoring.\n"
    return text


# =============================================================================
# HELPER FUNCTIES VOOR ANALYSE
# =============================================================================
def categorize_slope_trend(val, threshold):
    """Bepaalt de trendcategorie op basis van een drempelwaarde."""
    if val > threshold:
        return "Verbeterend ↗️"
    elif val < -threshold:
        return "Verslechterend ↘️"
    return "Stabiel ➡️"


def get_color_absolute(val, min_v, max_v):
    """Geeft RGB kleur terug van Rood (laag) naar Groen (hoog)."""
    if pd.isna(val):
        return [200, 200, 200, 100]
    norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
    norm = max(0, min(1, norm))
    r = int(255 * (1 - norm))
    g = int(255 * norm)
    b = 0
    return [r, g, b, 200]


def get_color_diff(val):
    """Geeft Rood (verslechtering), Grijs (stabiel), Groen (verbetering)."""
    threshold = 0.5
    if val < -threshold:
        return [255, 0, 0, 200]
    elif val > threshold:
        return [0, 255, 0, 200]
    return [128, 128, 128, 100]


# =============================================================================
# AGGREGATIE EN KPI FUNCTIES
# =============================================================================
def get_location_metric_mean(dataframe, metric_col):
    """
    Berekent gemiddelde van een locatie-parameter (zoals WATPTN/Totale Bedekking).
    Stap 1: Unieke waarde per monstername (CollectieReferentie) pakken.
    Stap 2: Gemiddelde daarvan nemen.
    """
    if dataframe.empty:
        return 0.0
    per_sample = dataframe.groupby("CollectieReferentie")[metric_col].first()
    return per_sample.mean()


def calculate_kpi(curr_df, prev_df, metric_col, is_loc_metric=False):
    if curr_df.empty:
        return 0.0, 0.0
    if is_loc_metric:
        curr_val = get_location_metric_mean(curr_df, metric_col)
        prev_val = get_location_metric_mean(prev_df, metric_col) if not prev_df.empty else curr_val
    else:
        curr_val = curr_df[metric_col].mean()
        prev_val = prev_df[metric_col].mean() if not prev_df.empty else curr_val
    delta = curr_val - prev_val
    return curr_val, delta


# =============================================================================
# KAART VISUALISATIE FUNCTIES
# =============================================================================
def get_color_vegetation(value):
    """Rood (0%) -> Groen (100%)"""
    if value == 0:
        return "#d73027"
    elif value <= 5:
        return "#fc8d59"
    elif value <= 15:
        return "#fee08b"
    elif value <= 40:
        return "#d9ef8b"
    elif value <= 75:
        return "#91cf60"
    return "#1a9850"


def get_color_depth(value):
    """Lichtblauw (ondiep) -> Donkerblauw (diep)"""
    if pd.isna(value):
        return "gray"
    elif value < 0.5:
        return "#eff3ff"
    elif value < 1.5:
        return "#bdd7e7"
    elif value < 2.5:
        return "#6baed6"
    elif value < 4.0:
        return "#3182bd"
    return "#08519c"


def get_color_transparency(value):
    """Bruin (weinig zicht) -> Groen (veel zicht)"""
    if pd.isna(value):
        return "gray"
    elif value < 0.5:
        return "#8c510a"
    elif value < 1.0:
        return "#d8b365"
    elif value < 1.5:
        return "#f6e8c3"
    elif value < 2.0:
        return "#c7eae5"
    elif value < 3.0:
        return "#5ab4ac"
    return "#01665e"


def get_color_krw(score):
    """KRW-score 1-5: 1-2 groen, 3-4 oranje, 5 rood."""
    if pd.isna(score):
        return "gray"
    try:
        s = float(score)
    except Exception:
        return "gray"
    if s <= 2:
        return "#1a9850"
    elif s <= 4:
        return "#ff7f0e"
    return "#d73027"


def _polar_to_cart(cx, cy, r, angle_rad):
    return (cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad))


def _wedge_path(cx, cy, r, start_angle, end_angle):
    large_arc = 1 if (end_angle - start_angle) > math.pi else 0
    x1, y1 = _polar_to_cart(cx, cy, r, start_angle)
    x2, y2 = _polar_to_cart(cx, cy, r, end_angle)
    return f"M {cx:.2f},{cy:.2f} L {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f} Z"


def pie_svg(
    counts: dict,
    color_map: dict,
    order=None,
    size=30,
    border=1,
    border_color="#333",
    fixed_total=None,
    fill_gap=False,
    gap_color="transparent",
):
    """
    SVG pie chart.
    - Default: normaliseert op som(counts) -> altijd volle cirkel (geschikt voor records).
    - fixed_total=100 + fill_gap=True: sectoren zijn absolute percentages, rest blijft leeg/transparant.
    """
    nonzero = [(k, float(v)) for k, v in counts.items() if v is not None and float(v) > 0]

    if not nonzero:
        r = (size / 2) - border
        cx = cy = size / 2
        return (
            f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
            f"xmlns='http://www.w3.org/2000/svg'>"
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#cccccc' stroke='{border_color}' stroke-width='{border}' />"
            f"</svg>"
        )

    r = (size / 2) - border
    cx = cy = size / 2

    if fixed_total is not None:
        denom = float(fixed_total)
    else:
        denom = sum(v for _, v in nonzero)
        if denom <= 0:
            return (
                f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
                f"xmlns='http://www.w3.org/2000/svg'>"
                f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#cccccc' stroke='{border_color}' stroke-width='{border}' />"
                f"</svg>"
            )

    sum_vals = sum(v for _, v in nonzero)

    if fixed_total is not None:
        if (sum_vals / denom) >= 0.999 and len(nonzero) == 1:
            cat, _ = nonzero[0]
            fill = color_map.get(cat, "#999999")
            return (
                f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
                f"xmlns='http://www.w3.org/2000/svg'>"
                f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{fill}' stroke='{border_color}' stroke-width='{border}' />"
                f"</svg>"
            )
    else:
        if len(nonzero) == 1:
            cat, _ = nonzero[0]
            fill = color_map.get(cat, "#999999")
            return (
                f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
                f"xmlns='http://www.w3.org/2000/svg'>"
                f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{fill}' stroke='{border_color}' stroke-width='{border}' />"
                f"</svg>"
            )

    cats = order if order else list(counts.keys())
    start = -math.pi / 2
    paths = []

    if fixed_total is not None and fill_gap and gap_color != "transparent":
        paths.append(f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{gap_color}' />")

    for cat in cats:
        val = float(counts.get(cat, 0) or 0)
        if val <= 0:
            continue
        frac = val / denom
        if frac <= 0:
            continue
        end = start + frac * 2 * math.pi
        color = color_map.get(cat, "#999999")
        d = _wedge_path(cx, cy, r, start, end)
        paths.append(f"<path d='{d}' fill='{color}' />")
        start = end
        if fixed_total is not None and (start - (-math.pi / 2)) >= 2 * math.pi * 0.999:
            break

    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        + "".join(paths)
        + f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{border_color}' stroke-width='{border}' />"
        + f"</svg>"
    )


def create_pie_map(
    df_locs: pd.DataFrame,
    counts_by_loc: dict,
    label: str,
    color_map: dict,
    order=None,
    size_px: int = 30,
    zoom_start: int = 10,
    fixed_total=None,
    fill_gap=False,
    gap_color="transparent",
):
    """
    Folium kaart met pie-chart markers (SVG via DivIcon) per locatie.
    FIX: return pas na loop (anders slechts 1 marker).
    """
    if df_locs["lat"].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat, center_lon = df_locs["lat"].mean(), df_locs["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)

    for row in df_locs.dropna(subset=["lat", "lon"]).itertuples():
        loc_id = getattr(row, "locatie_id")
        wb = getattr(row, "Waterlichaam", "")
        counts = counts_by_loc.get(loc_id, {})

        svg = pie_svg(
            counts,
            color_map=color_map,
            order=order,
            size=size_px,
            fixed_total=fixed_total,
            fill_gap=fill_gap,
            gap_color=gap_color,
        )

        parts = [f"{escape(str(k))}: {int(v)}" for k, v in counts.items() if v]
        dist_txt = "<br/>".join(parts) if parts else "Geen data"

        diepte = getattr(row, "diepte_m", float("nan"))
        doorzicht = getattr(row, "doorzicht_m", float("nan"))
        diepte_txt = "n.v.t." if pd.isna(diepte) else f"{diepte:.2f} m"
        doorzicht_txt = "n.v.t." if pd.isna(doorzicht) else f"{doorzicht:.2f} m"

        tooltip_html = (
            f"<b>Locatie:</b> {escape(str(loc_id))}<br/>"
            f"<b>Water:</b> {escape(str(wb))}<br/>"
            f"<b>🌊 Diepte:</b> {escape(diepte_txt)}<br/>"
            f"<b>👁️ Doorzicht:</b> {escape(doorzicht_txt)}<br/>"
            f"<b>{escape(label)}:</b><br/>{dist_txt}"
        )

        icon = folium.DivIcon(
            html=f"""
            <div style="width:{size_px}px;height:{size_px}px;transform: translate(-50%, -50%);">
            {svg}
            </div>
            """
        )
        folium.Marker(
            location=[getattr(row, "lat"), getattr(row, "lon")],
            icon=icon,
            tooltip=tooltip_html,
        ).add_to(m)

    return m


def create_map(dataframe, mode, label_veg="Vegetatie", value_style="vegetation", category_col=None, category_color_map=None):
    """
    Genereert een Folium kaart (OSM-tiles).
    FIX: return pas na loop (anders slechts 1 marker).
    """
    if dataframe["lat"].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat = dataframe["lat"].mean()
        center_lon = dataframe["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)

    for row in dataframe.itertuples():
        radius = 5
        fill_opacity = 0.8

        if mode == "Vegetatie":
            if value_style == "categorical" and category_col:
                cat = getattr(row, category_col, None)
                color = (category_color_map or {}).get(cat, "#999999")
                main_line = f"<b>🌱 {label_veg}:</b> {cat}"
                radius = 6
            else:
                val = getattr(row, "waarde_veg", 0.0)
                if value_style == "krw":
                    color = get_color_krw(val)
                    main_line = f"<b>🌱 {label_veg}:</b> {val:.2f}"
                    radius = 6
                else:
                    color = get_color_vegetation(val)
                    main_line = f"<b>🌱 {label_veg}:</b> {val:.1f}%"
                    radius = 4 + (min(val, 100) / 100 * 6) if val > 0 else 4

        elif mode == "Diepte":
            val = getattr(row, "diepte_m", float("nan"))
            color = get_color_depth(val)
            main_line = f"<b>🌊 Diepte:</b> {val:.2f} m"

        else:  # Doorzicht
            val = getattr(row, "doorzicht_m", float("nan"))
            color = get_color_transparency(val)
            main_line = f"<b>👁️ Doorzicht:</b> {val:.2f} m"

        depth_line = f"<b>🌊 Diepte:</b> {getattr(row, 'diepte_m', float('nan')):.2f} m"
        trans_line = f"<b>👁️ Doorzicht:</b> {getattr(row, 'doorzicht_m', float('nan')):.2f} m"

        tooltip_html = (
            f"<b>Locatie:</b> {getattr(row, 'locatie_id', '')}<br>"
            f"<b>Water:</b> {getattr(row, 'Waterlichaam', '')}<br>"
            f"{main_line}<br>"
            f"{depth_line}<br>"
            f"{trans_line}"
        )

        if getattr(row, "lat") is not None and getattr(row, "lon") is not None:
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=radius,
                color="#333333",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                tooltip=tooltip_html,
            ).add_to(m)

    return m


def df_to_geojson_points(df: pd.DataFrame, value_col: str, id_col: str = "locatie_id"):
    """
    Zet punten (lat/lon) om naar een GeoJSON FeatureCollection.
    """
    features = []
    for row in df.dropna(subset=["lat", "lon"]).itertuples(index=False):
        props = {
            "locatie_id": getattr(row, id_col),
            "value": float(getattr(row, value_col)) if getattr(row, value_col) is not None else None,
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row.lon), float(row.lat)]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def render_swipe_map_html(
    geojson_left: dict,
    geojson_right: dict,
    year_left: int,
    year_right: int,
    metric_label: str,
    min_val: float,
    max_val: float,
    center_lat: float,
    center_lon: float,
    zoom: float = 9.0,
    height_px: int = 650,
    bounds=None,
):
    """
    Rendert een swipe-map met dragbare divider/handle op de kaart zelf (MapLibre in Streamlit html component).
    (Ongewijzigd t.o.v. je bestaande implementatie)
    """
    style_url = "https://tiles.openfreemap.org/styles/liberty"
    left_json = json.dumps(geojson_left)
    right_json = json.dumps(geojson_right)
    bounds_json = "null" if bounds is None else json.dumps(bounds)

    if max_val == min_val:
        max_val = min_val + 1e-6

    html_str = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
  <style>
    body {{ margin: 0; padding: 0; }}
    #wrap {{
      position: relative;
      width: 100%;
      height: {height_px}px;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 1px 10px rgba(0,0,0,0.08);
      background: #f7f7f7;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}
    #map_left {{ position: absolute; inset: 0; }}
    #map_right {{ position: absolute; inset: 0; clip-path: inset(0 0 0 50%); }}
    #divider {{
      position: absolute; top: 0; bottom: 0; left: 50%;
      width: 2px; background: rgba(230,230,230,0.95);
      box-shadow: 0 0 0 1px rgba(0,0,0,0.08);
      z-index: 10; cursor: ew-resize;
    }}
    #handle {{
      position: absolute; left: 50%; top: 50%;
      transform: translate(-50%, -50%);
      width: 16px; height: 120px; border-radius: 10px;
      background: rgba(255,255,255,0.95);
      border: 1px solid rgba(0,0,0,0.15);
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
      z-index: 11; cursor: ew-resize;
    }}
    .year-label {{
      position: absolute; top: 18px;
      font-size: 44px; font-weight: 700;
      color: rgba(0,0,0,0.78);
      text-shadow: 0 1px 0 rgba(255,255,255,0.6);
      z-index: 12; pointer-events: none;
    }}
    #label_left {{ left: 30px; opacity: 0.40; }}
    #label_right {{ right: 30px; opacity: 0.95; }}
    #legend {{
      position: absolute; left: 50%; bottom: 20px;
      transform: translateX(-50%);
      width: 520px; max-width: calc(100% - 40px);
      background: rgba(255,255,255,0.90);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 12px;
      padding: 12px 14px;
      z-index: 12;
      backdrop-filter: blur(3px);
    }}
    #legend .title {{
      text-align: center; font-size: 22px; font-weight: 700;
      margin-bottom: 8px;
    }}
    #legend .bar {{
      height: 14px; border-radius: 8px;
      border: 1px solid rgba(0,0,0,0.12);
      background: linear-gradient(90deg, #d73027 0%, #fee08b 50%, #1a9850 100%);
    }}
    #legend .labels {{
      display: flex; justify-content: space-between;
      margin-top: 8px;
      font-size: 16px; font-weight: 650;
      color: rgba(0,0,0,0.80);
    }}
    #legend .sub {{
      text-align: center; margin-top: 3px;
      font-size: 14px; color: rgba(0,0,0,0.60);
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="map_left"></div>
    <div id="map_right"></div>
    <div id="divider"></div>
    <div id="handle"></div>
    <div id="label_left" class="year-label">{year_left}</div>
    <div id="label_right" class="year-label">{year_right}</div>
    <div id="legend">
      <div class="title">{metric_label}</div>
      <div class="bar"></div>
      <div class="labels"><span>{min_val:.1f}</span><span>{max_val:.1f}</span></div>
      <div class="sub">Laag → Hoog</div>
    </div>
  </div>

<script>
  const styleUrl = "{style_url}";
  const leftData = {left_json};
  const rightData = {right_json};
  const minVal = {min_val};
  const maxVal = {max_val};
  const bounds = {bounds_json};

  const mapLeft = new maplibregl.Map({{
    container: 'map_left',
    style: styleUrl,
    center: [{center_lon}, {center_lat}],
    zoom: {zoom},
    attributionControl: true
  }});

  const mapRight = new maplibregl.Map({{
    container: 'map_right',
    style: styleUrl,
    center: [{center_lon}, {center_lat}],
    zoom: {zoom},
    attributionControl: true,
    interactive: true
  }});

  mapRight.scrollZoom.disable();
  mapRight.boxZoom.disable();
  mapRight.dragRotate.disable();
  mapRight.dragPan.disable();
  mapRight.keyboard.disable();
  mapRight.doubleClickZoom.disable();
  mapRight.touchZoomRotate.disable();

  function sync() {{
    const c = mapLeft.getCenter();
    mapRight.jumpTo({{
      center: c,
      zoom: mapLeft.getZoom(),
      bearing: mapLeft.getBearing(),
      pitch: mapLeft.getPitch()
    }});
  }}
  mapLeft.on('move', sync);
  mapLeft.on('moveend', sync);

  function applyBasemapGray(map) {{
    const style = map.getStyle();
    const layers = (style && style.layers) ? style.layers : [];
    const GRAY_BG = "#e7e7e7";
    const GRAY_FILL = "#cdcdcd";
    const GRAY_LINE = "#9c9c9c";
    const GRAY_TEXT = "#666666";
    const GRAY_HALO = "#f2f2f2";

    layers.forEach((ly) => {{
      if (!ly || !ly.id) return;
      try {{
        switch (ly.type) {{
          case "background":
            map.setPaintProperty(ly.id, "background-color", GRAY_BG);
            break;
          case "fill":
            try {{ map.setPaintProperty(ly.id, "fill-pattern", null); }} catch(e) {{}}
            map.setPaintProperty(ly.id, "fill-color", GRAY_FILL);
            try {{ map.setPaintProperty(ly.id, "fill-outline-color", "#b0b0b0"); }} catch(e) {{}}
            break;
          case "fill-extrusion":
            map.setPaintProperty(ly.id, "fill-extrusion-color", GRAY_FILL);
            break;
          case "line":
            try {{ map.setPaintProperty(ly.id, "line-pattern", null); }} catch(e) {{}}
            map.setPaintProperty(ly.id, "line-color", GRAY_LINE);
            break;
          case "symbol":
            try {{ map.setPaintProperty(ly.id, "text-color", GRAY_TEXT); }} catch(e) {{}}
            try {{ map.setPaintProperty(ly.id, "text-halo-color", GRAY_HALO); }} catch(e) {{}}
            try {{ map.setPaintProperty(ly.id, "icon-color", GRAY_TEXT); }} catch(e) {{}}
            try {{ map.setPaintProperty(ly.id, "text-opacity", 0.85); }} catch(e) {{}}
            break;
          case "circle":
            map.setPaintProperty(ly.id, "circle-color", GRAY_LINE);
            try {{ map.setPaintProperty(ly.id, "circle-opacity", 0.35); }} catch(e) {{}}
            break;
          case "heatmap":
            try {{ map.setPaintProperty(ly.id, "heatmap-opacity", 0.25); }} catch(e) {{}}
            break;
          case "raster":
            try {{ map.setPaintProperty(ly.id, "raster-saturation", -1); }} catch(e) {{}}
            break;
          default:
            break;
        }}
      }} catch (e) {{
      }}
    }});
  }}

  function addPoints(map, sourceName, layerName, data, dim=false) {{
    if (map.getSource(sourceName)) {{
      map.getSource(sourceName).setData(data);
      return;
    }}
    map.addSource(sourceName, {{
      type: 'geojson',
      data: data
    }});
    map.addLayer({{
      id: layerName,
      type: 'circle',
      source: sourceName,
      paint: {{
        'circle-radius': 6,
        'circle-stroke-color': dim ? 'rgba(40,40,40,0.35)' : 'rgba(40,40,40,0.75)',
        'circle-stroke-width': 1,
        'circle-opacity': dim ? 0.45 : 0.85,
        'circle-color': [
          'interpolate', ['linear'], ['get', 'value'],
          minVal, '#d73027',
          (minVal + maxVal) / 2.0, '#fee08b',
          maxVal, '#1a9850'
        ]
      }}
    }});
  }}

  mapLeft.on('load', () => {{
    applyBasemapGray(mapLeft);
    addPoints(mapLeft, 'leftPts', 'leftLayer', leftData, true);

    if (bounds && bounds.length === 4) {{
      const sw = [bounds[0], bounds[1]];
      const ne = [bounds[2], bounds[3]];
      mapLeft.fitBounds([sw, ne], {{
        padding: 70,
        maxZoom: 12,
        duration: 0
      }});
    }}

    const popup = new maplibregl.Popup({{ closeButton: false, closeOnClick: false }});
    mapLeft.on('mousemove', 'leftLayer', (e) => {{
      mapLeft.getCanvas().style.cursor = 'pointer';
      const p = e.features[0].properties;
      popup
        .setLngLat(e.lngLat)
        .setHTML(`<b>Locatie:</b> ${{p.locatie_id}}<br/><b>Waarde:</b> ${{Number(p.value).toFixed(1)}}`)
        .addTo(mapLeft);
    }});
    mapLeft.on('mouseleave', 'leftLayer', () => {{
      mapLeft.getCanvas().style.cursor = '';
      popup.remove();
    }});
  }});

  
  mapRight.on('load', () => {{
    addPoints(mapRight, 'rightPts', 'rightLayer', rightData, false);

    const popupR = new maplibregl.Popup({{ closeButton: false, closeOnClick: false }});
    mapRight.on('mousemove', 'rightLayer', (e) => {{
      mapRight.getCanvas().style.cursor = 'pointer';
      if (!e.features || !e.features.length) return;
      const p = e.features[0].properties;
      const v = (p.value === null || p.value === undefined || p.value === "" || isNaN(Number(p.value)))
        ? "n.v.t."
        : Number(p.value).toFixed(1);
      popupR
        .setLngLat(e.lngLat)
        .setHTML(`<b>Locatie:</b> ${{p.locatie_id}}<br/><b>Waarde ({year_right}):</b> ${{v}}`)
        .addTo(mapRight);
    }});
    mapRight.on('mouseleave', 'rightLayer', () => {{
      mapRight.getCanvas().style.cursor = '';
      popupR.remove();
    }});
  }});

  // ------------------------------------------------------------------
  // Swipe divider/handle (DOM)
  // ------------------------------------------------------------------
  const wrap = document.getElementById('wrap');
  const mapRightDiv = document.getElementById('map_right');
  const divider = document.getElementById('divider');
  const handle = document.getElementById('handle');
  let isDragging = false;

  function setSwipe(p) {{
    const pct = Math.max(0, Math.min(1, p));
    const x = pct * wrap.clientWidth;
    divider.style.left = x + 'px';
    handle.style.left = x + 'px';
    mapRightDiv.style.clipPath = `inset(0 0 0 ${{(pct*100).toFixed(2)}}%)`;
  }}

  function pointerToPct(clientX) {{
    const rect = wrap.getBoundingClientRect();
    return (clientX - rect.left) / rect.width;
  }}

  function onDown(e) {{
    isDragging = true;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    setSwipe(pointerToPct(x));
    e.preventDefault();
  }}

  function onMove(e) {{
    if (!isDragging) return;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    setSwipe(pointerToPct(x));
    e.preventDefault();
  }}

  function onUp() {{
    isDragging = false;
  }}

  divider.addEventListener('mousedown', onDown);
  handle.addEventListener('mousedown', onDown);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);

  divider.addEventListener('touchstart', onDown, {{passive:false}});
  handle.addEventListener('touchstart', onDown, {{passive:false}});
  window.addEventListener('touchmove', onMove, {{passive:false}});
  window.addEventListener('touchend', onUp);

  // init
  setSwipe(0.5);
</script>
</body>
</html>
"""
    return html_str