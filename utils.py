# utils.py
from __future__ import annotations

import re
import json
import math
import urllib.parse
import csv
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
FILE_PATH = "AquaDeskMeasurementExport_RWS_20260222163559.csv"
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
PIPELINE_VERSION = "2026-03-27_duckdb_parquet_coords_v4_n2000_as_entity"


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
# BATHYMETRIE / BASEMAP HELPERS
# =============================================================================
WMS_BASE_URL = 'https://geo.rijkswaterstaat.nl/services/ogc/gdr/bodemhoogte_ijsselmeergebied/ows'
WMS_LAYER_NAME = 'bodemhoogte_ijg_2022'
WMS_ATTRIBUTION = 'Rijkswaterstaat bathymetrie IJsselmeergebied'


def build_bathymetry_legend_url() -> str:
    """Legend URL voor dezelfde bathymetrie-WMS als in st_plot.py."""
    params = {
        'SERVICE': 'WMS',
        'REQUEST': 'GetLegendGraphic',
        'VERSION': '1.0.0',
        'FORMAT': 'image/png',
        'LAYER': WMS_LAYER_NAME,
        'STYLE': '',
    }
    return f"{WMS_BASE_URL}?{urllib.parse.urlencode(params)}"


def add_bathymetry_wms(map_obj: folium.Map) -> folium.Map:
    """Voeg de RWS-bathymetrie als basislaag toe aan een Folium-kaart."""
    if map_obj is None:
        return map_obj
    folium.raster_layers.WmsTileLayer(
        url=WMS_BASE_URL,
        name='Bathymetrie IJsselmeergebied',
        layers=WMS_LAYER_NAME,
        fmt='image/png',
        transparent=True,
        version='1.3.0',
        attr=WMS_ATTRIBUTION,
        overlay=False,
        control=True,
        show=True,
    ).add_to(map_obj)
    return map_obj


def create_folium_base_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 10,
    control_scale: bool = True,
    basemap: str = 'default',
) -> folium.Map:
    """Maak een Folium-basiskaart.

    basemap='default'     -> bestaand gedrag (OSM)
    basemap='bathymetry'  -> geen OSM, wel RWS bathymetrie-WMS
    """
    use_bathymetry = str(basemap).strip().lower() == 'bathymetry'
    if use_bathymetry:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            control_scale=control_scale,
            tiles=None,
        )
        add_bathymetry_wms(m)
        return m
    return folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        control_scale=control_scale,
    )
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
        # --- LEMNIDEN ---
        "Lemna gibba": "lemniden",
        "Lemna minor": "lemniden",
        "Lemna trisulca": "lemniden",
        "Spirodela polyrhiza": "lemniden"
    }




def add_species_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt soortgroep-kolommen toe aan de dataset, exclusief de algemene groeivorm-codes.

    Belangrijk:
    - 'soortgroep' behoudt het bestaande fallback-gedrag ('Overig / Individueel') voor compatibiliteit.
    - Kenmerkende soorten (N2000) tellen niet meer mee als soortgroep, maar worden als aparte entiteit gemarkeerd.
    - 'soortgroep_weergave' toont expliciet 'Geen match' voor soorten zonder mapping,
      zodat deze in grafieken en tellingen zichtbaar kunnen worden.
    - 'soortgroep_match_status' geeft per record aan of er wel/geen mapping is gevonden.
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
        df["soortgroep_weergave"] = pd.Series(dtype="object")
        df["soortgroep_match_status"] = pd.Series(dtype="object")
        df["is_kenmerkende_soort_n2000"] = pd.Series(dtype="bool")
        df["kenmerkende_soort_n2000_weergave"] = pd.Series(dtype="object")
        df["kenmerkende_soort_n2000_match_status"] = pd.Series(dtype="object")
        df["bedekkingsgraad_proc"] = pd.Series(dtype="float")
        return df

    soort = df["soort"].fillna("").astype(str).str.strip()
    genus = soort.str.split().str[0].fillna("")
    soortgroep = pd.Series("Overig / Individueel", index=df.index, dtype="object")

    if "Grootheid" in df.columns:
        mask_aanw = df["Grootheid"].astype(str) == "AANWZHD"
    else:
        mask_aanw = pd.Series(False, index=df.index)

    direct = soort.map(mapping)
    mask_direct = direct.notna() & (~mask_aanw)
    soortgroep.loc[mask_direct] = direct.loc[mask_direct].astype(str)

    mask_need = (soortgroep == "Overig / Individueel") & (~mask_aanw)
    genus_map = genus.map(mapping)
    mask_genus = mask_need & (genus != "Potamogeton") & genus_map.notna()
    soortgroep.loc[mask_genus] = genus_map.loc[mask_genus].astype(str)

    mask_match = mask_direct | mask_genus
    display_source = df["soort_display"] if "soort_display" in df.columns else soort

    df["soortgroep"] = soortgroep
    df["soortgroep_match_status"] = np.where(mask_match, "Match", "Geen match")
    df["soortgroep_weergave"] = np.where(mask_match, df["soortgroep"], "Geen match")

    # Kenmerkende soorten (N2000) als aparte entiteit
    df["is_kenmerkende_soort_n2000"] = mask_aanw.astype(bool)
    df["kenmerkende_soort_n2000_match_status"] = np.where(mask_aanw, "Match", "Geen match")
    df["kenmerkende_soort_n2000_weergave"] = np.where(mask_aanw, display_source.astype(str), "Geen match")

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


# =============================================================================
# SCHEMA HARDENING / MATCH DISPLAY KOL0MMEN
# =============================================================================
def _ensure_match_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Zorg dat match-/weergavekolommen voor KRW, trofieniveau en N2000 altijd aanwezig zijn.

    Dit maakt de app robuust voor oudere cached parquet-bestanden en voorkomt KeyErrors
    in pagina's die de nieuwe kolommen gebruiken.
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    if "trofisch_niveau" not in df.columns:
        df["trofisch_niveau"] = np.nan
    if "trofisch_niveau_match_status" not in df.columns:
        df["trofisch_niveau_match_status"] = np.where(
            df["trofisch_niveau"].notna() & (df["trofisch_niveau"].astype(str).str.strip() != ""),
            "Match",
            "Geen match",
        )
    if "trofisch_niveau_weergave" not in df.columns:
        df["trofisch_niveau_weergave"] = np.where(
            df["trofisch_niveau_match_status"] == "Match",
            df["trofisch_niveau"].astype(str),
            "Geen match",
        )

    if "krw_score" not in df.columns:
        df["krw_score"] = np.nan
    if "krw_class" not in df.columns:
        df["krw_class"] = pd.cut(
            pd.to_numeric(df["krw_score"], errors="coerce"),
            bins=[0, 2, 4, 5],
            labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
            include_lowest=True,
        )
    if "krw_match_status" not in df.columns:
        df["krw_match_status"] = np.where(pd.to_numeric(df["krw_score"], errors="coerce").notna(), "Match", "Geen match")
    if "krw_class_weergave" not in df.columns:
        df["krw_class_weergave"] = df["krw_class"].astype(object)
        df.loc[df["krw_class_weergave"].isna(), "krw_class_weergave"] = "Geen match"

    if "is_kenmerkende_soort_n2000" not in df.columns:
        if "Grootheid" in df.columns:
            df["is_kenmerkende_soort_n2000"] = df["Grootheid"].astype(str).eq("AANWZHD")
        else:
            df["is_kenmerkende_soort_n2000"] = False
    if "kenmerkende_soort_n2000_match_status" not in df.columns:
        df["kenmerkende_soort_n2000_match_status"] = np.where(
            df["is_kenmerkende_soort_n2000"].fillna(False),
            "Match",
            "Geen match",
        )
    if "kenmerkende_soort_n2000_weergave" not in df.columns:
        display_source = df["soort_display"] if "soort_display" in df.columns else (df["soort"] if "soort" in df.columns else pd.Series("", index=df.index))
        df["kenmerkende_soort_n2000_weergave"] = np.where(
            df["is_kenmerkende_soort_n2000"].fillna(False),
            display_source.astype(str),
            "Geen match",
        )

    return df


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
                df_final = _ensure_match_display_columns(df_final)
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
                final_df["trofisch_niveau_match_status"] = np.where(
                    final_df["trofisch_niveau"].notna() & (final_df["trofisch_niveau"].astype(str).str.strip() != ""),
                    "Match",
                    "Geen match",
                )
                final_df["trofisch_niveau_weergave"] = np.where(
                    final_df["trofisch_niveau_match_status"] == "Match",
                    final_df["trofisch_niveau"].astype(str),
                    "Geen match",
                )

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
                final_df["krw_match_status"] = np.where(final_df["krw_score"].notna(), "Match", "Geen match")
                final_df["krw_class_weergave"] = final_df["krw_class"].astype(object)
                final_df.loc[final_df["krw_class_weergave"].isna(), "krw_class_weergave"] = "Geen match"

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
                    "trofisch_niveau_weergave", "trofisch_niveau_match_status",
                    "krw_watertype", "krw_score", "krw_class", "krw_class_weergave", "krw_match_status", "soort_display"
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
    final_df["trofisch_niveau_match_status"] = np.where(
        final_df["trofisch_niveau"].notna() & (final_df["trofisch_niveau"].astype(str).str.strip() != ""),
        "Match",
        "Geen match",
    )
    final_df["trofisch_niveau_weergave"] = np.where(
        final_df["trofisch_niveau_match_status"] == "Match",
        final_df["trofisch_niveau"].astype(str),
        "Geen match",
    )

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
    final_df["krw_match_status"] = np.where(final_df["krw_score"].notna(), "Match", "Geen match")
    final_df["krw_class_weergave"] = final_df["krw_class"].astype(object)
    final_df.loc[final_df["krw_class_weergave"].isna(), "krw_class_weergave"] = "Geen match"
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
        "trofisch_niveau_weergave", "trofisch_niveau_match_status",
        "krw_watertype", "krw_score", "krw_class", "krw_class_weergave", "krw_match_status", "soort_display"
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
    """Public API: laadt verrijkte dataset (DuckDB+Parquet waar mogelijk)."""
    sig = _file_signature()
    return _load_data_cached(sig)


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in_clause(values: tuple[str, ...] | list[str]) -> str:
    vals = [v for v in values if str(v) != ""]
    if not vals:
        return "('')"
    return "(" + ", ".join(_sql_quote(v) for v in vals) + ")"


@st.cache_data(show_spinner=False)
def load_filtered_ecology_base(projects: tuple[str, ...] = tuple(), bodies: tuple[str, ...] = tuple()) -> pd.DataFrame:
    """Laad direct de gefilterde ecologische basisset, bij voorkeur via DuckDB op parquet."""
    if FINAL_PARQUET.exists() and duckdb is not None:
        con = _get_duckdb()
        if con is not None:
            where_parts = []
            if projects:
                where_parts.append(f"Project IN {_sql_in_clause(projects)}")
            if bodies:
                where_parts.append(f"Waterlichaam IN {_sql_in_clause(bodies)}")
            where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
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
        out = out[out["Project"].isin(projects)].copy()
    if bodies:
        out = out[out["Waterlichaam"].isin(bodies)].copy()
    return out


@st.cache_data(show_spinner=False)
def get_bubble_yearly_filtered(projects: tuple[str, ...] = tuple(), bodies: tuple[str, ...] = tuple()) -> pd.DataFrame:
    """Geaggregeerde bubble-brondata per soort x jaar, bij voorkeur direct via DuckDB."""
    if FINAL_PARQUET.exists() and duckdb is not None:
        con = _get_duckdb()
        if con is not None:
            where_parts = ["type = 'Soort'"]
            if projects:
                where_parts.append(f"Project IN {_sql_in_clause(projects)}")
            if bodies:
                where_parts.append(f"Waterlichaam IN {_sql_in_clause(bodies)}")
            where_sql = " AND ".join(where_parts)
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
        return pd.DataFrame(columns=["soort", "jaar", "doorzicht_m", "bedekking_pct", "diepte_m"])
    df_s = df_base[df_base["type"] == "Soort"].copy()
    for col in ["doorzicht_m", "diepte_m", "bedekking_pct"]:
        if col in df_s.columns:
            df_s[col] = pd.to_numeric(df_s[col], errors="coerce")
    if df_s.empty:
        return pd.DataFrame(columns=["soort", "jaar", "doorzicht_m", "bedekking_pct", "diepte_m"])
    return (
        df_s.groupby(["soort", "jaar"], as_index=False)
        .agg(
            doorzicht_m=("doorzicht_m", "mean"),
            bedekking_pct=("bedekking_pct", "mean"),
            diepte_m=("diepte_m", "mean"),
        )
    )


@st.cache_data(show_spinner=False)
def load_ecology_timeseries_data_filtered(
    project_sel: tuple[str, ...] = tuple(),
    body_sel: tuple[str, ...] = tuple(),
) -> pd.DataFrame:
    """Laad ecologische tijdreeksdata, maar filter zo vroeg mogelijk via DuckDB/parquet."""
    base = load_filtered_ecology_base(project_sel, body_sel)
    if base is None or base.empty:
        return pd.DataFrame()
    out = _ensure_match_display_columns(base.copy())
    out = _add_time_columns(out, "datum")
    out["bedekking_pct"] = pd.to_numeric(out.get("bedekking_pct"), errors="coerce")
    out["totaal_bedekking_locatie"] = pd.to_numeric(out.get("totaal_bedekking_locatie"), errors="coerce")
    out["krw_score"] = pd.to_numeric(out.get("krw_score"), errors="coerce")
    out["__row_id__"] = np.arange(len(out))
    mask_species = out.get("type", pd.Series(index=out.index, dtype="object")).astype(str).eq("Soort")
    species = out.loc[mask_species].copy()
    if not species.empty:
        species = add_species_group_columns(species)
        add_cols = [
            c for c in [
                "__row_id__", "soortgroep", "soortgroep_weergave", "soortgroep_match_status",
                "is_kenmerkende_soort_n2000", "kenmerkende_soort_n2000_weergave",
                "kenmerkende_soort_n2000_match_status", "bedekkingsgraad_proc",
            ] if c in species.columns
        ]
        if add_cols:
            out = out.merge(species[add_cols], on="__row_id__", how="left", suffixes=("", "_new"))
            for col in [c for c in add_cols if c != "__row_id__"]:
                new_col = f"{col}_new"
                if new_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].where(out[col].notna(), out[new_col])
                        out = out.drop(columns=[new_col])
                    else:
                        out = out.rename(columns={new_col: col})
    defaults = {
        "soortgroep": "Overig / Individueel",
        "soortgroep_weergave": "Geen match",
        "soortgroep_match_status": "Geen match",
        "kenmerkende_soort_n2000_weergave": "Geen match",
        "kenmerkende_soort_n2000_match_status": "Geen match",
        "bedekkingsgraad_proc": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    if "is_kenmerkende_soort_n2000" not in out.columns:
        out["is_kenmerkende_soort_n2000"] = False
    out["is_kenmerkende_soort_n2000"] = out["is_kenmerkende_soort_n2000"].fillna(False).astype(bool)
    out = out.drop(columns=["__row_id__"], errors="ignore")
    return out


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


def get_color_total_bedekking(value):
    """Specifieke kleurindeling voor totale bedekking in de ruimtelijke analyse."""
    if pd.isna(value):
        return "gray"
    try:
        v = float(value)
    except Exception:
        return "gray"
    if v <= 0:
        return "#808080"  # grijs
    elif v < 1:
        return "#006400"  # donkergroen
    elif v < 5:
        return "#2ca02c"  # groen
    elif v < 15:
        return "#ffd700"  # geel
    elif v < 25:
        return "#fdb462"  # lichtoranje
    elif v < 50:
        return "#ff7f0e"  # oranje
    elif v < 75:
        return "#d95f02"  # donkeroranje
    return "#d73027"  # rood


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
    basemap: str = "default",
):
    """
    Folium kaart met pie-chart markers (SVG via DivIcon) per locatie.
    basemap="bathymetry" gebruikt de RWS bathymetrie-WMS i.p.v. OSM.
    FIX: return pas na loop (anders slechts 1 marker).
    """
    if df_locs["lat"].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat, center_lon = df_locs["lat"].mean(), df_locs["lon"].mean()

    m = create_folium_base_map(center_lat, center_lon, zoom_start=zoom_start, control_scale=True, basemap=basemap)

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


def create_map(dataframe, mode, label_veg="Vegetatie", value_style="vegetation", category_col=None, category_color_map=None, basemap: str = "default"):
    """
    Genereert een Folium kaart.
    basemap="default" behoudt OSM; basemap="bathymetry" gebruikt de RWS bathymetrie-WMS.
    FIX: return pas na loop (anders slechts 1 marker).
    """
    if dataframe["lat"].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat = dataframe["lat"].mean()
        center_lon = dataframe["lon"].mean()

    m = create_folium_base_map(center_lat, center_lon, zoom_start=10, control_scale=True, basemap=basemap)

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
                elif value_style == "total_bedekking":
                    color = get_color_total_bedekking(val)
                    main_line = f"<b>🌱 {label_veg}:</b> {val:.1f}%"
                    radius = 4 + (min(val, 100) / 100 * 6) if val > 0 else 4
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


# =============================================================================
# CHEMIE VS ECOLOGIE – TIJDREEKSHELPERS
# =============================================================================
CHEMISTRY_FILE_PATH = "FC Waterplanten IJG.csv"
CHEMISTRY_PARQUET = CACHE_DIR / "chemistry_timeseries.parquet"
CHEMISTRY_PIPELINE_VERSION = "2026-03-30_fc_waterplanten_ijg_eventwaarde_text_hotfix_v2"
CHEM_PARAM_SUGGESTIONS = ["Ntot", "Ptot", "TOC", "NO3", "NO2", "NH4", "PO4", "CHLFa", "O2", "HCO3"]
CHEM_LOCATION_PREFERENCES = {
 "Drontermeer": ["drontermeerdijk.km0p4", "reve", "reevediep"],
 "Vossemeer": ["reve", "drontermeerdijk.km0p4", "reevediep"],
 "Eemmeer": ["eemmeerdijk.km23"],
 "IJmeer": ["pampus.oost", "markermeer.midden"],
 "Veluwemeer": ["veluwemeer.midden"],
 "Ketelmeer": ["swifterbant.ketelmeer", "drontermeerdijk.km0p4"],
 "Zwartemeer": ["ramsdiep", "swifterbant.ketelmeer"],
 "Wolderwijd": ["veluwemeer.midden"],
 "Nuldernauw": ["veluwemeer.midden", "eemmeerdijk.km23"],
 "Nijkerkernauw": ["veluwemeer.midden", "eemmeerdijk.km23"],
 "Gouwzee": ["markengouwzee", "markermeer.midden"],
 "Gooimeer": ["eemmeerdijk.km23", "pampus.oost"],
 "IJsselmeer": ["vrouwezand", "andijk.ijsselmeer", "lelystad.houtribhoek"],
 "Markermeer": ["markermeer.midden", "hoornschehop", "markengouwzee", "lelystad.haven"],
 "Randmeren": ["drontermeerdijk.km0p4", "eemmeerdijk.km23", "ramsdiep", "reevediep", "reve", "swifterbant.ketelmeer", "veluwemeer.midden"],
}
CHEM_MARKER_COLOR = "#8e44ad"

SEASON_ORDER = ["Voorjaar", "Zomer", "Najaar", "Winter"]
SEASON_MONTH_MAP = {
    3: "Voorjaar", 4: "Voorjaar", 5: "Voorjaar",
    6: "Zomer", 7: "Zomer", 8: "Zomer",
    9: "Najaar", 10: "Najaar", 11: "Najaar",
    12: "Winter", 1: "Winter", 2: "Winter",
}


def _normalize_season_value(value: str) -> str:
    lookup = {
        "voorjaar": "Voorjaar",
        "lente": "Voorjaar",
        "zomer": "Zomer",
        "najaar": "Najaar",
        "herfst": "Najaar",
        "winter": "Winter",
    }
    key = str(value or "").strip().lower()
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
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["maand"] = out[date_col].dt.month
    out["jaar"] = pd.to_numeric(out.get("jaar"), errors="coerce")
    out.loc[out["jaar"].isna(), "jaar"] = out.loc[out["jaar"].isna(), date_col].dt.year
    out["seizoen"] = out["maand"].map(SEASON_MONTH_MAP)
    return out


def _filter_seasons(df: pd.DataFrame, seasons: tuple[str, ...] | list[str] | None) -> pd.DataFrame:
    normalized = _normalize_seasons(seasons)
    if not normalized:
        return df.copy()
    if "seizoen" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["seizoen"].isin(normalized)].copy()


def _chemistry_file_signature(path: str = CHEMISTRY_FILE_PATH) -> tuple[str, float, str]:
    csv_path = Path(path)
    return (
        str(csv_path.resolve()) if csv_path.exists() else str(csv_path),
        _mtime_or_zero(csv_path),
        CHEMISTRY_PIPELINE_VERSION,
    )


def _detect_csv_delimiter(path: Path, default: str = ",") -> str:
    """Probeer delimiter robuust te detecteren op basis van een kleine sample."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            sample = f.read(65536)
        if not sample:
            return default
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;	|")
            return getattr(dialect, "delimiter", default) or default
        except Exception:
            counts = {sep: sample.count(sep) for sep in [";", ",", "	", "|"]}
            return max(counts, key=counts.get) if counts else default
    except Exception:
        return default


def _read_chemistry_csv_pandas(path: Path, usecols_func, sep: str, chunksize: int | None = None) -> pd.DataFrame:
    dtype_map = {
        "locatie_code": "string",
        "locatie_lat_etrs89": "string",
        "locatie_lon_etrs89": "string",
        "parameter_code": "string",
        "parameter_omschrijving": "string",
        "hoedanigheid_code": "string",
        "eenheid_code": "string",
        "eenheid_omschrijving": "string",
        "status_waarde": "string",
        "eventdatum": "string",
        "event_datum": "string",
        "event_waarde": "string",
        "event_waarde_limietsymbool": "string",
    }
    kwargs = dict(
        sep=sep,
        encoding="utf-8-sig",
        low_memory=True,
        usecols=usecols_func,
        dtype=dtype_map,
        on_bad_lines="skip",
    )
    if chunksize is not None:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _read_chemistry_csv_duckdb(path: Path, preferred_cols: list[str]) -> pd.DataFrame:
    """Gebruik DuckDB als snelle fallback voor grote CSV-bestanden."""
    if duckdb is None or not path.exists():
        return pd.DataFrame()
    con = _get_duckdb()
    if con is None:
        return pd.DataFrame()
    select_parts = [f'"{c}"' for c in preferred_cols]
    select_sql = ", ".join(select_parts)
    safe_path = path.as_posix().replace("'", "''")
    for delim in [None, ";", ",", "	", "|"]:
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
    # Belangrijk: hoedanigheid_code moet expliciet meegelezen worden voor de
    # speciale NVT-mapping (o.a. waterstofcarbonaat).
    if "hoedanigheid_code" not in preferred_cols:
        preferred_cols.append("hoedanigheid_code")
    usecols_func = lambda c: str(c).strip() in set(preferred_cols)

    delimiters = []
    detected = _detect_csv_delimiter(csv_path, default=",")
    for sep in [detected, ";", ",", "	", "|"]:
        if sep not in delimiters:
            delimiters.append(sep)

    # 1) Snel pad: direct relevante kolommen inlezen
    for sep in delimiters:
        try:
            df = _read_chemistry_csv_pandas(csv_path, usecols_func=usecols_func, sep=sep, chunksize=None)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    # 2) Chunked fallback voor zeer grote bestanden / parserproblemen
    for sep in delimiters:
        try:
            df = _read_chemistry_csv_pandas(csv_path, usecols_func=usecols_func, sep=sep, chunksize=250_000)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    # 3) DuckDB fallback
    try:
        df = _read_chemistry_csv_duckdb(csv_path, preferred_cols=preferred_cols)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 4) Ultieme fallback: volledige read_csv in pandas (traag maar compatibel)
    for sep in delimiters:
        try:
            df = pd.read_csv(
                csv_path,
                sep=sep,
                encoding="utf-8-sig",
                low_memory=False,
                on_bad_lines="skip",
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()


def _clean_chem_numeric_strings(series: pd.Series) -> pd.Series:
    """Maak numerieke chemie-strings parse-klaar zonder broninformatie te verliezen.
    Ondersteunt o.a. waarden als '+.1460E+00', komma-decimalen en limietsymbolen.
    """
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
    """Interne cached loader met parquet-cache voor grote chemiebestanden."""
    csv_path = Path(path)

    if CHEMISTRY_PARQUET.exists() and _mtime_or_zero(CHEMISTRY_PARQUET) >= _mtime_or_zero(csv_path):
        try:
            cached = _read_parquet_to_pandas(CHEMISTRY_PARQUET)
            if cached is not None and not cached.empty:
                return cached
        except Exception:
            pass

    required = [
        'locatie_code',
        'locatie_lat_etrs89',
        'locatie_lon_etrs89',
        'parameter_code',
        'parameter_omschrijving',
        'hoedanigheid_code',
        'eenheid_code',
        'eenheid_omschrijving',
        'status_waarde',
        'eventdatum',
        'event_datum',
        'event_waarde',
        'event_waarde_tekst',
        'event_waarde_limietsymbool',
    ]

    df = _read_chemistry_raw(path, required=required)
    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    # Ondersteun zowel eventdatum als event_datum
    if 'event_datum' in df.columns and 'eventdatum' not in df.columns:
        df = df.rename(columns={'event_datum': 'eventdatum'})

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Standaardiseer tijd en numerieke waarde
    df = _add_time_columns(df, 'eventdatum')

    # Gebruik event_waarde_tekst als primaire bron voor numerieke parsing,
    # omdat dit veld in de bronbestanden vaak de meest precieze representatie
    # bevat (bijv. '+.1460E+00'), terwijl event_waarde incidenteel decimalen
    # kan missen of verschuiven (bijv. 146 i.p.v. 0.146).
    raw_num = pd.to_numeric(
        _clean_chem_numeric_strings(df.get('event_waarde', pd.Series(np.nan, index=df.index))),
        errors='coerce',
    )
    text_num = pd.to_numeric(
        _clean_chem_numeric_strings(df.get('event_waarde_tekst', pd.Series(np.nan, index=df.index))),
        errors='coerce',
    )

    # Primaire keuze: event_waarde_tekst; fallback: event_waarde.
    df['event_waarde_num'] = text_num.combine_first(raw_num)

    # Diagnosekolommen voor datakwaliteit / auditing.
    df['event_waarde_conflict'] = (
        raw_num.notna()
        & text_num.notna()
        & ((raw_num - text_num).abs() > (text_num.abs() * 0.01 + 1e-12))
    )
    df['event_waarde_bron'] = np.where(text_num.notna(), 'event_waarde_tekst', 'event_waarde')

    # Veiligheidsgrens voor onrealistische waarden.
    df.loc[df['event_waarde_num'] >= 1e10, 'event_waarde_num'] = np.nan

    # Basisvelden / aliases conform gevraagde mapping
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
    stofnaam = np.where(mask_nvt & hoedanigheid_code.eq('CaCO3'), 'waterstofcarbonaat', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('dm'), 'doorzicht', stofnaam)
    stofnaam = np.where(mask_nvt & eenheid_code.eq('oC'), 'temperatuur', stofnaam)
    stofnaam = np.where(
        mask_nvt & eenheid_code.eq('DIMSLS') & df['event_waarde_num'].gt(6) & df['event_waarde_num'].lt(10),
        'zuurgraad',
        stofnaam,
    )

    df['stofnaam'] = pd.Series(stofnaam, index=df.index).astype(str).str.strip()
    df.loc[df['stofnaam'].eq('') & parameter_code.ne(''), 'stofnaam'] = parameter_code[parameter_code.ne('')]
    df.loc[df['stofnaam'].eq(''), 'stofnaam'] = 'Onbekend'

    df['parameter_omschrijving'] = df['stofnaam']
    df['parameter_code'] = parameter_code

    df['eenheid_code'] = df['eenheid_code'].fillna('').astype(str).str.strip()
    df['eenheid_omschrijving'] = df['eenheid_omschrijving'].fillna('').astype(str).str.strip()
    df['eenheid_label'] = np.where(
        df['eenheid_omschrijving'] != '',
        df['eenheid_omschrijving'],
        df['eenheid_code'],
    )

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

    # Houd alleen relevante kolommen over om geheugen te sparen voor grote bronbestanden.
    keep_cols = [
        'locatie_code', 'locatie_lat_etrs89', 'locatie_lon_etrs89', 'meetlocatie_naam',
        'x_coord', 'y_coord', 'parameter_code', 'parameter_omschrijving', 'stofnaam',
        'hoedanigheid_code', 'eenheid_code', 'eenheid_omschrijving', 'eenheid_label', 'eenheid',
        'status_waarde', 'eventdatum', 'datum_bemonstering', 'event_waarde',
        'event_waarde_limietsymbool', 'event_waarde_num', 'event_waarde_conflict',
        'event_waarde_bron', 'resultaat', 'jaar', 'maand',
        'seizoen', 'chem_label',
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
    """Laad en verrijk chemische data voor koppeling met ecologische tijdreeksen.

    Robuust voor grote bronbestanden zoals FC Waterplanten IJG.csv door:
    - alleen relevante kolommen te lezen waar mogelijk;
    - delimiterdetectie + meerdere fallbacks;
    - chunked pandas fallback;
    - DuckDB fallback;
    - een parquet-cache van de genormaliseerde chemische tijdreeks.
    """
    sig = _chemistry_file_signature(path)
    return _load_chemistry_data_cached(sig, path)

@st.cache_data(show_spinner=False)
def load_ecology_timeseries_data() -> pd.DataFrame:
    """Laad ecologische data voor tijdreekskoppelingen zonder groeivormen te verliezen."""
    df = load_data()
    if df is None or df.empty:
        return pd.DataFrame()

    out = _ensure_match_display_columns(df.copy())
    out = _add_time_columns(out, "datum")
    out["bedekking_pct"] = pd.to_numeric(out.get("bedekking_pct"), errors="coerce")
    out["totaal_bedekking_locatie"] = pd.to_numeric(out.get("totaal_bedekking_locatie"), errors="coerce")
    out["krw_score"] = pd.to_numeric(out.get("krw_score"), errors="coerce")
    out["__row_id__"] = np.arange(len(out))

    mask_species = out.get("type", pd.Series(index=out.index, dtype="object")).astype(str).eq("Soort")
    species = out.loc[mask_species].copy()
    if not species.empty:
        species = add_species_group_columns(species)
        add_cols = [
            c for c in [
                "__row_id__", "soortgroep", "soortgroep_weergave", "soortgroep_match_status",
                "is_kenmerkende_soort_n2000", "kenmerkende_soort_n2000_weergave",
                "kenmerkende_soort_n2000_match_status", "bedekkingsgraad_proc",
            ] if c in species.columns
        ]
        if add_cols:
            out = out.merge(species[add_cols], on="__row_id__", how="left", suffixes=("", "_new"))
            for col in [c for c in add_cols if c != "__row_id__"]:
                new_col = f"{col}_new"
                if new_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].where(out[col].notna(), out[new_col])
                        out = out.drop(columns=[new_col])
                    else:
                        out = out.rename(columns={new_col: col})

    defaults = {
        "soortgroep": "Overig / Individueel",
        "soortgroep_weergave": "Geen match",
        "soortgroep_match_status": "Geen match",
        "kenmerkende_soort_n2000_weergave": "Geen match",
        "kenmerkende_soort_n2000_match_status": "Geen match",
        "bedekkingsgraad_proc": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)

    if "is_kenmerkende_soort_n2000" not in out.columns:
        out["is_kenmerkende_soort_n2000"] = False
    out["is_kenmerkende_soort_n2000"] = out["is_kenmerkende_soort_n2000"].fillna(False).astype(bool)
    out = out.drop(columns=["__row_id__"], errors="ignore")
    return out


@st.cache_data(show_spinner=False)
def get_chemistry_location_points(
    body_sel: tuple[str, ...] | list[str] | None = None,
    preferred_only: bool = False,
    df_chem: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Retourneer unieke chemische meetlocaties met kaartcoördinaten.

    Indien preferred_only=True en er waterlichamen zijn geselecteerd, wordt de set
    beperkt tot de in CHEM_LOCATION_PREFERENCES gedefinieerde gekoppelde locaties.
    """
    if df_chem is None or df_chem.empty:
        df_chem = load_chemistry_data()
    if df_chem is None or df_chem.empty:
        return pd.DataFrame(columns=["locatie_code", "meetlocatie_naam", "chem_lat", "chem_lon", "n_records", "n_parameters"])

    d = df_chem.copy()
    for col in ["locatie_code", "meetlocatie_naam", "x_coord", "y_coord", "parameter_code"]:
        if col not in d.columns:
            d[col] = np.nan

    d["locatie_code"] = d["locatie_code"].fillna("").astype(str).str.strip()
    d["meetlocatie_naam"] = d["meetlocatie_naam"].fillna(d["locatie_code"]).astype(str).str.strip()
    d["chem_lat"] = pd.to_numeric(d["x_coord"], errors="coerce")
    d["chem_lon"] = pd.to_numeric(d["y_coord"], errors="coerce")
    d = d[(d["locatie_code"] != "") & d["chem_lat"].notna() & d["chem_lon"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["locatie_code", "meetlocatie_naam", "chem_lat", "chem_lon", "n_records", "n_parameters"])

    selected_bodies = [str(x).strip() for x in (body_sel or []) if str(x).strip()]
    if preferred_only and selected_bodies:
        preferred = []
        for body in selected_bodies:
            preferred.extend(CHEM_LOCATION_PREFERENCES.get(body, []))
        preferred = [x for x in preferred if str(x).strip()]
        if preferred:
            d = d[d["locatie_code"].isin(preferred)].copy()

    stats = (
        d.groupby("locatie_code", as_index=False)
        .agg(
            meetlocatie_naam=("meetlocatie_naam", "first"),
            chem_lat=("chem_lat", "first"),
            chem_lon=("chem_lon", "first"),
            n_records=("locatie_code", "size"),
            n_parameters=("parameter_code", lambda s: s.dropna().astype(str).str.strip().replace('', np.nan).dropna().nunique()),
        )
        .sort_values("locatie_code")
        .reset_index(drop=True)
    )
    return stats


def add_chemistry_locations_to_map(
    map_obj: folium.Map,
    df_points: pd.DataFrame | None,
    color: str = CHEM_MARKER_COLOR,
    label: str = "Chemische meetlocatie",
    radius: int = 8,
) -> folium.Map:
    """Voeg chemische meetlocaties toe als paarse ruitjes aan een bestaande Folium-kaart."""
    if map_obj is None or df_points is None or df_points.empty:
        return map_obj

    for row in df_points.dropna(subset=["chem_lat", "chem_lon"]).itertuples(index=False):
        loc_code = getattr(row, "locatie_code", "")
        meetnaam = getattr(row, "meetlocatie_naam", loc_code) or loc_code
        n_records = getattr(row, "n_records", None)
        n_parameters = getattr(row, "n_parameters", None)

        parts = [f"<b>{escape(label)}:</b> {escape(str(meetnaam))}"]
        if str(meetnaam) != str(loc_code) and str(loc_code).strip():
            parts.append(f"<b>Code:</b> {escape(str(loc_code))}")
        if n_records is not None and pd.notna(n_records):
            parts.append(f"<b>Records:</b> {int(n_records)}")
        if n_parameters is not None and pd.notna(n_parameters):
            parts.append(f"<b>Parameters:</b> {int(n_parameters)}")
        tooltip_html = "<br/>".join(parts)

        folium.RegularPolygonMarker(
            location=[float(getattr(row, "chem_lat")), float(getattr(row, "chem_lon"))],
            number_of_sides=4,
            rotation=45,
            radius=radius,
            color="#5b2c6f",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=tooltip_html,
        ).add_to(map_obj)
    return map_obj


def get_preferred_chemistry_locations(
    body_sel: tuple[str, ...] | list[str] | None,
    available_locations: list[str] | None,
) -> tuple[list[str], str | None, bool]:
    """Bepaal voorkeurslocaties voor chemie op basis van geselecteerde waterlichamen.

    Retourneert: (options, default_location, user_must_choose).
    Bij Randmeren of meerdere geselecteerde waterlichamen zonder eenduidige default
    moet de gebruiker zelf een locatie kiezen.
    """
    available = [str(x) for x in (available_locations or []) if str(x).strip()]
    if not available:
        return [], None, True

    selected_bodies = [str(x).strip() for x in (body_sel or []) if str(x).strip()]
    if not selected_bodies:
        default_loc = "veluwemeer.midden" if "veluwemeer.midden" in available else (available[0] if available else None)
        return available, default_loc, False

    if "Randmeren" in selected_bodies:
        return available, None, True

    preferred: list[str] = []
    for body in selected_bodies:
        preferred.extend(CHEM_LOCATION_PREFERENCES.get(body, []))

    preferred_available: list[str] = []
    for loc in preferred:
        if loc in available and loc not in preferred_available:
            preferred_available.append(loc)

    if not preferred_available:
        default_loc = "veluwemeer.midden" if "veluwemeer.midden" in available else (available[0] if available else None)
        return available, default_loc, False

    if len(selected_bodies) == 1:
        return preferred_available, preferred_available[0], False

    return preferred_available, None, True


@st.cache_data(show_spinner=False)
def get_available_chemistry_locations(df_chem: pd.DataFrame | None = None) -> list[str]:
    if df_chem is None or df_chem.empty:
        df_chem = load_chemistry_data()
    if df_chem.empty or "locatie_code" not in df_chem.columns:
        return []
    return sorted(df_chem["locatie_code"].dropna().astype(str).unique().tolist())


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
        df_chem[["chem_label", "parameter_code"]]
        .dropna(subset=["chem_label"])
        .drop_duplicates()
        .sort_values(["parameter_code", "chem_label"])
    )

    ordered: list[str] = []
    for code in suggestions:
        ordered.extend(labels.loc[labels["parameter_code"] == code, "chem_label"].tolist())
    ordered.extend([x for x in labels["chem_label"].tolist() if x not in ordered])
    return ordered


@st.cache_data(show_spinner=False)
def aggregate_chemistry_yearly(
    df_chem: pd.DataFrame,
    chemistry_labels: tuple[str, ...],
    location: str | None = None,
    definitive_only: bool = False,
    seasons: tuple[str, ...] = tuple(),
) -> pd.DataFrame:
    """Aggregeer 1..n chemische parameters naar jaargemiddelden.

    Belangrijk: er wordt óók op eenheid gegroepeerd, zodat varianten van dezelfde stof
    met verschillende eenheden nooit onbedoeld gemengd worden.
    """
    if df_chem is None or df_chem.empty or not chemistry_labels:
        return pd.DataFrame(columns=["jaar", "serie", "chem_value", "eenheid_code", "eenheid_omschrijving", "parameter_code"])

    d = df_chem.copy()
    d = d[d["chem_label"].isin(list(chemistry_labels))].copy()
    if location:
        d = d[d["locatie_code"].astype(str) == str(location)].copy()
    if definitive_only and "status_waarde" in d.columns:
        d = d[d["status_waarde"].astype(str).str.lower() == "definitief"].copy()
    d = _filter_seasons(d, seasons)
    d = d.dropna(subset=["jaar", "event_waarde_num"])
    if d.empty:
        return pd.DataFrame(columns=["jaar", "serie", "chem_value", "eenheid_code", "eenheid_omschrijving", "parameter_code"])

    # groepssleutels inclusief eenheid => juiste unit blijft behouden
    group_cols = ["jaar", "chem_label", "parameter_code", "eenheid_code", "eenheid_omschrijving"]
    for col in group_cols:
        if col not in d.columns:
            d[col] = ""

    out = (
        d.groupby(group_cols, as_index=False)
        .agg(chem_value=("event_waarde_num", "mean"))
        .rename(columns={"chem_label": "serie"})
    )
    out["jaar"] = pd.to_numeric(out["jaar"], errors="coerce").astype(int)
    return out[["jaar", "serie", "chem_value", "eenheid_code", "eenheid_omschrijving", "parameter_code"]]

@st.cache_data(show_spinner=False)
def aggregate_ecology_yearly(
    df_eco: pd.DataFrame,
    project_sel: tuple[str, ...],
    body_sel: tuple[str, ...],
    metric: str,
    mode: str = "default",
    seasons: tuple[str, ...] = tuple(),
    top_n: int | None = None,
) -> pd.DataFrame:
    """Aggregeer ecologische indicatoren naar jaartotalen/gemiddelden voor de dual-axis grafiek."""
    if df_eco is None or df_eco.empty:
        return pd.DataFrame(columns=["jaar", "serie", "waarde"])

    d = df_eco.copy()
    if project_sel:
        d = d[d["Project"].isin(project_sel)].copy()
    if body_sel:
        d = d[d["Waterlichaam"].isin(body_sel)].copy()
    d = _filter_seasons(d, seasons)
    d = d.dropna(subset=["jaar"])
    if d.empty:
        return pd.DataFrame(columns=["jaar", "serie", "waarde"])
    d["jaar"] = pd.to_numeric(d["jaar"], errors="coerce")
    d = d.dropna(subset=["jaar"])
    d["jaar"] = d["jaar"].astype(int)

    if metric == "Totale bedekking":
        x = d.dropna(subset=["totaal_bedekking_locatie"]).copy()
        if x.empty:
            return pd.DataFrame(columns=["jaar", "serie", "waarde"])
        x = x.groupby(["jaar", "CollectieReferentie"], as_index=False)["totaal_bedekking_locatie"].first()
        out = x.groupby("jaar", as_index=False)["totaal_bedekking_locatie"].mean()
        out = out.rename(columns={"totaal_bedekking_locatie": "waarde"})
        out["serie"] = "Totale bedekking"
        return out[["jaar", "serie", "waarde"]]

    if metric == "Groeivormen":
        x = d[d.get("type", "").astype(str) == "Groep"].copy()
        x = x.dropna(subset=["groeivorm", "bedekking_pct"])
        out = x.groupby(["jaar", "groeivorm"], as_index=False)["bedekking_pct"].mean()
        out = out.rename(columns={"groeivorm": "serie", "bedekking_pct": "waarde"})
    elif metric == "Soortgroep":
        x = d[d.get("type", "").astype(str) == "Soort"].copy()
        x["serie"] = x.get("soortgroep_weergave", pd.Series("Geen match", index=x.index)).fillna("Geen match").astype(str)
        x = x.dropna(subset=["bedekking_pct"])
        out = x.groupby(["jaar", "serie"], as_index=False)["bedekking_pct"].mean().rename(columns={"bedekking_pct": "waarde"})
    elif metric == "Trofieniveau":
        x = d[d.get("type", "").astype(str) == "Soort"].copy()
        serie = x.get("trofisch_niveau_weergave")
        if serie is None:
            serie = x.get("trofisch_niveau", pd.Series("Geen match", index=x.index))
        x["serie"] = serie.fillna("Geen match").astype(str)
        x = x.dropna(subset=["bedekking_pct"])
        out = x.groupby(["jaar", "serie"], as_index=False)["bedekking_pct"].mean().rename(columns={"bedekking_pct": "waarde"})
    elif metric == "KRW score":
        x = d[d.get("type", "").astype(str) == "Soort"].copy()
        if mode == "index":
            out = x.groupby("jaar", as_index=False)["krw_score"].mean().rename(columns={"krw_score": "waarde"})
            out["serie"] = "Gemiddelde KRW-score"
        else:
            serie = x.get("krw_class_weergave")
            if serie is None:
                serie = x.get("krw_class", pd.Series("Geen match", index=x.index))
            x["serie"] = serie.fillna("Geen match").astype(str)
            out = x.groupby(["jaar", "serie"], as_index=False)["bedekking_pct"].mean().rename(columns={"bedekking_pct": "waarde"})
    elif metric == "Kenmerkende soort (N2000)":
        x = d[d.get("is_kenmerkende_soort_n2000", pd.Series(False, index=d.index)).fillna(False)].copy()
        if x.empty:
            return pd.DataFrame(columns=["jaar", "serie", "waarde"])
        if mode == "records":
            out = x.groupby("jaar", as_index=False).size().rename(columns={"size": "waarde"})
            out["serie"] = "N2000 aanwezigheidsrecords"
            return out[["jaar", "serie", "waarde"]]
        serie = x.get("kenmerkende_soort_n2000_weergave")
        if serie is None:
            serie = x.get("soort_display", x.get("soort", pd.Series("N2000", index=x.index)))
        x["serie"] = serie.fillna("Geen match").astype(str)
        out = x.groupby(["jaar", "serie"], as_index=False).size().rename(columns={"size": "waarde"})
    else:
        return pd.DataFrame(columns=["jaar", "serie", "waarde"])

    if out.empty:
        return pd.DataFrame(columns=["jaar", "serie", "waarde"])

    if top_n is not None and "serie" in out.columns and metric in {"Groeivormen", "Soortgroep", "Trofieniveau", "Kenmerkende soort (N2000)"}:
        totals = out.groupby("serie", as_index=False)["waarde"].sum().sort_values("waarde", ascending=False)
        keep = totals.head(int(top_n))["serie"].tolist()
        out = out[out["serie"].isin(keep)].copy()

    return out[["jaar", "serie", "waarde"]]




@st.cache_data(show_spinner=False)
def summarize_chemistry_period_average(
    df_chem_year: pd.DataFrame,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    """Bereken per geselecteerde stof het gemiddelde over de gekozen periode.

    Verwacht jaargeaggregeerde chemische data (bijv. jaargemiddelden na seizoensfilter).
    Als slechts Zomer is geselecteerd en de periode 2015-2020 is, dan wordt hier dus
    per stof het gemiddelde van de zomergemiddelden over 2015 t/m 2020 berekend.
    """
    if df_chem_year is None or df_chem_year.empty:
        return pd.DataFrame(columns=["stof", "gemiddelde_geselecteerde_periode", "eenheid_omschrijving", "aantal_jaren"])

    d = df_chem_year.copy()
    d["jaar"] = pd.to_numeric(d.get("jaar"), errors="coerce")
    d = d.dropna(subset=["jaar", "chem_value"])
    if year_min is not None:
        d = d[d["jaar"] >= int(year_min)]
    if year_max is not None:
        d = d[d["jaar"] <= int(year_max)]
    if d.empty:
        return pd.DataFrame(columns=["stof", "gemiddelde_geselecteerde_periode", "eenheid_omschrijving", "aantal_jaren"])

    out = (
        d.groupby("serie", as_index=False)
        .agg(
            gemiddelde_geselecteerde_periode=("chem_value", "mean"),
            eenheid_omschrijving=("eenheid_omschrijving", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
            aantal_jaren=("jaar", "nunique"),
        )
        .rename(columns={"serie": "stof"})
        .sort_values("stof")
    )
    return out

@st.cache_data(show_spinner=False)
def get_chem_ecology_timeseries(
    df_eco: pd.DataFrame,
    df_chem: pd.DataFrame,
    project_sel: tuple[str, ...],
    body_sel: tuple[str, ...],
    ecology_metric: str,
    chemistry_labels: tuple[str, ...],
    chemistry_location: str | None = None,
    ecology_mode: str = "default",
    definitive_only: bool = False,
    seasons: tuple[str, ...] = tuple(),
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Retourneer ecologische en chemische jaarreeksen op overlappende jaren.

    Belangrijk: `seasons` wordt hier bewust alléén op de chemische data toegepast.
    De ecologische bedekkingsgraad / indices blijven dus jaarrond beschikbaar en worden
    niet verwijderd wanneer een seizoen in de UI wordt uit- of aangezet.
    """
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

    common_years = sorted(set(pd.to_numeric(eco_year["jaar"], errors="coerce").dropna().astype(int)) & set(pd.to_numeric(chem_year["jaar"], errors="coerce").dropna().astype(int)))
    if not common_years:
        return eco_year.iloc[0:0].copy(), chem_year.iloc[0:0].copy(), []

    eco_year = eco_year[eco_year["jaar"].isin(common_years)].copy()
    chem_year = chem_year[chem_year["jaar"].isin(common_years)].copy()
    return eco_year, chem_year, common_years

