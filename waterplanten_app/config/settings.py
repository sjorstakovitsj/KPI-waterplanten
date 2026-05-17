from __future__ import annotations

from pathlib import Path

# ============================================================================
# BRONBESTANDEN / PADEN
# ============================================================================
FILE_PATH = 'AquaDeskMeasurementExport_RWS_20260222163559.csv'
SPECIES_LOOKUP_PATH = 'Koppeltabel_score_namen.csv'
BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / '.cache_waterplanten'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MEAS_PARQUET = CACHE_DIR / 'measurements.parquet'
FINAL_PARQUET = CACHE_DIR / 'final_df.parquet'
LOOKUP_PARQUET = CACHE_DIR / 'species_lookup.parquet'
COORD_CACHE_PARQUET = CACHE_DIR / 'coord_cache.parquet'
PIPELINE_VERSION = '2026-03-27_duckdb_parquet_coords_v4_n2000_as_entity'

# ============================================================================
# BRONSELECTIE / DD-ECO API V3
# ============================================================================
# Primaire bron voor ecologische data. Mogelijke waarden:
# - 'dd_eco_api_v3': observations + references ophalen via DD-ECO / DD-API V3
# - 'csv': bestaand AquaDesk-exportbestand gebruiken
DATA_SOURCE = 'dd_eco_api_v3'

# Hybride fallback: als de API-bron tijdelijk niet beschikbaar is, val terug op CSV.
# Zet op None of '' om fallback uit te schakelen.
DATA_SOURCE_FALLBACK = 'csv'

# DD-ECO / DD-API V3 providerinstellingen
DD_API_BASE_URL = 'https://ddapi-rws.ecosys.nl/v3/odata'
DD_API_OBSERVATIONS_ENDPOINT = '/observations'
DD_API_REFERENCES_ENDPOINT = '/references'
DD_API_ACCEPT_CRS = '4258'
DD_API_TIMEOUT_SECONDS = 60
DD_API_PAGE_SIZE = 5000
DD_API_MAX_PAGES = 1000

# Optionele query-tuning. Laat op None voor provider-defaults.
DD_API_DEFAULT_FILTER = None
DD_API_DEFAULT_SELECT = None
DD_API_DEFAULT_EXPAND = None

# Lokale cache voor ruwe API-downloads (voordat de bestaande verrijkingspipeline draait)
DD_API_MEAS_PARQUET = CACHE_DIR / 'dd_eco_measurements_raw.parquet'
DD_API_REFERENCES_PARQUET = CACHE_DIR / 'dd_eco_references.parquet'
DD_API_PIPELINE_VERSION = '2026-04-04_dd_eco_v3_raw_contract_v1'

# ============================================================================
# BATHYMETRIE / WMS
# ============================================================================
WMS_BASE_URL = 'https://geo.rijkswaterstaat.nl/services/ogc/gdr/bodemhoogte_ijsselmeergebied/ows'
WMS_LAYER_NAME = 'bodemhoogte_ijg_2022'
WMS_ATTRIBUTION = 'Rijkswaterstaat bathymetrie IJsselmeergebied'

# ============================================================================
# CHEMIE
# ============================================================================
CHEMISTRY_FILE_PATH = 'FC Waterplanten IJG.csv'
CHEMISTRY_PARQUET = CACHE_DIR / 'chemistry_timeseries.parquet'
CHEMISTRY_PIPELINE_VERSION = '2026-03-30_fc_waterplanten_ijg_eventwaarde_text_hotfix_v2'

# ============================================================================
# RUIMTELIJKE ANALYSE / KAARTDEFAULTS
# ============================================================================
SPATIAL_ANALYSIS_BASEMAP = 'bathymetry'
SPATIAL_PIE_SIZE_PX = 30
SPATIAL_PIE_ZOOM_START = 10
SPATIAL_PIE_FIXED_TOTAL = 100
SPATIAL_PIE_FILL_GAP = True
SPATIAL_PIE_GAP_COLOR = 'transparent'
