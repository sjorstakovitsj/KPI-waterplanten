from __future__ import annotations

# ============================================================================
# PROJECT / WATERLICHAAM
# ============================================================================
PROJECT_MAPPING = {
    'MWTL_WOP': 'KRW',
    'GRID_WOP': 'N2000',
}

WATERBODY_MAPPING = {
    'DRNMR': 'Drontermeer',
    'EEMMR': 'Eemmeer',
    'GOOIM': 'Gooimeer',
    'GOUWZ': 'Gouwzee',
    'IJMR': 'IJmeer',
    'IJSMR': 'IJsselmeer',
    'KETMR': 'Ketelmeer',
    'MRKMR': 'Markermeer',
    'NIJKN': 'Nijkerkernauw',
    'NULDN': 'Nuldernauw',
    'RNDMRN': 'Randmeren',
    'RNDMR': 'Randmeren',
    'VELWM': 'Veluwemeer',
    'VOSSM': 'Vossemeer',
    'WOLDW': 'Wolderwijd',
    'ZWTMR': 'Zwartemeer',
}

# ============================================================================
# ECOLOGIE / GROEIVORM / KRW
# ============================================================================
GROWTH_FORM_MAPPING = {
    'DRAADAGN': 'Draadalgen',
    'DRIJFBPTN': 'Drijvend',
    'EMSPTN': 'Emergent',
    'SUBMSPTN': 'Ondergedoken',
    'FLAB': 'FLAB',
    'KROOS': 'Kroos',
}

EXCLUDED_SPECIES_CODES = ['FLAB', 'KROOS', 'SUBMSPTN', 'DRAADAGN', 'DRIJFBPTN', 'EMSPTN', 'WATPTN']
RWS_GROEIVORM_CODES = EXCLUDED_SPECIES_CODES

KRW_WATERTYPE_BY_WATERLICHAAM = {
    'Markermeer': 'M21',
    'Gouwzee': 'M21',
    'IJmeer': 'M21',
    'IJsselmeer': 'M21',
    'Drontermeer': 'M14',
    'Eemmeer': 'M14',
    'Gooimeer': 'M14',
    'Ketelmeer': 'M14',
    'Nijkerkernauw': 'M14',
    'Nuldernauw': 'M14',
    'Randmeren': 'M14',
    'Veluwemeer': 'M14',
    'Vossemeer': 'M14',
    'Wolderwijd': 'M14',
    'Zwartemeer': 'M14',
}

# ============================================================================
# CHEMIE / SEIZOENEN
# ============================================================================
CHEM_PARAM_SUGGESTIONS = ['Ntot', 'Ptot', 'TOC', 'NO3', 'NO2', 'NH4', 'PO4', 'CHLFa', 'O2', 'HCO3']

CHEM_LOCATION_PREFERENCES = {
    'Drontermeer': ['drontermeerdijk.km0p4', 'reve', 'reevediep'],
    'Vossemeer': ['reve', 'drontermeerdijk.km0p4', 'reevediep'],
    'Eemmeer': ['eemmeerdijk.km23'],
    'IJmeer': ['pampus.oost', 'markermeer.midden'],
    'Veluwemeer': ['veluwemeer.midden'],
    'Ketelmeer': ['swifterbant.ketelmeer', 'drontermeerdijk.km0p4'],
    'Zwartemeer': ['ramsdiep', 'swifterbant.ketelmeer'],
    'Wolderwijd': ['veluwemeer.midden'],
    'Nuldernauw': ['veluwemeer.midden', 'eemmeerdijk.km23'],
    'Nijkerkernauw': ['veluwemeer.midden', 'eemmeerdijk.km23'],
    'Gouwzee': ['markengouwzee', 'markermeer.midden'],
    'Gooimeer': ['eemmeerdijk.km23', 'pampus.oost'],
    'IJsselmeer': ['vrouwezand', 'andijk.ijsselmeer', 'lelystad.houtribhoek'],
    'Markermeer': ['markermeer.midden', 'hoornschehop', 'markengouwzee', 'lelystad.haven'],
    'Randmeren': ['drontermeerdijk.km0p4', 'eemmeerdijk.km23', 'ramsdiep', 'reevediep', 'reve', 'swifterbant.ketelmeer', 'veluwemeer.midden'],
}

CHEM_MARKER_COLOR = '#8e44ad'

SEASON_ORDER = ['Voorjaar', 'Zomer', 'Najaar', 'Winter']
SEASON_MONTH_MAP = {
    3: 'Voorjaar', 4: 'Voorjaar', 5: 'Voorjaar',
    6: 'Zomer', 7: 'Zomer', 8: 'Zomer',
    9: 'Najaar', 10: 'Najaar', 11: 'Najaar',
    12: 'Winter', 1: 'Winter', 2: 'Winter',
}

# ============================================================================
# RUIMTELIJKE ANALYSE / LEGENDA'S / PIE-MAPS
# ============================================================================
PIE_TYPES = {'KRW score', 'Trofieniveau', 'Groeivormen', 'soortgroepen'}

TOTAL_BEDEKKING_LEGEND = [
    ('0%', '#808080'),
    ('0.01–1%', '#006400'),
    ('1–5%', '#2ca02c'),
    ('5–15%', '#ffd700'),
    ('15–25%', '#fdb462'),
    ('25–50%', '#ff7f0e'),
    ('50–75%', '#d95f02'),
    ('75–100%', '#d73027'),
]

GROEI_COLORS = {
    'Ondergedoken': '#2ca02c',
    'Draadalgen': '#c2a5cf',
    'Drijvend': '#ff7f0e',
    'Kroos': '#8c510a',
    'Emergent': '#ffd700',
    'FLAB': '#d62728',
    'Geen match': 'transparent',
    'Onbekend': 'transparent',
}

TROFIE_COLORS = {
    'oligotroof': '#1b9e77',
    'sterk eutroof': '#e7298a',
    'kroos': '#a6761d',
    'mesotroof': '#d95f02',
    'brak': '#66a61e',
    'Onbekend': '#666666',
    'eutroof': '#7570b3',
    'marien': '#e6ab02',
    'Geen match': 'transparent',
}

KRW_COLORS = {
    'Gunstig (1-2)': '#2ca02c',
    'Geen match': 'transparent',
    'Neutraal (3-4)': '#ff7f0e',
    'Ongewenst (5)': '#d62728',
}

SOORTGROEP_ORDER = [
    'chariden',
    'parvopotamiden',
    'myriophylliden',
    'elodeiden',
    'pepliden',
    'nymphaeiden',
    'lemniden',
    'isoetiden',
    'magnopotamiden',
    'vallisneriiden',
    'stratiotiden',
    'batrachiiden',
    'haptofyten',
    'Overig / Individueel',
    'Geen match',
]

SOORTGROEP_PALETTE = [
    '#1b9e77', '#7570b3', '#66a61e', '#a6761d', '#b2df8a', '#cab2d6', '#6a3d9a',
    '#d95f02', '#e7298a', '#e6ab02', '#1f78b4', '#fb9a99', '#fdbf6f', '#FFD700', '#bbbbbb',
    '#ff7f00', '#b15928', '#17becf',
]
