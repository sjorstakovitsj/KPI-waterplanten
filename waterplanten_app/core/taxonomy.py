from __future__ import annotations

"""Definitieve taxonomie-/soortgroephelperlaag.

Deze module is de enige bron voor:
- get_species_group_mapping()
- add_species_group_columns()
- get_sorted_species_list()

Repositories en services horen deze helpers direct uit
`waterplanten_app.core.taxonomy` te importeren.
"""

from typing import Dict

import numpy as np
import pandas as pd

from waterplanten_app.config.mappings import EXCLUDED_SPECIES_CODES, RWS_GROEIVORM_CODES


def get_species_group_mapping() -> Dict[str, str]:
    """Retourneert een dictionary die Latijnse namen mapt naar soortgroepen."""
    return {
        # --- CHARIDEN ---
        'Chara': 'chariden',
        'Chara aspera': 'chariden',
        'Chara canescens': 'chariden',
        'Chara connivens': 'chariden',
        'Chara contraria': 'chariden',
        'Chara globularis': 'chariden',
        'Chara hispida': 'chariden',
        'Chara major': 'chariden',
        'Chara vulgaris': 'chariden',
        'Chara virgata': 'chariden',
        'Nitella': 'chariden',
        'Nitella flexilis': 'chariden',
        'Nitella hyalina': 'chariden',
        'Nitella mucronata': 'chariden',
        'Nitella opaca': 'chariden',
        'Nitella translucens': 'chariden',
        'Nitellopsis obtusa': 'chariden',
        'Tolypella': 'chariden',
        'Tolypella intricata': 'chariden',
        'Tolypella prolifera': 'chariden',
        # --- ISOETIDEN ---
        'Isoetes': 'iseotiden',
        'Isoetes lacustris': 'iseotiden',
        'Isoetes echinospora': 'iseotiden',
        'Littorella uniflora': 'iseotiden',
        'Lobelia dortmanna': 'iseotiden',
        'Subularia aquatica': 'iseotiden',
        'Pilularia globulifera': 'iseotiden',
        # --- PARVOPOTAMIDEN ---
        'Potamogeton berchtoldii': 'parvopotamiden',
        'Potamogeton compressus': 'parvopotamiden',
        'Potamogeton acutifolius': 'parvopotamiden',
        'Potamogeton friesii': 'parvopotamiden',
        'Potamogeton pusillus': 'parvopotamiden',
        'Potamogeton trichoides': 'parvopotamiden',
        'Potamogeton obtusifolius': 'parvopotamiden',
        'Potamogeton pectinatus': 'parvopotamiden',
        'Stuckenia pectinata': 'parvopotamiden',
        'Zannichellia palustris': 'parvopotamiden',
        'Zannichellia palustris ssp. palustris': 'parvopotamiden',
        'Zannichellia palustris ssp. pedicellata': 'parvopotamiden',
        'Zannichellia': 'parvopotamiden',
        'Ruppia': 'parvopotamiden',
        'Ruppia cirrhosa': 'parvopotamiden',
        'Ruppia maritima': 'parvopotamiden',
        'Najas': 'parvopotamiden',
        'Najas marina': 'parvopotamiden',
        'Najas minor': 'parvopotamiden',
        # --- MAGNOPOTAMIDEN ---
        'Potamogeton lucens': 'magnopotamiden',
        'Potamogeton perfoliatus': 'magnopotamiden',
        'Potamogeton alpinus': 'magnopotamiden',
        'Potamogeton praelongus': 'magnopotamiden',
        'Potamogeton gramineus': 'magnopotamiden',
        'Potamogeton coloratus': 'magnopotamiden',
        'Potamogeton nodosus': 'magnopotamiden',
        'Potamogeton crispus': 'magnopotamiden',
        'Groenlandia densa': 'magnopotamiden',
        # --- MYRIOPHYLLIDEN ---
        'Myriophyllum': 'myriophylliden',
        'Myriophyllum spicatum': 'myriophylliden',
        'Myriophyllum verticillatum': 'myriophylliden',
        'Myriophyllum alterniflorum': 'myriophylliden',
        'Myriophyllum heterophyllum': 'myriophylliden',
        'Hottonia palustris': 'myriophylliden',
        # --- VALLISNERIIDEN ---
        'Vallisneria': 'vallisneriiden',
        'Vallisneria spiralis': 'vallisneriiden',
        # --- ELODEIDEN ---
        'Elodea': 'elodeiden',
        'Elodea canadensis': 'elodeiden',
        'Elodea nuttallii': 'elodeiden',
        'Elodea callitrichoides': 'elodeiden',
        'Egeria densa': 'elodeiden',
        'Lagarosiphon major': 'elodeiden',
        'Hydrilla verticillata': 'elodeiden',
        'Ceratophyllum': 'elodeiden',
        'Ceratophyllum demersum': 'elodeiden',
        'Ceratophyllum submersum': 'elodeiden',
        # --- STRATIOTIDEN ---
        'Stratiotes aloides': 'stratiotiden',
        # --- PEPLIDEN ---
        'Peplis portula': 'pepliden',
        'Lythrum portula': 'pepliden',
        # --- BATRACHIIDEN ---
        'Ranunculus': 'batrachiiden',
        'Ranunculus aquatilis': 'batrachiiden',
        'Ranunculus circinatus': 'batrachiiden',
        'Ranunculus fluitans': 'batrachiiden',
        'Ranunculus peltatus': 'batrachiiden',
        'Ranunculus penicillatus': 'batrachiiden',
        'Ranunculus trichophyllus': 'batrachiiden',
        'Ranunculus baudotii': 'batrachiiden',
        'Callitriche': 'batrachiiden',
        'Callitriche stagnalis': 'batrachiiden',
        'Callitriche platycarpa': 'batrachiiden',
        'Callitriche obtusangula': 'batrachiiden',
        'Callitriche cophocarpa': 'batrachiiden',
        'Callitriche hamulata': 'batrachiiden',
        'Callitriche truncata': 'batrachiiden',
        # --- NYMPHAEIDEN ---
        'Nuphar lutea': 'nymphaeiden',
        'Nymphaea alba': 'nymphaeiden',
        'Nymphaea candida': 'nymphaeiden',
        'Nymphoides peltata': 'nymphaeiden',
        'Potamogeton natans': 'nymphaeiden',
        'Persicaria amphibia': 'nymphaeiden',
        'Sparganium emersum': 'nymphaeiden',
        'Sagittaria sagittifolia': 'nymphaeiden',
        # --- HAPTOFYTEN ---
        'Fontinalis antipyretica': 'haptofyten',
        'Enteromorpha': 'haptofyten',
        'Enteromorpha intestinalis': 'haptofyten',
        'Ulva intestinalis': 'haptofyten',
        'Hydrodictyon reticulatum': 'haptofyten',
        'Cladophora': 'haptofyten',
        'Vaucheria': 'haptofyten',
        'Amblystegium varium': 'haptofyten',
        'Amblystegium fluviatile': 'haptofyten',
        'Leptodictyum riparium': 'haptofyten',
        # --- LEMNIDEN ---
        'Lemna gibba': 'lemniden',
        'Lemna minor': 'lemniden',
        'Lemna trisulca': 'lemniden',
        'Spirodela polyrhiza': 'lemniden',
    }


def add_species_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt soortgroep- en kenmerkende-soortkolommen toe aan de dataset."""
    mapping = get_species_group_mapping()
    df = df.copy()
    if 'soort' not in df.columns:
        return df

    # Filter verzamelcodes en type Groep
    df = df[~df['soort'].isin(EXCLUDED_SPECIES_CODES)]
    if 'type' in df.columns:
        df = df[df['type'] != 'Groep']

    if df.empty:
        df['soortgroep'] = pd.Series(dtype='object')
        df['soortgroep_weergave'] = pd.Series(dtype='object')
        df['soortgroep_match_status'] = pd.Series(dtype='object')
        df['is_kenmerkende_soort_n2000'] = pd.Series(dtype='bool')
        df['kenmerkende_soort_n2000_weergave'] = pd.Series(dtype='object')
        df['kenmerkende_soort_n2000_match_status'] = pd.Series(dtype='object')
        df['bedekkingsgraad_proc'] = pd.Series(dtype='float')
        return df

    soort = df['soort'].fillna('').astype(str).str.strip()
    genus = soort.str.split().str[0].fillna('')
    soortgroep = pd.Series('Overig / Individueel', index=df.index, dtype='object')

    if 'Grootheid' in df.columns:
        mask_aanw = df['Grootheid'].astype(str) == 'AANWZHD'
    else:
        mask_aanw = pd.Series(False, index=df.index)

    direct = soort.map(mapping)
    mask_direct = direct.notna() & (~mask_aanw)
    soortgroep.loc[mask_direct] = direct.loc[mask_direct].astype(str)

    mask_need = (soortgroep == 'Overig / Individueel') & (~mask_aanw)
    genus_map = genus.map(mapping)
    mask_genus = mask_need & (genus != 'Potamogeton') & genus_map.notna()
    soortgroep.loc[mask_genus] = genus_map.loc[mask_genus].astype(str)

    mask_match = mask_direct | mask_genus
    display_source = df['soort_display'] if 'soort_display' in df.columns else soort

    df['soortgroep'] = soortgroep
    df['soortgroep_match_status'] = np.where(mask_match, 'Match', 'Geen match')
    df['soortgroep_weergave'] = np.where(mask_match, df['soortgroep'], 'Geen match')

    # Kenmerkende soorten (N2000) als aparte entiteit
    df['is_kenmerkende_soort_n2000'] = mask_aanw.astype(bool)
    df['kenmerkende_soort_n2000_match_status'] = np.where(mask_aanw, 'Match', 'Geen match')
    df['kenmerkende_soort_n2000_weergave'] = np.where(mask_aanw, display_source.astype(str), 'Geen match')

    target_col = 'bedekking_pct'
    if target_col not in df.columns:
        target_col = 'waarde_bedekking' if 'waarde_bedekking' in df.columns else ('WaardeGemeten' if 'WaardeGemeten' in df.columns else None)

    if target_col is None:
        df['bedekkingsgraad_proc'] = 0.0
    else:
        s = df[target_col].fillna(0).astype(str).str.replace(',', '.', regex=False)
        s = s.str.replace('<', '', regex=False).str.replace('>', '', regex=False).str.strip()
        df['bedekkingsgraad_proc'] = pd.to_numeric(s, errors='coerce').fillna(0.0).astype(float)

    return df


def get_sorted_species_list(df: pd.DataFrame) -> list:
    """Gesorteerde lijst met individuele soorten (excl. verzamelcodes)."""
    if 'soort' not in df.columns:
        return []
    mask_not_excluded = ~df['soort'].isin(EXCLUDED_SPECIES_CODES)
    mask_type = (df['type'] != 'Groep') if ('type' in df.columns) else True
    species_df = df[mask_not_excluded & mask_type]
    return sorted(species_df['soort'].dropna().unique())





def normalize_krw_category(category) -> str | None:
    """Normaliseer diverse KRW-labels/scores naar de vaste presentatiecategorieën."""
    if category is None:
        return None
    text = str(category).strip()
    if not text:
        return None
    lower = text.lower()
    compact = lower.replace(' ', '').replace('_', '')
    if 'geenmatch' in compact or 'nomatch' in compact:
        return 'Geen match'
    if 'gunstig' in compact or '1-2' in compact or '(1-2)' in compact:
        return 'Gunstig (1-2)'
    if 'neutraal' in compact or '3-4' in compact or '(3-4)' in compact:
        return 'Neutraal (3-4)'
    if 'ongewenst' in compact or compact in {'5', '5.0'} or '(5)' in compact:
        return 'Ongewenst (5)'
    try:
        num = float(text.replace(',', '.'))
    except Exception:
        return text
    if num <= 2:
        return 'Gunstig (1-2)'
    if num <= 4:
        return 'Neutraal (3-4)'
    return 'Ongewenst (5)'



def build_soortgroep_color_map(dist: pd.DataFrame | None, soortgroep_order: list[str], soortgroep_palette: list[str]) -> tuple[dict[str, str], list[str]]:
    """Bouw een kleurmapping en presentatievolgorde voor soortgroepen op basis van aanwezige categorieën."""
    present = [] if dist is None or getattr(dist, 'empty', True) else [c for c in dist['categorie'].dropna().astype(str).unique().tolist()]
    ordered_present = [c for c in soortgroep_order if c in present]
    remaining = [c for c in present if c not in ordered_present]
    ordered = ordered_present + remaining

    color_map: dict[str, str] = {}
    for idx, category in enumerate(soortgroep_order):
        if idx < len(soortgroep_palette):
            color_map[category] = soortgroep_palette[idx]

    extra_palette = soortgroep_palette[len(soortgroep_order):] + soortgroep_palette[:len(soortgroep_order)]
    for idx, category in enumerate(remaining):
        color_map[category] = extra_palette[idx % len(extra_palette)]

    return color_map, ordered

__all__ = [
    'EXCLUDED_SPECIES_CODES',
    'RWS_GROEIVORM_CODES',
    'get_species_group_mapping',
    'add_species_group_columns',
    'get_sorted_species_list',
    'normalize_krw_category',
    'build_soortgroep_color_map',
]
