# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium 
import re
import json
import math
from html import escape


# --- CONFIGURATIE ---
FILE_PATH = "AquaDeskMeasurementExport_RWS_20260129204403.csv"
SPECIES_LOOKUP_PATH = "Koppeltabel_score_namen.csv"

# --- MAPPINGS ---
PROJECT_MAPPING = {
    'MWTL_WOP': 'KRW',
    'GRID_WOP': 'N2000'
}

WATERBODY_MAPPING = {
    'DRNMR': 'Drontermeer', 'EEMMR': 'Eemmeer', 'GOOIM': 'Gooimeer',
    'GOUWZ': 'Gouwzee', 'IJMR': 'IJmeer', 'IJSMR': 'IJsselmeer',
    'KETMR': 'Ketelmeer', 'MRKMR': 'Markermeer', 'NIJKN': 'Nijkerkernauw',
    'NULDN': 'Nuldernauw', 'RNDMRN': 'Randmeren', 'RNDMR': 'Randmeren',
    'VELWM': 'Veluwemeer', 'VOSSM': 'Vossemeer', 'WOLDW': 'Wolderwijd', 
    'ZWTMR': 'Zwartemeer'
}

# Groeivormen mapping voor specifieke groepen
GROWTH_FORM_MAPPING = {
    'DRAADAGN': 'Draadalgen',
    'DRIJFBPTN': 'Drijvend',
    'EMSPTN': 'Emergent',
    'SUBMSPTN': 'Ondergedoken',
    'FLAB': 'FLAB',
    'KROOS': 'Kroos'
}

# Lijst met codes die géén individuele soort zijn, maar een RWS-verzamelgroep
EXCLUDED_SPECIES_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]
RWS_GROEIVORM_CODES = EXCLUDED_SPECIES_CODES

def rd_to_wgs84(x, y):
    """Converteert Rijksdriehoek (RD) coördinaten naar WGS84 (Lat/Lon)."""
    try:
        x0 = 155000
        y0 = 463000
        phi0 = 52.15517440
        lam0 = 5.38720621
        dx = (x - x0) * 10**-5
        dy = (y - y0) * 10**-5
        sum_phi = (3235.65389 * dy) + (-32.58297 * dx**2) + (-0.24750 * dy**2) + \
                  (-0.84978 * dx**2 * dy) + (-0.06550 * dy**3) + \
                  (1.70776 * dx**2 * dy**2) + (-0.10715 * dy**4) + (0.009 * dy**5)
        sum_lam = (5260.52916 * dx) + (105.94684 * dx * dy) + (2.45656 * dx * dy**2) + \
                  (-0.81885 * dx**3) + (0.05594 * dx * dy**3) + \
                  (-0.05607 * dx**3 * dy) + (0.01199 * dy * dx**4) + (-0.00256 * dx**3 * dy**2)
        lat = phi0 + sum_phi / 3600
        lon = lam0 + sum_lam / 3600
        return lat, lon
    except:
        return None, None

def parse_coordinates(geo_str):
    """Parset coordinaten string naar X, Y, Lat, Lon"""
    try:
        clean_str = re.sub(r'[^\d\s.]', '', str(geo_str)).strip()
        parts = clean_str.split()
        if len(parts) >= 2:
            x = float(parts[0])
            y = float(parts[1])
            lat, lon = rd_to_wgs84(x, y)
            return x, y, lat, lon
    except:
        pass
    return None, None, None, None

def determine_waterbody(meetobject_code):
    for code, name in WATERBODY_MAPPING.items():
        if code in str(meetobject_code):
            return name
    return str(meetobject_code)

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


def get_species_group_mapping():
    """
    Retourneert een dictionary die Latijnse namen mapt naar de gevraagde soortgroepen.
    """
    return {
        # --- CHARIDEN (Kranswieren) ---
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

        # --- ISOETIDEN (Biesvormigen) ---
        'Isoetes': 'iseotiden',
        'Isoetes lacustris': 'iseotiden',
        'Isoetes echinospora': 'iseotiden',
        'Littorella uniflora': 'iseotiden',
        'Lobelia dortmanna': 'iseotiden',
        'Subularia aquatica': 'iseotiden',
        'Pilularia globulifera': 'iseotiden',

        # --- PARVOPOTAMIDEN (Smalbladige fonteinkruiden) ---
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
        
        # --- MAGNOPOTAMIDEN (Breedbladige fonteinkruiden) ---
        'Potamogeton lucens': 'magnopotamiden',
        'Potamogeton perfoliatus': 'magnopotamiden',
        'Potamogeton alpinus': 'magnopotamiden',
        'Potamogeton praelongus': 'magnopotamiden',
        'Potamogeton gramineus': 'magnopotamiden',
        'Potamogeton coloratus': 'magnopotamiden',
        'Potamogeton nodosus': 'magnopotamiden',
        'Potamogeton crispus': 'magnopotamiden',
        'Groenlandia densa': 'magnopotamiden',

        # --- MYRIOPHYLLIDEN (Vederkruiden) ---
        'Myriophyllum': 'myriophylliden',
        'Myriophyllum spicatum': 'myriophylliden',
        'Myriophyllum verticillatum': 'myriophylliden',
        'Myriophyllum alterniflorum': 'myriophylliden',
        'Myriophyllum heterophyllum': 'myriophylliden',
        'Hottonia palustris': 'myriophylliden',

        # --- VALLISNERIIDEN ---
        'Vallisneria': 'vallisneriiden',
        'Vallisneria spiralis': 'vallisneriiden',

        # --- ELODEIDEN (Waterpestachtigen) ---
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

        # --- STRATIOTIDEN (Krabbenscheer) ---
        'Stratiotes aloides': 'stratiotiden',

        # --- PEPLIDEN ---
        'Peplis portula': 'pepliden',
        'Lythrum portula': 'pepliden',

        # --- BATRACHIIDEN (Waterranonkels) ---
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

        # --- NYMPHAEIDEN (Drijfbladplanten) ---
        'Nuphar lutea': 'nymphaeiden',
        'Nymphaea alba': 'nymphaeiden',
        'Nymphaea candida': 'nymphaeiden',
        'Nymphoides peltata': 'nymphaeiden',
        'Potamogeton natans': 'nymphaeiden',
        'Persicaria amphibia': 'nymphaeiden',
        'Sparganium emersum': 'nymphaeiden',
        'Sagittaria sagittifolia': 'nymphaeiden',

        # --- HAPTOFYTEN (Vastzittende wieren/mossen) ---
        'Fontinalis antipyretica': 'haptofyten',
        'Enteromorpha': 'haptofyten',
        'Enteromorpha intestinalis': 'haptofyten',
        'Ulva intestinalis': 'haptofyten',
        'Hydrodictyon reticulatum': 'haptofyten',
        'Cladophora': 'haptofyten',
        'Vaucheria': 'haptofyten',
        'Chara': 'chariden',
        'Amblystegium varium': 'haptofyten',
        'Amblystegium fluviatile': 'haptofyten',
        'Leptodictyum riparium': 'haptofyten'
    }

def determine_group(row, mapping_dict):
    """
    Hulpfunctie om de groep te bepalen.
    CHECK: Kijkt naar 'soort', 'Parameter' en 'Grootheid'.
    """
    # Eerst checken op kenmerkende soorten o.b.v. Grootheid AANWZHD
    grootheid = row.get('Grootheid', '')
    if grootheid == 'AANWZHD':
        return 'Kenmerkende soort (N2000)'

    # Haal de waarde op uit 'soort' (nieuwe naam) of 'Parameter' (fallback)
    param_val = row.get('soort', row.get('Parameter', ''))
    param = str(param_val).strip()
    
    # 1. Directe match
    if param in mapping_dict:
        return mapping_dict[param]
    
    # 2. Check op Genus (eerste woord)
    genus = param.split(' ')[0]
    # Uitzondering voor Potamogeton (is opgesplitst)
    if genus == 'Potamogeton':
        pass 
    elif genus in mapping_dict:
        return mapping_dict[genus]
    
    # 3. Check bestaande mapping (GROWTH_FORM_MAPPING)
    if 'GROWTH_FORM_MAPPING' in globals() and param in globals()['GROWTH_FORM_MAPPING']:
        original_group = globals()['GROWTH_FORM_MAPPING'][param]
        if 'draadalg' in str(original_group).lower():
            return 'haptofyten'
        if 'kranswier' in str(original_group).lower():
            return 'chariden'
            
    return 'Overig / Individueel'

def add_species_group_columns(df):
    """
    Voegt 'soortgroep' toe aan de dataset, maar EXCLUSIEF de algemene groeivorm-codes.
    """
    mapping = get_species_group_mapping()
    df = df.copy()
    
    # --- STAP 1: FILTEREN ---
    # We sluiten rijen uit die algemene verzamelgroepen zijn.
    # We filteren op de kolom 'soort' (wat voorheen 'Parameter' was)
    df = df[~df['soort'].isin(EXCLUDED_SPECIES_CODES)]
    
    # Extra check op de kolom 'type' als back-up
    if 'type' in df.columns:
        df = df[df['type'] != 'Groep']

    # --- STAP 2: MAPPING TOEPASSEN ---
    # Nu bepalen we de soortgroep voor de overgebleven echte soorten
    df['soortgroep'] = df.apply(lambda row: determine_group(row, mapping), axis=1)
    
    # --- STAP 3: PERCENTAGES ---
    # Als Grootheid AANWZHD is, is de waarde vaak 1.0 (aanwezig) of 0.0 (afwezig).
    # We willen dit wel parsen, maar bewust zijn dat dit geen percentage bedekking is.
    target_col = 'bedekking_pct'
    if target_col not in df.columns:
        target_col = 'waarde_bedekking' if 'waarde_bedekking' in df.columns else 'WaardeGemeten'
    
    def parse_percentage(val):
        if pd.isna(val): return 0.0
        try:
            return float(str(val).replace(',', '.').replace('<', '').replace('>', '').strip())
        except: return 0.0

    df['bedekkingsgraad_proc'] = df[target_col].apply(parse_percentage)
    
    return df

def get_sorted_species_list(df):
    """
    Genereert een gesorteerde lijst van individuele soorten (uniek),
    waarbij verzamelgroepen (FLAB, etc.) worden uitgesloten.
    """
    # Filter op rijen die GEEN groep zijn (dus type='Soort') of 
    # expliciet in de exclude lijst staan
    mask_not_excluded = ~df['soort'].isin(EXCLUDED_SPECIES_CODES)
    mask_type_sort = df['type'] != 'Groep' if 'type' in df.columns else True
    
    species_df = df[mask_not_excluded & mask_type_sort]
    
    # Unieke waarden ophalen en sorteren
    unique_species = sorted(species_df['soort'].dropna().unique())
    return unique_species

@st.cache_data
def load_data():
    """
    Laadt data, splitst WATPTN van soorten, en merged alles terug.
    Inclusief verwerking van Grootheid 'AANWZHD' voor N2000 soorten.
    """
    try:
        df_raw = pd.read_csv(FILE_PATH, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        st.error(f"Fout bij inlezen '{FILE_PATH}': {e}")
        return pd.DataFrame()

    df_raw.columns = df_raw.columns.str.strip()
    
    if 'MetingDatumTijd' in df_raw.columns:
        df_raw['MetingDatumTijd'] = pd.to_datetime(df_raw['MetingDatumTijd'], dayfirst=True, errors='coerce')
        df_raw['datum'] = df_raw['MetingDatumTijd'].dt.floor('D')
        df_raw['jaar'] = df_raw['datum'].dt.year
    else:
        st.error("Kolom 'MetingDatumTijd' mist.")
        return pd.DataFrame()

    if 'Projecten' in df_raw.columns:
        df_raw['Project'] = df_raw['Projecten'].map(PROJECT_MAPPING).fillna(df_raw['Projecten'])
    
    if 'MeetObject' in df_raw.columns:
        df_raw['Waterlichaam'] = df_raw['MeetObject'].apply(determine_waterbody)

    if 'GeografieVorm' in df_raw.columns:
        coords = df_raw['GeografieVorm'].apply(parse_coordinates)
        df_raw['x_rd'] = [c[0] for c in coords]
        df_raw['y_rd'] = [c[1] for c in coords]
        df_raw['lat'] = [c[2] for c in coords]
        df_raw['lon'] = [c[3] for c in coords]

    # --- ABIOTIEK (Diepte, Zicht) ---
    df_abiotic = df_raw[df_raw['Grootheid'].isin(['DIEPTE', 'ZICHT'])].copy()
    if not df_abiotic.empty:
        df_env = df_abiotic.pivot_table(
            index='CollectieReferentie', 
            columns='Grootheid', 
            values='WaardeGemeten', 
            aggfunc='mean'
        ).reset_index()
    else:
        df_env = pd.DataFrame(columns=['CollectieReferentie'])

    if 'DIEPTE' in df_env.columns:
        df_env['diepte_m'] = df_env['DIEPTE'] / 100 
    else:
        df_env['diepte_m'] = np.nan

    if 'ZICHT' in df_env.columns:
        df_env['doorzicht_m'] = df_env['ZICHT'] / 10 
    else:
        df_env['doorzicht_m'] = np.nan

    # --- TOTALE BEDEKKING (WATPTN) ---
    df_total = df_raw[df_raw['Parameter'] == 'WATPTN'].copy()
    df_total = df_total[['CollectieReferentie', 'WaardeGemeten']].rename(
        columns={'WaardeGemeten': 'totaal_bedekking_locatie'}
    )
    df_total = df_total.groupby('CollectieReferentie', as_index=False).mean()

    # --- INDIVIDUELE SOORTEN EN KENMERKENDE SOORTEN ---
    # Hier filteren we op BEDKG (Bedekking) ÉN AANWZHD (Aanwezigheid)
    df_bedkg = df_raw[
        (df_raw['Grootheid'].isin(['BEDKG', 'AANWZHD'])) & 
        (df_raw['Parameter'] != 'WATPTN')
    ].copy()

    def classify_row(row):
        param = row['Parameter']
        grootheid = row['Grootheid']
        
        # Mapping voor Kenmerkende Soorten (N2000)
        if grootheid == 'AANWZHD':
            return 'Kenmerkende soort (N2000)', 'Soort'
            
        # Mapping voor Groeivormen
        if param in GROWTH_FORM_MAPPING:
            return GROWTH_FORM_MAPPING[param], 'Groep'
        else:
            return 'Individuele soort', 'Soort'

    classificatie = df_bedkg.apply(classify_row, axis=1)
    df_bedkg['groeivorm'] = [x[0] for x in classificatie]
    df_bedkg['type'] = [x[1] for x in classificatie]

    # Mergen van alle data
    df_merged = pd.merge(df_bedkg, df_env, on='CollectieReferentie', how='left')
    df_merged = pd.merge(df_merged, df_total, on='CollectieReferentie', how='left')   
    
    final_df = df_merged.rename(columns={
        'MeetObject': 'locatie_id',
        'Parameter': 'soort',
        'WaardeGemeten': 'waarde_bedekking', # Tijdelijke rename, wordt zo gefixed
        'EenheidGemeten': 'eenheid'
    })
    
    # Consistent maken van bedekkingskolom
    if 'waarde_bedekking' in final_df.columns:
        final_df['bedekking_pct'] = final_df['waarde_bedekking']

    cols_to_keep = [
        'datum', 'jaar', 'locatie_id', 'Waterlichaam', 'Project', 'CollectieReferentie',
        'soort', 'bedekking_pct', 'waarde_bedekking', 'totaal_bedekking_locatie', 
        'diepte_m', 'doorzicht_m', 'lat', 'lon', 'x_rd', 'y_rd',
        'groeivorm', 'type', 'Grootheid', 'soort_triviaal', 'trofisch_niveau', 'krw_watertype',
         'krw_score', 'krw_class', 'soort_display'
    ]
    
    # --- Verrijking: NL naam, trofisch niveau, KRW score (M14/M21) ---
    lookup = load_species_lookup()

    final_df["soort_norm"] = (
        final_df["soort"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    final_df = final_df.merge(lookup, on="soort_norm", how="left")

    final_df = final_df.rename(columns={
        "NL naam": "soort_triviaal",
        "Watertype": "trofisch_niveau"
    })

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
        include_lowest=True
    )

    final_df["soort_display"] = np.where(
        final_df["soort_triviaal"].notna() & (final_df["soort_triviaal"].astype(str).str.len() > 0),
        final_df["soort_triviaal"] + " (" + final_df["soort"] + ")",
        final_df["soort"]
    )

    for col in cols_to_keep:
        if col not in final_df.columns:
            final_df[col] = np.nan

    return final_df[cols_to_keep]

@st.cache_data
def load_species_lookup():
    """Laad koppeltabel met NL naam, trofie (Watertype) en KRW-scores (M14/M21). Robuust voor delimiters/ontbrekende kolommen."""
    try:
        df_lu = pd.read_csv(SPECIES_LOOKUP_PATH, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"⚠️ Koppeltabel kon niet worden ingelezen ({SPECIES_LOOKUP_PATH}): {e}")
        return pd.DataFrame(columns=["soort_norm", "NL naam", "Watertype", "M14", "M21"])

    df_lu.columns = df_lu.columns.str.strip()

    # Optioneel: vang alternatieve kolomnamen af
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

    # Minimaal required
    if "Wetenschappelijke naam" not in df_lu.columns:
        st.warning("⚠️ Koppeltabel mist kolom 'Wetenschappelijke naam'. Verrijking wordt overgeslagen.")
        return pd.DataFrame(columns=["soort_norm", "NL naam", "Watertype", "M14", "M21"])

    # Optioneel aanwezige kolommen: voeg toe als ze ontbreken (blijven NaN)
    if "NL naam" not in df_lu.columns:
        df_lu["NL naam"] = np.nan
    if "Watertype" not in df_lu.columns:
        df_lu["Watertype"] = np.nan
    for c in ["M14", "M21"]:
        if c not in df_lu.columns:
            df_lu[c] = np.nan
        df_lu[c] = pd.to_numeric(df_lu[c], errors="coerce")

    # Normaliseer sleutel
    df_lu["soort_norm"] = (
        df_lu["Wetenschappelijke naam"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Voorkom row-explosion door dubbelen
    sort_cols = [c for c in ["NL naam", "Watertype", "M14", "M21"] if c in df_lu.columns]
    if sort_cols:
        df_lu = df_lu.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    df_lu = df_lu.drop_duplicates(subset=["soort_norm"], keep="first")

    return df_lu[["soort_norm", "NL naam", "Watertype", "M14", "M21"]]

# --- PLOT FUNCTIES ---
def plot_trend_line(df, x_col, y_col, color=None, title="Trend"):
    """Genereert een standaard trendlijn plot."""
    fig = px.line(df, x=x_col, y=y_col, color=color, markers=True, title=title)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def interpret_soil_state(df_loc):
    """Genereert automatische tekstinterpretatie van bodemconditie."""
    if df_loc.empty:
        return "Geen data beschikbaar."
        
    total_cover = df_loc['totaal_bedekking_locatie'].mean()
    modes = df_loc['groeivorm'].mode()
    dom_type = modes[0] if not modes.empty else 'Onbekend'
    
    text = f"**Automatische Interpretatie:**\n"
    
    if pd.isna(total_cover):
        text += "Geen bedekkingsgegevens beschikbaar.\n"
    elif total_cover < 5:
        text += "⚠️ **Zeer kale bodem** (<5% bedekking).\n"
    elif dom_type == 'Ondergedoken':
        text += f"✅ Goede ontwikkeling (**{total_cover:.0f}%**).\n"
    elif dom_type == 'Drijvend':
        text += f"⚠️ Veel drijfbladplanten (**{total_cover:.0f}%**).\n"
    elif dom_type == 'Draadalgen':
        text += "❌ Dominantie van draadalgen wijst op verstoring.\n"
        
    return text

# --- HELPER FUNCTIES VOOR ANALYSE ---

def categorize_slope_trend(val, threshold):
    """Bepaalt de trendcategorie op basis van een drempelwaarde."""
    if val > threshold: return 'Verbeterend ↗️'
    elif val < -threshold: return 'Verslechterend ↘️'
    else: return 'Stabiel ➡️'

def get_color_absolute(val, min_v, max_v):
    """Geeft RGB kleur terug van Rood (laag) naar Groen (hoog)."""
    if pd.isna(val): return [200, 200, 200, 100]
    
    # Normaliseren 0-1
    norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
    norm = max(0, min(1, norm))
    
    # Simpele interpolatie Rood [255,0,0] naar Groen [0,255,0]
    r = int(255 * (1 - norm))
    g = int(255 * norm)
    b = 0
    return [r, g, b, 200] # alpha 200

def get_color_diff(val):
    """Geeft Rood (verslechtering), Grijs (stabiel), Groen (verbetering)."""
    threshold = 0.5 # Drempelwaarde voor relevant verschil
    if val < -threshold:
        return [255, 0, 0, 200] # Rood
    elif val > threshold:
        return [0, 255, 0, 200] # Groen
    else:
        return [128, 128, 128, 100] # Grijs

# --- AGGREGATIE EN KPI FUNCTIES (Vanuit 1_Overzicht.py) ---

def get_location_metric_mean(dataframe, metric_col):
    """
    Berekent gemiddelde van een locatie-parameter (zoals WATPTN/Totale Bedekking).
    Stap 1: Unieke waarde per monstername (CollectieReferentie) pakken.
    Stap 2: Gemiddelde daarvan nemen.
    """
    if dataframe.empty: return 0.0
    # Group by Sample ID -> take first value (want die is constant voor het sample)
    per_sample = dataframe.groupby('CollectieReferentie')[metric_col].first()
    return per_sample.mean()

def calculate_kpi(curr_df, prev_df, metric_col, is_loc_metric=False):
    if curr_df.empty: return 0.0, 0.0
    
    if is_loc_metric:
        curr_val = get_location_metric_mean(curr_df, metric_col)
        prev_val = get_location_metric_mean(prev_df, metric_col) if not prev_df.empty else curr_val
    else:
        # Voor metrics op soort-niveau (zoals Eco Score)
        curr_val = curr_df[metric_col].mean()
        prev_val = prev_df[metric_col].mean() if not prev_df.empty else curr_val
        
    delta = curr_val - prev_val
    return curr_val, delta

# --- KAART VISUALISATIE FUNCTIES (Vanuit 2_Ruimtelijke_analyse.py) ---

def get_color_vegetation(value):
    """Rood (0%) -> Groen (100%)"""
    if value == 0: return '#d73027'       # Rood (Afwezig)
    elif value <= 5: return '#fc8d59'     # Oranje
    elif value <= 15: return '#fee08b'    # Geel
    elif value <= 40: return '#d9ef8b'    # Lichtgroen
    elif value <= 75: return '#91cf60'    # Medium Groen
    else: return '#1a9850'                # Donkergroen

def get_color_depth(value):
    """Lichtblauw (ondiep) -> Donkerblauw (diep)"""
    if pd.isna(value): return 'gray'
    elif value < 0.5: return '#eff3ff'    # Heel licht
    elif value < 1.5: return '#bdd7e7'
    elif value < 2.5: return '#6baed6'
    elif value < 4.0: return '#3182bd'
    else: return '#08519c'                # Donkerblauw

def get_color_transparency(value):
    """Bruin (weinig zicht) -> Groen (veel zicht)"""
    if pd.isna(value): return 'gray'
    elif value < 0.5: return '#8c510a'    # Donkerbruin
    elif value < 1.0: return '#d8b365'    # Lichtbruin
    elif value < 1.5: return '#f6e8c3'    # Beige
    elif value < 2.0: return '#c7eae5'    # Lichtgroen/blauw
    elif value < 3.0: return '#5ab4ac'    # Medium Groen/Teal
    else: return '#01665e'                # Donkergroen/Teal

def get_color_krw(score):
    """KRW-score 1-5: 1-2 groen, 3-4 oranje, 5 rood."""
    if pd.isna(score): 
        return 'gray'
    try:
        s = float(score)
    except:
        return 'gray'
    if s <= 2:
        return '#1a9850'  # groen
    elif s <= 4:
        return '#ff7f0e'  # oranje
    else:
        return '#d73027'  # rood

import math
from html import escape

def _polar_to_cart(cx, cy, r, angle_rad):
    return (cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad))

def _wedge_path(cx, cy, r, start_angle, end_angle):
    # Grote boog?
    large_arc = 1 if (end_angle - start_angle) > math.pi else 0
    x1, y1 = _polar_to_cart(cx, cy, r, start_angle)
    x2, y2 = _polar_to_cart(cx, cy, r, end_angle)
    # Move center -> line to start -> arc -> close
    return f"M {cx:.2f},{cy:.2f} L {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f} Z"

def pie_svg(
    counts: dict,
    color_map: dict,
    order=None,
    size=30,
    border=1,
    border_color="#333",
    fixed_total=None,          # <-- nieuw: bv. 100 voor bedekking%
    fill_gap=False,            # <-- nieuw: laat rest leeg (transparant) als True
    gap_color="transparent"    # <-- nieuw: kun je ook '#ffffff' maken
):
    """
    SVG pie chart.
    - Default: normaliseert op som(counts) -> altijd volle cirkel (geschikt voor records).
    - fixed_total=100 + fill_gap=True: sectoren zijn absolute percentages, rest blijft leeg/transparant.
    """

    # Filter nonzero values
    nonzero = [(k, float(v)) for k, v in counts.items() if v is not None and float(v) > 0]

    # Als helemaal geen waarden
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

    # Kies de schaal waarop we normaliseren
    if fixed_total is not None:
        denom = float(fixed_total)
        # Cap: als som > fixed_total, dan knippen we af op fixed_total (geen rescale)
        # (alternatief: rescale zodat het altijd 100% wordt, maar dat wil jij juist niet)
        # We tekenen wedges op basis van val/denom; rest blijft leeg.
    else:
        denom = sum(v for _, v in nonzero)

    if denom <= 0:
        # fallback
        return (
            f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
            f"xmlns='http://www.w3.org/2000/svg'>"
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#cccccc' stroke='{border_color}' stroke-width='{border}' />"
            f"</svg>"
        )

    # ✅ Special-case: 100% volle cirkel (alleen als werkelijk (bijna) volledig)
    # - Voor fixed_total=100: alleen vol als som >= 99.9
    # - Voor default: alleen vol als één categorie ~100% van denom
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
        # default gedrag: als één categorie alles is -> volle cirkel
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

    # Start op -90° (bovenaan)
    start = -math.pi / 2
    paths = []

    # (optioneel) achtergrondvulling voor "gap" (meestal transparant)
    # Als fill_gap=False doen we niets; de ondergrond blijft transparant.
    if fixed_total is not None and fill_gap and gap_color != "transparent":
        paths.append(f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{gap_color}' />")

    # Wedges tekenen: angle = (val/denom)*2π
    # Bij fixed_total: som kan < denom -> overblijvende sector blijft leeg.
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

        # Bij fixed_total cap: stop als we (bijna) rond zijn
        if fixed_total is not None and (start - (-math.pi / 2)) >= 2 * math.pi * 0.999:
            break

    # Rand bovenop
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        + "".join(paths) +
        f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{border_color}' stroke-width='{border}' />"
        f"</svg>"
    )

def create_pie_map(
    df_locs: pd.DataFrame,
    counts_by_loc: dict,
    label: str,
    color_map: dict,
    order=None,
    size_px: int = 30,
    zoom_start: int = 10,
    fixed_total=None,     # <-- nieuw
    fill_gap=False,       # <-- nieuw
    gap_color="transparent"
):
    """
    Folium kaart met pie-chart markers (SVG via DivIcon) per locatie.
    Verwacht df_locs met kolommen: locatie_id, Waterlichaam, lat, lon (+ evt. diepte/doorzicht)
    counts_by_loc: dict locatie_id -> dict categorie -> count
    """
    # Center
    if df_locs["lat"].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat, center_lon = df_locs["lat"].mean(), df_locs["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)

    for row in df_locs.dropna(subset=["lat", "lon"]).itertuples():
        loc_id = getattr(row, "locatie_id")
        wb = getattr(row, "Waterlichaam", "")
        counts = counts_by_loc.get(loc_id, {})
        svg = pie_svg(counts,
            color_map=color_map,
            order=order,
            size=size_px,
            fixed_total=fixed_total,
            fill_gap=fill_gap,
            gap_color=gap_color
)

        # Tooltip: korte samenvatting
        # Toon alleen categorieën met >0
        parts = [f"{escape(str(k))}: {int(v)}" for k, v in counts.items() if v]
        dist_txt = "<br/>".join(parts) if parts else "Geen data"

        # Diepte & doorzicht (kunnen NaN zijn)
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

    Args:
        dataframe: Pandas DataFrame met kolommen 'lat', 'lon', 'waarde_veg', 'diepte_m', 'doorzicht_m', 'Waterlichaam', 'locatie_id'
        mode: "Vegetatie", "Diepte" of "Doorzicht"
        label_veg: Label voor de hoofdwaarde in de tooltip
    """
    # Centreer kaart
    if dataframe['lat'].isnull().all():
        center_lat, center_lon = 52.5, 5.5  # Fallback NL
    else:
        center_lat = dataframe['lat'].mean()
        center_lon = dataframe['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)

    for row in dataframe.itertuples():
        # Bepaal hoofdwaarde + kleur + radius door modus
        radius = 5
        fill_opacity = 0.8
        if mode == "Vegetatie":
            # CATEGORISCH
            if value_style == "categorical" and category_col:
                cat = getattr(row, category_col, None)
                color = (category_color_map or {}).get(cat, '#999999')
                main_line = f"<b>🌱 {label_veg}:</b> {cat}"
                radius = 6
            else:
                # NUMERIEK
                val = getattr(row, 'waarde_veg', 0.0)
                if value_style == "krw":
                    color = get_color_krw(val)
                    main_line = f"<b>🌱 {label_veg}:</b> {val:.2f}"
                    radius = 6
                else:
                    color = get_color_vegetation(val)
                    main_line = f"<b>🌱 {label_veg}:</b> {val:.1f}%"
                    radius = 4 + (min(val, 100) / 100 * 6) if val > 0 else 4
        elif mode == "Diepte":
            val = getattr(row, 'diepte_m', float('nan'))
            color = get_color_depth(val)
            main_line = f"<b>🌊 Diepte:</b> {val:.2f} m"
        else:  # Doorzicht
            val = getattr(row, 'doorzicht_m', float('nan'))
            color = get_color_transparency(val)
            main_line = f"<b>👁️ Doorzicht:</b> {val:.2f} m"

        depth_line = f"<b>🌊 Diepte:</b> {getattr(row, 'diepte_m', float('nan')):.2f} m"
        trans_line = f"<b>👁️ Doorzicht:</b> {getattr(row, 'doorzicht_m', float('nan')):.2f} m"

        tooltip_html = (
            f"<b>Locatie:</b> {row.locatie_id}<br>"
            f"<b>Water:</b> {row.Waterlichaam}<br>"
            f"{main_line}<br>"
            f"{depth_line}<br>"
            f"{trans_line}"
        )

        if getattr(row, 'lat') is not None and getattr(row, 'lon') is not None:
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=radius,
                color='#333333',  # rand
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                tooltip=tooltip_html
            ).add_to(m)

    return m

def df_to_geojson_points(df: pd.DataFrame, value_col: str, id_col: str = "locatie_id"):
    """
    Zet punten (lat/lon) om naar een GeoJSON FeatureCollection.
    Verwacht kolommen: lat, lon, id_col, value_col
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
    bounds=None,  # [min_lon, min_lat, max_lon, max_lat] of None
):
    """
    Rendert een swipe-map met dragbare divider/handle op de kaart zelf (MapLibre in Streamlit html component).

    - Basemap: OpenFreeMap Liberty style URL (direct bruikbaar in MapLibre). [1](https://www.npmjs.com/package/@stadiamaps/maplibre-search-box)
    - map_left: basemap grijs via setPaintProperty (punten blijven kleur). [2](https://github.com/maptiler/tileserver-gl)[3](https://www.nationaalgeoregister.nl/geonetwork/srv/api/records/c82a783a-9a58-4761-a809-b4c5d90dcd35)
    - bounds: optional [min_lon, min_lat, max_lon, max_lat] -> fitBounds
    """

    # OpenFreeMap style URL (vector basemap)
    style_url = "https://tiles.openfreemap.org/styles/liberty"  # [1](https://www.npmjs.com/package/@stadiamaps/maplibre-search-box)

    left_json = json.dumps(geojson_left)
    right_json = json.dumps(geojson_right)
    bounds_json = "null" if bounds is None else json.dumps(bounds)

    # Veilig: als min == max, maak kleine marge zodat interpolate werkt
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

    /* Onderliggende kaart (jaar links) */
    #map_left {{
      position: absolute;
      inset: 0;
    }}

    /* Bovenliggende kaart (jaar rechts) */
    #map_right {{
      position: absolute;
      inset: 0;
      clip-path: inset(0 0 0 50%);
    }}

    /* Divider lijn */
    #divider {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 2px;
      background: rgba(230,230,230,0.95);
      box-shadow: 0 0 0 1px rgba(0,0,0,0.08);
      z-index: 10;
      cursor: ew-resize;
    }}

    /* Handle */
    #handle {{
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      width: 16px;
      height: 120px;
      border-radius: 10px;
      background: rgba(255,255,255,0.95);
      border: 1px solid rgba(0,0,0,0.15);
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
      z-index: 11;
      cursor: ew-resize;
    }}

    /* Jaarlabels */
    .year-label {{
      position: absolute;
      top: 18px;
      font-size: 44px;
      font-weight: 700;
      color: rgba(0,0,0,0.78);
      text-shadow: 0 1px 0 rgba(255,255,255,0.6);
      z-index: 12;
      pointer-events: none;
    }}
    #label_left {{ left: 30px; opacity: 0.40; }}
    #label_right {{ right: 30px; opacity: 0.95; }}

    /* Legenda */
    #legend {{
      position: absolute;
      left: 50%;
      bottom: 20px;
      transform: translateX(-50%);
      width: 520px;
      max-width: calc(100% - 40px);
      background: rgba(255,255,255,0.90);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 12px;
      padding: 12px 14px;
      z-index: 12;
      backdrop-filter: blur(3px);
    }}
    #legend .title {{
      text-align: center;
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    #legend .bar {{
      height: 14px;
      border-radius: 8px;
      border: 1px solid rgba(0,0,0,0.12);
      background: linear-gradient(90deg, #d73027 0%, #fee08b 50%, #1a9850 100%);
    }}
    #legend .labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      font-size: 16px;
      font-weight: 650;
      color: rgba(0,0,0,0.80);
    }}
    #legend .sub {{
      text-align: center;
      margin-top: 3px;
      font-size: 14px;
      color: rgba(0,0,0,0.60);
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
    interactive: false
  }});

  // Sync view (links -> rechts)
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

  // --- Basemap (OpenFreeMap layers) grijs maken op mapLeft ---
  // We overschrijven paint-properties per layer type via setPaintProperty. [2](https://github.com/maptiler/tileserver-gl)[3](https://www.nationaalgeoregister.nl/geonetwork/srv/api/records/c82a783a-9a58-4761-a809-b4c5d90dcd35)
  function applyBasemapGray(map) {{
    const style = map.getStyle();
    const layers = (style && style.layers) ? style.layers : [];

    const GRAY_BG   = "#e7e7e7";
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
            // POI's etc. in basemap dimmen
            map.setPaintProperty(ly.id, "circle-color", GRAY_LINE);
            try {{ map.setPaintProperty(ly.id, "circle-opacity", 0.35); }} catch(e) {{}}
            break;

          case "heatmap":
            try {{ map.setPaintProperty(ly.id, "heatmap-opacity", 0.25); }} catch(e) {{}}
            break;

          case "raster":
            // Voor het geval een style raster layers bevat
            try {{ map.setPaintProperty(ly.id, "raster-saturation", -1); }} catch(e) {{}}
            break;

          default:
            break;
        }}
      }} catch (e) {{
        // stil negeren
      }}
    }});
  }}

  // --- Punten toevoegen (kleur blijft) ---
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
    // 1) eerst basemap links grijs
    applyBasemapGray(mapLeft);

    // 2) punten links toevoegen (kleur houden, desnoods iets dimmen)
    addPoints(mapLeft, 'leftPts', 'leftLayer', leftData, true);

    // 3) autozoom
    if (bounds && bounds.length === 4) {{
      const sw = [bounds[0], bounds[1]];
      const ne = [bounds[2], bounds[3]];
      mapLeft.fitBounds([sw, ne], {{
        padding: 70,
        maxZoom: 12,
        duration: 0
      }});
    }}

    // tooltip op linkerpuntlaag
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
    // overlay (rechts) normaal, niet dimmen
    addPoints(mapRight, 'rightPts', 'rightLayer', rightData, false);
  }});

  // --- Swipe mechanics ---
  const wrap = document.getElementById('wrap');
  const mapRightDiv = document.getElementById('map_right');
  const divider = document.getElementById('divider');
  const handle = document.getElementById('handle');

  let isDragging = false;
  let pct = 0.5;

  function setSwipe(p) {{
    pct = Math.max(0, Math.min(1, p));
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
