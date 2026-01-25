# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import folium 
import re

# --- CONFIGURATIE ---
FILE_PATH = "AquaDeskMeasurementExport_RWS_20251227211036.csv"

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
        'diepte_m', 'doorzicht_m', 
        'lat', 'lon', 'x_rd', 'y_rd',
        'groeivorm', 'type', 'Grootheid' # Grootheid toegevoegd om onderscheid te kunnen maken
    ]
    
    for col in cols_to_keep:
        if col not in final_df.columns:
            final_df[col] = np.nan

    return final_df[cols_to_keep]

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
        text += "⚠️ **Zeer kale bodem** (<5% bedekking). Mogelijk lichtgebrek of woeling.\n"
    elif dom_type == 'Ondergedoken':
        text += f"✅ Goede ontwikkeling (**{total_cover:.0f}%**). Dominantie van ondergedoken planten.\n"
    elif dom_type == 'Drijvend':
        text += f"⚠️ Veel drijfbladplanten (**{total_cover:.0f}%**). Mogelijk slibrijke bodem.\n"
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

def create_year_map_deck(df_source, year, metric, min_val, max_val):
    """
    Genereert een Pydeck Scatterplot object voor een specifiek jaar.
    """
    df_year = df_source[df_source['jaar'] == year].copy()
    
    # Afronden voor nette weergave in tooltips
    df_year[metric] = df_year[metric].round(1)
    
    # Kleur berekenen
    df_year['color'] = df_year[metric].apply(lambda x: get_color_absolute(x, min_val, max_val))
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_year,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=150,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
        get_line_color=[50, 50, 50],
    )

    tooltip = {
            "html": f"<b>Locatie:</b> {{locatie_id}}<br/>"
                    f"<b>Jaar:</b> {year}<br/>"
                    f"<b>Waarde:</b> {{{metric}}} %", # Drie accolades nodig door de f-string
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

    view_state = pdk.ViewState(
        latitude=df_source['lat'].mean(),
        longitude=df_source['lon'].mean(),
        zoom=9
    )
    
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")

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

def create_map(dataframe, mode, label_veg="Vegetatie"):
    """
    Genereert een Folium kaart.
    Argumenten:
    - dataframe: Pandas DataFrame met kolom 'lat', 'lon', 'waarde_veg', 'diepte_m', 'doorzicht_m'
    - mode: De visualisatiemodus ("Vegetatie", "Diepte", "Doorzicht")
    - label_veg: Label voor de vegetatie tooltip (default: "Vegetatie")
    """
    # Centreer kaart
    if dataframe['lat'].isnull().all():
        center_lat, center_lon = 52.5, 5.5 # Fallback NL
    else:
        center_lat = dataframe['lat'].mean()
        center_lon = dataframe['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
    
    for row in dataframe.itertuples():
        # Tooltip altijd compleet
        tooltip_html = (
            f"<b>Locatie:</b> {row.locatie_id}<br>"
            f"<b>Water:</b> {row.Waterlichaam}<br>"
            f"<b>🌱 {label_veg}:</b> {row.waarde_veg:.1f}%<br>"
            f"<b>🌊 Diepte:</b> {row.diepte_m:.2f} m<br>"
            f"<b>👁️ Doorzicht:</b> {row.doorzicht_m:.2f} m"
        )
        
        # Bepaal waarde en kleur op basis van modus
        radius = 5 # Standaard grootte
        fill_opacity = 0.8
        
        if mode == "Vegetatie":
            val = row.waarde_veg
            color = get_color_vegetation(val)
            # Optioneel: puntjes iets groter maken als er veel vegetatie is
            if val > 0:
                radius = 4 + (min(val, 100) / 100 * 6) # Max radius 10
            else:
                radius = 4
            
        elif mode == "Diepte":
            val = row.diepte_m
            color = get_color_depth(val)
            
        elif mode == "Doorzicht":
            val = row.doorzicht_m
            color = get_color_transparency(val)
        
        if pd.notna(row.lat) and pd.notna(row.lon):
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=radius,
                color='#333333',     # Dun grijs randje om de stip
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                tooltip=tooltip_html
            ).add_to(m)

    return m