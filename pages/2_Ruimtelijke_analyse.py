# 2_Ruimtelijke_analyse.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from utils import load_data, add_species_group_columns, create_map, get_sorted_species_list

st.set_page_config(layout="wide", page_title="Ruimtelijke analyse")

st.title("🗺️ Ruimtelijke analyse")
st.markdown("Vergelijk de vegetatieontwikkeling met diepte en doorzicht.")

# --- DATA INLADEN ---
df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# 1. Jaar Filter
all_years = sorted(df['jaar'].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Selecteer jaar", all_years)

# 2. Project Filter
all_projects = sorted(df['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)", 
    options=all_projects, 
    default=all_projects
)

# Filter basis dataset (Ruwe data met RWS codes)
df_filtered = df[
    (df['jaar'] == selected_year) & 
    (df['Project'].isin(selected_projects))
].copy()

if df_filtered.empty:
    st.warning("Geen data gevonden voor deze selectie.")
    st.stop()

# --- DATA VERRIJKING VOOR SOORTGROEPEN ---
# We maken een extra dataset aan waarin individuele soorten zijn ingedeeld in groepen
df_species_groups = add_species_group_columns(df_filtered)

st.sidebar.markdown("---")
st.sidebar.header("Kaartinstellingen")

# 3. KEUZE ANALYSE NIVEAU (NIEUWE FUNCTIONALITEIT)
# Hiermee splitsen we de dropdown om de UI schoon te houden.
analysis_level = st.sidebar.radio(
    "Kies analyseniveau",
    options=["groepen & aggregaties", "individuele soorten"]
)

selected_coverage_type = None

# A. Logica voor Groepen
if analysis_level == "groepen & aggregaties":
    # Optie A1: Totale bedekking
    opt_general = ["totale bedekking"]
    
    # Optie A2: RWS Groeivormen (uit ruwe data waar type='Groep')
    rws_forms = sorted(df_filtered[df_filtered['type'] == 'Groep']['groeivorm'].unique())
    
    # Optie A3: Taxonomische Soortgroepen (uit de verrijkte dataset)
    species_groups_list = sorted(df_species_groups['soortgroep'].dropna().unique())
    
    all_options = opt_general + rws_forms + species_groups_list
    selected_coverage_type = st.sidebar.selectbox(
        "selecteer groep",
        options=all_options
    )

# B. Logica voor Individuele Soorten
else:
    # We halen de schone lijst met soorten op via de helper in utils.py
    # Dit toont alleen soorten die ook daadwerkelijk voorkomen in de gefilterde dataset (df_filtered)
    species_list = get_sorted_species_list(df_filtered)
    
    if not species_list:
        st.sidebar.warning("Geen individuele soorten gevonden in deze selectie.")
        st.stop()
        
    selected_coverage_type = st.sidebar.selectbox(
        "selecteer soort",
        options=species_list
    )

# 4. Kaartlaag Keuze
layer_mode = st.sidebar.radio(
    "kies kaartlaag",
    options=["Vegetatie", "Diepte", "Doorzicht"]
)

# --- DATA VOORBEREIDING VOOR KAART ---

# Stap 1: Maak een 'Base' dataframe met unieke locaties en abiotiek
# We gebruiken df_filtered omdat die alle bezochte locaties bevat (ook waar soorten 0% zijn)
df_locs = df_filtered.groupby(['locatie_id', 'Waterlichaam']).agg({
    'lat': 'first',
    'lon': 'first',
    'diepte_m': 'mean',
    'doorzicht_m': 'mean'
}).reset_index()

# Stap 2: Bepaal de vegetatiewaarde op basis van de selectie
df_veg_calc = pd.DataFrame()

# LOGICA SELECTIE:
if analysis_level == "groepen & aggregaties":
    
    if selected_coverage_type == "totale bedekking":
        # Gebruik WATPTN kolom
        df_veg_calc = df_filtered.groupby('locatie_id')['totaal_bedekking_locatie'].mean().reset_index()
        df_veg_calc.rename(columns={'totaal_bedekking_locatie': 'waarde_veg'}, inplace=True)

    elif selected_coverage_type in rws_forms:
        # Filter de RUWE dataset op RWS groeivorm
        df_subset = df_filtered[df_filtered['groeivorm'] == selected_coverage_type]
        df_veg_calc = df_subset.groupby('locatie_id')['bedekking_pct'].mean().reset_index()
        df_veg_calc.rename(columns={'bedekking_pct': 'waarde_veg'}, inplace=True)

    elif selected_coverage_type in species_groups_list:
        # Filter de VERRIJKTE dataset op taxonomische groep (sommeren want groep bestaat uit meerdere soorten)
        df_subset = df_species_groups[df_species_groups['soortgroep'] == selected_coverage_type]
        df_veg_calc = df_subset.groupby('locatie_id')['bedekkingsgraad_proc'].sum().reset_index()
        df_veg_calc.rename(columns={'bedekkingsgraad_proc': 'waarde_veg'}, inplace=True)

else: 
    # LOGICA VOOR INDIVIDUELE SOORTEN
    # We filteren de ruwe dataset op de specifieke soortnaam
    df_subset = df_filtered[df_filtered['soort'] == selected_coverage_type]
    
    # We nemen het gemiddelde (meestal is er maar 1 meting per locatie per jaar)
    df_veg_calc = df_subset.groupby('locatie_id')['bedekking_pct'].mean().reset_index()
    df_veg_calc.rename(columns={'bedekking_pct': 'waarde_veg'}, inplace=True)

# Stap 3: Merge alles samen
# Left join op locaties: zodat we OOK punten zien waar wel diepte is gemeten, maar de soort NIET voorkomt (waarde 0)
df_map_data = pd.merge(df_locs, df_veg_calc, on='locatie_id', how='left')

# BELANGRIJK: Vul NaN op met 0. Dit betekent: "Op deze locatie is de soort niet waargenomen"
df_map_data['waarde_veg'] = df_map_data['waarde_veg'].fillna(0)

# --- WEERGAVE ---

st.subheader(f"Kaartweergave: {layer_mode}")
if layer_mode == "Vegetatie":
    if analysis_level == "Individuele Soorten":
        st.info(f"Je bekijkt de verspreiding van de soort: **{selected_coverage_type}**")
    elif selected_coverage_type != "Totale Bedekking":
        st.info(f"Je bekijkt de verspreiding van de groep: **{selected_coverage_type}**")

# Legenda omschrijving
if layer_mode == "Vegetatie":
    st.caption("Legenda: Rood (0%) → Geel → Donkergroen (Hoge bedekking)")
elif layer_mode == "Diepte":
    st.caption("Legenda: Lichtblauw (Ondiep) → Donkerblauw (Diep)")
elif layer_mode == "Doorzicht":
    st.caption("Legenda: Bruin (Troebel) → Groen (Helder)")

# Map aanmaken
map_obj = create_map(df_map_data, layer_mode, label_veg=selected_coverage_type)
st_folium(map_obj, height=600, width=None)

# --- TABEL ---
st.divider()
with st.expander(f"Toon data voor {selected_coverage_type}"):
    # Filter de tabel om alleen locaties te tonen waar de soort/groep ook echt voorkomt (>0)
    # Tenzij de gebruiker expliciet Diepte/Doorzicht wil zien, dan alles tonen.
    df_display = df_map_data.copy()
    
    # Optioneel: Sorteer op hoogste vegetatiewaarde om direct te zien waar het groeit
    df_display = df_display.sort_values(by='waarde_veg', ascending=False)

    st.dataframe(
        df_display[['locatie_id', 'Waterlichaam', 'waarde_veg', 'diepte_m', 'doorzicht_m']], 
        use_container_width=True,
        column_config={
            "waarde_veg": st.column_config.NumberColumn(f"{selected_coverage_type} (%)", format="%.1f%%"),
            "diepte_m": st.column_config.NumberColumn("Diepte (m)", format="%.2f"),
            "doorzicht_m": st.column_config.NumberColumn("Doorzicht (m)", format="%.2f")
        }
    )