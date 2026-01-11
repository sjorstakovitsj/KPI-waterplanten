import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import time
from utils import (
    load_data, 
    get_color_absolute, 
    get_color_diff, 
    categorize_slope_trend, 
    create_year_map_deck
)

st.set_page_config(layout="wide")
st.title("📈 Tijd- en Trendanalyse")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Trend Filters")

if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# 1. Selectie Waterlichaam
all_bodies = sorted(df['Waterlichaam'].unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer Waterlichaam / Waterlichamen", 
    options=all_bodies, 
    default=all_bodies[:1] if all_bodies else []
)

# 2. Selectie Indicator Categorie
all_cats = sorted(df['indicator_cat'].unique())
selected_cats = st.sidebar.multiselect("Filter op Indicatorsoort", all_cats, default=all_cats)

# 3. Metriek Keuze
metric_options = {
    'Bedekkingsgraad (%)': 'bedekking_pct',
    'Indicator Score': 'eco_score',
    'Doorzicht (m)': 'doorzicht_m',
    'Soortenrijkdom': 'soort_count'
}
selected_metric_label = st.sidebar.selectbox("Kies Analyse Variabele", list(metric_options.keys()))
selected_metric = metric_options[selected_metric_label]

# --- DATA FILTEREN ---
df_filtered = df[
    (df['Waterlichaam'].isin(selected_bodies)) & 
    (df['indicator_cat'].isin(selected_cats))
]

if df_filtered.empty:
    st.warning("Geen data beschikbaar voor deze selectie.")
    st.stop()

# Voorbereiden data voor plotting (aggregatie per jaar/locatie)
if selected_metric == 'soort_count':
    df_trend = df_filtered.groupby(['locatie_id', 'jaar'])['soort'].nunique().reset_index(name='waarde')
else:
    df_trend = df_filtered.groupby(['locatie_id', 'jaar'])[selected_metric].mean().reset_index(name='waarde')

# --- 1. TIJDREEKS TRENDLIJNEN ---
st.subheader(f"Verloop {selected_metric_label} door de jaren heen")
fig_line = px.line(df_trend, x='jaar', y='waarde', color='locatie_id', markers=True,
                   title=f"Trendontwikkeling per meetpunt in geselecteerde wateren")
st.plotly_chart(fig_line, use_container_width=True)

# --- 2. SLOPE ANALYSE (Berekening) ---
st.subheader("Slope Analyse: Verbetert of verslechtert de toestand?")
st.markdown("Analyse per meetpunt over de beschikbare jaren.")

# Uitleg over de Slope
with st.expander("ℹ️ Hoe interpreteer ik de Slope (Trendgetal)?", expanded=False):
    st.markdown(f"""
    De **slope** (richtingscoëfficiënt) is een getal dat de gemiddelde verandering per jaar aangeeft op basis van een lineaire trendlijn.
    
    * **Positief getal (+):** De waarde stijgt gemiddeld elk jaar. 
        * *Voorbeeld:* Een slope van `0.5` bij Bedekkingsgraad betekent dat de bedekking gemiddeld met 0.5% per jaar toeneemt.
    * **Negatief getal (-):** De waarde daalt gemiddeld elk jaar.
    * **Nul (0):** Er is geen stijgende of dalende trend (stabiel).
    
    *Let op: In deze tabel worden alleen locaties getoond met **minimaal 5 meetjaren**.*
    """)

slopes = []
unique_locs = df_trend['locatie_id'].unique()
MIN_JAREN = 5  # Filter criterium

for loc in unique_locs:
    df_loc = df_trend[df_trend['locatie_id'] == loc]
    n_measurements = len(df_loc)
    
    # Check: Alleen berekenen als er minimaal 5 meetjaren zijn
    if n_measurements >= MIN_JAREN:
        slope, intercept = np.polyfit(df_loc['jaar'], df_loc['waarde'], 1)
        slopes.append({
            'locatie_id': loc, 
            'slope': slope,
            'n_jaren': n_measurements # Toevoegen aantal metingen
        })

# Maak dataframe van slopes
df_slopes = pd.DataFrame(slopes) if slopes else pd.DataFrame(columns=['locatie_id', 'slope', 'n_jaren'])

if not df_slopes.empty:
    # Categorie bepalen (nu via util functie)
    threshold = 0.1 if selected_metric == 'eco_score' else 0.5 
    
    df_slopes['Trend'] = df_slopes['slope'].apply(lambda x: categorize_slope_trend(x, threshold))

    # Kolommen voor de layout
    col1, col2 = st.columns([1, 2])
    with col1:
        fig_pie = px.pie(df_slopes, names='Trend', title="Verdeling Trends (Aantal Locaties)", 
                         color='Trend',
                         color_discrete_map={'Verbeterend ↗️': 'green', 'Verslechterend ↘️': 'red', 'Stabiel ➡️': 'grey'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Dataframe tonen met opmaak
        st.write(f"**Detailtabel (Locaties met >={MIN_JAREN} jaar data)**")
        st.dataframe(
            df_slopes.sort_values('slope', ascending=False).style.background_gradient(subset=['slope'], cmap='RdYlGn')
            .format({'slope': "{:.4f}", 'n_jaren': "{:.0f}"}), # Format slope op 4 decimalen, jaren als geheel getal
            use_container_width=True,
            hide_index=True
        )
else:
    st.warning(f"Geen locaties gevonden met minimaal {MIN_JAREN} jaar aan meetgegevens in de huidige selectie.")

# --- 3. GEOGRAFISCHE PATROON HERKENNING (PYDECK) ---
st.divider()
st.subheader("Geografische Patroonanalyse")

# Data voorbereiding voor de kaart: Aangepaste logica voor soortenrijkdom vs andere metrieken
if selected_metric == 'soort_count':
    # Als we soortenrijkdom bekijken, moeten we unieke soorten tellen ('nunique')
    df_map_source = df_filtered.groupby(['locatie_id', 'jaar']).agg({
        'lat': 'first',
        'lon': 'first',
        'soort': 'nunique'
    }).reset_index()
    
    # De resultaatkolom heet nu 'soort', hernoem deze naar 'soort_count' zodat de rest van de code werkt
    df_map_source = df_map_source.rename(columns={'soort': 'soort_count'})
    
else:
    # Voor andere metrieken (bedekking, doorzicht) nemen we gewoon het gemiddelde
    df_map_source = df_filtered.groupby(['locatie_id', 'jaar']).agg({
        'lat': 'first',
        'lon': 'first',
        selected_metric: 'mean'
    }).reset_index()

df_map_source = df_map_source.dropna(subset=['lat', 'lon'])

# UI Controls voor de kaart
col_mode, col_legenda = st.columns([2, 1])

with col_mode:
    map_mode = st.radio(
        "Selecteer Analyse Modus:",
        ["⏱️ Tijdlijn Animatie (Absoluut)", "⚖️ Verschil Modus (Jaar A vs Jaar B)"],
        horizontal=True
    )

# Container voor de kaart
map_container = st.empty()
legend_container = st.empty()

# --- LOGICA PER MODUS ---

if "Verschil" in map_mode:
    # --- VERSCHIL MODUS ---
    available_years = sorted(df_map_source['jaar'].unique())
    if len(available_years) < 2:
        st.warning("Onvoldoende jaren voor een vergelijking.")
    else:
        # 1. Selectie jaren
        start_y, end_y = st.select_slider(
            "Vergelijk Jaar A met Jaar B",
            options=available_years,
            value=(available_years[0], available_years[-1])
        )
        
        # 2. Data filteren per jaar
        df_start = df_map_source[df_map_source['jaar'] == start_y].copy()
        df_end = df_map_source[df_map_source['jaar'] == end_y].copy()
        
        # 3. Mergen van de twee jaren op locatie_id
        df_diff = pd.merge(
            df_start[['locatie_id', 'lat', 'lon', selected_metric]], 
            df_end[['locatie_id', selected_metric]], 
            on='locatie_id', 
            suffixes=('_start', '_end')
        )

        if df_diff.empty:
            st.error(f"Geen locaties gevonden die data hebben in zowel {start_y} als {end_y}.")
        else:
            # 4. Berekenen en direct afronden
            df_diff['waarde_start'] = df_diff[f'{selected_metric}_start'].round(1)
            df_diff['waarde_end'] = df_diff[f'{selected_metric}_end'].round(1)
            df_diff['delta'] = (df_diff['waarde_end'] - df_diff['waarde_start']).round(1)

            # 5. Kleuren en Radius (met util functie)
            df_diff['color'] = df_diff['delta'].apply(get_color_diff)
            df_diff['radius'] = 150 

            # 6. Pydeck Layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                df_diff,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
                get_line_color=[0, 0, 0],
            )

            # 7. Tooltip
            tooltip = {
                "html": f"""
                    <b>Locatie:</b> {{locatie_id}}<br/>
                    <b>{start_y}:</b> {{waarde_start}}<br/>
                    <b>{end_y}:</b> {{waarde_end}}<br/>
                    <hr>
                    <b>Verschil:</b> {{delta}}
                """,
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

            view_state = pdk.ViewState(
                latitude=df_diff['lat'].mean(),
                longitude=df_diff['lon'].mean(),
                zoom=9
            )

            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")
            map_container.pydeck_chart(r)

        # Statische Legenda HTML
        st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
                <strong>Legenda (Verschil)</strong><br>
                <span style='color:green'>●</span> Verbetering (> +0.5)<br>
                <span style='color:grey'>●</span> Stabiel<br>
                <span style='color:red'>●</span> Verslechtering (< -0.5)
            </div>
            """, unsafe_allow_html=True)

else:
    # --- ANIMATIE / TIJDLIJN MODUS ---
    available_years = sorted(df_map_source['jaar'].unique())
    min_val = df_map_source[selected_metric].min()
    max_val = df_map_source[selected_metric].max()

    # Controls
    col_play, col_slider = st.columns([0.2, 0.8])
    with col_play:
        play_btn = st.button("▶️ Afspelen")
    
    with col_slider:
        selected_year = st.slider("Selecteer Jaar", min_value=min(available_years), max_value=max(available_years), value=min(available_years))

    # Animatie Logica
    if play_btn:
        for y in available_years:
            # Gebruik nu de functie uit utils
            deck = create_year_map_deck(df_map_source, y, selected_metric, min_val, max_val)
            
            map_container.pydeck_chart(deck)
            legend_container.markdown(f"### Huidig Jaar: {y}")
            time.sleep(2.0)
    else:
        # Statische weergave gebaseerd op slider
        deck = create_year_map_deck(df_map_source, selected_year, selected_metric, min_val, max_val)
        map_container.pydeck_chart(deck)

    # Legenda voor absolute waarden
    st.markdown(f"""
        <div style="display: flex; align-items: center; background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px;">
            <div style="margin-right: 10px;"><strong>Legenda ({selected_metric_label}):</strong></div>
            <div style="background: linear-gradient(90deg, red, green); width: 150px; height: 20px; border: 1px solid #ccc;"></div>
            <div style="margin-left: 10px; font-size: 0.8em;">
                {min_val:.1f} (Laag) &nbsp;&nbsp; ➡️ &nbsp;&nbsp; {max_val:.1f} (Hoog)
            </div>
        </div>
        """, unsafe_allow_html=True)


# --- 4. VOOR / NA VERGELIJKING ---
st.divider()
st.subheader("Voor / Na Vergelijking")
available_years = sorted(df_filtered['jaar'].unique())

if len(available_years) >= 2:
    c_year1, c_year2 = st.columns(2)
    year_start = c_year1.selectbox("Referentiejaar (Voor)", available_years, index=0)
    year_end = c_year2.selectbox("Vergelijkingsjaar (Na)", available_years, index=len(available_years)-1)

    df_compare = df_trend[df_trend['jaar'].isin([year_start, year_end])]

    fig_bar = px.bar(df_compare, x='locatie_id', y='waarde', color=df_compare['jaar'].astype(str),
                     barmode='group', title=f"Vergelijking {year_start} vs {year_end} per locatie",
                     labels={'waarde': selected_metric_label, 'color': 'Jaar'})
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Niet genoeg jaren aan data voor een vergelijking.")