# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import time
from streamlit.components.v1 import html

from utils import (
    load_data,
    get_color_diff,
    categorize_slope_trend,
    create_year_map_deck,
    get_sorted_species_list,
    create_map,
)

st.set_page_config(layout="wide")
st.title("📈 Tijd- en trendanalyse")

# -------------------------------------------------------------
# Data laden
# -------------------------------------------------------------
df = load_data()

# -------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------
st.sidebar.header("Selectiefilters")
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# 1) Waterlicha(a)m(en)
all_bodies = sorted(df['Waterlichaam'].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam / waterlichamen",
    options=all_bodies,
    default=all_bodies[:1] if all_bodies else []
)

# 2) Individuele soort (nieuw)
species_options = ["— Alle soorten —"] + get_sorted_species_list(df)
selected_species = st.sidebar.selectbox(
    "Selecteer individuele soort (optioneel)",
    species_options,
    index=0
)
species_is_selected = (selected_species and selected_species != "— Alle soorten —")

# 3) Metriek-keuze (als soort is gekozen, beperken we tot bedekking)
if species_is_selected:
    metric_options = {
        'Bedekkingsgraad (%)': 'bedekking_pct',
    }
else:
    metric_options = {
        'Bedekkingsgraad (%)': 'bedekking_pct',
        'Doorzicht (m)': 'doorzicht_m',
        'Diepte (m)': 'diepte_m',
        'Soortenrijkdom': 'soort_count',
    }
selected_metric_label = st.sidebar.selectbox("Kies analysevariabele", list(metric_options.keys()))
selected_metric = metric_options[selected_metric_label]

# -------------------------------------------------------------
# Data filteren
# -------------------------------------------------------------
df_filtered = df[df['Waterlichaam'].isin(selected_bodies)] if selected_bodies else df.copy()

# Soortfilter – alleen echte soorten met bedekking (BEDKG)
if species_is_selected:
    mask = (df_filtered['soort'] == selected_species)
    if 'type' in df_filtered.columns:
        mask &= (df_filtered['type'] == 'Soort')
    if 'Grootheid' in df_filtered.columns:
        mask &= (df_filtered['Grootheid'] == 'BEDKG')
    df_filtered = df_filtered[mask]

# Zorg dat bedekking numeriek is
if 'bedekking_pct' in df_filtered.columns:
    df_filtered['bedekking_pct'] = pd.to_numeric(df_filtered['bedekking_pct'], errors='coerce')

if df_filtered.empty:
    st.warning("Geen data beschikbaar voor deze selectie.")
    st.stop()

# -------------------------------------------------------------
# Voorbereiden trend-data (aggregatie per jaar/locatie)
# -------------------------------------------------------------
if selected_metric == 'soort_count':
    df_trend = (
        df_filtered.groupby(['locatie_id', 'jaar'])['soort']
        .nunique()
        .reset_index(name='waarde')
    )
else:
    df_trend = (
        df_filtered.groupby(['locatie_id', 'jaar'])[selected_metric]
        .mean()
        .reset_index(name='waarde')
    )

# -------------------------------------------------------------
# 0) Kaart (OpenStreetMap) – vóór regressieanalyse
# -------------------------------------------------------------
st.subheader("Kaart – resultaten voor een gekozen jaar")

years_available = sorted([int(y) for y in df_filtered['jaar'].dropna().unique()])
if not years_available:
    st.info("Er zijn geen jaartallen beschikbaar om op de kaart te tonen.")
else:
    c1, c2 = st.columns([3, 1])
    with c2:
        y_min, y_max = int(min(years_available)), int(max(years_available))
        if y_min == y_max:
            # Eén jaar: geen slider nodig
            yr = y_min
            st.caption(f"Enkel jaar beschikbaar: **{yr}**")
        else:
            yr = st.slider(
                "Jaar voor kaartweergave",
                min_value=y_min, max_value=y_max, value=y_min
            )

    with c1:
        # Aggregatie per locatie in gekozen jaar
        df_y = df_filtered[df_filtered['jaar'] == yr].copy()
        agg_cols = {
            'lat': 'first', 'lon': 'first', 'Waterlichaam': 'first',
            'diepte_m': 'mean', 'doorzicht_m': 'mean'
        }

        if selected_metric == 'soort_count':
            # Soortenrijkdom per locatie in jaar
            species_count = (
                df_y.groupby('locatie_id')['soort']
                .nunique()
                .rename('waarde_veg')
            )
            base = df_y.groupby('locatie_id').agg(agg_cols)
            df_map_loc = base.join(species_count).reset_index()
            mode = "Vegetatie"
            label = "Soortenrijkdom (aantal)"
        else:
            base_vals = (
                df_y.groupby('locatie_id')[selected_metric]
                .mean()
                .rename('waarde_veg')
            )
            base = df_y.groupby('locatie_id').agg(agg_cols)
            df_map_loc = base.join(base_vals).reset_index()

            if selected_metric == 'bedekking_pct':
                mode = "Vegetatie"
                label = f"Bedekking (%) – {selected_species}" if species_is_selected else "Bedekking (%)"
            elif selected_metric == 'diepte_m':
                mode = "Diepte"
                label = "Diepte (m)"
            else:
                mode = "Doorzicht"
                label = "Doorzicht (m)"

        osm_map = create_map(df_map_loc, mode=mode, label_veg=label)
        html(osm_map._repr_html_(), height=600)

# -------------------------------------------------------------
# 1) Tijdreeks trendlijnen
# -------------------------------------------------------------
st.subheader(f"Verloop {selected_metric_label} door de jaren heen")
if species_is_selected and selected_metric == 'bedekking_pct':
    st.caption(f"Weergave voor soort: **{selected_species}**")

fig_line = px.line(
    df_trend, x='jaar', y='waarde', color='locatie_id', markers=True,
    title="Trendontwikkeling per meetpunt in geselecteerde wateren"
)
st.plotly_chart(fig_line, use_container_width=True)

# -------------------------------------------------------------
# 2) Regressieanalyse – verbetert of verslechtert de toestand?
# -------------------------------------------------------------
st.subheader("Regressieanalyse: verbetert of verslechtert de toestand?")
st.markdown("Analyse per meetpunt over de beschikbare jaren.")

with st.expander("ℹ️ Hoe interpreteer ik de hellingwaarde?", expanded=False):
    st.markdown(f"""
    De **helling** (richtingscoëfficiënt) is een getal dat de gemiddelde verandering per meetjaar aangeeft op basis van lineaire regressie.

    * **Positief getal (+):** De waarde stijgt gemiddeld elk meetjaar.  
      * *Voorbeeld:* Een slope van `5.0` bij bedekkingsgraad betekent dat de bedekking gemiddeld met 5% per meetjaar toeneemt.
    * **Negatief getal (-):** De waarde daalt gemiddeld elk meetjaar.
    * **Nul (0):** Er is geen stijgende of dalende trend (stabiel).

    Drempelwaarden voor trendclassificatie:
    * **Diepte**/**Doorzicht**: ±0,1 per jaar  
    * **Soortenrijkdom**: ±0,4 per jaar  
    * **Bedekkingsgraad**: ±1,0 per jaar  

    *Let op: hieronder worden alleen locaties getoond met **minimaal 5 meetjaren**.*
    """)

slopes = []
unique_locs = df_trend['locatie_id'].unique()
MIN_JAREN = 5
for loc in unique_locs:
    df_loc = df_trend[df_trend['locatie_id'] == loc]
    n_measurements = len(df_loc)
    if n_measurements >= MIN_JAREN:
        slope, intercept = np.polyfit(df_loc['jaar'], df_loc['waarde'], 1)
        slopes.append({
            'locatie_id': loc,
            'slope': slope,
            'n_jaren': n_measurements
        })

df_slopes = pd.DataFrame(slopes) if slopes else pd.DataFrame(columns=['locatie_id', 'slope', 'n_jaren'])

if not df_slopes.empty:
    # Specifieke thresholds per metriek
    if selected_metric in ['diepte_m', 'doorzicht_m']:
        threshold = 0.1
    elif selected_metric == 'soort_count':
        threshold = 0.4
    else:
        threshold = 1.0

    df_slopes['Trend'] = df_slopes['slope'].apply(lambda x: categorize_slope_trend(x, threshold))

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_pie = px.pie(
            df_slopes, names='Trend', title="Trends meetlocaties per geselecteerd waterlichaam",
            color='Trend',
            color_discrete_map={'Verbeterend ↗️': 'green', 'Verslechterend ↘️': 'red', 'Stabiel ➡️': 'grey'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.write(f"**Detailtabel (meetlocaties met >= {MIN_JAREN} jaar data)**")
        st.dataframe(
            df_slopes.sort_values('slope', ascending=False)
                     .style.background_gradient(subset=['slope'], cmap='RdYlGn')
                     .format({'slope': "{:.4f}", 'n_jaren': "{:.0f}"}),
            use_container_width=True, hide_index=True
        )
else:
    st.warning(f"Geen locaties gevonden met minimaal {MIN_JAREN} jaar aan meetgegevens in de huidige selectie.")

# -------------------------------------------------------------
# 3) Geografische patroonherkenning (Pydeck)
# -------------------------------------------------------------
st.divider()
st.subheader("Geografische patroonanalyse (Pydeck)")

# Data voorbereiding voor de kaart
if selected_metric == 'soort_count':
    df_map_source = df_filtered.groupby(['locatie_id', 'jaar']).agg({
        'lat': 'first',
        'lon': 'first',
        'soort': 'nunique'
    }).reset_index().rename(columns={'soort': 'soort_count'})
else:
    df_map_source = df_filtered.groupby(['locatie_id', 'jaar']).agg({
        'lat': 'first',
        'lon': 'first',
        selected_metric: 'mean'
    }).reset_index()

df_map_source = df_map_source.dropna(subset=['lat', 'lon'])

col_mode, col_legenda = st.columns([2, 1])
with col_mode:
    map_mode = st.radio(
        "Selecteer analysemodus:",
        ["⏱️ Tijdlijn animatie", "⚖️ Verschilmodus (jaar A vs jaar B)"],
        horizontal=True
    )

map_container = st.empty()
legend_container = st.empty()

if "Verschil" in map_mode:
    available_years = sorted(df_map_source['jaar'].unique())
    if len(available_years) < 2:
        st.warning("Onvoldoende jaren voor een vergelijking.")
    else:
        start_y, end_y = st.select_slider(
            "Vergelijk Jaar A met Jaar B",
            options=available_years,
            value=(available_years[0], available_years[-1])
        )

        df_start = df_map_source[df_map_source['jaar'] == start_y].copy()
        df_end = df_map_source[df_map_source['jaar'] == end_y].copy()

        df_diff = pd.merge(
            df_start[['locatie_id', 'lat', 'lon', selected_metric]],
            df_end[['locatie_id', selected_metric]],
            on='locatie_id',
            suffixes=('_start', '_end')
        )
        if df_diff.empty:
            st.error(f"Geen locaties gevonden die data hebben in zowel {start_y} als {end_y}.")
        else:
            df_diff['waarde_start'] = df_diff[f'{selected_metric}_start'].round(1)
            df_diff['waarde_end'] = df_diff[f'{selected_metric}_end'].round(1)
            df_diff['delta'] = (df_diff['waarde_end'] - df_diff['waarde_start']).round(1)
            df_diff['color'] = df_diff['delta'].apply(get_color_diff)
            df_diff['radius'] = 150

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

            st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
<strong>Legenda (Verschil)</strong><br>
<span style='color:green'>●</span> Verbetering<br>
<span style='color:grey'>●</span> Stabiel<br>
<span style='color:red'>●</span> Verslechtering<br>
</div>
""", unsafe_allow_html=True)
else:
    # --- ANIMATIE / TIJDLIJN MODUS ---
    available_years = sorted(df_map_source['jaar'].unique())

    if len(available_years) == 0:
        st.info("Geen jaren beschikbaar voor de tijdlijn.")
    elif len(available_years) == 1:
        # Eén jaar: render statisch zonder slider
        min_val = df_map_source[selected_metric].min()
        max_val = df_map_source[selected_metric].max()
        selected_year = int(available_years[0])
        deck = create_year_map_deck(df_map_source, selected_year, selected_metric, min_val, max_val)
        map_container.pydeck_chart(deck)
        st.caption(f"Enkel jaar beschikbaar: **{selected_year}**")
    else:
        min_val = df_map_source[selected_metric].min()
        max_val = df_map_source[selected_metric].max()

        # Controls
        col_play, col_slider = st.columns([0.2, 0.8])
        with col_play:
            play_btn = st.button("▶️ Afspelen")
        with col_slider:
            selected_year = st.slider(
                "Selecteer jaar",
                min_value=int(min(available_years)),
                max_value=int(max(available_years)),
                value=int(min(available_years))
            )

        # Animatie Logica
        if play_btn:
            for y in available_years:
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
    {min_val:.1f} (Laag) ➡️ {max_val:.1f} (Hoog)
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4) Voor/na vergelijking (balken)
# -------------------------------------------------------------
st.divider()
st.subheader("Vergelijking versus een historisch meetjaar")

available_years = sorted(df_filtered['jaar'].dropna().unique())
if len(available_years) >= 2:
    c_year1, c_year2 = st.columns(2)
    year_start = c_year1.selectbox("Referentiejaar", available_years, index=0)
    year_end = c_year2.selectbox("Vergelijkingsjaar", available_years, index=len(available_years)-1)

    df_compare = df_trend[df_trend['jaar'].isin([year_start, year_end])]
    fig_bar = px.bar(
        df_compare, x='locatie_id', y='waarde', color=df_compare['jaar'].astype(str),
        barmode='group', title=f"Vergelijking {year_start} vs {year_end} per locatie",
        labels={'waarde': selected_metric_label, 'color': 'Jaar'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Niet genoeg jaren aan data voor een vergelijking.")