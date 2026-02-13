# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html

from utils import (
    load_data,
    categorize_slope_trend,
    get_sorted_species_list,
    create_map,
    df_to_geojson_points,
    render_swipe_map_html,

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

# 2b) Plantgroep filter (nieuw) – alleen zinvol als GEEN individuele soort gekozen is
st.sidebar.subheader("Extra kaartfilter")

if species_is_selected:
    st.sidebar.caption("Plantgroep-filter is uitgeschakeld bij een gekozen soort.")
    show_ondergedoken = st.sidebar.checkbox("Onder­gedoken waterplanten", value=False, disabled=True)
    show_chariden = st.sidebar.checkbox("Chariden (kranswieren)", value=False, disabled=True)
else:
    show_ondergedoken = st.sidebar.checkbox("Onder­gedoken waterplanten", value=False)
    show_chariden = st.sidebar.checkbox("Chariden (kranswieren)", value=False)

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
    
# -------------------------------------------------------------
# Extra filter: plantgroepen (ondergedoken / chariden)
# Alleen toepassen als GEEN individuele soort is gekozen
# -------------------------------------------------------------
if not species_is_selected:
    # Chariden herkennen op basis van (Latijnse) genusnamen in kolom 'soort'
    # (Chara, Nitella, Nitellopsis, Tolypella) – zie mapping in utils.py
    chara_genera = ("Chara", "Nitella", "Nitellopsis", "Tolypella")
    mask_chariden = (
        df_filtered["soort"].fillna("").astype(str).str.strip()
        .str.startswith(chara_genera)
    )

    # Onder­gedoken waterplanten komen in jouw data als groeivorm "Ondergedoken"
    # (GROWTH_FORM_MAPPING: SUBMSPTN -> Ondergedoken) en zijn doorgaans type='Groep'
    mask_ondergedoken = (
        (df_filtered.get("groeivorm", pd.Series(False, index=df_filtered.index)) == "Ondergedoken")
    )

    # Als één of beide checkboxes aangevinkt zijn, filter daarop
    if show_ondergedoken or show_chariden:
        mask = pd.Series(False, index=df_filtered.index)

        if show_ondergedoken:
            mask |= mask_ondergedoken

        if show_chariden:
            mask |= mask_chariden

        df_filtered = df_filtered[mask].copy()

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

# ---------------------------------------------------------------------------
# 0) Kaart (Swipe Map) – vervangt “resultaten voor een gekozen jaar”
# ---------------------------------------------------------------------------
st.subheader("Kaart – resultaten voor een gekozen jaar")
years_available = sorted([int(y) for y in df_filtered["jaar"].dropna().unique()])

if not years_available:
    st.info("Er zijn geen jaartallen beschikbaar om op de kaart te tonen.")
else:
    # --- 1) Bepaal defaults: twee meest recente jaren binnen de HUIDIGE selectie ---
    
    if selected_metric == "soort_count":
        df_map_source = (
            df_filtered.groupby(["locatie_id", "jaar"])
            .agg({"lat": "first", "lon": "first", "soort": "nunique"})
            .reset_index()
            .rename(columns={"soort": "soort_count"})
        )
    else:
        df_map_source = (
            df_filtered.groupby(["locatie_id", "jaar"])
            .agg({"lat": "first", "lon": "first", selected_metric: "mean"})
            .reset_index()
        )

    df_map_source = df_map_source.dropna(subset=["lat", "lon"])

    if len(years_available) >= 2:
        default_left = years_available[-2]
        default_right = years_available[-1]
    else:
        default_left = years_available[0]
        default_right = years_available[0]

    # --- 2) Maak een context-id die uniek is voor de huidige selectie ---
    # Multi-waterbody: sorteert zodat volgorde in multiselect geen invloed heeft
    bodies_key = "|".join(sorted(selected_bodies)) if selected_bodies else "ALL"
    species_key = selected_species if (selected_species and selected_species != "— Alle soorten —") else "ALLSPECIES"
    context_id = f"bodies={bodies_key}__metric={selected_metric}__species={species_key}"

    # --- 3) Init/restore per context ---
    if "swipe_years_by_context" not in st.session_state:
        st.session_state["swipe_years_by_context"] = {}

    if context_id not in st.session_state["swipe_years_by_context"]:
        # eerste keer voor deze selectie: pak defaults (laatste twee)
        st.session_state["swipe_years_by_context"][context_id] = {
            "left": default_left,
            "right": default_right,
        }

    # --- 4) Valideer: als jaren niet meer bestaan (door andere waterlichamen), herstel defaults ---
    stored_left = st.session_state["swipe_years_by_context"][context_id]["left"]
    stored_right = st.session_state["swipe_years_by_context"][context_id]["right"]

    if stored_left not in years_available:
        stored_left = default_left
    if stored_right not in years_available:
        stored_right = default_right

    # Zorg dat left != right als dat kan
    if stored_left == stored_right and len(years_available) >= 2:
        stored_left = years_available[-2]
        stored_right = years_available[-1]

    # --- 5) UI: selectboxen met defaults uit context ---
    c_left, c_right = st.columns([1, 1])

    with c_left:
        year_left = st.selectbox(
            "Jaar links",
            years_available,
            index=years_available.index(stored_left),
            key=f"swipe_year_left__{context_id}",
        )

    with c_right:
        year_right = st.selectbox(
            "Jaar rechts",
            years_available,
            index=years_available.index(stored_right),
            key=f"swipe_year_right__{context_id}",
        )

    # --- 6) Schrijf de keuze terug naar context state ---
    st.session_state["swipe_years_by_context"][context_id]["left"] = year_left
    st.session_state["swipe_years_by_context"][context_id]["right"] = year_right

    if year_left == year_right:
        st.warning("Kies twee verschillende jaren voor de swipe‑vergelijking.")
    else:
        df_left = df_map_source[df_map_source["jaar"] == year_left][
            ["locatie_id", "lat", "lon", selected_metric]
        ].copy()
        df_right = df_map_source[df_map_source["jaar"] == year_right][
            ["locatie_id", "lat", "lon", selected_metric]
        ].copy()

        # 👉 Standaardiseer de kolomnaam voor GeoJSON-properties
        df_left = df_left.rename(columns={selected_metric: "value"})
        df_right = df_right.rename(columns={selected_metric: "value"})

        gj_left = df_to_geojson_points(df_left, value_col="value", id_col="locatie_id")
        gj_right = df_to_geojson_points(df_right, value_col="value", id_col="locatie_id")

        # bounds (links + rechts samen) -> autozoom
        df_bounds = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["lat", "lon"])
        if len(df_bounds) >= 2:
            bounds = [
                float(df_bounds["lon"].min()),
                float(df_bounds["lat"].min()),
                float(df_bounds["lon"].max()),
                float(df_bounds["lat"].max()),
            ]
        else:
            bounds = None

        # fallback center/scale
        center_lat = float(df_map_source["lat"].mean())
        center_lon = float(df_map_source["lon"].mean())
        min_val = float(df_map_source[selected_metric].min())
        max_val = float(df_map_source[selected_metric].max())

        swipe_html = render_swipe_map_html(
            geojson_left=gj_left,
            geojson_right=gj_right,
            year_left=year_left,
            year_right=year_right,
            metric_label=selected_metric_label,
            min_val=min_val,
            max_val=max_val,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=9.0,
            height_px=650,
            bounds=bounds,
        )
        html(swipe_html, height=670)


    # Eenvoudige legenda (zelfde schaal links en rechts)
    st.markdown(f"""
        <div style="display:flex;align-items:center;background-color: rgba(255,255,255,0.85);
                    padding:10px;border-radius:6px;border:1px solid #ddd;margin-top:6px;">
            <div style="margin-right:10px;"><strong>Legenda ({selected_metric_label}):</strong></div>
            <div style="background: linear-gradient(90deg, red, green); width: 160px; height: 14px; border: 1px solid #ccc;"></div>
            <div style="margin-left:10px; font-size: 0.9em;">
                {min_val:.1f} (Laag) → {max_val:.1f} (Hoog)
            </div>
        </div>
        """, unsafe_allow_html=True)

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
# 4) Voor/na vergelijking (balken) – default 2 meest recente jaren
# -------------------------------------------------------------
st.divider()
st.subheader("Vergelijking versus een historisch meetjaar")

available_years = sorted(df_filtered["jaar"].dropna().unique())

if len(available_years) >= 2:
    c_year1, c_year2 = st.columns(2)

    # defaults: laatste twee
    default_start = int(available_years[-2])
    default_end = int(available_years[-1])

    year_start = c_year1.selectbox(
        "Referentiejaar",
        available_years,
        index=available_years.index(default_start),
        key="bar_ref_year",
    )
    year_end = c_year2.selectbox(
        "Vergelijkingsjaar",
        available_years,
        index=available_years.index(default_end),
        key="bar_cmp_year",
    )

    df_compare = df_trend[df_trend["jaar"].isin([year_start, year_end])]

    fig_bar = px.bar(
        df_compare,
        x="locatie_id",
        y="waarde",
        color=df_compare["jaar"].astype(str),
        barmode="group",
        title=f"Vergelijking {year_start} vs {year_end} per locatie",
        labels={"waarde": selected_metric_label, "color": "Jaar"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Niet genoeg jaren aan data voor een vergelijking.")
    
# -------------------------------------------------------------
# 5) TOP 50 HEATMAP
# -------------------------------------------------------------
st.divider()
st.subheader("Soortenaanwezigheid heatmap (top 50)")

# Basis voor heatmap: alleen filter op waterlichamen (niet op selected_species)
df_heat_base = df[df["Waterlichaam"].isin(selected_bodies)] if selected_bodies else df.copy()

# Alleen individuele soorten en bij voorkeur alleen bedekking (BEDKG)
df_species_only = df_heat_base[df_heat_base["type"] == "Soort"].copy()
if "Grootheid" in df_species_only.columns:
    df_species_only = df_species_only[df_species_only["Grootheid"] == "BEDKG"].copy()

# Zorg dat bedekking numeriek is
if "bedekking_pct" in df_species_only.columns:
    df_species_only["bedekking_pct"] = pd.to_numeric(df_species_only["bedekking_pct"], errors="coerce")

if df_species_only.empty:
    st.info("Geen soortdata gevonden voor de huidige selectie.")
else:
    # 1) Top 50 bepalen o.b.v. gemiddelde bedekking over alle jaren
# --- Gewogen TOP 50: elk (locatie_id, jaar) telt even zwaar ---
# 1) eerst per soort-locatie-jaar een gemiddelde bedekking
    df_cells = (
        df_species_only
        .groupby(["soort", "locatie_id", "jaar"])["bedekking_pct"]
        .mean()
        .reset_index()
    )

    # 2) score per soort: gemiddelde over unieke locatie-jaar cellen
    weighted_score = (
        df_cells.groupby("soort")["bedekking_pct"]
        .mean()
        .sort_values(ascending=False)
    )

    top_species = weighted_score.head(50).index

    # 2) Filter op top 50
    df_top_heatmap = df_species_only[df_species_only["soort"].isin(top_species)].copy()

    # 3) Aggregeer per soort en jaar
    heatmap_data = (
        df_top_heatmap.groupby(["soort", "jaar"])["bedekking_pct"]
        .mean()
        .reset_index()
    )

    if heatmap_data.empty:
        st.warning("Geen data beschikbaar voor de heatmap.")
    else:
        # Pivot naar matrix
        heatmap_matrix = (
            heatmap_data.pivot(index="soort", columns="jaar", values="bedekking_pct")
            .fillna(0)
        )

        # --- Maskers maken ---
        # 0-cellen (exact nul) markeren
        zero_mask = (heatmap_matrix == 0)

        # Voor de "groene" laag: zet nullen op NaN zodat ze niet groen kleuren
        matrix_green = heatmap_matrix.mask(zero_mask, other=np.nan)

        # --- Basis heatmap (alleen >0) ---
        fig_heat = px.imshow(
            matrix_green,
            color_continuous_scale="Greens",
            aspect="auto",
            title="Ontwikkeling bedekking (top 50 meest voorkomende soorten)",
            labels=dict(x="Jaar", y="Soort", color="Gem. Bedekking (%)")
        )

        # --- Grijze laag voor 0-cellen ---
        # Maak een overlay-matrix: 1 op plekken waar 0 zit, anders NaN
        matrix_zero = zero_mask.astype(int).replace({0: np.nan})

        fig_heat.add_trace(
            go.Heatmap(
                z=matrix_zero.values,
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(200,200,200,0.65)"]],
                showscale=False,
                hoverinfo="skip"
            )
        )

        # --- Gearceerd effect: X-markers bovenop 0-cellen ---
        # Coördinaten van 0-cellen (y=soortnaam, x=jaar)
        ys, xs = np.where(zero_mask.values)

        x_vals = [heatmap_matrix.columns[i] for i in xs]
        y_vals = [heatmap_matrix.index[i] for i in ys]

        fig_heat.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=6,
                    color="rgba(120,120,120,0.55)"
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # Leesbaarheid
        fig_heat.update_layout(
            height=1200,
            yaxis=dict(side="left")
        )

        st.plotly_chart(fig_heat, use_container_width=True)