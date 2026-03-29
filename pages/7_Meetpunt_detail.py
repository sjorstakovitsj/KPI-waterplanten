import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from utils import (
    load_data,
    load_chemistry_data,
    get_chemistry_location_points,
    add_chemistry_locations_to_map,
    interpret_soil_state,
    create_folium_base_map,
    build_bathymetry_legend_url,
)

st.set_page_config(layout="wide", page_title="Meetpunt Detail")
st.title("📍 Meetpunt detailniveau analyse")
st.markdown("Gedetailleerde analyse van een specifiek monitoringspunt.")

# --- DATA INLADEN ---
df = load_data()
df_chem = load_chemistry_data()
df_chem_points = get_chemistry_location_points(df_chem=df_chem)

# --- SIDEBAR: FILTERS ---
st.sidebar.header("Filters")
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# 1) Project
all_projects = sorted(df["Project"].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)",
    options=all_projects,
    default=all_projects
)
df_proj_filtered = df[df["Project"].isin(selected_projects)].copy()

# 2) Waterlichaam
all_bodies = sorted(df_proj_filtered["Waterlichaam"].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam / waterlichamen",
    options=all_bodies,
    default=all_bodies
)
df_body_filtered = df_proj_filtered[df_proj_filtered["Waterlichaam"].isin(selected_bodies)].copy()

# 2b) Jaarfilter (range slider, default = laatste 10 jaar)
year_values = sorted(pd.to_numeric(df_body_filtered["jaar"], errors="coerce").dropna().astype(int).unique().tolist())
if not year_values:
    st.warning("Geen jaardata beschikbaar voor de geselecteerde filters.")
    st.stop()

min_year_dataset = int(min(year_values))
max_year_dataset = int(max(year_values))
default_year_start = max(min_year_dataset, max_year_dataset - 10)
selected_year_range = st.sidebar.slider(
    "Selecteer jaarbereik",
    min_value=min_year_dataset,
    max_value=max_year_dataset,
    value=(default_year_start, max_year_dataset),
    step=1,
)
df_body_filtered = df_body_filtered[
    pd.to_numeric(df_body_filtered["jaar"], errors="coerce").between(
        selected_year_range[0], selected_year_range[1], inclusive="both"
    )
].copy()
if df_body_filtered.empty:
    st.warning("Geen meetpunten gevonden binnen het geselecteerde jaarbereik.")
    st.stop()

# 2c) Meetjarenfilter per locatie (range slider, default = minimaal 5 jaar)
loc_year_counts = (
    df_body_filtered[["locatie_id", "jaar"]]
    .dropna(subset=["locatie_id", "jaar"])
    .assign(jaar_num=lambda d: pd.to_numeric(d["jaar"], errors="coerce"))
    .dropna(subset=["jaar_num"])
    .drop_duplicates(subset=["locatie_id", "jaar_num"])
    .groupby("locatie_id", as_index=False)["jaar_num"]
    .count()
    .rename(columns={"jaar_num": "n_meetjaren"})
)
if loc_year_counts.empty:
    st.warning("Geen meetjaren beschikbaar voor de geselecteerde filters.")
    st.stop()

min_meetjaren = int(loc_year_counts["n_meetjaren"].min())
max_meetjaren = int(loc_year_counts["n_meetjaren"].max())
default_min_meetjaren = min(max(3, min_meetjaren), max_meetjaren)
selected_meetjaren_range = st.sidebar.slider(
    "Selecteer bereik aantal meetjaren per locatie",
    min_value=min_meetjaren,
    max_value=max_meetjaren,
    value=(default_min_meetjaren, max_meetjaren),
    step=1,
)
valid_locs = loc_year_counts[
    loc_year_counts["n_meetjaren"].between(
        selected_meetjaren_range[0], selected_meetjaren_range[1], inclusive="both"
    )
]["locatie_id"]
df_body_filtered = df_body_filtered[df_body_filtered["locatie_id"].isin(valid_locs)].copy()
if df_body_filtered.empty:
    st.warning("Geen meetpunten gevonden voor het gekozen bereik in meetjaren en jaartallen.")
    st.stop()

# 3) Meetlocaties overzicht + selectie via kaart
available_locs = sorted(df_body_filtered["locatie_id"].dropna().unique())
if not available_locs:
    st.warning("Geen meetpunten gevonden voor de geselecteerde filters.")
    st.stop()

locs_overview = (
    df_body_filtered.groupby(["locatie_id", "Waterlichaam"], as_index=False)
    .agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        jaar_min=("jaar", "min"),
        jaar_max=("jaar", "max"),
        n_records=("locatie_id", "count"),
    )
    .dropna(subset=["lat", "lon"])
)

state_key = "meetpunt_detail_selected_loc"
if state_key not in st.session_state or st.session_state[state_key] not in available_locs:
    st.session_state[state_key] = available_locs[0]

st.subheader("🗺️ Overzicht meetpunten")
st.caption("Hover over een meetpunt voor details en klik op een meetpunt om de tijdreeksen, diagnose en aangetroffen soorten hieronder te tonen.")
st.caption("Paarse ruitjes op de kaart geven de locaties van chemische metingen weer.")

if locs_overview.empty:
    st.info("Geen kaartcoördinaten beschikbaar voor de meetpunten in deze selectie.")
else:
    center_lat = float(locs_overview["lat"].mean())
    center_lon = float(locs_overview["lon"].mean())
    m = create_folium_base_map(center_lat, center_lon, zoom_start=10, control_scale=True, basemap="bathymetry")

    for row in locs_overview.itertuples(index=False):
        jaar_min = "n.v.t." if pd.isna(row.jaar_min) else int(row.jaar_min)
        jaar_max = "n.v.t." if pd.isna(row.jaar_max) else int(row.jaar_max)
        tooltip_html = (
            f"<b>Meetpunt:</b> {row.locatie_id}<br>"
            f"<b>Waterlichaam:</b> {row.Waterlichaam}<br>"
            f"<b>Periode:</b> {jaar_min} - {jaar_max}<br>"
            f"<b>Aantal records:</b> {int(row.n_records)}"
        )
        is_selected = str(row.locatie_id) == str(st.session_state[state_key])
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=8 if is_selected else 6,
            color="#000000",
            weight=3 if is_selected else 2,
            fill=True,
            fill_color="#000000",
            fill_opacity=0.12 if is_selected else 0.05,
            tooltip=tooltip_html,
            popup=str(row.locatie_id),
        ).add_to(m)

    # Overlay chemische meetlocaties als paarse ruitjes, zonder bestaande functionaliteit te veranderen.
    m = add_chemistry_locations_to_map(m, df_chem_points)

    map_event = st_folium(
        m,
        height=520,
        width=None,
        returned_objects=["last_object_clicked", "last_object_clicked_popup", "last_object_clicked_tooltip"],
        key="meetpunt_detail_map",
    )

    clicked_loc = None
    if isinstance(map_event, dict):
        clicked_loc = map_event.get("last_object_clicked_popup")
        if not clicked_loc and map_event.get("last_object_clicked"):
            clicked = map_event["last_object_clicked"]
            if isinstance(clicked, dict) and {"lat", "lng"}.issubset(clicked.keys()):
                lat_click = float(clicked["lat"])
                lon_click = float(clicked["lng"])
                nearest_idx = ((locs_overview["lat"] - lat_click) ** 2 + (locs_overview["lon"] - lon_click) ** 2).idxmin()
                clicked_loc = str(locs_overview.loc[nearest_idx, "locatie_id"])

    if clicked_loc and clicked_loc in set(map(str, available_locs)):
        clicked_loc = str(clicked_loc)
        current_loc = str(st.session_state.get(state_key, ""))
        if clicked_loc != current_loc:
            st.session_state[state_key] = clicked_loc
            st.session_state["meetpunt_detail_selectbox"] = clicked_loc
            st.rerun()

legend_url = build_bathymetry_legend_url()
with st.expander("Legenda bathymetrie", expanded=False):
    st.markdown(
        f'<img src="{legend_url}" alt="Legenda bathymetrie" style="max-width:360px; width:100%; height:auto;"/>',
        unsafe_allow_html=True,
    )
    st.caption("Legenda uit de WMS-kaartservice van Rijkswaterstaat.")

selected_loc = st.session_state[state_key]
st.caption(f"Geselecteerd meetpunt: **{selected_loc}**")
if "meetpunt_detail_selectbox" not in st.session_state or st.session_state["meetpunt_detail_selectbox"] not in available_locs:
    st.session_state["meetpunt_detail_selectbox"] = selected_loc

selected_loc = st.selectbox(
    "Of kies handmatig een meetpunt",
    available_locs,
    index=available_locs.index(st.session_state["meetpunt_detail_selectbox"]),
    key="meetpunt_detail_selectbox",
)
st.session_state[state_key] = selected_loc

# --- DATA VOORBEREIDING ---
# ✅ Belangrijk: gebruik de gefilterde set (niet de volledige df)
df_loc = df_body_filtered[df_body_filtered["locatie_id"] == selected_loc].copy()
if df_loc.empty:
    st.warning("Geen data voor deze meetlocatie binnen de huidige filters.")
    st.stop()

# --- HEADER INFO (KPI'S) ---
mean_diepte = pd.to_numeric(df_loc["diepte_m"], errors="coerce").mean()
mean_doorzicht = pd.to_numeric(df_loc["doorzicht_m"], errors="coerce").mean()
col1, col2, col3 = st.columns(3)
col1.metric("Aantal metingen (jaren)", len(df_loc["jaar"].dropna().unique()))
col2.metric("Gemiddelde diepte", f"{mean_diepte:.2f} m" if pd.notna(mean_diepte) else "n.v.t.")
col3.metric("Gemiddeld doorzicht", f"{mean_doorzicht:.2f} m" if pd.notna(mean_doorzicht) else "n.v.t.")

# --- TABS ---
tab1, tab2 = st.tabs(["📈 Tijdreeksen", "📝 Diagnose en aangetroffen soorten"])

with tab1:
    st.subheader(f"Trendontwikkeling: {selected_loc}")
    df_trend = (
        df_loc.groupby("jaar", as_index=False)
        .agg(
            totaal_bedekking_locatie=("totaal_bedekking_locatie", "mean"),
            doorzicht_m=("doorzicht_m", "mean"),
            diepte_m=("diepte_m", "mean"),
        )
        .sort_values("jaar")
    )

    max_diepte = pd.to_numeric(df_trend["diepte_m"], errors="coerce").max()
    y2_max = 3.0 if pd.isna(max_diepte) else max(3.0, float(max_diepte) * 1.2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend["jaar"],
        y=df_trend["diepte_m"],
        name="Waterdiepte (m)",
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(200, 230, 255, 0.3)",
        yaxis="y2",
    ))
    fig.add_trace(go.Bar(
        x=df_trend["jaar"],
        y=df_trend["totaal_bedekking_locatie"],
        name="Totale bedekking (%)",
        marker_color="rgba(34, 139, 34, 0.6)",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=df_trend["jaar"],
        y=df_trend["doorzicht_m"],
        name="Doorzicht (m)",
        mode="lines+markers",
        line=dict(color="#1E90FF", width=3),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Interactie: vegetatie (staven) vs. waterkolom (lijn/vlak)",
        xaxis=dict(title="Jaar"),
        yaxis=dict(
            title=dict(text="Bedekking (%)", font=dict(color="#228B22")),
            tickfont=dict(color="#228B22"),
            range=[0, 105],
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Meters (Doorzicht / Diepte)", font=dict(color="#1E90FF")),
            tickfont=dict(color="#1E90FF"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, y2_max],
        ),
        legend=dict(x=0.01, y=1.1, orientation="h"),
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    latest_year = df_loc["jaar"].dropna().max()
    if pd.isna(latest_year):
        st.info("Geen jaarinformatie beschikbaar voor deze locatie.")
    else:
        df_latest = df_loc[df_loc["jaar"] == latest_year].copy()
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(f"### Bodemdiagnose ({int(latest_year)})")
            st.write(interpret_soil_state(df_latest))
        with col_b:
            st.markdown("### Soortenlijst en historie")
            df_hist = df_loc[df_loc["type"] == "Soort"][["soort", "jaar"]].dropna().copy()
            df_hist = df_hist.drop_duplicates(subset=["soort", "jaar"]).sort_values(["soort", "jaar"], ascending=[True, False])
            if df_hist.empty:
                species_history = pd.DataFrame(columns=["soort", "Gemeten in jaren"])
            else:
                species_history = (
                    df_hist.groupby("soort", as_index=False)["jaar"]
                    .agg(list)
                    .rename(columns={"jaar": "Gemeten in jaren"})
                )
                species_history["Gemeten in jaren"] = species_history["Gemeten in jaren"].apply(lambda yrs: ", ".join(map(str, yrs)))

            df_species_now = df_latest[df_latest["type"] == "Soort"][["soort", "bedekking_pct", "groeivorm"]].copy()
            if not df_species_now.empty:
                df_combined = df_species_now.merge(species_history, on="soort", how="left")
                df_combined["bedekking_pct"] = pd.to_numeric(df_combined["bedekking_pct"], errors="coerce").fillna(0.0)
                st.dataframe(
                    df_combined.sort_values("bedekking_pct", ascending=False)
                    .style.background_gradient(subset=["bedekking_pct"], cmap="Greens"),
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.info(f"Geen specifieke soorten geregistreerd in {int(latest_year)}.")
            if not species_history.empty:
                st.write("Historisch aangetroffen soorten (niet in laatste jaar):")
                st.dataframe(species_history, width='stretch', hide_index=True)
