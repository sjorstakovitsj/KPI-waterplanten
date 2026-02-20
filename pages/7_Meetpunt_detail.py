import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import load_data, interpret_soil_state

st.set_page_config(layout="wide", page_title="Meetpunt Detail")
st.title("📍 Meetpunt detailniveau analyse")
st.markdown("Gedetailleerde analyse van een specifiek monitoringspunt.")

# --- DATA INLADEN ---
df = load_data()

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

# 3) Meetlocatie
available_locs = sorted(df_body_filtered["locatie_id"].dropna().unique())
if not available_locs:
    st.warning("Geen meetpunten gevonden voor de geselecteerde filters.")
    st.stop()

selected_loc = st.sidebar.selectbox("Selecteer meetlocatie", available_locs)

# --- DATA VOORBEREIDING ---
# ✅ Belangrijk: gebruik de gefilterde set (niet de volledige df)
df_loc = df_body_filtered[df_body_filtered["locatie_id"] == selected_loc].copy()

if df_loc.empty:
    st.warning("Geen data voor deze meetlocatie binnen de huidige filters.")
    st.stop()

# --- HEADER INFO (KPI'S) ---
# Precompute means (micro-optimalisatie)
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

    # Data aggregatie per jaar
    df_trend = (
        df_loc.groupby("jaar", as_index=False)
        .agg(
            totaal_bedekking_locatie=("totaal_bedekking_locatie", "mean"),
            doorzicht_m=("doorzicht_m", "mean"),
            diepte_m=("diepte_m", "mean"),
        )
        .sort_values("jaar")
    )

    # veilige y2-range (avoid NaN)
    max_diepte = pd.to_numeric(df_trend["diepte_m"], errors="coerce").max()
    y2_max = 3.0 if pd.isna(max_diepte) else max(3.0, float(max_diepte) * 1.2)

    fig = go.Figure()

    # 1) Diepte (vlak, achtergrond) rechter as
    fig.add_trace(go.Scatter(
        x=df_trend["jaar"],
        y=df_trend["diepte_m"],
        name="Waterdiepte (m)",
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(200, 230, 255, 0.3)",
        yaxis="y2",
    ))

    # 2) Bedekking (staven) linker as
    fig.add_trace(go.Bar(
        x=df_trend["jaar"],
        y=df_trend["totaal_bedekking_locatie"],
        name="Totale bedekking (%)",
        marker_color="rgba(34, 139, 34, 0.6)",
        yaxis="y",
    ))

    # 3) Doorzicht (lijn) rechter as
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

    st.plotly_chart(fig, use_container_width=True)

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

            # --- 1) Historie: unieke jaren per soort (sneller/cleaner dan groupby.apply)
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

            # --- 2) Laatste jaar soorten
            df_species_now = df_latest[df_latest["type"] == "Soort"][["soort", "bedekking_pct", "groeivorm"]].copy()

            if not df_species_now.empty:
                df_combined = df_species_now.merge(species_history, on="soort", how="left")
                # sorteer op bedekking
                df_combined["bedekking_pct"] = pd.to_numeric(df_combined["bedekking_pct"], errors="coerce").fillna(0.0)

                st.dataframe(
                    df_combined.sort_values("bedekking_pct", ascending=False)
                    .style.background_gradient(subset=["bedekking_pct"], cmap="Greens"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"Geen specifieke soorten geregistreerd in {int(latest_year)}.")
                if not species_history.empty:
                    st.write("Historisch aangetroffen soorten (niet in laatste jaar):")
                    st.dataframe(species_history, use_container_width=True, hide_index=True)