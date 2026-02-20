import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_data, RWS_GROEIVORM_CODES

st.set_page_config(layout="wide")
st.title("ℹ️ Onderliggende (meta)data")
st.markdown("Controle op volledigheid, meetgaten en taxonomische consistentie.")

df = load_data()
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# Zorg dat jaar numeriek is (voor min/max/plots)
df["jaar"] = pd.to_numeric(df["jaar"], errors="coerce")

# --- 1. ALGEMENE METADATA ---
min_year = df["jaar"].min()
max_year = df["jaar"].max()
year_range_txt = "n.v.t." if (pd.isna(min_year) or pd.isna(max_year)) else f"{int(min_year)} - {int(max_year)}"

col1, col2, col3, col4 = st.columns(4)
col1.metric("totaal aantal records", f"{len(df):,}")
col2.metric("unieke locaties", int(df["locatie_id"].nunique()))
col3.metric("unieke soorten", int(df["soort"].nunique()))
col4.metric("jaren bereik", year_range_txt)
st.divider()

# --- 2. MEETGATEN ANALYSE (HEATMAP) ---
st.subheader("📅 Tijdruimtelijke meetinspanning")
st.markdown(
    "Blauwe vlakken geven aan dat er gemeten is. Witte gaten zijn ontbrekende jaren per locatie. "
    "**NB.** dit zijn niet alle meetlocaties."
)

# Optionele limiter om extreme plots te vermijden (geen functieverlies: je kunt altijd "alles" tonen)
show_all = st.checkbox("Toon alle locaties in heatmap (kan traag zijn)", value=False)
top_n = st.slider("Max. aantal locaties in heatmap", 50, 2000, 300, 50, disabled=show_all)

if show_all:
    df_heat = df[["locatie_id", "jaar"]].dropna()
else:
    # Top locaties op record-aantal (meetinspanning)
    top_locs = df["locatie_id"].value_counts().head(int(top_n)).index
    df_heat = df[df["locatie_id"].isin(top_locs)][["locatie_id", "jaar"]].dropna()

coverage_matrix = df_heat.groupby(["locatie_id", "jaar"]).size().unstack(fill_value=0)

fig_heat = px.imshow(
    coverage_matrix,
    labels=dict(x="Jaar", y="Locatie", color="Aantal waarnemingen"),
    x=coverage_matrix.columns,
    y=coverage_matrix.index,
    aspect="auto",
    color_continuous_scale="Blues",
)
fig_heat.update_layout(height=800)
st.plotly_chart(fig_heat, use_container_width=True)

# --- 3. WAARNEMINGEN PER JAAR ---
st.subheader("📊 Waarnemingsinspanning per jaar")

obs_per_year = df.groupby("jaar").size().reset_index(name="aantal_records")
locs_per_year = df.groupby("jaar")["locatie_id"].nunique().reset_index(name="aantal_locaties")

c1, c2 = st.columns(2)
with c1:
    fig_obs = px.bar(obs_per_year, x="jaar", y="aantal_records", title="Totaal aantal records per jaar")
    st.plotly_chart(fig_obs, use_container_width=True)

with c2:
    fig_locs = px.line(
        locs_per_year,
        x="jaar",
        y="aantal_locaties",
        markers=True,
        title="Aantal bezochte meetlocaties per jaar",
        line_shape="spline",
    )
    fig_locs.update_yaxes(range=[0, int(df["locatie_id"].nunique()) + 5])
    st.plotly_chart(fig_locs, use_container_width=True)

# --- 4. CONSISTENTIE SOORTENNAAM ---
st.divider()
st.subheader("Taxonomische consistentie")
st.markdown("Controleer hieronder op zeldzame spellingen of dubbele namen (mogelijk invoerfouten).")

# ✅ Alleen individuele soorten meenemen (geen groeivorm-/aggregatiecodes en geen type='Groep')
if "type" in df.columns:
    df_tax = df[(df["type"] == "Soort") & (~df["soort"].isin(RWS_GROEIVORM_CODES))].copy()
else:
    # Fallback: als 'type' ontbreekt, filter alleen de bekende aggregatiecodes
    df_tax = df[~df["soort"].isin(RWS_GROEIVORM_CODES)].copy()

species_counts = df_tax["soort"].value_counts().reset_index()
species_counts.columns = ["Soortnaam", "Aantal Records"]

total_records = len(df_tax)
# Vermijd deling door 0
if total_records == 0:
    st.info("Geen individuele soorten aanwezig in de huidige selectie.")
else:
    species_counts["Percentage"] = (species_counts["Aantal Records"] / total_records) * 100

    # Klassen op basis van percentage (in %)
    p = species_counts["Percentage"]

    conditions = [
        (p < 0.01),                  # <0,01%
        (p >= 0.01) & (p < 0.1),     # 0,01–0,1%
        (p >= 0.1) & (p < 1.0),      # 0,1–1%
        (p >= 1.0) & (p < 2.5),      # 1–2.5%
        (p >= 2.5),                  # >2.5%
    ]

    choices = [
        "🚨 Extreem zeldzaam (<0,01%)",
        "🚨 Zeer zeldzaam (0,01–0,1%)",
        "⚠️ Zeldzaam (0,1–1%)",
        "🟡 Vaak voorkomend (1–2.5%)",
        "🟢 Algemeen (>2.5%)",
    ]

    species_counts["Status"] = np.select(conditions, choices, default="Onbekend")

    # (optioneel) handige context in de UI
    st.caption(f"Taxonomische consistentie gebaseerd op {total_records:,} records van individuele soorten (excl. aggregatiecodes).")

    st.dataframe(species_counts[["Soortnaam", "Aantal Records", "Percentage", "Status"]], use_container_width=True)

# --- 5. RUIMTELIJKE SPREIDING ---
st.subheader("Ruimtelijke dekking meetnet")
st.markdown("Overzicht van alle unieke meetpunten in de dataset.")

unique_locs = (
    df.groupby("locatie_id", as_index=False)
    .agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        jaar_min=("jaar", "min"),
        jaar_max=("jaar", "max"),
        soort=("soort", "count"),
    )
)

unique_locs["periode"] = np.where(
    unique_locs["jaar_min"].notna() & unique_locs["jaar_max"].notna(),
    unique_locs["jaar_min"].astype(int).astype(str) + "-" + unique_locs["jaar_max"].astype(int).astype(str),
    "n.v.t."
)

# st.map verwacht lat/lon; we tonen extra kolommen blijven zichtbaar in tooltip waar supported
st.map(unique_locs.rename(columns={"soort": "totaal_waarnemingen"}))