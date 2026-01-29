import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data

st.set_page_config(layout="wide")
st.title("ℹ️ Onderliggende (meta)data")
st.markdown("Controle op volledigheid, meetgaten en taxonomische consistentie.")

df = load_data()

# --- 1. ALGEMENE METADATA ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("totaal aantal records", f"{len(df):,}")
col2.metric("unieke locaties", df['locatie_id'].nunique())
col3.metric("unieke soorten", df['soort'].nunique())
col4.metric("jaren bereik", f"{df['jaar'].min()} - {df['jaar'].max()}")

st.divider()

# --- 2. MEETGATEN ANALYSE (HEATMAP) ---
st.subheader("📅 Tijdruimtelijke meetinspanning")
st.markdown("Blauwe vlakken geven aan dat er gemeten is. Witte gaten zijn ontbrekende jaren per locatie. **NB.** dit zijn niet alle meetlocaties.")

# Pivot table: Index=Locatie, Kolom=Jaar, Waarde=Aantal metingen (of aanwezigheid)
coverage_matrix = df.groupby(['locatie_id', 'jaar']).size().unstack(fill_value=0)

# Heatmap plotten
fig_heat = px.imshow(coverage_matrix, 
                     labels=dict(x="Jaar", y="Locatie", color="Aantal waarnemingen"),
                     x=coverage_matrix.columns,
                     y=coverage_matrix.index,
                     aspect="auto",
                     color_continuous_scale="Blues")
fig_heat.update_layout(height=800) # Langere plot omdat er veel locaties kunnen zijn
st.plotly_chart(fig_heat, use_container_width=True)

# --- 3. WAARNEMINGEN PER JAAR ---
st.subheader("📊 Waarnemingsinspanning per jaar")
obs_per_year = df.groupby('jaar').size().reset_index(name='aantal_records')
locs_per_year = df.groupby('jaar')['locatie_id'].nunique().reset_index(name='aantal_locaties')

c1, c2 = st.columns(2)
with c1:
    fig_obs = px.bar(obs_per_year, x='jaar', y='aantal_records', title="Totaal aantal records per jaar")
    st.plotly_chart(fig_obs, use_container_width=True)
with c2:
    fig_locs = px.line(locs_per_year, x='jaar', y='aantal_locaties', markers=True, 
                       title="Aantal bezochte meetlocaties per jaar", line_shape='spline')
    fig_locs.update_yaxes(range=[0, df['locatie_id'].nunique() + 5])
    st.plotly_chart(fig_locs, use_container_width=True)

# --- 4. CONSISTENTIE SOORTENNAAM ---
st.divider()
st.subheader("Taxonomische consistentie")
st.markdown("Controleer hieronder op zeldzame spellingen of dubbele namen (mogelijk invoerfouten).")

species_counts = df['soort'].value_counts().reset_index()
species_counts.columns = ['Soortnaam', 'Aantal Records']

# Markeer potentiele fouten (soorten die heel weinig voorkomen, bijv. < 10 keer in 200k records)
species_counts['Status'] = species_counts['Aantal Records'].apply(
    lambda x: '⚠️ Check (Zeldzaam)' if x < (len(df)*0.001) else '✅ OK'
)

st.dataframe(species_counts, use_container_width=True)

# --- 5. RUIMTELIJKE SPREIDING ---
st.subheader("Ruimtelijke dekking meetnet")
st.markdown("Overzicht van alle unieke meetpunten in de dataset.")

# Simpele kaart met alle unieke punten (ongeacht jaar)
unique_locs = df.groupby('locatie_id').agg({
    'lat': 'first', 
    'lon': 'first', 
    'jaar': lambda x: f"{min(x)}-{max(x)}", # Periode
    'soort': 'count' # Totaal waarnemingen
}).reset_index()

st.map(unique_locs)