import streamlit as st
import plotly.express as px
from utils import load_data

st.set_page_config(layout="wide")
st.title("🌿 Ecologische Duiding & Indicatoren")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Selectie Filters")

# 1. Project Filter
all_projects = sorted(df['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer Project(en)", 
    options=all_projects, 
    default=all_projects
)

# Filter data op basis van project voor de volgende stap
df_project = df[df['Project'].isin(selected_projects)]

# 2. Waterlichaam Filter
all_bodies = sorted(df_project['Waterlichaam'].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer Waterlichaam",
    options=all_bodies,
    default=all_bodies
)

# Filter data op basis van waterlichaam
df_filtered_base = df_project[df_project['Waterlichaam'].isin(selected_bodies)]

# 3. Indicator Filter (Verplaatst naar sidebar voor consistentie)
categories = sorted(df_filtered_base['indicator_cat'].unique())
selected_cats = st.sidebar.multiselect(
    "Filter op indicatorcategorie", 
    categories, 
    default=categories
)

# Uitsluiten van groepen (zoals EMSPTN) en toepassen van alle filters
df_species_only = df_filtered_base[
    (df_filtered_base['type'] == 'Soort') & 
    (df_filtered_base['indicator_cat'].isin(selected_cats))
].copy()

# --- BUBBLE PLOT ---
st.subheader("Relatie Doorzicht vs Bedekking")

# 1. Aggregeren
df_bubble = df_species_only.groupby(['soort', 'indicator_cat', 'jaar']).agg({
    'doorzicht_m': 'mean',
    'bedekking_pct': 'mean',
    'diepte_m': 'mean'
}).reset_index()

# 2. JAAR SELECTIE (RANGE SLIDER)
if not df['jaar'].empty:
    min_year = int(df['jaar'].min())
    max_year = int(df['jaar'].max())
    sel_years = st.slider("Selecteer periode", min_year, max_year, [min_year, max_year])

    # Filter op jaar-range
    df_bubble_range = df_bubble[
        (df_bubble['jaar'] >= sel_years[0]) & (df_bubble['jaar'] <= sel_years[1])
    ].copy()

    # Gemiddelde over de periode
    df_bubble_plot = df_bubble_range.groupby(['soort', 'indicator_cat']).agg({
        'doorzicht_m': 'mean',
        'bedekking_pct': 'mean',
        'diepte_m': 'mean'
    }).reset_index()

    # Plotly fix voor NaNs in size
    df_bubble_plot['diepte_m'] = df_bubble_plot['diepte_m'].fillna(0.1)
    df_bubble_plot.loc[df_bubble_plot['diepte_m'] <= 0, 'diepte_m'] = 0.1

    if not df_bubble_plot.empty:
        fig_bubble = px.scatter(
            df_bubble_plot, 
            x="doorzicht_m", 
            y="bedekking_pct",
            size="diepte_m", 
            color="indicator_cat",
            hover_name="soort",
            size_max=40,
            title=f"Ecologische Niche ({sel_years[0]} - {sel_years[1]})",
            labels={"doorzicht_m": "Gem. Doorzicht (m)", "bedekking_pct": "Gem. Bedekking (%)"}
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("Geen data gevonden voor deze filtercombinatie.")

# --- HEATMAP ---
st.subheader("Soortenaanwezigheid Heatmap")
heatmap_data = df_species_only.groupby(['soort', 'jaar'])['bedekking_pct'].mean().reset_index()

if not heatmap_data.empty:
    heatmap_matrix = heatmap_data.pivot(index='soort', columns='jaar', values='bedekking_pct').fillna(0)
    fig_heat = px.imshow(
        heatmap_matrix, 
        color_continuous_scale='Greens',
        aspect="auto",
        title="Ontwikkeling bedekking per soort"
    )
    st.plotly_chart(fig_heat, use_container_width=True)