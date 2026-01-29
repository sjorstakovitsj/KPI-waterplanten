import streamlit as st
import plotly.express as px
from utils import load_data

st.set_page_config(layout="wide")
st.title("🌿 Ecologische indices")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Selectie filters")

# 1. Project Filter
all_projects = sorted(df['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)", 
    options=all_projects, 
    default=all_projects
)

# Filter data op basis van project voor de volgende stap
df_project = df[df['Project'].isin(selected_projects)]

# 2. Waterlichaam Filter
all_bodies = sorted(df_project['Waterlichaam'].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam",
    options=all_bodies,
    default=all_bodies
)

# Filter data op basis van waterlichaam
df_filtered_base = df_project[df_project['Waterlichaam'].isin(selected_bodies)]

# Uitsluiten van groepen (zoals EMSPTN) en toepassen van alle filters
df_species_only = df_filtered_base[
    (df_filtered_base['type'] == 'Soort')
].copy()

# --- BUBBLE PLOT ---
st.subheader("Relatie doorzicht vs bedekking")

# 1. Aggregeren
df_bubble = df_species_only.groupby(['soort','jaar']).agg({
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
    df_bubble_plot = df_bubble_range.groupby(['soort']).agg({
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
            hover_name="soort",
            size_max=40,
            title=f"Ecologische indices ({sel_years[0]} - {sel_years[1]})",
            labels={"doorzicht_m": "gem. doorzicht (m)", "bedekking_pct": "gem. bedekking (%)"}
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("Geen data gevonden voor deze filtercombinatie.")

# --- HEATMAP ---
st.subheader("Soortenaanwezigheid heatmap (top 50)")

if not df_species_only.empty:
    # 1. Bereken de gemiddelde bedekking per soort over alle jaren om de 'Top 100' te bepalen
    top_species = (
        df_species_only.groupby('soort')['bedekking_pct']
        .mean()
        .sort_values(ascending=False)
        .head(50)
        .index
    )

    # 2. Filter de dataset zodat alleen deze top 100 overblijft
    df_top_heatmap = df_species_only[df_species_only['soort'].isin(top_species)]

    # 3. Aggregeer de data per soort en jaar voor de heatmap
    heatmap_data = df_top_heatmap.groupby(['soort', 'jaar'])['bedekking_pct'].mean().reset_index()

    if not heatmap_data.empty:
        # Pivot naar matrix-vorm
        heatmap_matrix = heatmap_data.pivot(index='soort', columns='jaar', values='bedekking_pct').fillna(0)
        
        # Sorteer de matrix ook op de gemiddelde bedekking zodat de meest voorkomende bovenaan staan
        heatmap_matrix = heatmap_matrix.loc[top_species.intersection(heatmap_matrix.index)]

        fig_heat = px.imshow(
            heatmap_matrix, 
            color_continuous_scale='Greens',
            aspect="auto",
            title="Ontwikkeling bedekking (top 50 meest voorkomende soorten)",
            labels=dict(x="Jaar", y="Soort", color="Gem. Bedekking (%)")
        )
        
        # Verbeter de leesbaarheid van de y-as (soorten)
        fig_heat.update_layout(
            height=1200,  # Verhoogde hoogte voor 100 soorten
            yaxis={'side': 'left'}
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Geen data beschikbaar voor de heatmap.")
else:
    st.info("Geen soortdata gevonden voor de huidige filters.")