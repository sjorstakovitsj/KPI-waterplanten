import streamlit as st
import plotly.express as px
import numpy as np
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

    df_bubble_plot = df_bubble_range.groupby(['soort']).agg({
    'doorzicht_m': 'mean',
    'bedekking_pct': 'mean',
    'diepte_m': 'mean'
}).reset_index()

    # --- NIEUW: ratio doorzicht / diepte voor x-as ---
    # Zorg dat diepte niet 0 of negatief is om deling door 0 te voorkomen
    df_bubble_plot["diepte_safe"] = df_bubble_plot["diepte_m"].fillna(0.0)
    df_bubble_plot.loc[df_bubble_plot["diepte_safe"] <= 0, "diepte_safe"] = np.nan

    df_bubble_plot["doorzicht_diepte_ratio"] = df_bubble_plot["doorzicht_m"] / df_bubble_plot["diepte_safe"]

    # Optioneel: verwijder rijen waar ratio niet berekend kan worden
    df_bubble_plot = df_bubble_plot.dropna(subset=["doorzicht_diepte_ratio"])

    # Plotly fix voor NaNs in size (bubble size blijft diepte_m)
    df_bubble_plot['diepte_m'] = df_bubble_plot['diepte_m'].fillna(0.1)
    df_bubble_plot.loc[df_bubble_plot['diepte_m'] <= 0, 'diepte_m'] = 0.1


    # Plotly fix voor NaNs in size
    df_bubble_plot['diepte_m'] = df_bubble_plot['diepte_m'].fillna(0.1)
    df_bubble_plot.loc[df_bubble_plot['diepte_m'] <= 0, 'diepte_m'] = 0.1

    if not df_bubble_plot.empty:
        fig_bubble = px.scatter(
            df_bubble_plot,
            x="doorzicht_diepte_ratio",
            y="bedekking_pct",
            size="diepte_m",
            hover_name="soort",
            size_max=40,
            title=f"Ecologische indices ({sel_years[0]} - {sel_years[1]})",
            labels={
                "doorzicht_diepte_ratio": "gem. doorzicht / gem. diepte",
                "bedekking_pct": "gem. bedekking (%)",
                "diepte_m": "gem. diepte (m)"
            }
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("Geen data gevonden voor deze filtercombinatie.")
    
    # --- Shaded band: gewenst bereik 0.6–0.8 ---
    fig_bubble.add_vrect(
        x0=0.6, x1=0.8,
        fillcolor="rgba(0, 200, 0, 0.12)",  # lichtgroen transparant
        line_width=0,
        annotation_text="Gewenst bereik (0.6–0.8)",
        annotation_position="top left"
    )

    # Optioneel: extra stippellijn op de ideale waarde 0.8
    fig_bubble.add_vline(
        x=0.8,
        line_width=2,
        line_dash="dot",
        line_color="green",
        annotation_text="Ideaal 0.8",
        annotation_position="top left"
    )

    # Optioneel: ook een lijn op 0.6 (minimum)
    fig_bubble.add_vline(
        x=0.6,
        line_width=2,
        line_dash="dot",
        line_color="orange",
        annotation_text="Minimaal 0.6",
        annotation_position="top left"
    )

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