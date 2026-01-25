import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
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

# 1. Project Filter
all_projects = sorted(df['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)", 
    options=all_projects, 
    default=all_projects
)

# 2. Waterlichaam Filter
df_proj_filtered = df[df['Project'].isin(selected_projects)]
all_bodies = sorted(df_proj_filtered['Waterlichaam'].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam / waterlichamen",
    options=all_bodies,
    default=all_bodies
)

# 3. Meetlocatie Selectie
df_body_filtered = df_proj_filtered[df_proj_filtered['Waterlichaam'].isin(selected_bodies)]
available_locs = sorted(df_body_filtered['locatie_id'].unique())

if not available_locs:
    st.warning("Geen meetpunten gevonden voor de geselecteerde filters.")
    st.stop()

selected_loc = st.sidebar.selectbox("Selecteer meetlocatie", available_locs)

# --- DATA VOORBEREIDING ---
df_loc = df[df['locatie_id'] == selected_loc]

# --- HEADER INFO (KPI'S) ---
col1, col2, col3 = st.columns(3)
col1.metric("Aantal metingen", len(df_loc['jaar'].unique()))
col2.metric("Gemiddelde diepte", f"{df_loc['diepte_m'].mean():.2f} m")
col3.metric("Gemiddeld doorzicht", f"{df_loc['doorzicht_m'].mean():.2f} m")

# --- TABS ---
tab1, tab2 = st.tabs(["📈 Tijdreeksen", "📝 Diagnose en aangetroffen soorten"])

with tab1:
    st.subheader(f"Trendontwikkeling: {selected_loc}")
    
    # Data aggregatie per jaar
    df_trend = df_loc.groupby('jaar').agg({
        'totaal_bedekking_locatie': 'mean', 
        'doorzicht_m': 'mean',
        'diepte_m': 'mean'
    }).reset_index()

    fig = go.Figure()

    # 1. Diepte (Als gevuld vlak op de achtergrond - Rechter as)
    fig.add_trace(go.Scatter(
        x=df_trend['jaar'],
        y=df_trend['diepte_m'],
        name='Waterdiepte (m)',
        fill='tozeroy',
        mode='none', # Geen lijn, alleen vlak
        fillcolor='rgba(200, 230, 255, 0.3)', # Zeer lichtblauw
        yaxis='y2'
    ))

    # 2. Bedekking (Staven - Linker as)
    fig.add_trace(go.Bar(
        x=df_trend['jaar'],
        y=df_trend['totaal_bedekking_locatie'],
        name='Totale bedekking (%)',
        marker_color='rgba(34, 139, 34, 0.6)',
        yaxis='y'
    ))

    # 3. Doorzicht (Lijn - Rechter as)
    fig.add_trace(go.Scatter(
        x=df_trend['jaar'],
        y=df_trend['doorzicht_m'],
        name='Doorzicht (m)',
        mode='lines+markers',
        line=dict(color='#1E90FF', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))

    # Layout met dubbele as
    fig.update_layout(
        title="Interactie: vegetatie (staven) vs. waterkolom (lijn/vlak)",
        xaxis=dict(title="Jaar"),
        
        # Linker Y-as: Bedekking
        yaxis=dict(
            title=dict(text="Bedekking (%)", font=dict(color="#228B22")),
            tickfont=dict(color="#228B22"),
            range=[0, 105],
            side="left"
        ),
        
        # Rechter Y-as: Doorzicht en Diepte (beide in meters)
        yaxis2=dict(
            title=dict(text="Meters (Doorzicht / Diepte)", font=dict(color="#1E90FF")),
            tickfont=dict(color="#1E90FF"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, max(3.0, df_trend['diepte_m'].max() * 1.2)]
        ),
        legend=dict(x=0.01, y=1.1, orientation='h'),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    latest_year = df_loc['jaar'].max()
    df_latest = df_loc[df_loc['jaar'] == latest_year]
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown(f"### Bodemdiagnose ({latest_year})")
        st.write(interpret_soil_state(df_latest))
    
    with col_b:
        st.markdown(f"### Soortenlijst en historie")

        # 1. Haal alle unieke jaren per soort op voor DEZE locatie
        # We groeperen op soort en maken een lijst van unieke jaren, gesorteerd van nieuw naar oud
        species_history = (
            df_loc[df_loc['type'] == 'Soort']
            .groupby('soort')['jaar']
            .apply(lambda x: sorted(list(set(x)), reverse=True))
            .reset_index()
            .rename(columns={'jaar': 'Gemeten in jaren'})
        )

        # 2. Pak de data van het laatste jaar
        df_species_now = df_latest[df_latest['type'] == 'Soort'][['soort', 'bedekking_pct', 'groeivorm']]

        # 3. Combineer de actuele data met de historie-lijst
        # We gebruiken een 'left join' zodat we de details van nu behouden en de historie eraan plakken
        if not df_species_now.empty:
            df_combined = pd.merge(df_species_now, species_history, on='soort', how='left')

            # Optioneel: Maak van de lijst met jaren een mooie leesbare string
            df_combined['Gemeten in jaren'] = df_combined['Gemeten in jaren'].apply(lambda x: ", ".join(map(str, x)))

            st.dataframe(
                df_combined.sort_values('bedekking_pct', ascending=False)
                .style.background_gradient(subset=['bedekking_pct'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"Geen specifieke soorten geregistreerd in {latest_year}.")
            
            # Toon eventueel wel de soorten uit het verleden als er nu niets staat
            if not species_history.empty:
                st.write("Historisch aangetroffen soorten (niet in laatste jaar):")
                species_history['Gemeten in jaren'] = species_history['Gemeten in jaren'].apply(lambda x: ", ".join(map(str, x)))
                st.dataframe(species_history, use_container_width=True, hide_index=True)