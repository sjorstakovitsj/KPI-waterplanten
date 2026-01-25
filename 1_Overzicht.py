# 1_Overzicht.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data, plot_trend_line, add_species_group_columns, calculate_kpi

st.set_page_config(page_title="Waterplanten Monitor", layout="wide")

st.title("🌱 Waterplanten dashboard IJsselmeergebied")
st.markdown("Gemiddelden van geselecteerd meetjaar.")

# --- DATA INLADEN ---
df = load_data()

# --- SIDEBAR: GLOBALE FILTERS ---
st.sidebar.header("Algemene filters")

if not df.empty:
    # 1. Peiljaar (voor de KPI's en Tabel)
    all_years = sorted(df['jaar'].dropna().unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Selecteer meetjaar", all_years)

    # 2. Project Filter (KRW / N2000)
    all_projects = sorted(df['Project'].dropna().unique())
    selected_projects = st.sidebar.multiselect(
        "Selecteer project(en)", 
        options=all_projects, 
        default=all_projects 
    )
    
    # Pas projectfilter toe op de hele dataset
    df_filtered = df[df['Project'].isin(selected_projects)]
    
    # Dataframes voor KPI's (Huidig vs Vorig jaar)
    df_year = df_filtered[df_filtered['jaar'] == selected_year]
    df_prev = df_filtered[df_filtered['jaar'] == selected_year - 1]
else:
    st.error("Geen data geladen. Controleer utils.py en het bronbestand.")
    st.stop()

# --- 1. GLOBALE KPI'S ---
# KPI Berekeningen
# Gebruikt 'totaal_bedekking_locatie' (de WATPTN waarde uit utils.py)
avg_bedekking, d_bedekking = calculate_kpi(df_year, df_prev, 'totaal_bedekking_locatie', is_loc_metric=True)
avg_doorzicht, d_doorzicht = calculate_kpi(df_year, df_prev, 'doorzicht_m', is_loc_metric=True)
n_soorten = df_year['soort'].nunique()
d_soorten = n_soorten - (df_prev['soort'].nunique() if not df_prev.empty else n_soorten)

# KPI Weergave
c1, c2, c3 = st.columns(3)
with c1: st.metric("gem. totale bedekking", f"{avg_bedekking:.1f}%", f"{d_bedekking:.1f}%")
with c2: st.metric("gem. doorzicht", f"{avg_doorzicht:.2f}m", f"{d_doorzicht:.2f}m")
with c3: st.metric("gem. soortenrijkdom", n_soorten, d_soorten)


st.divider()

# --- 2. DETAILOVERZICHT PER WATERLICHAAM ---
st.subheader(f"📊 Opsomming per waterlichaam ({selected_year})")

if not df_year.empty:
    # Aggregatie stap 1: Per monstername (uniek maken van locatie-variabelen)
    df_samples = df_year.groupby(['Waterlichaam', 'CollectieReferentie']).agg({
        'totaal_bedekking_locatie': 'first', # Dit is de WATPTN waarde
        'diepte_m': 'first',
        'doorzicht_m': 'first'
    }).reset_index()
    
    # Aggregatie stap 2: Per Waterlichaam (gemiddelde van de samples)
    df_water_stats = df_samples.groupby('Waterlichaam').agg({
        'totaal_bedekking_locatie': 'mean',
        'diepte_m': 'mean',
        'doorzicht_m': 'mean'
    }).reset_index()
      
    # 2. Bereken de Soortenrijkdom (unieke soorten per waterlichaam)
    # Filter eerst op type 'Soort' om te voorkomen dat groepen als 'FLAB' worden meegeteld
    df_species_only = df_year[df_year['type'] == 'Soort']
    df_richness = df_species_only.groupby('Waterlichaam')['soort'].nunique().reset_index()
    df_richness.columns = ['Waterlichaam', 'Soortenrijkdom']
        
    # 3. Voeg de twee tabellen samen (Merge)
    overview_df = pd.merge(df_water_stats, df_richness, on='Waterlichaam', how='left')

    overview_df = overview_df.rename(columns={
        'totaal_bedekking_locatie': 'Bedekking (Totaal %)',
        'diepte_m': 'Gem. Diepte (m)',
        'doorzicht_m': 'Gem. Doorzicht (m)',
    })

    # Zorg dat lege waarden (NaN) in rijkdom op 0 staan
    overview_df['Soortenrijkdom'] = overview_df['Soortenrijkdom'].fillna(0).astype(int)
    
    # Tabel weergave
    st.dataframe(
        overview_df[['Waterlichaam', 'Bedekking (Totaal %)', 'Gem. Diepte (m)', 'Gem. Doorzicht (m)', 'Soortenrijkdom']],
        use_container_width=True, hide_index=True,
        column_config={
            "Bedekking (Totaal %)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Gem. Diepte (m)": st.column_config.NumberColumn(format="%.2f m"),
            "Gem. Doorzicht (m)": st.column_config.NumberColumn(format="%.2f m"),
            "Soortenrijkdom": st.column_config.NumberColumn(format="%d soorten")
        }
    )
else:
    st.info("Geen data beschikbaar voor de huidige filters.")

st.divider()

# --- 3. TREND ANALYSE (Met specifieke selectie) ---
st.subheader("📈 Basale trendanalyse")

# Lijst van specifieke RWS-groeivormcodes (nodig voor filtering 3e grafiek)
RWS_GROEIVORM_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]

# A. Trend Selectie Filter (Specifiek voor de grafieken)
available_bodies = sorted(df_filtered['Waterlichaam'].unique())
selected_trend_bodies = st.multiselect(
    "Selecteer waterlichaam / waterlichamen voor trendlijn:",
    options=available_bodies,
    default=available_bodies[:3] if len(available_bodies) > 0 else available_bodies
)

if selected_trend_bodies:
    # Filter de data voor de plots
    df_trend_base = df_filtered[df_filtered['Waterlichaam'].isin(selected_trend_bodies)]

    c_trend1, c_trend2 = st.columns(2)

    # --- GRAFIEK 1: TOTALE BEDEKKING (WATPTN) ---
    with c_trend1:
        st.markdown("**Totale bedekking**")
        # Stap 1: Dedup per sample (anders tellen we WATPTN dubbel per soort)
        df_trend_samples = df_trend_base.groupby(['jaar', 'Waterlichaam', 'CollectieReferentie'])['totaal_bedekking_locatie'].first().reset_index()
        
        # Stap 2: Mean per Jaar per Waterlichaam
        df_trend_cover = df_trend_samples.groupby(['jaar', 'Waterlichaam'])['totaal_bedekking_locatie'].mean().reset_index()
        
        fig_cover = px.line(
            df_trend_cover, 
            x='jaar', 
            y='totaal_bedekking_locatie', 
            color='Waterlichaam',
            markers=True,
            title="Trend totale Bedekking (%) per waterlichaam"
        )
        fig_cover.update_layout(height=350, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_cover, use_container_width=True)

    # --- GRAFIEK 2: BEDEKKINGSVORMEN (GROWTH FORMS) ---
    with c_trend2:
        st.markdown("**Samenstelling groeivormen**")
        
        # STAP A: Filter eerst ALLE individuele soorten eruit.
        # We willen alleen de hoofdgroepen (Submers, Drijvend, etc.) zien.
        df_forms_only = df_trend_base[df_trend_base['type'] == 'Groep'].copy()
        
        if not df_forms_only.empty:
            # Stap 1: Sommeer bedekking per groeivorm per sample
            df_form_sample = df_forms_only.groupby(['jaar', 'CollectieReferentie', 'groeivorm'])['bedekking_pct'].sum().reset_index()
            
            # Stap 2: Gemiddelde per jaar
            df_form_trend = df_form_sample.groupby(['jaar', 'groeivorm'])['bedekking_pct'].mean().reset_index()
            
            # Plot
            fig_forms = px.area(
                df_form_trend, 
                x='jaar', 
                y='bedekking_pct', 
                color='groeivorm',
                markers=True,
                title="Trend groeivormen (gemiddelden)"
            )
            fig_forms.update_layout(height=350, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_forms, use_container_width=True)
        else:
            st.info("Geen groeivorm-groepen (zoals 'Ondergedoken', 'Drijvend') gevonden in de selectie.")

    # --- GRAFIEK 3: RELATIEVE SAMENSTELLING SOORTGROEPEN (NIEUW) ---
    st.divider()
    st.markdown("**Relatieve samenstelling soortgroepen**")
    
    # 1. Data voorbereiden: Filter RWS codes & 'Groep' types eruit
    df_species_raw = df_trend_base[~df_trend_base['soort'].isin(RWS_GROEIVORM_CODES)].copy()
    df_species_raw = df_species_raw[df_species_raw['type'] != 'Groep']

    if not df_species_raw.empty:
        # 2. Voeg soortgroepen toe (Chariden, etc.)
        df_species_mapped = add_species_group_columns(df_species_raw)
        
        # 3. Aggregeren voor 100% Stacked Bar
        # Som van bedekking per jaar per soortgroep binnen de geselecteerde waterlichamen
        df_trend_species = df_species_mapped.groupby(['jaar', 'soortgroep'])['bedekkingsgraad_proc'].sum().reset_index()

        # 4. Bereken totaal per jaar voor normalisatie
        df_totals = df_trend_species.groupby('jaar')['bedekkingsgraad_proc'].transform('sum')
        
        # 5. Bereken percentage aandeel
        df_trend_species['percentage_relatief'] = 0.0
        mask = df_totals > 0
        df_trend_species.loc[mask, 'percentage_relatief'] = (
            df_trend_species.loc[mask, 'bedekkingsgraad_proc'] / df_totals[mask]
        ) * 100

        # 6. Grafiek tekenen
        fig_stack = px.bar(
            df_trend_species,
            x='jaar',
            y='percentage_relatief',
            color='soortgroep',
            title='Relatieve samenstelling soortgroepen van geselecteerde waterlichamen',
            labels={
                'percentage_relatief': 'Aandeel (%)', 
                'jaar': 'Jaar',
                'soortgroep': 'Groep'
            },
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=500
        )
        
        fig_stack.update_layout(yaxis=dict(range=[0, 100], ticksuffix="%"))
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Geen soort-specifieke data gevonden voor deze selectie.")