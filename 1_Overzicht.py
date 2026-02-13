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
    # --- MATCH: vorige meetronde per waterlichaam (in plaats van altijd jaar-1) ---
    df_year = df_filtered[df_filtered['jaar'] == selected_year]

    # Waterlichamen die in het geselecteerde jaar voorkomen
    current_wbs = sorted(df_year['Waterlichaam'].dropna().unique())

    prev_parts = []
    prev_year_map = {}

    for wb in current_wbs:
        # alle jaren waarin dit waterlichaam voorkomt
        years_wb = sorted(df_filtered.loc[df_filtered['Waterlichaam'] == wb, 'jaar'].dropna().unique())
        # laatste jaar vóór selected_year
        prev_years = [y for y in years_wb if y < selected_year]
        if prev_years:
            prev_y = max(prev_years)
            prev_year_map[wb] = prev_y
            prev_parts.append(df_filtered[(df_filtered['Waterlichaam'] == wb) & (df_filtered['jaar'] == prev_y)])

    # Samengevoegde "vorige meting" dataset
    df_prev_matched = pd.concat(prev_parts, ignore_index=True) if prev_parts else pd.DataFrame(columns=df_filtered.columns)

    # Alleen waterlichamen meenemen waarvoor ook echt een vorige meting bestaat
    matched_wbs = sorted(prev_year_map.keys())
    df_year_matched = df_year[df_year['Waterlichaam'].isin(matched_wbs)].copy()
else:
    st.error("Geen data geladen. Controleer utils.py en het bronbestand.")
    st.stop()

# --- 1. GLOBALE KPI'S ---
# KPI Berekeningen
# Gebruikt 'totaal_bedekking_locatie' (de WATPTN waarde uit utils.py)
avg_bedekking, d_bedekking = calculate_kpi(df_year_matched, df_prev_matched, 'totaal_bedekking_locatie', is_loc_metric=True)
avg_doorzicht, d_doorzicht = calculate_kpi(df_year_matched, df_prev_matched, 'doorzicht_m', is_loc_metric=True)
avg_diepte, d_diepte = calculate_kpi(df_year_matched, df_prev_matched, 'diepte_m', is_loc_metric=True)

# Soortenrijkdom: liefst alleen individuele soorten tellen
RWS_GROEIVORM_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]

df_year_species = df_year_matched[(df_year_matched['type'] == 'Soort') & (~df_year_matched['soort'].isin(RWS_GROEIVORM_CODES))]
df_prev_species = df_prev_matched[(df_prev_matched['type'] == 'Soort') & (~df_prev_matched['soort'].isin(RWS_GROEIVORM_CODES))]

n_soorten = df_year_species['soort'].nunique()
prev_soorten = df_prev_species['soort'].nunique() if not df_prev_species.empty else n_soorten
d_soorten = n_soorten - prev_soorten

# --- EXTRA: Taartdiagrammen (KRW-score & Trofieniveau) ---
st.subheader("🥧 Samenstelling waarnemingen (individuele soorten)")

# Zelfde exclude-lijst als elders (RWS-verzamelcodes + WATPTN)
RWS_GROEIVORM_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]

# Filter naar individuele soorten (geen groepen/codes)
df_ind = df_year[(df_year["type"] == "Soort") & (~df_year["soort"].isin(RWS_GROEIVORM_CODES))].copy()

c_pie1, c_pie2 = st.columns(2)

# 1) KRW-score verdeling (alleen waar score beschikbaar is)
with c_pie1:
    st.markdown("**Verdeling waarnemingen per KRW-score**")

    if "krw_class" not in df_ind.columns and "krw_score" not in df_ind.columns:
        st.info("KRW-score is nog niet beschikbaar in de dataset (controleer verrijking in utils.py).")
    else:
        # Gebruik krw_class als die bestaat; anders classificeer op basis van krw_score
        if "krw_class" in df_ind.columns:
            s = df_ind["krw_class"]
        else:
            # fallback classificatie op basis van score
            s = pd.cut(
                df_ind["krw_score"],
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True
            )

        # Alleen waarden meenemen die beschikbaar zijn
        s = s.dropna()
        if s.empty:
            st.info("Geen KRW-scores beschikbaar voor de huidige selectie.")
        else:
            pie_df = s.value_counts().rename_axis("KRW-klasse").reset_index(name="Aantal waarnemingen")

            color_map = {
                "Gunstig (1-2)": "#2ca02c",   # groen
                "Neutraal (3-4)": "#ff7f0e",  # oranje
                "Ongewenst (5)": "#d62728"    # rood
            }

            fig_krw = px.pie(
                pie_df,
                names="KRW-klasse",
                values="Aantal waarnemingen",
                color="KRW-klasse",
                color_discrete_map=color_map,
                hole=0.35
            )
            fig_krw.update_traces(textposition="inside", textinfo="percent+label")
            fig_krw.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_krw, use_container_width=True)

# 2) Trofieniveau verdeling (alleen waar beschikbaar)
with c_pie2:
    st.markdown("**Verdeling waarnemingen per trofieniveau**")

    if "trofisch_niveau" not in df_ind.columns:
        st.info("Trofieniveau is nog niet beschikbaar in de dataset (controleer verrijking in utils.py).")
    else:
        t = df_ind["trofisch_niveau"].dropna()
        if t.empty:
            st.info("Geen trofieniveaus beschikbaar voor de huidige selectie.")
        else:
            pie_df = t.value_counts().rename_axis("Trofieniveau").reset_index(name="Aantal waarnemingen")

            fig_trofie = px.pie(
                pie_df,
                names="Trofieniveau",
                values="Aantal waarnemingen",
                hole=0.35
            )
            fig_trofie.update_traces(textposition="inside", textinfo="percent+label")
            fig_trofie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_trofie, use_container_width=True)

# KPI Weergave
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("gem. totale bedekking", f"{avg_bedekking:.1f}%", f"{d_bedekking:.1f}%")

with c2:
    st.metric("gem. doorzicht", f"{avg_doorzicht:.2f} m", f"{d_doorzicht:.2f} m")

with c3:
    st.metric("gem. diepte", f"{avg_diepte:.2f} m", f"{d_diepte:.2f} m")

with c4:
    st.metric("gem. soortenrijkdom", n_soorten, d_soorten)


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

    with st.expander("ℹ️ Hoe komt deze grafiek tot stand?"):
        st.markdown("""
    **Wat zie je?**  
    Per jaar: de bijdrage van soortgroepen aan de totale bedekking (WATPTN).

    **Stap 1 — Filtering:** groeivormcodes en `type='Groep'` worden uitgesloten (alleen echte soorten).  
    **Stap 2 — Indeling:** soorten krijgen een `soortgroep` via mapping en een numerieke bedekking (`bedekkingsgraad_proc`).  
    **Stap 3 — Teller:** per jaar en soortgroep wordt bedekking opgeteld.  
    **Stap 4 — Noemer:** totale bedekking (WATPTN) wordt per monstername (`CollectieReferentie`) 1× meegeteld en per jaar gesommeerd.  
    **Stap 5 — Fractie:** teller / noemer = fractie t.o.v. totale bedekking.  

    **Waarom geen 100%-stack?**  
    De staafhoogte mag <1 blijven: zo zie je ook welk deel van WATPTN niet door de getoonde soortgroepen wordt verklaard.
    """)
    
    # 1. Data voorbereiden: Filter RWS codes & 'Groep' types eruit
    df_species_raw = df_trend_base[~df_trend_base['soort'].isin(RWS_GROEIVORM_CODES)].copy()
    df_species_raw = df_species_raw[df_species_raw['type'] != 'Groep']

    if not df_species_raw.empty:
        # 2. Voeg soortgroepen toe (Chariden, etc.)
        df_species_mapped = add_species_group_columns(df_species_raw)
        
        # 3. Aggregeren: som van bedekking per jaar per soortgroep (teller)
        df_trend_species = (
            df_species_mapped
            .groupby(['jaar', 'soortgroep'])['bedekkingsgraad_proc']
            .sum()
            .reset_index()
        )

        # 4. Noemer: totale bedekking (WATPTN) per jaar, zonder dubbel tellen per monstername
        df_year_totals = (
            df_species_mapped
            .groupby(['jaar', 'CollectieReferentie'])['totaal_bedekking_locatie']
            .first()
            .reset_index()
        )

        df_year_totals = (
            df_year_totals
            .groupby('jaar')['totaal_bedekking_locatie']
            .sum()
            .reset_index()
            .rename(columns={'totaal_bedekking_locatie': 'totaal_bedekking_jaar'})
        )

        # 5. Merge noemer in df_trend_species en bereken fractie t.o.v. WATPTN
        df_trend_species = df_trend_species.merge(df_year_totals, on='jaar', how='left')

        df_trend_species['fractie_tov_totaal'] = 0.0
        mask = df_trend_species['totaal_bedekking_jaar'].notna() & (df_trend_species['totaal_bedekking_jaar'] > 0)
        df_trend_species.loc[mask, 'fractie_tov_totaal'] = (
            df_trend_species.loc[mask, 'bedekkingsgraad_proc'] / df_trend_species.loc[mask, 'totaal_bedekking_jaar']
        )

        # 6. Grafiek tekenen (NIET 100%-genormaliseerd; fractie t.o.v. totale bedekking)
        fig_stack = px.bar(
            df_trend_species,
            x='jaar',
            y='fractie_tov_totaal',
            color='soortgroep',
            title='Samenstelling soortgroepen t.o.v. totale bedekking (WATPTN) – geselecteerde waterlichamen',
            labels={
                'fractie_tov_totaal': 'Fractie van totale bedekking',
                'jaar': 'Jaar',
                'soortgroep': 'Groep'
            },
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=500
        )

        fig_stack.update_layout(yaxis=dict(range=[0, 1]))

        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Geen soort-specifieke data gevonden voor deze selectie.")