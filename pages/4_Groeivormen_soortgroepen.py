import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import load_data, interpret_soil_state, add_species_group_columns

st.set_page_config(layout="wide", page_title="Groeivormen & Bodem")

st.title("🌱 Groeivormen en soortgroepen")
st.markdown("Analyse van vegetaties. Boven: functionele groeivormen. Onder: taxonomische soortgroepen.")

# --- 1. DATA INLADEN ---
# We laden de ruwe data. De functie load_data() zorgt al voor de basiskolommen.
df_raw = load_data()

if df_raw.empty:
    st.error("Geen data geladen. Controleer het bronbestand.")
    st.stop()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Jaar Filter
all_years = sorted(df_raw['jaar'].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Selecteer meetjaar", all_years)

# Project Filter
all_projects = sorted(df_raw['Project'].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)", 
    options=all_projects, 
    default=all_projects 
)

# Waterlichaam Filter
# We tonen alleen waterlichamen die voorkomen in de geselecteerde projecten
available_bodies = sorted(df_raw[df_raw['Project'].isin(selected_projects)]['Waterlichaam'].unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam / waterlichamen",
    options=available_bodies,
    default=available_bodies
)

# --- 3. FILTER TOEPASSEN ---
df_filtered = df_raw[
    (df_raw['Project'].isin(selected_projects)) & 
    (df_raw['Waterlichaam'].isin(selected_bodies))
].copy()

if df_filtered.empty:
    st.warning("Geen data gevonden voor de huidige selectie.")
    st.stop()

# Lijst van specifieke RWS-groeivormcodes
RWS_GROEIVORM_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]

# --- 4. LOGICA VOOR GROEIVORMEN (BOVENSTE GRAFIEKEN) ---
# Doel: Bepalen hoe we de groeivormen (Ondergedoken, Drijvend, etc.) berekenen.

# Stap A: Kijk of de dataset de specifieke RWS-codes bevat
df_rws_codes = df_filtered[df_filtered['soort'].isin(RWS_GROEIVORM_CODES)].copy()

use_aggregated_species = False
df_trend_growth = pd.DataFrame()

if not df_rws_codes.empty:
    # SCENARIO 1: RWS-codes zijn aanwezig. We gebruiken deze voor de trend.
    # We groeperen op jaar en groeivorm.
    # Meestal zijn deze waardes al locatietotalen, dus nemen we het gemiddelde over het gebied.
    df_trend_growth = df_rws_codes.groupby(['jaar', 'groeivorm'])['bedekking_pct'].mean().reset_index()
    source_label = "Bron: ruwe data Aquadesk"
    
    # Voor de radar plot (specifiek jaar)
    df_radar_source = df_rws_codes[df_rws_codes['jaar'] == selected_year]

else:
    # SCENARIO 2: Geen RWS-codes. We moeten individuele soorten optellen.
    # We filteren de soorten (alles wat GEEN RWS-code is en GEEN 'Groep' type)
    df_species_only = df_filtered[
        (~df_filtered['soort'].isin(RWS_GROEIVORM_CODES)) & 
        (df_filtered['type'] != 'Groep')
    ].copy()
    
    if df_species_only.empty:
        st.error("Geen data beschikbaar voor groeivorm-analyse (noch codes, noch soorten).")
        st.stop()
        
    use_aggregated_species = True
    source_label = "Berekend: Som van soorten"
    
    # We sommeren de bedekking van alle soorten per groeivorm per jaar
    df_trend_growth = df_species_only.groupby(['jaar', 'groeivorm'])['bedekking_pct'].sum().reset_index()
    
    # Voor de radar plot
    df_radar_source = df_species_only[df_species_only['jaar'] == selected_year]


# --- 5. VISUALISATIE: GROEIVORMEN & RADAR ---
c1, c2 = st.columns([2, 1])

# Mapping voor vaste kleuren en volgorde
GROWTH_ORDER = ['Ondergedoken', 'Drijvend', 'Emergent', 'Draadalgen', 'Kroos', 'FLAB']

with c1:
    st.subheader(f"Trend in groeivormen")
    st.caption(f"Methode: {source_label}")
    
    if not df_trend_growth.empty:
        fig_area = px.area(
            df_trend_growth, 
            x="jaar", 
            y="bedekking_pct", 
            color="groeivorm", 
            category_orders={"groeivorm": GROWTH_ORDER},
            color_discrete_map={
                'Ondergedoken': '#2ca02c', # Groen
                'Drijvend': '#1f77b4',     # Blauw
                'Emergent': '#ff7f0e',     # Oranje
                'Draadalgen': '#d62728',   # Rood
                'FLAB': '#7f7f7f',         # Grijs
                'Kroos': '#bcbd22'         # Geelgroen
            }
        )
        fig_area.update_layout(yaxis_title="Bedekking (%)", xaxis_title="Jaar", height=400)
        st.plotly_chart(fig_area, use_container_width=True)

with c2:
    st.subheader(f"Profiel {selected_year}")
    
    if not df_radar_source.empty:
        # Data voorbereiden voor radar: gemiddelde of som afhankelijk van bron
        if use_aggregated_species:
            # Bij soorten moeten we eerst sommeren per locatie, dan gemiddelde nemen? 
            # Voor radar over het hele gebied: Totale som per groeivorm / Aantal locaties is complex.
            # Eenvoudige benadering voor profiel: Verdeling van de totale bedekking.
            
            # Stap 1: Sommeer per groeivorm
            current_dist = df_radar_source.groupby('groeivorm')['bedekking_pct'].sum()
        else:
            # Bij RWS codes: Gemiddelde bedekking
            current_dist = df_radar_source.groupby('groeivorm')['bedekking_pct'].mean()
            
        # Normaliseren naar relatieve verdeling (0-1) voor de vorm
        total_val = current_dist.sum()
        if total_val > 0:
            current_dist = current_dist / total_val

        # Referentie (voorbeeld streefbeeld helder water)
        ref_dict = {'Ondergedoken': 0.6, 'Drijvend': 0.2, 'Emergent': 0.15, 'Draadalgen': 0.05}
        
        categories = GROWTH_ORDER
        r_vals = [current_dist.get(c, 0) for c in categories]
        ref_vals = [ref_dict.get(c, 0) for c in categories]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals, theta=categories, fill='toself', name=f'Data {selected_year}'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=ref_vals, theta=categories, fill='toself', name='Referentie', line=dict(dash='dot')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(r_vals), 0.6)])),
            showlegend=True, height=400, margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info(f"Geen data beschikbaar voor radarplot in {selected_year}")
        
    with st.expander("ℹ️ Hoe lees ik deze spingrafiek?"):
        st.markdown("""
        **Wat wordt er weergegeven?**
        Deze spingrafiek toont de *relatieve verdeling* van functionele groeivormen. 
        Hoe verder de punt naar de buitenrand van de cirkel staat, hoe groter het aandeel van die specifieke groeivorm binnen het totale plantenbestand van het geselecteerde jaar.

        **De referentielijn (streefbeeld):**
        De gestippelde lijn vertegenwoordigt een theoretisch streefbeeld voor een **ecologisch gezond, helder watersysteem**:
        * **Dominantie van ondergedoken planten (60%);** 
        * **Beperkt aandeel drijvend/emergent (15-20%);**
        * **Maximaal aandeel draadalgen (5%).**
      
        """)


st.divider()

# --- 6. LOGICA VOOR SOORTGROEPEN (ONDERSTE GRAFIEK) ---
# Doel: Taxonomische groepen tonen (Chariden, etc.). Hier MOETEN we de RWS-codes negeren.

st.subheader("🌿 Samenstelling soortgroepen (relatief)")

# A. Data voorbereiden via de utility functie
# Deze functie voegt 'soortgroep' toe. 
# BELANGRIJK: Zorg dat utils.py ook is bijgewerkt om RWS-codes te negeren in deze functie.
# Voor de zekerheid filteren we hier NOGMAALS de RWS codes eruit.
df_species_raw = df_filtered[~df_filtered['soort'].isin(RWS_GROEIVORM_CODES)].copy()
df_species_raw = df_species_raw[df_species_raw['type'] != 'Groep'] # Dubbele check

if df_species_raw.empty:
    st.info("Geen soort-specifieke data gevonden (alleen groepscodes aanwezig?).")
else:
    # Voeg soortgroepen toe (Chariden, etc.)
    df_species_mapped = add_species_group_columns(df_species_raw)
    
    # B. Aggregeren voor 100% Stacked Bar
    # Som van bedekking per jaar per soortgroep
    df_trend_species = df_species_mapped.groupby(['jaar', 'soortgroep'])['bedekkingsgraad_proc'].sum().reset_index()

    # Bereken totaal per jaar voor normalisatie
    df_totals = df_trend_species.groupby('jaar')['bedekkingsgraad_proc'].transform('sum')
    
    # Bereken percentage aandeel
    df_trend_species['percentage_relatief'] = 0.0
    mask = df_totals > 0
    df_trend_species.loc[mask, 'percentage_relatief'] = (
        df_trend_species.loc[mask, 'bedekkingsgraad_proc'] / df_totals[mask]
    ) * 100

    # C. Grafiek tekenen
    fig_stack = px.bar(
        df_trend_species,
        x='jaar',
        y='percentage_relatief',
        color='soortgroep',
        title='Relatieve samenstelling soortgroepen per jaar (excl. algemene groeivormen)',
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

    # D. Detail "Overig"
    with st.expander("🔍 Analyse 'overig / individueel' (soorten die nog niet zijn ingedeeld)"):
        df_overig = df_species_mapped[df_species_mapped['soortgroep'] == 'Overig / Individueel']
        if not df_overig.empty:
            missing_stats = df_overig.groupby('soort').agg(
                Aantal_Metingen=('bedekkingsgraad_proc', 'count'),
                Max_Bedekking=('bedekkingsgraad_proc', 'max')
            ).sort_values('Max_Bedekking', ascending=False).reset_index()
            st.dataframe(missing_stats, use_container_width=True)
        else:
            st.success("Alle aangetroffen soorten zijn succesvol ingedeeld in een groep!")
            
    # --- 6E. TOELICHTING SOORTGROEPEN (EXPLAINER) ---
with st.expander("ℹ️ Toelichting op de soortgroepen"):
    st.write("Hieronder vind je een beschrijving van de verschillende ecologische soortgroepen die in de grafiek worden getoond. Bron: waterplanten en waterkwaliteit, van Geest, G. et al.")
    
    # Maak kolommen voor een mooie layout of gebruik tabs
    tab1, tab2 = st.tabs(["Wortelend in sediment", "Overigen/mossen en vrijzwevende groeivormen"])

    with tab1:
        st.markdown("""
        **CHARIDEN (Kranswieren)** *Toelichting:* ondergedoken waterplanten met kransvormige vertakkingen, die in sommige wateren uitgestrekte onderwaterweiden kunnen vormen. Deze soorten behoren tot de macro-algen (en niet tot de hogere planten, waartoe veel andere waterplanten behoren). De planten bezitten dunne wortelachtige structuren (de zogeheten rhizoiden) waarmee ze oppervlakkig in de waterbodem groeien. Veel kranswieren concentreren hun biomassa dichtbij het sediment, waardoor ze gevoelig zijn voor troebel water. Voorbeelden: Gewoon kransblad, Sterkranswier, en Buigzaam glanswier.

        **ISOETIDEN (Biesvormigen)** *Toelichting:* waterplanten met een uitgebreid wortelstelsel, met bovengronds een korte stengel en rozet van stevige, lijn- of priemvormige bladeren. Deze groeivorm is kenmerkend voor wateren met een zeer lage beschikbaarheid van kooldioxide in de waterlaag. Karakteristieke soorten zijn onder meer Oeverkruid, Waterlobelia en Grote biesvaren.

        **PARVOPOTAMIDEN (Smalbladige fonteinkruiden)** *Toelichting:* ondergedoken, wortelende waterplanten met lange scheuten en lijnvormige of langwerpige bladeren. Ze hebben geen drijfbladeren. Sommige soorten zoals Schedefonteinkruid hebben een zogeheten horizontale groeiwijze, waardoor het merendeel van hun biomassa zich net onder het wateroppervlak bevindt. Hierdoor zijn sommige soorten binnen deze groep minder gevoelig voor troebel water. Voorbeelden: Plat fonteinkruid en Tenger fonteinkruid.

        **MAGNOPOTAMIDEN (Breedbladige fonteinkruiden)** *Toelichting:* wortelende, tamelijk grote waterplanten met langwerpige of lancetvormige ondergedoken bladeren, en een lange stengel. Deze soorten groeien vaak in dieper water, en zijn daarom gevoelig voor troebeling. Voorbeelden: Glanzig fonteinkruid en Doorgroeid fonteinkruid. 

        **MYRIOPHYLLIDEN (Vederkruiden)** *Toelichting:* wortelende waterplanten met lange stengels en fijn gedeelde, ondergedoken bladeren, maar zonder drijfbladeren. Deze fijn gedeelde bladeren hebben een hoog oppervlak tot inhoud ratio, wat de nutriëntenenopname stimuleert. De bloemen steken altijd boven het water uit. Voorbeelden: Kransvederkruid, Waterviolier en Stijve waterranonkel.
        
        **VALLISNERIIDEN (Rozetvormende waterplantjes)** *Toelichting:* zijn ondergedoken, wortelende waterplanten met een korte stengel en een rozet of bundel van lange, slappe, lijnvormige bladeren, al dan niet met uitlopers.
        
        **ELODEIDEN (Waterpest)** *Toelichting:* zijn ondergedoken, al dan niet wortelende waterplanten met lange, rechtopstaande scheuten met spiraalgewijs gerangschikte, lijn-, lancetvormige of langwerpige bladeren. Ze hebben geen drijfbladeren. Voorbeeld: Smalle waterpest.
        
        **STRATIOTIDEN (Stugge waterplanten)** *Toelichting:* zijn wortelende waterplanten met uitlopers, en met een rozet van stugge, spitse bladeren, waarvan de toppen doorgaans boven het water uitsteken. Ze zijn door middel van wortels losjes verankerd in organisch sediment, en ze zakken in het najaar naar de bodem. Voorbeeld: Krabbenscheer.
        
        **PEPLIDEN (Rozet met spatelvormige blaadjes)** *Toelichting:* zijn wortelende waterplanten met stengels en langwerpige, spatelvormige bladeren, waarvan de bovenste een drijvend rozet kunnen vormen en zijn aangepast aan de lucht. Planten van deze groeivormen blijven echter regelmatig ook permanent ondergedoken tijdens hun gehele levenscyclus. Voorbeelden: Waterpostelein en Stomphoekig sterrekroos.
        
        **BATRACHIIDEN (Amfibische waterplanten)** *Toelichting:* zijn wortelende waterplanten voorzien van stengels met gespecialiseerde drijfbladeren en fijn gedeelde ondergedoken waterbladeren. Een aantal van deze soorten ontwikkelt regelmatig landvormen. Voorbeeld: Middelste waterranonkel.      
       
        """)

    with tab2:
        st.markdown("""
        **HAPTOFYTEN (Vastzittende wieren/mossen)** *Toelichting:* zijn ondergedoken planten die aan hard substraat (zoals stenen of hout) vastgehecht zitten. In de gematige streken van het Noordelijk Halfrond betreft dit verschillende mossoorten, zoals Bronmos en diverse Kribmossen.

        **NYMPHAEIDEN (Drijfbladplanten)** *Toelichting:* zijn wortelende waterplanten met drijfbladeren en een lange steel. Soms hebben ze lijn-, lancet- of ruitvormige bladeren, of ronde ondergedoken bladeren. Deze groep is zeer heterogeen en moet nog verder worden onderverdeeld. Er wordt onderscheid gemaakt tussen magnonymphaeiden met grote drijfbladeren die in rozetten ontspringen uit grote wortelstokkenm en parvonymphaeidenm met kleinere drijfbladeren. Voorbeelden: Gele plomp (magno-) en Drijvend fonteinkruid (parvonymphaeide).

        """)        

# --- 7. BODEMDIAGNOSE ---
st.divider()
st.subheader("🕵️ Bodemdiagnose")

# We gebruiken de gefilterde set van het geselecteerde jaar
df_year_locs = df_filtered[df_filtered['jaar'] == selected_year]
available_locs = sorted(df_year_locs['locatie_id'].unique()) if not df_year_locs.empty else []

if available_locs:
    c_loc, c_txt = st.columns([1, 2])
    with c_loc:
        selected_loc = st.selectbox("Selecteer specifieke locatie voor diagnose", available_locs)
    
    with c_txt:
        if selected_loc:
            df_sample = df_year_locs[df_year_locs['locatie_id'] == selected_loc]
            
            # We moeten zorgen dat de interpretatie functie werkt met wat we hebben.
            # Als we geen 'groeivorm' kolom hebben (omdat RWS codes misten), moeten we hopen dat 'interpret_soil_state'
            # de fallback aankan of dat 'groeivorm' goed is ingevuld via load_data().
            # Door de fix in utils.py (load_data) zou 'groeivorm' altijd gevuld moeten zijn.
            
            interpretation = interpret_soil_state(df_sample)
            st.markdown(f"**Diagnose voor {selected_loc} ({selected_year}):**")
            st.markdown(interpretation)
else:
    st.write("Selecteer een jaar met beschikbare data voor de diagnose.")