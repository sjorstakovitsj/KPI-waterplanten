import streamlit as st
from utils import load_data
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.services.overview_service import build_overview_result
from waterplanten_app.ui.charts import render_pie
from waterplanten_app.ui.filters import select_projects, select_year
from waterplanten_app.ui.metrics import render_overview_kpis
from waterplanten_app.ui.tables import render_overview_table

st.set_page_config(page_title='Waterplanten Monitor', layout='wide')
st.title('🌱 Waterplanten dashboard IJsselmeergebied')
st.markdown('Gemiddelden van geselecteerd meetjaar.')
NO_MATCH = 'Geen match'
KRW = {'Gunstig (1-2)': '#2ca02c', 'Neutraal (3-4)': '#ff7f0e', 'Ongewenst (5)': '#d62728', NO_MATCH: '#9e9e9e'}
TROFIE = {'oligotroof': '#2ca02c', 'mesotroof': '#1f77b4', 'eutroof': '#ff7f0e', 'sterk eutroof': '#d62728', 'brak': '#ffd700', 'marien': '#8c510a', 'kroos': '#7f7f7f', 'Onbekend': '#999999', NO_MATCH: '#9e9e9e'}
SOORTGROEP = {'chariden': '#1b9e77', 'iseotiden': '#7570b3', 'parvopotamiden': '#d95f02', 'magnopotamiden': '#66a61e', 'myriophylliden': '#e7298a', 'vallisneriiden': '#e6ab02', 'elodeiden': '#a6761d', 'stratiotiden': '#1f78b4', 'pepliden': '#b2df8a', 'batrachiiden': '#fb9a99', 'nymphaeiden': '#cab2d6', 'haptofyten': '#fdbf6f', 'Overig / Individueel': '#999999', NO_MATCH: '#9e9e9e'}

RELEASE_NOTES = [
        {
        "date": "29 september 2026",
        "title": "Bodemdiagnose en trends",
        "sections": {
            "Nieuwe bodemdiagnose": [
                (
                    "Een nieuwe bodemdiagnose toont op basis van aangetroffen "
                    "plantensoorten aanwijzingen voor kenmerken van de waterlaag "
                    "en waterbodem."
                ),
                (
                    "De resultaten zijn beschikbaar per waterlichaam, meetlocatie "
                    "en meetjaar."
                ),
            ],
            "Trends door de tijd": [
                (
                    "Ontwikkelingen worden weergegeven met boxplots per meetjaar. "
                    "Onderbrekingen zonder meetdata zijn gemarkeerd met gestippelde lijnen."
                ),
                (
                    "Een optionele formele analyse berekent een robuuste trend, "
                    "onzekerheidsinterval en gevoeligheid voor verschillende "
                    "berekeningsmethoden."
                ),
            ],
            "Duiding van resultaten": [
                (
                    "De resultaten bevatten uitleg over het tijdspatroon, verschillen "
                    "tussen locaties, bemonsteringskwaliteit en onzekerheid."
                ),
                (
                    "Numerieke indicaties zijn waar mogelijk vertaald naar begrijpelijke "
                    "ecologische klassen, met aanvullende uitleg per karakteristiek."
                ),
            ],
            "Snelheid en gebruiksgemak": [
                (
                    "Gegevens en berekende resultaten worden hergebruikt, waardoor "
                    "schakelen tussen waterlichaam, trends en meetlocaties sneller verloopt."
                ),
                (
                    "De formele trendanalyse wordt alleen uitgevoerd wanneer de gebruiker "
                    "daar expliciet voor kiest."
                ),
            ],
        },
    },
    {
        "date": "10 september 2026",
        "title": "Kaarten, trends en gebruiksgemak",
        "sections": {
            "Tijd- en trendanalyse": [
                "N2000-meetlocaties kunnen ook zonder waterlichaamkoppeling op de swipe map worden weergegeven.",
                "De projectnaam is zichtbaar in de popup van meetlocaties op de swipe map.",
                "Inzoomen, slepen, dubbelklikken en pinch-to-zoom werken op beide helften van de swipe map; ook zijn zoomknoppen toegevoegd.",
                "Nijkerkernauw is toegevoegd aan de selectielijst met waterlichamen.",
                "De kleuren voor bedekkingsgraad volgen dezelfde klassen als voor de KRW.",
            ],
            "Regressie en soortenaanwezigheid": [
                "Bij de regressiekaart en soortenaanwezigheid-heatmap is duidelijk aangegeven welke waterlichamen zijn geselecteerd.",
                "De punten op de regressiekaart zijn kleiner gemaakt en hebben een dunne zwarte visuele rand.",
            ],
            "Groeivormen en soortgroepen": [
                "De titels, assen, aswaarden en legenda's van alle grafieken onder Trend over de jaren zijn vergroot voor een betere leesbaarheid.",
            ],
            "Technische verbeteringen": [
                "Projectgegevens blijven behouden bij de verwerking van kaartpunten, zodat meetlocaties uit verschillende projecten niet ongemerkt worden samengevoegd.",
                "De verwerking van coördinaten, meetwaarden en kaartpopups is robuuster gemaakt.",
            ],
        },
    },
    {
        "date": "4 september 2026",
        "title": "Ecologische indices en trendgrafieken",
        "sections": {
            "Ecologische indices": [
                "De gekozen ecologische metric wordt correct weergegeven.",
                "Stofnamen en eenheden worden automatisch opgeschoond.",
                "Technische stofcodes worden niet meer getoond.",
                "De legenda is gecentreerd en de grafiek gebruikt meer paginabreedte.",
            ],
            "Trendgrafieken": [
                "Jaren zonder gegevens worden overgeslagen.",
                "Tijdshiaten worden gemarkeerd met een onderbroken lijn.",
                "Legenda's staan gecentreerd onder de grafieken.",
                "De grafieken zijn hoger en beter leesbaar.",
            ],
        },
    },
]

@st.dialog("Nieuw in deze versie", width="large")
def show_release_notes():
    max_index = len(RELEASE_NOTES) - 1
    current_index = min(
        max(int(st.session_state.get("release_notes_index", 0)), 0),
        max_index,
    )
    release = RELEASE_NOTES[current_index]

    st.caption(
        f"Update {current_index + 1} van {len(RELEASE_NOTES)} | "
        f"{release['date']}"
    )
    st.subheader(release["title"])

    for section, items in release["sections"].items():
        st.markdown(f"### {section}")
        st.markdown("\n".join(f"- {item}" for item in items))

    st.divider()
    previous_col, status_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button(
            "← Nieuwer",
            disabled=current_index == 0,
            use_container_width=True,
            key="release_notes_newer",
        ):
            st.session_state["release_notes_index"] = current_index - 1
            st.rerun()

    with status_col:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.45rem;'>"
            f"<strong>{release['date']}</strong></div>",
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button(
            "Ouder →",
            disabled=current_index == max_index,
            use_container_width=True,
            key="release_notes_older",
        ):
            st.session_state["release_notes_index"] = current_index + 1
            st.rerun()

    if st.button(
        "Begrepen",
        type="primary",
        use_container_width=True,
        key="release_notes_close",
    ):
        st.session_state["release_notes_seen_version"] = RELEASE_NOTES[0]["date"]
        st.session_state["release_notes_index"] = 0
        st.rerun()

LATEST_RELEASE = RELEASE_NOTES[0]["date"]
if st.session_state.get("release_notes_seen_version") != LATEST_RELEASE:
    st.session_state.setdefault("release_notes_index", 0)
    show_release_notes()

df = load_data()
st.sidebar.header('Algemene filters')
if df.empty:
    st.error('Geen data geladen. Controleer utils.py en het bronbestand.')
    st.stop()
filters = DashboardFilters(year=select_year(df, 'Selecteer meetjaar', default=2024), projects=select_projects(df))
result = build_overview_result(filters)

st.subheader('🥧 Samenstelling waarnemingen (individuele soorten)')
c1, c2, c3, c4 = st.columns(4)
with c1:
    d = result.krw_pie.rename(columns={'categorie': 'KRW-klasse', 'aantal': 'Aantal waarnemingen'})
    render_pie(d, 'KRW-klasse', 'Aantal waarnemingen', 'Verdeling waarnemingen per KRW-score', colors=KRW, caption=f"Aantal zonder KRW-match: {int(d.loc[d['KRW-klasse'] == NO_MATCH, 'Aantal waarnemingen'].sum())}" if not d.empty else None)
with c2:
    d = result.trophic_pie.rename(columns={'categorie': 'Trofieniveau', 'aantal': 'Aantal waarnemingen'})
    render_pie(d, 'Trofieniveau', 'Aantal waarnemingen', 'Verdeling waarnemingen per trofieniveau', colors=TROFIE, caption=f"Aantal zonder trofieniveau-match: {int(d.loc[d['Trofieniveau'] == NO_MATCH, 'Aantal waarnemingen'].sum())}" if not d.empty else None)
with c3:
    d = result.species_group_pie.rename(columns={'categorie': 'Soortgroep', 'aantal': 'Aantal waarnemingen'})
    render_pie(d, 'Soortgroep', 'Aantal waarnemingen', 'Verdeling waarnemingen per soortgroep', colors=SOORTGROEP, caption=f"Aantal zonder soortgroep-match: {int(d.loc[d['Soortgroep'] == NO_MATCH, 'Aantal waarnemingen'].sum())}" if not d.empty else None)
with c4:
    d = result.n2000_pie.rename(columns={'categorie': 'Kenmerkende soort (N2000)', 'label_kort': 'Label kort', 'aantal': 'Aantal waarnemingen'})
    render_pie(d, 'Kenmerkende soort (N2000)', 'Aantal waarnemingen', 'Verdeling waarnemingen van Kenmerkende soorten (N2000)', short_label_col='Label kort', caption=f"Aantal Kenmerkende soorten (N2000)-waarnemingen: {int(d['Aantal waarnemingen'].sum())}" if not d.empty else None)
render_overview_kpis(result.kpis); st.divider()
st.subheader(f'📊 Opsomming per waterlichaam ({filters.year})')
if result.overview_table.empty:
    st.info('Geen data beschikbaar voor de huidige filters.')
else:
    render_overview_table(result.overview_table)
