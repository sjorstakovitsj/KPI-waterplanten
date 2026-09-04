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

@st.dialog("Nieuw in deze versie", width="large")
def show_release_notes():
    st.markdown(
        """
        ### Ecologische indices

        - KRW en N2000 zijn standaard geselecteerd.
        - Doorzicht en chlorofyl-a zijn standaard geselecteerd.
        - De gekozen ecologische metric wordt correct weergegeven.
        - Stofnamen en eenheden worden automatisch opgeschoond.
        - Technische stofcodes worden niet meer getoond.
        - De legenda is gecentreerd en de grafiek gebruikt meer paginabreedte.

        ### Trendgrafieken

        - Jaren zonder gegevens worden overgeslagen.
        - Tijdshiaten worden gemarkeerd met een onderbroken lijn.
        - Legenda's staan gecentreerd onder de grafieken.
        - De grafieken zijn hoger en beter leesbaar.
        """
    )

    if st.button("Begrepen", type="primary", use_container_width=True):
        st.session_state["release_notes_seen"] = True
        st.rerun()


if not st.session_state.get("release_notes_seen", False):
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
