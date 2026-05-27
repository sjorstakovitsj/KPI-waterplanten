import streamlit as st
from streamlit.components.v1 import html
from utils import get_sorted_species_list, load_data
from waterplanten_app.services.time_trend_service import build_all_trophic_species_trend_figure, build_compare_figure, build_heatmap_figure, build_regression_map_figure, build_regression_outputs, build_swipe_map_view, compute_trend_table, filter_time_selection
from waterplanten_app.ui.filters import select_optional_species
from waterplanten_app.ui.legends import render_gradient_legend

st.set_page_config(layout='wide')
st.title('📈 Tijd- en trendanalyse')

df = load_data()
if df.empty:
    st.error('Geen data geladen.')
    st.stop()

st.sidebar.header('Selectiefilters')

WATERLICHAAM_KARAKTERISTIEKEN = {
    'IJsselmeer': {'voedselrijkdom': 'matig', 'diepte': 'diep', 'grootte': 'groot'},
    'Markermeer': {'voedselrijkdom': 'matig', 'diepte': 'diep', 'grootte': 'groot'},
    'IJmeer': {'voedselrijkdom': 'matig', 'diepte': 'diep', 'grootte': 'groot'},
    'Gouwzee': {'voedselrijkdom': 'matig', 'diepte': 'diep', 'grootte': 'groot'},
    'Gooimeer': {'voedselrijkdom': 'rijk', 'diepte': 'ondiep', 'grootte': 'middel'},
    'Eemmeer': {'voedselrijkdom': 'zeer rijk', 'diepte': 'ondiep', 'grootte': 'middel'},
    'Wolderwijd': {'voedselrijkdom': 'matig', 'diepte': 'ondiep', 'grootte': 'middel'},
    'Veluwemeer': {'voedselrijkdom': 'matig', 'diepte': 'ondiep', 'grootte': 'middel'},
    'Drontermeer': {'voedselrijkdom': 'matig', 'diepte': 'ondiep', 'grootte': 'klein'},
    'Vossemeer': {'voedselrijkdom': 'rijk', 'diepte': 'ondiep', 'grootte': 'klein'},
    'Ketelmeer': {'voedselrijkdom': 'zeer rijk', 'diepte': 'ondiep', 'grootte': 'middel'},
    'Zwartemeer': {'voedselrijkdom': 'rijk', 'diepte': 'ondiep', 'grootte': 'middel'},
}

voedselrijkdom_filter = st.sidebar.multiselect('Voedselrijkdom', ['matig', 'rijk', 'zeer rijk'], default=['matig', 'rijk', 'zeer rijk'])
diepte_filter = st.sidebar.multiselect('Diepte', ['diep', 'ondiep'], default=['diep', 'ondiep'])
grootte_filter = st.sidebar.multiselect('Grootte', ['klein', 'middel', 'groot'], default=['klein', 'middel', 'groot'])

all_waterbodies = sorted(df['Waterlichaam'].dropna().astype(str).unique()) if 'Waterlichaam' in df.columns else []
waterbody_options = [
    wb for wb in all_waterbodies
    if wb in WATERLICHAAM_KARAKTERISTIEKEN
    and WATERLICHAAM_KARAKTERISTIEKEN[wb]['voedselrijkdom'] in voedselrijkdom_filter
    and WATERLICHAAM_KARAKTERISTIEKEN[wb]['diepte'] in diepte_filter
    and WATERLICHAAM_KARAKTERISTIEKEN[wb]['grootte'] in grootte_filter
]

bodies = tuple(st.sidebar.multiselect('Selecteer waterlichaam / waterlichamen', waterbody_options, default=waterbody_options[:1]))
species = select_optional_species(df, get_sorted_species_list(df))

st.sidebar.subheader('Extra kaartfilter')
show_ondergedoken = st.sidebar.checkbox('Onder­gedoken waterplanten', value=False, disabled=bool(species))
show_chariden = st.sidebar.checkbox('Chariden (kranswieren)', value=False, disabled=bool(species))

metric_options = {'Bedekkingsgraad (%)': 'bedekking_pct'} if species else {'Bedekkingsgraad (%)': 'bedekking_pct', 'Doorzicht (m)': 'doorzicht_m', 'Diepte (m)': 'diepte_m', 'Soortenrijkdom': 'soort_count'}
metric_label = st.sidebar.selectbox('Kies analysevariabele', list(metric_options.keys()))
metric = metric_options[metric_label]

filtered = filter_time_selection(df, bodies, species, show_ondergedoken, show_chariden)
if filtered.empty:
    st.warning('Geen data beschikbaar voor deze selectie.')
    st.stop()

trend = compute_trend_table(filtered, metric)
years = sorted(filtered['jaar'].dropna().astype(int).unique().tolist()) if 'jaar' in filtered.columns else []

st.subheader('Kaart – resultaten voor een gekozen jaar')
if len(years) >= 2:
    c1, c2 = st.columns(2)
    y1 = c1.selectbox('Jaar links', years, index=max(0, len(years) - 2))
    y2 = c2.selectbox('Jaar rechts', years, index=len(years) - 1)
    if y1 == y2:
        st.warning('Kies twee verschillende jaren voor de swipe-vergelijking.')
    else:
        swipe_html, legend = build_swipe_map_view(filtered, metric, metric_label, int(y1), int(y2))
        if swipe_html:
            html(swipe_html, height=670)
            render_gradient_legend(legend['label'], legend['min'], legend['max'])

    with st.expander('Vergelijking versus een historisch meetjaar', expanded=False):
        if y1 == y2:
            st.info('Kies twee verschillende jaren boven de kaart om de vergelijking te tonen.')
        else:
            fig = build_compare_figure(trend, metric_label, int(y1), int(y2))
            if fig:
                st.plotly_chart(fig, width='stretch')
            else:
                st.info('Geen vergelijkingsdata beschikbaar voor de geselecteerde jaren.')
else:
    st.info('Er zijn niet genoeg jaartallen beschikbaar om op de kaart te tonen.')

st.subheader('📊 Trend individuele soorten per trofieniveau')
st.caption('Per geselecteerd waterlichaam worden zes trofieniveau-panelen naast elkaar getoond. Elke y-as is individueel geschaald en toont percentages. De trend is genormaliseerd rond 0% en wordt berekend met alle beschikbare jaren binnen de huidige sidebarselectie. Groen = positieve ontwikkeling, rood = negatieve ontwikkeling.')
with st.expander('Hoe worden deze trends berekend?', expanded=False):
    st.markdown('''
    **Kort samengevat:** per soort wordt een lineaire trend berekend over alle beschikbare jaren binnen de huidige selectie.

    **Rekenregels:**
    - De analyse gebruikt alleen soortrecords met bedekking (`type == "Soort"` en, indien aanwezig, `Grootheid == "BEDKG"`).
    - De soorten worden ingedeeld naar trofieniveau, bijvoorbeeld oligotroof, mesotroof, eutroof, sterk eutroof, brak en marien.
    - Per waterlichaam, trofieniveau, soort en jaar wordt de gemiddelde bedekking berekend.
    - Ontbreekt een soort in een jaar, dan telt dat jaar mee als `0%` bedekking.
    - Vervolgens wordt over alle jaren een lineaire helling (`slope`) berekend.
    - Die helling wordt omgerekend naar een relatieve trend:

      `Trend (%) = slope × (laatste jaar - eerste jaar) / gemiddelde bedekking × 100`

    **Interpretatie:**
    - `> 0%` = positieve ontwikkeling, groen.
    - `< 0%` = negatieve ontwikkeling, rood.
    - `= 0%` = stabiel, grijs.

    Let op: dit is dus geen simpele vergelijking tussen alleen het eerste en laatste jaar. Alle beschikbare jaren in de selectie tellen mee.
    ''')

if species:
    st.info('Let op: de sidebar-soortfilter is actief. Deze analyse gebruikt daardoor alleen de geselecteerde soort.')

if len(years) < 2:
    st.info('Er zijn minimaal twee jaren nodig om soorttrends per trofieniveau te berekenen.')
else:
    selected_waterbodies = sorted(filtered['Waterlichaam'].dropna().astype(str).unique().tolist()) if 'Waterlichaam' in filtered.columns else ['Onbekend']
    for wb in selected_waterbodies:
        wb_df = filtered[filtered['Waterlichaam'].astype(str).eq(str(wb))].copy() if 'Waterlichaam' in filtered.columns else filtered.copy()
        fig_trophic = build_all_trophic_species_trend_figure(wb_df, str(wb))
        if fig_trophic:
            st.plotly_chart(fig_trophic, width='content')
        else:
            st.info(f'Geen geschikte trofieniveau-trenddata gevonden voor {wb}.')

st.subheader('Regressieanalyse: verbetert of verslechtert de toestand?')
pie, slopes, _ = build_regression_outputs(trend, metric)
if pie is None:
    st.warning('Geen locaties gevonden met minimaal 5 jaar aan meetgegevens in de huidige selectie.')
else:
    regression_map, regression_map_msg = build_regression_map_figure(filtered, slopes)
    if regression_map:
        st.plotly_chart(regression_map, width='stretch')
    else:
        st.info(regression_map_msg)

    with st.expander('Toon taartdiagram en regressietabel', expanded=False):
        l, r = st.columns([1, 2])
        l.plotly_chart(pie, width='stretch')
        r.dataframe(slopes.style.background_gradient(subset=['slope'], cmap='RdYlGn').format({'slope': '{:.4f}', 'n_jaren': '{:.0f}'}), width='stretch', hide_index=True)

st.divider()
st.subheader('Soortenaanwezigheid heatmap (top 50)')
fig, msg = build_heatmap_figure(df, bodies)
if fig:
    st.plotly_chart(fig, width='stretch')
else:
    st.info(msg)
