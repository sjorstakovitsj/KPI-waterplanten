import streamlit as st
from streamlit.components.v1 import html
from utils import get_sorted_species_list, load_data
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.services.time_trend_service import build_compare_figure, build_cover_figure, build_heatmap_figure, build_meetpunt_trend_figure, build_regression_outputs, build_swipe_map_view, compute_trend_table, filter_time_selection
from waterplanten_app.ui.filters import select_optional_species
from waterplanten_app.ui.legends import render_gradient_legend

st.set_page_config(layout='wide')
st.title('📈 Tijd- en trendanalyse')

df = load_data()
if df.empty:
    st.error('Geen data geladen.')
    st.stop()
st.sidebar.header('Selectiefilters')
bodies = tuple(st.sidebar.multiselect('Selecteer waterlichaam / waterlichamen', sorted(df['Waterlichaam'].dropna().unique()) if 'Waterlichaam' in df.columns else [], default=(sorted(df['Waterlichaam'].dropna().unique())[:1] if 'Waterlichaam' in df.columns else [])))
species = select_optional_species(df, get_sorted_species_list(df))
st.sidebar.subheader('Extra kaartfilter')
show_ondergedoken = st.sidebar.checkbox('Onder­gedoken waterplanten', value=False, disabled=bool(species))
show_chariden = st.sidebar.checkbox('Chariden (kranswieren)', value=False, disabled=bool(species))
metric_options = {'Bedekkingsgraad (%)': 'bedekking_pct'} if species else {'Bedekkingsgraad (%)': 'bedekking_pct', 'Doorzicht (m)': 'doorzicht_m', 'Diepte (m)': 'diepte_m', 'Soortenrijkdom': 'soort_count'}
metric_label = st.sidebar.selectbox('Kies analysevariabele', list(metric_options.keys())); metric = metric_options[metric_label]
filtered = filter_time_selection(df, bodies, species, show_ondergedoken, show_chariden)
if filtered.empty:
    st.warning('Geen data beschikbaar voor deze selectie.')
    st.stop()
trend = compute_trend_table(filtered, metric); years = sorted(filtered['jaar'].dropna().astype(int).unique().tolist()) if 'jaar' in filtered.columns else []
st.subheader('Kaart – resultaten voor een gekozen jaar')
if len(years) >= 2:
    c1, c2 = st.columns(2); y1 = c1.selectbox('Jaar links', years, index=max(0, len(years) - 2)); y2 = c2.selectbox('Jaar rechts', years, index=len(years) - 1)
    if y1 == y2: st.warning('Kies twee verschillende jaren voor de swipe-vergelijking.')
    else:
        swipe_html, legend = build_swipe_map_view(filtered, metric, metric_label, int(y1), int(y2))
        if swipe_html: html(swipe_html, height=670); render_gradient_legend(legend['label'], legend['min'], legend['max'])
else:
    st.info('Er zijn niet genoeg jaartallen beschikbaar om op de kaart te tonen.')
st.subheader('📈 Basale trendanalyse – Totale bedekking')
cover = build_cover_figure(filtered, species)
if cover:
    st.plotly_chart(cover, width='stretch')
else:
    st.info('Geen (soort-)bedekkingsdata (BEDKG) gevonden binnen de huidige sidebar-filters.')
st.subheader(f'Verloop {metric_label} door de jaren heen')
if trend.empty: st.info('Geen trenddata beschikbaar voor de huidige selectie.')
else: st.plotly_chart(build_meetpunt_trend_figure(trend, metric_label, 'Trendontwikkeling per meetpunt in geselecteerde wateren'), width='stretch')
st.subheader('Regressieanalyse: verbetert of verslechtert de toestand?')
pie, slopes, _ = build_regression_outputs(trend, metric)
if pie is None: st.warning('Geen locaties gevonden met minimaal 5 jaar aan meetgegevens in de huidige selectie.')
else:
    l, r = st.columns([1, 2]); l.plotly_chart(pie, width='stretch'); r.dataframe(slopes.style.background_gradient(subset=['slope'], cmap='RdYlGn').format({'slope': '{:.4f}', 'n_jaren': '{:.0f}'}), width='stretch', hide_index=True)
st.divider(); st.subheader('Vergelijking versus een historisch meetjaar')
if len(years) >= 2:
    c1, c2 = st.columns(2); start = c1.selectbox('Referentiejaar', years, index=max(0, len(years) - 2), key='bar_ref_year'); end = c2.selectbox('Vergelijkingsjaar', years, index=len(years) - 1, key='bar_cmp_year')
    fig = build_compare_figure(trend, metric_label, int(start), int(end))
    if fig:
        st.plotly_chart(fig, width='stretch')
else: st.info('Niet genoeg jaren aan data voor een vergelijking.')
st.divider(); st.subheader('Soortenaanwezigheid heatmap (top 50)')
fig, msg = build_heatmap_figure(df, bodies)
if fig:
    st.plotly_chart(fig, width='stretch')
else:
    st.info(msg)
