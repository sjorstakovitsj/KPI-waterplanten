import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.services.meetpoint_detail_service import build_kpis, build_location_overview, build_location_trend_figure, build_overview_map, filter_locations_by_years, get_bathymetry_legend_url, get_chemistry_overlay, get_detail_base, get_latest_location_summary, get_soil_diagnosis
from waterplanten_app.ui.filters import select_projects, select_waterbodies, select_year_range, select_meetjaren_range
from waterplanten_app.ui.legends import render_bathymetry_legend
from waterplanten_app.ui.metrics import render_simple_metrics
from waterplanten_app.ui.tables import render_species_history

st.set_page_config(layout='wide', page_title='Meetpunt Detail')
st.title('📍 Meetpunt detailniveau analyse')
st.markdown('Gedetailleerde analyse van een specifiek monitoringspunt.')

base_df = get_detail_base(DashboardFilters())
if base_df.empty:
    st.error('Geen data geladen.')
    st.stop()
st.sidebar.header('Filters')
projects = select_projects(base_df); bodies = select_waterbodies(base_df, projects)
scoped = get_detail_base(DashboardFilters(projects=projects, waterbodies=bodies))
if scoped.empty:
    st.warning('Geen meetpunten gevonden voor de geselecteerde filters.')
    st.stop()
year_range = select_year_range(scoped, 'Selecteer jaarbereik', default_last_n=10)
scoped, counts = filter_locations_by_years(scoped, year_range, (1, max(1, len(pd.to_numeric(scoped['jaar'], errors='coerce').dropna().unique().tolist()))))
if counts.empty:
    st.warning('Geen meetjaren beschikbaar voor de geselecteerde filters.')
    st.stop()
scoped, counts = filter_locations_by_years(scoped, year_range, select_meetjaren_range(counts))
if scoped.empty:
    st.warning('Geen meetpunten gevonden voor het gekozen bereik in meetjaren en jaartallen.')
    st.stop()
locs = sorted(scoped['locatie_id'].dropna().astype(str).unique().tolist()); key = 'meetpunt_detail_selected_loc'
if key not in st.session_state or st.session_state[key] not in locs: st.session_state[key] = locs[0]
st.subheader('🗺️ Overzicht meetpunten'); st.caption('Paarse ruitjes op de kaart geven de locaties van chemische metingen weer.')
overview = build_location_overview(scoped)
if not overview.empty:
    event = st_folium(build_overview_map(overview, st.session_state[key], get_chemistry_overlay()), height=520, width=None, returned_objects=['last_object_clicked', 'last_object_clicked_popup'], key='meetpunt_detail_map')
    clicked = event.get('last_object_clicked_popup') if isinstance(event, dict) else None
    if clicked and str(clicked) in locs and str(clicked) != str(st.session_state[key]): st.session_state[key] = str(clicked); st.session_state['meetpunt_detail_selectbox'] = str(clicked); st.rerun()
render_bathymetry_legend(get_bathymetry_legend_url())
if 'meetpunt_detail_selectbox' not in st.session_state or st.session_state['meetpunt_detail_selectbox'] not in locs: st.session_state['meetpunt_detail_selectbox'] = st.session_state[key]
selected = st.selectbox('Of kies handmatig een meetpunt', locs, index=locs.index(st.session_state['meetpunt_detail_selectbox']), key='meetpunt_detail_selectbox'); st.session_state[key] = selected
df_loc = scoped[scoped['locatie_id'].astype(str) == str(selected)].copy()
if df_loc.empty: st.warning('Geen data voor deze meetlocatie binnen de huidige filters.'); st.stop()
kpis = build_kpis(df_loc)
render_simple_metrics([('Aantal metingen (jaren)', str(kpis['n_jaren'])), ('Gemiddelde diepte', f"{kpis['mean_diepte']:.2f} m" if kpis['mean_diepte'] is not None else 'n.v.t.'), ('Gemiddeld doorzicht', f"{kpis['mean_doorzicht']:.2f} m" if kpis['mean_doorzicht'] is not None else 'n.v.t.')])
t1, t2 = st.tabs(['📈 Tijdreeksen', '📝 Diagnose en aangetroffen soorten'])
with t1:
    fig = build_location_trend_figure(df_loc)
    if fig:
        st.plotly_chart(fig, width='stretch')
    else:
        st.info('Geen trendgegevens beschikbaar.')
with t2:
    now_species, history, latest_year = get_latest_location_summary(df_loc)
    a, b = st.columns([1, 1])
    with a: st.markdown(f"### Bodemdiagnose ({latest_year})" if latest_year else '### Bodemdiagnose'); st.write(get_soil_diagnosis(df_loc))
    with b: st.markdown('### Soortenlijst en historie'); render_species_history(now_species, history, latest_year)
