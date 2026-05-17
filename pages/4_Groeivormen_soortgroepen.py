import streamlit as st
from utils import load_data
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.services.growth_forms_service import build_species_group_view, build_trend_figure, get_soil_diagnosis, get_soil_locations
from waterplanten_app.ui.filters import select_projects, select_waterbodies, select_year

st.set_page_config(layout='wide', page_title='Groeivormen & Bodem')
st.title('🌱 Groeivormen en soortgroepen')
st.markdown('Analyse van vegetaties. Boven: functionele groeivormen. Onder: taxonomische soortgroepen.')

df = load_data()
if df.empty:
    st.error('Geen data geladen. Controleer het bronbestand.')
    st.stop()
st.sidebar.header('Filters')
projects = select_projects(df)
filters = DashboardFilters(year=select_year(df, 'Selecteer meetjaar'), projects=projects, waterbodies=select_waterbodies(df, projects))
st.subheader('Trend over de jaren')
trend_mode = st.selectbox('Kies trendweergave', ['Groeivormen', 'Totale bedekking', 'KRW score', 'Trofieniveau', 'Soortgroepen', 'Kenmerkende soorten (N2000)'], index=0, key='trend_mode_choice')
fig, msg, caption = build_trend_figure(filters, trend_mode)
if msg:
    st.info(msg)
else:
    st.plotly_chart(fig, width='stretch')
if caption:
    st.caption(caption)
st.divider(); st.subheader('🌿 Samenstelling soortgroepen (relatief)')
fig, missing, msg = build_species_group_view(filters)
if msg:
    st.info(msg)
else:
    st.plotly_chart(fig, width='stretch')
with st.expander("🔍 Analyse 'overig / individueel' (soorten die nog niet zijn ingedeeld)"):
    st.dataframe(missing, width='stretch') if missing is not None and not missing.empty else st.success('Alle aangetroffen soorten zijn succesvol ingedeeld in een groep!')
st.divider(); st.subheader('🕵️ Bodemdiagnose')
locs = get_soil_locations(filters)
if not locs:
    st.write('Selecteer een jaar met beschikbare data voor de diagnose.')
else:
    c1, c2 = st.columns([1, 2]); loc = c1.selectbox('Selecteer specifieke locatie voor diagnose', locs)
    c2.markdown(f'**Diagnose voor {loc} ({filters.year}):**'); c2.markdown(get_soil_diagnosis(filters, loc))
