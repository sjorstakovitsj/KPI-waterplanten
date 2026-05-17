import streamlit as st
from utils import CHEM_PARAM_SUGGESTIONS, SEASON_ORDER, get_available_chemistry_locations, get_available_chemistry_parameter_labels, get_preferred_chemistry_locations, load_chemistry_data, load_data
from waterplanten_app.services.ecological_indices_service import build_bubble_figure, build_dual_axis_view, build_heatmap_figure, get_shared_years
from waterplanten_app.ui.charts import render_plot
from waterplanten_app.ui.filters import select_projects, select_waterbodies

st.set_page_config(layout='wide')
st.title('🌿 Ecologische indices')

df = load_data()
if df.empty:
    st.error('Geen data geladen.')
    st.stop()
st.sidebar.header('Selectie filters')
projects = select_projects(df)
bodies = select_waterbodies(df, projects, 'Selecteer waterlichaam')
years = get_shared_years(projects, bodies)
period = None if not years else tuple(st.slider('Selecteer periode (voor chemie vs ecologie én bubble plot)', int(min(years)), int(max(years)), [int(min(years)), int(max(years))], key='ecol_shared_period'))

df_chem = load_chemistry_data(); st.subheader('🧪 Chemie vs ecologische indices')
if df_chem.empty:
    st.warning('Chemische data kon niet worden geladen voor de dubbele Y-as grafiek.')
else:
    st.sidebar.markdown('---'); st.sidebar.subheader('Chemie vs bedekking')
    options, default_loc, required = get_preferred_chemistry_locations(bodies, get_available_chemistry_locations(df_chem))
    options = options or get_available_chemistry_locations(df_chem)
    location = st.sidebar.selectbox('Meetlocatie chemie', options=options, index=(options.index(default_loc) if default_loc in options else None), key='chem_vs_eco_location', placeholder='Kies een meetlocatie') if options else None
    labels = get_available_chemistry_parameter_labels(df_chem)
    preferred = [x for code in CHEM_PARAM_SUGGESTIONS for x in labels if str(x).startswith(f'{code} ') or str(x) == code or str(x).startswith(f'{code}—') or str(x).startswith(f'{code} —')]
    chems = tuple(st.sidebar.multiselect('Selecteer stof(fen) (max. 5)', options=labels, default=(preferred[:2] if preferred else labels[:1]), key='chem_vs_eco_params', max_selections=5)[:5])
    left_metric = st.sidebar.selectbox('Linker Y-as', ['Totale bedekking', 'Soortgroep', 'Trofieniveau', 'KRW score', 'Kenmerkende soort (N2000)', 'Groeivormen'], index=0)
    display_mode = st.sidebar.radio('Weergave linker Y-as', ['Lijnen', 'Kolommen', 'Gestapeld gebied'], index=1)
    seasons = tuple(st.sidebar.multiselect('Seizoensgemiddelden chemie', options=SEASON_ORDER, default=SEASON_ORDER))
    krw_mode = st.sidebar.radio('KRW-score tonen als', ['index', 'klassen'], index=0, horizontal=True) if left_metric == 'KRW score' else 'index'
    n2000_mode = st.sidebar.radio('Kenmerkende soort (N2000) tonen als', ['records', 'soorten'], index=0, horizontal=True) if left_metric == 'Kenmerkende soort (N2000)' else 'records'
    top_n = st.sidebar.slider('Max. aantal ecologische series links', 1, 12, 6) if left_metric in {'Soortgroep', 'Trofieniveau', 'Groeivormen', 'Kenmerkende soort (N2000)'} else None
    fig, eco, chem, summary, msg = build_dual_axis_view(projects, bodies, df_chem, chems, location, left_metric, display_mode, period, seasons, krw_mode, n2000_mode, top_n)
    if required and location is None:
        st.info('Kies eerst een meetlocatie chemie voor het geselecteerde waterlichaam / de geselecteerde waterlichamen.')
    elif msg:
        st.info(msg)
    else:
        render_plot(fig); st.caption(f"Chemische lijnen tonen per jaar het gemiddelde van de geselecteerde seizoenen ({', '.join(seasons) if seasons else 'alle seizoenen'}).")
        if summary is not None and not summary.empty: st.dataframe(summary, width='stretch', hide_index=True)

st.subheader('Relatie doorzicht vs bedekking')
fig, msg = build_bubble_figure(projects, bodies, period); render_plot(fig, info=msg)
st.subheader('📊 Verdeling per jaar (heatmap)')
param = st.selectbox('Kies parameter voor heatmap', ['Trofieniveau', 'Groeivormen', 'Soortgroepen', 'KRW score', 'Kenmerkende soorten (N2000)'])
basis = st.radio('Bereken verdeling op basis van', ['Records (aantal waarnemingen)', 'Bedekking-gewogen (som bedekking)'], index=1, horizontal=True, key='heatmap_basis_choice')
normalize = st.checkbox('Normaliseer per jaar (100% verdeling)', value=True)
fig, msg, caption = build_heatmap_figure(projects, bodies, param, basis, normalize); render_plot(fig, info=msg)
if caption: st.caption(caption)
