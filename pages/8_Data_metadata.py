import streamlit as st
from waterplanten_app.services.data_metadata_service import build_effort_heatmap, build_effort_year_figures, build_general_metadata, build_spatial_coverage, build_taxonomic_consistency, load_metadata_base
from waterplanten_app.ui.charts import render_plot
from waterplanten_app.ui.metrics import render_simple_metrics

st.set_page_config(layout='wide')
st.title('ℹ️ Onderliggende (meta)data')
st.markdown('Controle op volledigheid, meetgaten en taxonomische consistentie.')

df = load_metadata_base()
if df.empty:
    st.error('Geen data geladen.')
    st.stop()
meta = build_general_metadata(df)
render_simple_metrics([('totaal aantal records', f"{meta['records']:,}"), ('unieke locaties', str(meta['locaties'])), ('unieke soorten', str(meta['soorten'])), ('jaren bereik', meta['jaar_range'])]); st.divider()
st.subheader('📅 Tijdruimtelijke meetinspanning')
show_all = st.checkbox('Toon alle locaties in heatmap (kan traag zijn)', value=False)
top_n = st.slider('Max. aantal locaties in heatmap', 50, 2000, 300, 50, disabled=show_all)
fig, msg = build_effort_heatmap(df, show_all=show_all, top_n=int(top_n)); render_plot(fig, info=msg)
st.subheader('📊 Waarnemingsinspanning per jaar')
fig_obs, fig_locs, msg = build_effort_year_figures(df)
if msg: st.info(msg)
else:
    l, r = st.columns(2); l.plotly_chart(fig_obs, width='stretch'); r.plotly_chart(fig_locs, width='stretch')
st.divider(); st.subheader('Taxonomische consistentie')
counts, msg, caption = build_taxonomic_consistency(df)
if msg: st.info(msg)
else: st.caption(caption); st.dataframe(counts, width='stretch')
st.subheader('Ruimtelijke dekking meetnet')
locs = build_spatial_coverage(df)
if not locs.empty:
    st.map(locs)
else:
    st.info('Geen ruimtelijke dekking beschikbaar.')
