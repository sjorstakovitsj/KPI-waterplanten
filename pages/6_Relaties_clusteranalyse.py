import streamlit as st
from utils import load_data
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.services.relations_cluster_service import build_pca_figure, build_scatter_cover_figure, build_scatter_richness_figure
from waterplanten_app.ui.charts import render_plot
from waterplanten_app.ui.filters import select_year

st.set_page_config(layout='wide', page_title='Relaties en multivariate analyse')
st.title('🔗 Relaties en multivariate analyse')

df = load_data()
if df.empty:
    st.error('Geen data beschikbaar.')
    st.stop()
filters = DashboardFilters(year=select_year(df, 'Kies jaar analyse'))
c1, c2 = st.columns(2)
with c1:
    fig, msg = build_scatter_cover_figure(filters); render_plot(fig, info=msg)
with c2:
    st.caption('Soortenrijkdom kan optioneel gefilterd worden (type=Soort, excl. RWS codes).')
    fig, msg = build_scatter_richness_figure(filters, strict_richness=st.checkbox('Soortenrijkdom: alleen echte soorten (excl. RWS codes)', value=False)); render_plot(fig, info=msg)
st.divider(); st.subheader('Multivariate clusteranalyse (PCA)')
use_top_n = st.checkbox('PCA: beperk tot Top-N soorten op totale bedekking (sneller bij veel soorten)', value=False)
top_n = st.slider('Top-N', min_value=10, max_value=200, value=50, step=10, disabled=not use_top_n)
fig, caption, colors, msg = build_pca_figure(filters, use_top_n=use_top_n, top_n=int(top_n), color_var=None)
if msg:
    st.warning(msg)
else:
    color_var = st.selectbox('Kleur PCA-punten op (optioneel)', ['(geen)'] + (colors or []))
    fig, caption, _, _ = build_pca_figure(filters, use_top_n=use_top_n, top_n=int(top_n), color_var=color_var)
    st.plotly_chart(fig, width='stretch'); st.caption(caption)
