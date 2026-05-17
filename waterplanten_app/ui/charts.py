from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _plotly_chart_stretch(fig) -> None:
    """Render Plotly-grafieken compatibel met oude en nieuwe Streamlit-versies."""
    try:
        st.plotly_chart(fig, width='stretch')
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def render_plot(fig, *, info: str | None = None) -> None:
    if fig is None:
        if info:
            st.info(info)
        return
    _plotly_chart_stretch(fig)


def render_pie(
    df: pd.DataFrame,
    name_col: str,
    value_col: str,
    title: str,
    *,
    colors: dict | None = None,
    short_label_col: str | None = None,
    caption: str | None = None,
) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.info('Geen data beschikbaar voor de huidige selectie.')
        return

    fig = px.pie(
        df,
        names=name_col,
        values=value_col,
        hole=0.35,
        color=name_col,
        color_discrete_map=colors,
    )

    if short_label_col:
        fig.update_traces(
            textposition='inside',
            textinfo='none',
            texttemplate='%{customdata[0]} %{percent:.1%}',
            customdata=df[[short_label_col]].to_numpy(),
            hovertemplate='%{label}<br>Aantal waarnemingen: %{value}<br>Percentage: %{percent:.1%}',
        )
    else:
        fig.update_traces(textposition='inside', textinfo='percent+label')

    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    _plotly_chart_stretch(fig)
    if caption:
        st.caption(caption)
