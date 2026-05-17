from __future__ import annotations

import streamlit as st


def render_overview_kpis(kpis: dict) -> None:
    cols = st.columns(len(kpis))
    for col, (_, kpi) in zip(cols, kpis.items()):
        suffix = f" {kpi.unit}" if getattr(kpi, 'unit', None) and kpi.unit != 'soorten' else ''
        precision = kpi.precision if getattr(kpi, 'precision', None) is not None else 0
        value = f"{kpi.value:.{precision}f}{suffix}" if isinstance(kpi.value, (int, float)) and precision else (f"{int(kpi.value)}{suffix}" if isinstance(kpi.value, (int, float)) else str(kpi.value))
        delta = f"{kpi.delta:.{precision}f}{suffix}" if isinstance(kpi.delta, (int, float)) and precision else (f"{int(kpi.delta)}{suffix}" if isinstance(kpi.delta, (int, float)) else str(kpi.delta))
        col.metric(kpi.label, value, delta)


def render_simple_metrics(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
