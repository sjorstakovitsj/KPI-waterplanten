from __future__ import annotations

import pandas as pd
import streamlit as st


def select_year(
    df: pd.DataFrame,
    label: str,
    *,
    reverse: bool = True,
    default: int | None = None,
) -> int:
    years = sorted(
        pd.to_numeric(df['jaar'], errors='coerce').dropna().astype(int).unique().tolist(),
        reverse=reverse,
    )
    index = years.index(int(default)) if default is not None and int(default) in years else 0
    return int(st.sidebar.selectbox(label, years, index=index))


def select_projects(
    df: pd.DataFrame,
    label: str = 'Selecteer project(en)',
    *,
    default: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    projects = sorted(df['Project'].dropna().astype(str).unique().tolist())
    selected_default = projects if default is None else [x for x in default if x in projects]
    return tuple(st.sidebar.multiselect(label, options=projects, default=selected_default))


def select_waterbodies(
    df: pd.DataFrame,
    projects: tuple[str, ...],
    label: str = 'Selecteer waterlichaam / waterlichamen',
    *,
    default: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    scoped = df[df['Project'].isin(projects)] if projects else df
    bodies = sorted(scoped['Waterlichaam'].dropna().astype(str).unique().tolist())
    selected_default = bodies if default is None else [x for x in default if x in bodies]
    return tuple(st.sidebar.multiselect(label, options=bodies, default=selected_default))


def select_year_range(df: pd.DataFrame, label: str, *, default_last_n: int = 10) -> tuple[int, int]:
    years = sorted(pd.to_numeric(df['jaar'], errors='coerce').dropna().astype(int).unique().tolist())
    if not years:
        return (0, 0)
    min_y, max_y = int(min(years)), int(max(years))
    return st.sidebar.slider(label, min_value=min_y, max_value=max_y, value=(max(min_y, max_y - default_last_n), max_y), step=1)


def select_meetjaren_range(counts: pd.DataFrame, label: str = 'Selecteer bereik aantal meetjaren per locatie') -> tuple[int, int]:
    min_v = int(counts['n_meetjaren'].min())
    max_v = int(counts['n_meetjaren'].max())
    default_min = min(max(3, min_v), max_v)
    return st.sidebar.slider(label, min_value=min_v, max_value=max_v, value=(default_min, max_v), step=1)


def select_optional_species(df: pd.DataFrame, species_options: list[str], label: str = 'Selecteer individuele soort (optioneel)') -> str | None:
    selected = st.sidebar.selectbox(label, ['— Alle soorten —'] + species_options, index=0)
    return None if selected == '— Alle soorten —' else str(selected)
