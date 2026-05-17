from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from waterplanten_app.config.mappings import RWS_GROEIVORM_CODES
from waterplanten_app.core.data_access import load_data
from waterplanten_app.domain.contracts import DashboardFilters


def load_metadata_base(filters: DashboardFilters | None = None) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return df
    out = df.copy()
    out['jaar'] = pd.to_numeric(out['jaar'], errors='coerce')
    if filters:
        if filters.projects:
            out = out[out['Project'].isin(filters.projects)].copy()
        if filters.waterbodies:
            out = out[out['Waterlichaam'].isin(filters.waterbodies)].copy()
        if filters.year is not None:
            out = out[out['jaar'].eq(int(filters.year))].copy()
    return out


def build_general_metadata(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'records': 0, 'locaties': 0, 'soorten': 0, 'jaar_range': 'n.v.t.'}
    min_year, max_year = df['jaar'].min(), df['jaar'].max()
    year_range = 'n.v.t.' if (pd.isna(min_year) or pd.isna(max_year)) else f'{int(min_year)} - {int(max_year)}'
    return {'records': int(len(df)), 'locaties': int(df['locatie_id'].nunique()), 'soorten': int(df['soort'].nunique()), 'jaar_range': year_range}


def build_effort_heatmap(df: pd.DataFrame, show_all: bool = False, top_n: int = 300):
    if df.empty:
        return None, 'Geen data beschikbaar voor meetinspanning.'
    heat = df[['locatie_id', 'jaar']].dropna()
    if not show_all:
        top_locs = df['locatie_id'].value_counts().head(int(top_n)).index
        heat = heat[heat['locatie_id'].isin(top_locs)].copy()
    matrix = heat.groupby(['locatie_id', 'jaar']).size().unstack(fill_value=0)
    if matrix.empty:
        return None, 'Geen data beschikbaar voor meetinspanning.'
    fig = px.imshow(matrix, labels=dict(x='Jaar', y='Locatie', color='Aantal waarnemingen'), x=matrix.columns, y=matrix.index, aspect='auto', color_continuous_scale='Blues')
    fig.update_layout(height=800)
    return fig, None


def build_effort_year_figures(df: pd.DataFrame):
    if df.empty:
        return None, None, 'Geen jaardata beschikbaar.'
    obs = df.groupby('jaar').size().reset_index(name='aantal_records')
    locs = df.groupby('jaar')['locatie_id'].nunique().reset_index(name='aantal_locaties')
    fig_obs = px.bar(obs, x='jaar', y='aantal_records', title='Totaal aantal records per jaar')
    fig_locs = px.line(locs, x='jaar', y='aantal_locaties', markers=True, title='Aantal bezochte meetlocaties per jaar', line_shape='spline')
    fig_locs.update_yaxes(range=[0, int(df['locatie_id'].nunique()) + 5])
    return fig_obs, fig_locs, None


def build_taxonomic_consistency(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), 'Geen individuele soorten aanwezig in de huidige selectie.', ''
    tax = df[(df['type'] == 'Soort') & (~df['soort'].isin(RWS_GROEIVORM_CODES))].copy() if 'type' in df.columns else df[~df['soort'].isin(RWS_GROEIVORM_CODES)].copy()
    total = len(tax)
    if total == 0:
        return pd.DataFrame(), 'Geen individuele soorten aanwezig in de huidige selectie.', ''
    counts = tax['soort'].value_counts().reset_index(); counts.columns = ['Soortnaam', 'Aantal Records']
    counts['Percentage'] = (counts['Aantal Records'] / total) * 100
    p = counts['Percentage']
    conditions = [(p < 0.01), (p >= 0.01) & (p < 0.1), (p >= 0.1) & (p < 1.0), (p >= 1.0) & (p < 2.5), (p >= 2.5)]
    choices = ['🚨 Extreem zeldzaam (<0,01%)', '🚨 Zeer zeldzaam (0,01–0,1%)', '⚠️ Zeldzaam (0,1–1%)', '🟡 Vaak voorkomend (1–2.5%)', '🟢 Algemeen (>2.5%)']
    counts['Status'] = np.select(conditions, choices, default='Onbekend')
    caption = f'Taxonomische consistentie gebaseerd op {total:,} records van individuele soorten (excl. aggregatiecodes).'
    return counts[['Soortnaam', 'Aantal Records', 'Percentage', 'Status']], None, caption


def build_spatial_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['locatie_id', 'lat', 'lon', 'jaar_min', 'jaar_max', 'totaal_waarnemingen', 'periode'])
    locs = df.groupby('locatie_id', as_index=False).agg(lat=('lat', 'first'), lon=('lon', 'first'), jaar_min=('jaar', 'min'), jaar_max=('jaar', 'max'), totaal_waarnemingen=('soort', 'count'))
    locs['periode'] = np.where(locs['jaar_min'].notna() & locs['jaar_max'].notna(), locs['jaar_min'].astype(int).astype(str) + '-' + locs['jaar_max'].astype(int).astype(str), 'n.v.t.')
    return locs
