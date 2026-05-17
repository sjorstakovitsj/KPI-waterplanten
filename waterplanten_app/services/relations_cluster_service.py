from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from waterplanten_app.config.mappings import RWS_GROEIVORM_CODES
from waterplanten_app.core.data_access import load_data
from waterplanten_app.domain.contracts import DashboardFilters


def _as_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype='float')
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(',', '.', regex=False).str.replace('<', '', regex=False).str.replace('>', '', regex=False).str.strip(), errors='coerce')


def _load_year_frame(filters: DashboardFilters) -> pd.DataFrame:
    df = load_data()
    if df.empty or filters.year is None:
        return pd.DataFrame()
    out = df[df['jaar'].astype(str) == str(filters.year)].copy()
    for c in [c for c in ['doorzicht_m', 'diepte_m', 'bedekking_pct', 'totaal_bedekking_locatie'] if c in out.columns]:
        out[c] = _as_numeric(out[c])
    out['zicht_per_diepte'] = np.where((out['diepte_m'].notna()) & (out['diepte_m'] > 0) & (out['doorzicht_m'].notna()), out['doorzicht_m'] / out['diepte_m'], np.nan) if {'doorzicht_m', 'diepte_m'}.issubset(out.columns) else np.nan
    return out


def build_scatter_cover_figure(filters: DashboardFilters):
    df = _load_year_frame(filters)
    if df.empty:
        return None, 'Geen data voor dit jaar.'
    return px.scatter(df, x='zicht_per_diepte', y='bedekking_pct', color='groeivorm' if 'groeivorm' in df.columns else None, title='Doorzicht/Diepte vs Bedekking', labels={'zicht_per_diepte': 'Doorzicht / Diepte (-)', 'bedekking_pct': 'Bedekking (%)'}), None


def build_scatter_richness_figure(filters: DashboardFilters, strict_richness: bool = False):
    df = _load_year_frame(filters)
    if df.empty:
        return None, 'Geen data voor dit jaar.'
    if strict_richness and 'type' in df.columns and 'soort' in df.columns:
        df = df[(df['type'] == 'Soort') & (~df['soort'].isin(RWS_GROEIVORM_CODES))].copy()
    div = df.groupby('locatie_id', as_index=False).agg(soort=('soort', 'nunique'), doorzicht_m=('doorzicht_m', 'mean'), diepte_m=('diepte_m', 'mean'))
    div['zicht_per_diepte'] = np.where((div['diepte_m'].notna()) & (div['diepte_m'] > 0) & (div['doorzicht_m'].notna()), div['doorzicht_m'] / div['diepte_m'], np.nan)
    return px.scatter(div, x='zicht_per_diepte', y='soort', title='Doorzicht/Diepte vs Soortenrijkdom', labels={'zicht_per_diepte': 'Doorzicht / Diepte (-)', 'soort': 'Soortenrijkdom (#)'}), None


def build_pca_figure(filters: DashboardFilters, use_top_n: bool = False, top_n: int = 50, color_var: str | None = None):
    df = _load_year_frame(filters)
    if df.empty:
        return None, None, None, 'Geen data voor dit jaar.'
    source = df[['locatie_id', 'soort', 'bedekking_pct']].copy()
    source['bedekking_pct'] = _as_numeric(source['bedekking_pct']).fillna(0.0)
    if use_top_n:
        top_species = source.groupby('soort', as_index=False)['bedekking_pct'].sum().sort_values('bedekking_pct', ascending=False).head(int(top_n))['soort'].tolist()
        source = source[source['soort'].isin(top_species)].copy()
    pivot_df = source.pivot_table(index='locatie_id', columns='soort', values='bedekking_pct', fill_value=0.0)
    if len(pivot_df) <= 5:
        return None, None, None, 'Te weinig meetpunten (>5 nodig) om een betrouwbare clusteranalyse uit te voeren voor dit jaar.'
    x = StandardScaler().fit_transform(pivot_df); pca = PCA(n_components=2); pc = pca.fit_transform(x)
    pca_df = pd.DataFrame(pc, columns=['PC1', 'PC2']); pca_df['locatie_id'] = pivot_df.index
    explained = pca.explained_variance_ratio_; meta = df.groupby('locatie_id', as_index=False).mean(numeric_only=True); final = pca_df.merge(meta, on='locatie_id', how='left')
    fig = px.scatter(final, x='PC1', y='PC2', hover_data=['locatie_id'], color=(None if not color_var or color_var == '(geen)' else color_var), color_continuous_scale='RdYlGn', title=f'PCA vegetatiesamenstelling {filters.year}', labels={'PC1': f'PC1 ({explained[0]:.1%})', 'PC2': f'PC2 ({explained[1]:.1%})'})
    fig.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.5); fig.add_vline(x=0, line_dash='dash', line_color='grey', opacity=0.5)
    caption = f'NB.: Verklaarde variantie: PC1 ({explained[0]:.1%}) + PC2 ({explained[1]:.1%}). Dit betekent dat deze 2D-weergave ongeveer {sum(explained):.0%} van de totale verschillen in vegetatie samenvat.'
    color_options = [c for c in ['totaal_bedekking_locatie', 'doorzicht_m', 'diepte_m'] if c in final.columns]
    return fig, caption, color_options, None
