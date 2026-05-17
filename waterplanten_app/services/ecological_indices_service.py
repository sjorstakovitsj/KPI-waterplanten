from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from waterplanten_app.core.taxonomy import add_species_group_columns
try:
    from waterplanten_app.core.data_access import get_bubble_yearly_filtered, load_ecology_timeseries_data_filtered, load_filtered_ecology_base
except Exception:
    from waterplanten_app.pipelines.ecology_pipeline import get_bubble_yearly_filtered, load_ecology_timeseries_data_filtered, load_filtered_ecology_base
try:
    from waterplanten_app.core.chemistry import get_chem_ecology_timeseries, summarize_chemistry_period_average
except Exception:
    from waterplanten_app.pipelines.chemistry_pipeline import get_chem_ecology_timeseries, summarize_chemistry_period_average


def ensure_nomatch_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if 'trofisch_niveau' not in df.columns:
        df['trofisch_niveau'] = np.nan
    if 'trofisch_niveau_weergave' not in df.columns:
        df['trofisch_niveau_weergave'] = np.where(df['trofisch_niveau'].notna() & (df['trofisch_niveau'].astype(str).str.strip() != ''), df['trofisch_niveau'].astype(str), 'Geen match')
    if 'krw_score' not in df.columns:
        df['krw_score'] = np.nan
    if 'krw_class' not in df.columns:
        df['krw_class'] = pd.cut(pd.to_numeric(df['krw_score'], errors='coerce'), bins=[0, 2, 4, 5], labels=['Gunstig (1-2)', 'Neutraal (3-4)', 'Ongewenst (5)'], include_lowest=True)
    if 'krw_class_weergave' not in df.columns:
        df['krw_class_weergave'] = df['krw_class'].astype(object); df.loc[df['krw_class_weergave'].isna(), 'krw_class_weergave'] = 'Geen match'
    return df


def get_shared_years(projects: tuple[str, ...], bodies: tuple[str, ...]) -> list[int]:
    df = get_bubble_yearly_filtered(projects, bodies)
    if df.empty or 'jaar' not in df.columns:
        return []
    return sorted(pd.to_numeric(df['jaar'], errors='coerce').dropna().astype(int).unique().tolist())


def _bubble_period_means(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['soort', 'doorzicht_m', 'bedekking_pct', 'diepte_m', 'doorzicht_diepte_ratio'])
    rng = df[(df['jaar'] >= year_min) & (df['jaar'] <= year_max)]
    if rng.empty:
        return pd.DataFrame(columns=['soort', 'doorzicht_m', 'bedekking_pct', 'diepte_m', 'doorzicht_diepte_ratio'])
    out = rng.groupby('soort', as_index=False).agg(doorzicht_m=('doorzicht_m', 'mean'), bedekking_pct=('bedekking_pct', 'mean'), diepte_m=('diepte_m', 'mean'))
    diepte = out['diepte_m'].astype(float).where(out['diepte_m'].astype(float) > 0, np.nan)
    out['doorzicht_diepte_ratio'] = out['doorzicht_m'] / diepte
    out = out.dropna(subset=['doorzicht_diepte_ratio']); out['diepte_m'] = out['diepte_m'].fillna(0.1).clip(lower=0.1)
    return out


def build_bubble_figure(projects: tuple[str, ...], bodies: tuple[str, ...], period: tuple[int, int] | None):
    df = get_bubble_yearly_filtered(projects, bodies)
    if df.empty or period is None:
        return None, 'Geen data beschikbaar voor bubbleplot na filtering.'
    plot_df = _bubble_period_means(df, int(period[0]), int(period[1]))
    if plot_df.empty:
        return None, 'Geen data gevonden voor deze filtercombinatie of periode.'
    fig = px.scatter(plot_df, x='doorzicht_diepte_ratio', y='bedekking_pct', size='diepte_m', hover_name='soort', size_max=40, title=f'Ecologische indices ({period[0]} - {period[1]})', labels={'doorzicht_diepte_ratio': 'gem. doorzicht / gem. diepte (-)', 'bedekking_pct': 'gem. bedekking (%)', 'diepte_m': 'gem. diepte (m)'})
    fig.add_vrect(x0=0.6, x1=0.8, fillcolor='rgba(46, 204, 113, 0.16)', line_width=0, layer='below', annotation_text='OK (0.6–0.8)', annotation_position='top left')
    fig.add_vrect(x0=0.8, x1=1.0, fillcolor='rgba(39, 174, 96, 0.24)', line_width=0, layer='below', annotation_text='Ideaal (≥0.8)', annotation_position='top left')
    fig.add_vline(x=0.6, line_width=2, line_dash='dot', line_color='rgba(255, 165, 0, 0.85)', annotation_text='Min 0.6', annotation_position='top left')
    fig.add_vline(x=0.8, line_width=2, line_dash='dot', line_color='rgba(0, 100, 0, 0.90)', annotation_text='Streef 0.8', annotation_position='top left')
    return fig, None


def build_dual_axis_view(projects: tuple[str, ...], bodies: tuple[str, ...], df_chem: pd.DataFrame, chemistry_labels: tuple[str, ...], chemistry_location: str | None, left_metric: str, display_mode: str, period: tuple[int, int] | None, seasons: tuple[str, ...], krw_mode: str = 'index', n2000_mode: str = 'records', top_n: int | None = None, show_markers: bool = True):
    if period is None:
        return None, None, None, None, 'Geen gedeelde periode beschikbaar voor chemie vs ecologie.'
    df_eco = load_ecology_timeseries_data_filtered(projects, bodies)
    if df_eco.empty:
        return None, None, None, None, 'Ecologische tijdreeksdata kon niet worden voorbereid.'
    if df_chem.empty:
        return None, None, None, None, 'Chemische data kon niet worden geladen.'
    if not chemistry_labels:
        return None, None, None, None, 'Selecteer minimaal één chemische stof om de dubbele Y-as grafiek te tonen.'
    eco_mode = krw_mode if left_metric == 'KRW score' else (n2000_mode if left_metric == 'Kenmerkende soort (N2000)' else 'default')
    eco_year, chem_year, common_years = get_chem_ecology_timeseries(df_eco=df_eco, df_chem=df_chem, project_sel=projects, body_sel=bodies, ecology_metric=left_metric, chemistry_labels=chemistry_labels, chemistry_location=chemistry_location, ecology_mode=eco_mode, definitive_only=False, seasons=seasons, top_n=top_n)
    if eco_year.empty or chem_year.empty or not common_years:
        return None, eco_year, chem_year, None, 'Geen overlappende jaren tussen ecologie en chemie voor deze filtercombinatie.'
    eco_year = eco_year[(eco_year['jaar'] >= int(period[0])) & (eco_year['jaar'] <= int(period[1]))].copy(); chem_year = chem_year[(chem_year['jaar'] >= int(period[0])) & (chem_year['jaar'] <= int(period[1]))].copy()
    if eco_year.empty or chem_year.empty:
        return None, eco_year, chem_year, None, 'Geen chemie- of ecologiedata binnen de gekozen gedeelde periode.'
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    eco_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939']
    chem_palette = ['#111111', '#4c4c4c', '#7f7f7f', '#555555', '#999999']; chem_dashes = ['dash', 'dot', 'dashdot', 'longdash', 'solid']
    for i, serie in enumerate(sorted(eco_year['serie'].dropna().astype(str).unique().tolist())):
        d = eco_year[eco_year['serie'] == serie].sort_values('jaar'); color = eco_palette[i % len(eco_palette)]
        trace = go.Bar(x=d['jaar'], y=d['waarde'], name=serie, marker_color=color, opacity=0.82) if display_mode == 'Kolommen' else go.Scatter(x=d['jaar'], y=d['waarde'], name=serie, mode='lines' if display_mode == 'Gestapeld gebied' else 'lines+markers', line=dict(color=color, width=1.5 if display_mode == 'Gestapeld gebied' else 2), marker=dict(size=6), stackgroup='one' if display_mode == 'Gestapeld gebied' else None)
        fig.add_trace(trace, secondary_y=False)
    for i, serie in enumerate([x for x in chemistry_labels if x in chem_year['serie'].unique().tolist()]):
        d = chem_year[chem_year['serie'] == serie].sort_values('jaar')
        if d.empty:
            continue
        unit = '' if 'eenheid_omschrijving' not in d.columns or d['eenheid_omschrijving'].dropna().empty else str(d['eenheid_omschrijving'].dropna().iloc[0])
        fig.add_trace(go.Scatter(x=d['jaar'], y=d['chem_value'], name=f'{serie} ({unit})' if unit else serie, mode='lines+markers' if show_markers else 'lines', line=dict(color=chem_palette[i % len(chem_palette)], width=3, dash=chem_dashes[i % len(chem_dashes)]), marker=dict(size=7, symbol='diamond')), secondary_y=True)
    left_title = 'Gemiddelde KRW-score' if left_metric == 'KRW score' and krw_mode == 'index' else ('Aantal aanwezigheidsrecords' if left_metric == 'Kenmerkende soort (N2000)' and n2000_mode == 'records' else f'{left_metric} / bedekking')
    units = [u for u in chem_year.get('eenheid_omschrijving', pd.Series(dtype='object')).dropna().astype(str).unique().tolist() if u]
    right_title = f'Concentratie ({units[0]})' if len(units) == 1 else ('Concentratie (eenheidsafhankelijk)' if len(units) > 1 else 'Concentratie')
    season_label = ', '.join(seasons) if seasons else 'alle seizoenen'
    fig.update_layout(title=f"Chemie vs {left_metric}<br><sup>Ecologie: {', '.join(projects) or 'geen project'} — {', '.join(bodies) or 'geen waterlichaam'} — Chemie: {chemistry_location or 'geen locatie'} — Seizoen: {season_label}</sup>", height=760, hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0), xaxis=dict(title='Jaar', tickmode='linear', rangeslider=dict(visible=True)), barmode='group' if display_mode == 'Kolommen' else None)
    fig.update_yaxes(title_text=left_title, secondary_y=False); fig.update_yaxes(title_text=right_title, secondary_y=True)
    summary = summarize_chemistry_period_average(chem_year, year_min=int(period[0]), year_max=int(period[1]))
    return fig, eco_year, chem_year, summary, None


def build_heatmap_figure(projects: tuple[str, ...], bodies: tuple[str, ...], param: str, basis: str, normalize_year: bool):
    df_base = ensure_nomatch_display_columns(load_filtered_ecology_base(projects, bodies))
    if df_base.empty:
        return None, 'Geen data beschikbaar voor deze heatmap-keuze (na filters).', ''
    if param == 'Groeivormen':
        df_h, cat_col = df_base[df_base['type'] == 'Groep'].copy(), 'groeivorm'; df_h['bedekking_num'] = pd.to_numeric(df_h['bedekking_pct'], errors='coerce').fillna(0).clip(lower=0)
    elif param == 'Soortgroepen':
        df_h = add_species_group_columns(df_base[df_base['type'] == 'Soort'].copy()); cat_col = 'soortgroep_weergave' if 'soortgroep_weergave' in df_h.columns else 'soortgroep'; df_h['bedekking_num'] = pd.to_numeric(df_h.get('bedekkingsgraad_proc', df_h['bedekking_pct']), errors='coerce').fillna(0).clip(lower=0)
    elif param == 'Kenmerkende soorten (N2000)':
        df_h = add_species_group_columns(df_base[df_base['type'] == 'Soort'].copy()); df_h = df_h[df_h.get('is_kenmerkende_soort_n2000', False).fillna(False)].copy(); cat_col = 'kenmerkende_soort_n2000_weergave' if 'kenmerkende_soort_n2000_weergave' in df_h.columns else ('soort_display' if 'soort_display' in df_h.columns else 'soort'); df_h['bedekking_num'] = pd.to_numeric(df_h.get('bedekkingsgraad_proc', df_h['bedekking_pct']), errors='coerce').fillna(0).clip(lower=0)
    elif param == 'Trofieniveau':
        df_h, cat_col = df_base[df_base['type'] == 'Soort'].copy(), 'trofisch_niveau_weergave'; df_h['bedekking_num'] = pd.to_numeric(df_h['bedekking_pct'], errors='coerce').fillna(0).clip(lower=0)
    else:
        df_h, cat_col = df_base[df_base['type'] == 'Soort'].copy(), 'krw_class_weergave'; df_h['bedekking_num'] = pd.to_numeric(df_h['bedekking_pct'], errors='coerce').fillna(0).clip(lower=0)
    if df_h.empty or 'jaar' not in df_h.columns:
        return None, 'Geen data beschikbaar voor deze heatmap-keuze (na filters).', ''
    df_h[cat_col] = df_h.get(cat_col, pd.Series(index=df_h.index, dtype='object')).astype(object); df_h.loc[df_h[cat_col].isna() | (df_h[cat_col].astype(str).str.strip() == ''), cat_col] = 'Geen match'
    agg = df_h.groupby([cat_col, 'jaar']).size().reset_index(name='waarde') if basis.startswith('Records') else df_h.groupby([cat_col, 'jaar'])['bedekking_num'].sum().reset_index(name='waarde')
    heat = agg.pivot(index=cat_col, columns='jaar', values='waarde').fillna(0)
    if normalize_year:
        heat = heat.div(heat.sum(axis=0).replace(0, np.nan), axis=1).fillna(0) * 100
    label = 'Aandeel (%)' if normalize_year else ('Aantal records' if basis.startswith('Records') else 'Som bedekking')
    fig = px.imshow(heat, color_continuous_scale='Viridis', aspect='auto', labels=dict(x='Jaar', y=param, color=label), title=f"Heatmap {param} per jaar" + (' (genormaliseerd)' if normalize_year else ''))
    text = heat.round(1).astype(str) + '%' if normalize_year else (heat.round(0).astype(int).astype(str) if basis.startswith('Records') else heat.round(1).astype(str))
    fig.update_traces(text=text.values, texttemplate='%{text}', textfont=dict(color='white', size=12)); fig.update_layout(uniformtext_minsize=8, uniformtext_mode='show', height=650, yaxis=dict(side='left'))
    caption = 'Bron trofieniveau-indeling: Verhofstad et al. (2025) – Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang. https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf' if param == 'Trofieniveau' else ''
    return fig, None, caption
