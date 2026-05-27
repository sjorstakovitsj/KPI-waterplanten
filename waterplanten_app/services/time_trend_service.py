from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from waterplanten_app.core.diagnostics import categorize_slope_trend
from waterplanten_app.core.maps import df_to_geojson_points, render_swipe_map_html
from waterplanten_app.core.taxonomy import add_species_group_columns


def as_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype='float')
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(',', '.', regex=False).str.replace('<', '', regex=False).str.replace('>', '', regex=False).str.strip(), errors='coerce')


def filter_time_selection(df: pd.DataFrame, bodies: tuple[str, ...], species: str | None, show_ondergedoken: bool, show_chariden: bool) -> pd.DataFrame:
    dfx = df[df['Waterlichaam'].isin(bodies)].copy() if bodies else df.copy()
    for c in [c for c in ['bedekking_pct', 'doorzicht_m', 'diepte_m', 'lat', 'lon', 'jaar'] if c in dfx.columns]:
        dfx[c] = as_numeric(dfx[c])
    if species:
        mask = dfx.get('soort', pd.Series(False, index=dfx.index)).eq(species)
        if 'type' in dfx.columns:
            mask &= dfx['type'].eq('Soort')
        if 'Grootheid' in dfx.columns:
            mask &= dfx['Grootheid'].eq('BEDKG')
        return dfx.loc[mask].copy()
    if show_ondergedoken or show_chariden:
        soort = dfx.get('soort', pd.Series('', index=dfx.index)).fillna('').astype(str).str.strip()
        groei = dfx.get('groeivorm', pd.Series('', index=dfx.index)).fillna('').astype(str)
        mask = pd.Series(False, index=dfx.index)
        if show_ondergedoken:
            mask |= groei.eq('Ondergedoken')
        if show_chariden:
            mask |= soort.str.startswith(("Chara", "Nitella", "Nitellopsis", "Tolypella"))
        return dfx.loc[mask].copy()
    return dfx


def compute_trend_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or not {'locatie_id', 'jaar'}.issubset(df.columns):
        return pd.DataFrame(columns=['Waterlichaam', 'locatie_id', 'jaar', 'waarde'])
    group_cols = [c for c in ['Waterlichaam', 'locatie_id', 'jaar'] if c in df.columns]
    if metric == 'soort_count':
        out = df[group_cols + ['soort']].dropna(subset=['locatie_id', 'jaar']).groupby(group_cols, sort=False, observed=True)['soort'].nunique().reset_index(name='waarde')
    else:
        if metric not in df.columns:
            return pd.DataFrame(columns=['Waterlichaam', 'locatie_id', 'jaar', 'waarde'])
        out = df[group_cols + [metric]].dropna(subset=['locatie_id', 'jaar']).groupby(group_cols, sort=False, observed=True)[metric].mean().reset_index(name='waarde')
    if 'Waterlichaam' not in out.columns:
        out['Waterlichaam'] = 'Onbekend'
    out['jaar'] = pd.to_numeric(out['jaar'], errors='coerce'); out['waarde'] = as_numeric(out['waarde'])
    return out.dropna(subset=['jaar', 'waarde', 'locatie_id']).assign(jaar=lambda x: x['jaar'].astype(int)).sort_values(['Waterlichaam', 'locatie_id', 'jaar']).reset_index(drop=True)


def trend_cover(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {'Waterlichaam', 'jaar', 'locatie_id', 'bedekking_pct'}.issubset(df.columns):
        return pd.DataFrame(columns=['Waterlichaam', 'jaar', 'totaal_bedekking_locatie'])
    dfx = df.copy()
    if 'type' in dfx.columns:
        dfx = dfx[dfx['type'].eq('Soort')].copy()
    if 'Grootheid' in dfx.columns:
        dfx = dfx[dfx['Grootheid'].eq('BEDKG')].copy()
    dfx = dfx[['Waterlichaam', 'jaar', 'locatie_id', 'bedekking_pct']].copy(); dfx['jaar'] = pd.to_numeric(dfx['jaar'], errors='coerce'); dfx['bedekking_pct'] = as_numeric(dfx['bedekking_pct'])
    dfx = dfx.dropna(subset=['Waterlichaam', 'jaar', 'locatie_id', 'bedekking_pct']).assign(jaar=lambda x: x['jaar'].astype(int))
    per_loc = dfx.groupby(['Waterlichaam', 'jaar', 'locatie_id'], sort=False, observed=True)['bedekking_pct'].sum().reset_index(name='totaal_bedekking_locatie')
    return per_loc.groupby(['Waterlichaam', 'jaar'], sort=False, observed=True)['totaal_bedekking_locatie'].mean().reset_index().sort_values(['Waterlichaam', 'jaar'])


def build_cover_figure(df: pd.DataFrame, species: str | None):
    trend = trend_cover(df)
    if trend.empty:
        return None
    fig = px.line(trend, x='jaar', y='totaal_bedekking_locatie', color='Waterlichaam', markers=True, title=f"Trend totale bedekking (%) per waterlichaam{' – ' + species if species else ''}", labels={'jaar': 'Jaar', 'totaal_bedekking_locatie': 'Totale bedekking (%)', 'Waterlichaam': 'Waterlichaam'})
    fig.update_layout(height=380, legend=dict(orientation='h', y=-0.25))
    return fig




DEFAULT_TROPHIC_LEVELS = ['oligotroof', 'mesotroof', 'eutroof', 'sterk eutroof', 'brak', 'marien']


def _prepare_species_with_trophic_level(df: pd.DataFrame) -> pd.DataFrame:
    """Selecteert soortrecords met bedekking en zorgt dat 'trofieniveau' beschikbaar is."""
    required = {'soort', 'jaar', 'bedekking_pct'}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    dfx = df.copy()
    if 'type' in dfx.columns:
        dfx = dfx[dfx['type'].eq('Soort')].copy()
    if 'Grootheid' in dfx.columns:
        dfx = dfx[dfx['Grootheid'].eq('BEDKG')].copy()
    if dfx.empty:
        return pd.DataFrame()

    if 'trofisch_niveau_weergave' not in dfx.columns and 'trofisch_niveau' not in dfx.columns:
        try:
            dfx = add_species_group_columns(dfx)
        except Exception:
            pass

    if 'trofisch_niveau_weergave' in dfx.columns:
        dfx['trofieniveau'] = dfx['trofisch_niveau_weergave']
    elif 'trofisch_niveau' in dfx.columns:
        dfx['trofieniveau'] = dfx['trofisch_niveau']
    else:
        return pd.DataFrame()

    dfx['trofieniveau'] = dfx['trofieniveau'].fillna('Onbekend').astype(str).str.strip()
    dfx.loc[dfx['trofieniveau'].eq(''), 'trofieniveau'] = 'Onbekend'
    dfx['jaar'] = pd.to_numeric(dfx['jaar'], errors='coerce')
    dfx['bedekking_pct'] = as_numeric(dfx['bedekking_pct']).fillna(0).clip(lower=0)
    dfx = dfx.dropna(subset=['soort', 'jaar'])
    if dfx.empty:
        return pd.DataFrame()

    dfx['jaar'] = dfx['jaar'].astype(int)
    if 'Waterlichaam' not in dfx.columns:
        dfx['Waterlichaam'] = 'Onbekend'
    return dfx


def compute_trophic_species_trends_all_years(
    df: pd.DataFrame,
    trophic_levels: list[str] | None = None,
    min_years: int = 2,
) -> pd.DataFrame:
    """
    Berekent per Waterlichaam, trofieniveau en soort een genormaliseerde trend
    over alle beschikbare jaren in de huidige selectie.
    """
    levels = trophic_levels or DEFAULT_TROPHIC_LEVELS
    empty_cols = ['Waterlichaam', 'trofieniveau', 'soort', 'trend_pct', 'slope', 'mean_cover', 'n_jaren', 'jaar_van', 'jaar_tot', 'kleurcategorie']

    dfx = _prepare_species_with_trophic_level(df)
    if dfx.empty:
        return pd.DataFrame(columns=empty_cols)

    dfx = dfx[dfx['trofieniveau'].isin(levels)].copy()
    if dfx.empty:
        return pd.DataFrame(columns=empty_cols)

    rows = []
    for wb, dfw in dfx.groupby('Waterlichaam', sort=True, observed=True):
        all_years = sorted(dfw['jaar'].dropna().astype(int).unique().tolist())
        if len(all_years) < min_years:
            continue
        year_min, year_max = min(all_years), max(all_years)
        year_span = max(1, year_max - year_min)

        for level in levels:
            dfl = dfw[dfw['trofieniveau'].eq(level)].copy()
            if dfl.empty:
                continue

            annual = dfl.groupby(['soort', 'jaar'], sort=False, observed=True)['bedekking_pct'].mean().reset_index()
            species_list = sorted(annual['soort'].dropna().astype(str).unique().tolist(), key=str.lower)

            for soort in species_list:
                ds = annual[annual['soort'].astype(str).eq(str(soort))][['jaar', 'bedekking_pct']].copy()
                full = pd.DataFrame({'jaar': all_years}).merge(ds, on='jaar', how='left')
                full['bedekking_pct'] = as_numeric(full['bedekking_pct']).fillna(0)

                if len(full) < min_years or float(full['bedekking_pct'].sum()) <= 0:
                    continue

                x = full['jaar'].astype(float).to_numpy()
                y = full['bedekking_pct'].astype(float).to_numpy()
                x_centered = x - x.mean()
                denom = float(np.sum(x_centered ** 2))
                if denom == 0:
                    continue

                slope = float(np.sum(x_centered * (y - y.mean())) / denom)
                mean_cover = float(np.mean(y))
                if mean_cover <= 0:
                    continue

                trend_pct = float((slope * year_span / mean_cover) * 100.0)
                rows.append({
                    'Waterlichaam': str(wb),
                    'trofieniveau': level,
                    'soort': str(soort),
                    'trend_pct': trend_pct,
                    'slope': slope,
                    'mean_cover': mean_cover,
                    'n_jaren': int(len(all_years)),
                    'jaar_van': int(year_min),
                    'jaar_tot': int(year_max),
                    'kleurcategorie': 'Toename' if trend_pct > 0 else ('Afname' if trend_pct < 0 else 'Stabiel'),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=empty_cols)

    return out.sort_values(
        ['Waterlichaam', 'trofieniveau', 'trend_pct'],
        ascending=[True, True, False]
    ).reset_index(drop=True)


def _nice_trend_range(values: pd.Series) -> list[float]:
    """Maakt per subplot een duidelijke, individuele y-as rond 0%."""
    vals = pd.to_numeric(values, errors='coerce').dropna()
    if vals.empty:
        return [-100.0, 100.0]
    ymin = min(0.0, float(vals.min()))
    ymax = max(0.0, float(vals.max()))
    span = ymax - ymin
    if span == 0:
        span = max(20.0, abs(ymax) * 0.2)
        ymin -= span / 2
        ymax += span / 2
    pad = max(5.0, span * 0.12)
    return [ymin - pad, ymax + pad]


def build_all_trophic_species_trend_figure(
    df: pd.DataFrame,
    waterbody: str,
    trophic_levels: list[str] | None = None,
):
    """
    Bouwt één compacte figuur met zes trofieniveau-panelen naast elkaar.
    Elke subplot krijgt een eigen y-as met percentages, zodat trends zichtbaar blijven.
    """
    levels = trophic_levels or DEFAULT_TROPHIC_LEVELS
    trends = compute_trophic_species_trends_all_years(df, trophic_levels=levels)
    if trends.empty:
        return None

    trends = trends[trends['Waterlichaam'].astype(str).eq(str(waterbody))].copy()
    if trends.empty:
        return None

    subplot_titles = [f"{chr(65 + i)}. {level.capitalize()}" for i, level in enumerate(levels)]
    fig = make_subplots(
        rows=1,
        cols=len(levels),
        shared_yaxes=False,
        horizontal_spacing=0.035,
        subplot_titles=subplot_titles,
    )

    for col, level in enumerate(levels, start=1):
        dfl = trends[trends['trofieniveau'].eq(level)].sort_values('trend_pct', ascending=False).copy()

        if dfl.empty:
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='markers', marker=dict(opacity=0), showlegend=False, hoverinfo='skip'),
                row=1,
                col=col,
            )
            fig.add_shape(
                type='rect',
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                xref=f'x{col} domain' if col > 1 else 'x domain',
                yref=f'y{col} domain' if col > 1 else 'y domain',
                fillcolor='rgba(235,235,235,0.85)',
                line=dict(color='rgba(120,120,120,0.9)', width=1.5, dash='dash'),
                layer='below',
            )
            fig.add_annotation(
                text=f'<b>Geen data beschikbaar</b><br>{level}',
                x=0.5,
                y=0.5,
                xref=f'x{col} domain' if col > 1 else 'x domain',
                yref=f'y{col} domain' if col > 1 else 'y domain',
                showarrow=False,
                align='center',
                font=dict(color='rgba(70,70,70,1)', size=15),
            )
            fig.update_yaxes(range=[-100, 100], tickmode='array', tickvals=[-100, -50, 0, 50, 100], ticksuffix='%', showticklabels=True, row=1, col=col)
            fig.update_xaxes(showticklabels=False, row=1, col=col)
        else:
            colors = np.where(dfl['trend_pct'] > 0, 'green', np.where(dfl['trend_pct'] < 0, 'red', 'gray'))
            fig.add_trace(
                go.Bar(
                    x=dfl['soort'],
                    y=dfl['trend_pct'],
                    width=[0.18] * len(dfl),
                    marker_color=colors,
                    marker_line_width=0,
                    name=level,
                    showlegend=False,
                    customdata=np.stack([
                        dfl['mean_cover'].round(3),
                        dfl['slope'].round(4),
                        dfl['n_jaren'],
                        dfl['jaar_van'],
                        dfl['jaar_tot'],
                    ], axis=-1),
                    hovertemplate=(
                        '<b>%{x}</b><br>'
                        'Trend: %{y:.1f}%<br>'
                        'Gem. bedekking: %{customdata[0]}%<br>'
                        'Slope: %{customdata[1]} %-punt/jaar<br>'
                        'Jaren: %{customdata[3]}–%{customdata[4]} (%{customdata[2]} jaar)'
                        '<extra></extra>'
                    ),
                ),
                row=1,
                col=col,
            )
            fig.update_yaxes(range=_nice_trend_range(dfl['trend_pct']), ticksuffix='%', showticklabels=True, row=1, col=col)
            fig.update_xaxes(tickangle=-90, tickfont=dict(size=13), title_text='', automargin=True, row=1, col=col)

        fig.add_hline(y=0, line_width=1.3, line_color='black', row=1, col=col)
        fig.update_yaxes(
            title_text='Trend (%)' if col == 1 else '',
            zeroline=True,
            zerolinecolor='black',
            gridcolor='rgba(0,0,0,0.12)',
            tickfont=dict(size=11),
            row=1,
            col=col,
        )

    year_from = int(trends['jaar_van'].min()) if 'jaar_van' in trends.columns and not trends.empty else None
    year_to = int(trends['jaar_tot'].max()) if 'jaar_tot' in trends.columns and not trends.empty else None
    period = f' ({year_from}–{year_to})' if year_from is not None and year_to is not None else ''

    fig.update_layout(
        title=f'Trend individuele soorten per trofieniveau – {waterbody}{period}',
        width=1380,
        height=700,
        bargap=0.88,
        bargroupgap=0.3,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=65, r=25, t=85, b=255),
    )

    for annotation in fig.layout.annotations:
        if annotation.text and annotation.text.startswith('<b>Geen data'):
            continue
        annotation.font = dict(size=14)

    return fig

def build_meetpunt_trend_figure(df_trend: pd.DataFrame, y_title: str, title: str):
    if df_trend.empty:
        return None
    dfp = df_trend.copy().sort_values(['Waterlichaam', 'locatie_id', 'jaar'])
    waterbodies = [wb for wb in dfp['Waterlichaam'].dropna().astype(str).unique().tolist() if wb] or ['Onbekend']
    if 'Waterlichaam' not in dfp.columns:
        dfp['Waterlichaam'] = 'Onbekend'
    fig = make_subplots(rows=len(waterbodies), cols=1, shared_xaxes=True, vertical_spacing=0.05 if len(waterbodies) > 1 else 0.08, subplot_titles=waterbodies if len(waterbodies) > 1 else None)
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly + px.colors.qualitative.D3
    cmap = {loc: palette[i % len(palette)] for i, loc in enumerate(sorted(dfp['locatie_id'].dropna().astype(str).unique().tolist()))}; shown = set()
    for r, wb in enumerate(waterbodies, start=1):
        dfw = dfp[dfp['Waterlichaam'].astype(str).eq(str(wb))].copy()
        for loc in dfw['locatie_id'].dropna().astype(str).unique().tolist():
            d = dfw[dfw['locatie_id'].astype(str).eq(loc)].copy(); c = cmap.get(loc, '#1f77b4')
            fig.add_trace(go.Scatter(x=d['jaar'], y=d['waarde'], mode='lines+markers', name=loc, legendgroup=loc, showlegend=loc not in shown, line=dict(color=c, width=1.8), marker=dict(size=5, color=c, line=dict(width=0.6, color='white')), opacity=0.9), row=r, col=1); shown.add(loc)
        fig.update_yaxes(title_text=y_title, row=r, col=1, rangemode='tozero', gridcolor='rgba(0,0,0,0.08)', zeroline=False)
        fig.update_xaxes(title_text='Jaar', tickmode='linear', dtick=1, gridcolor='rgba(0,0,0,0.08)')
    fig.update_layout(title=title, height=max(420, 260 * len(waterbodies)), hovermode='closest', plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=40, r=20, t=70, b=40), legend=dict(title='Meetpunt', orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.01, font=dict(size=10), itemsizing='constant', itemclick='toggleothers', itemdoubleclick='toggle'))
    return fig


def compute_slopes(df_trend: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    if df_trend.empty:
        return pd.DataFrame(columns=['locatie_id', 'slope', 'n_jaren'])
    tmp = df_trend[['locatie_id']].copy(); tmp['x'] = as_numeric(df_trend['jaar']).astype(float); tmp['y'] = as_numeric(df_trend['waarde']).astype(float)
    tmp = tmp.dropna(subset=['locatie_id', 'x', 'y'])
    if tmp.empty:
        return pd.DataFrame(columns=['locatie_id', 'slope', 'n_jaren'])
    tmp['xy'] = tmp['x'] * tmp['y']; tmp['xx'] = tmp['x'] * tmp['x']
    agg = tmp.groupby('locatie_id', sort=False, observed=True).agg(n=('x', 'count'), sum_x=('x', 'sum'), sum_y=('y', 'sum'), sum_xy=('xy', 'sum'), sum_xx=('xx', 'sum')).reset_index()
    agg = agg[agg['n'] >= min_years].copy()
    if agg.empty:
        return pd.DataFrame(columns=['locatie_id', 'slope', 'n_jaren'])
    denom = (agg['n'] * agg['sum_xx']) - (agg['sum_x'] ** 2); numer = (agg['n'] * agg['sum_xy']) - (agg['sum_x'] * agg['sum_y'])
    agg['slope'] = np.where(denom != 0, numer / denom, np.nan)
    return agg[['locatie_id', 'slope', 'n']].rename(columns={'n': 'n_jaren'}).dropna(subset=['slope'])


def build_regression_outputs(df_trend: pd.DataFrame, metric: str, min_years: int = 5):
    slopes = compute_slopes(df_trend, min_years=min_years)
    if slopes.empty:
        return None, None, None
    threshold = 0.2 if metric in ['diepte_m', 'doorzicht_m'] else (1.0 if metric == 'soort_count' else 2.0)
    slopes['Trend'] = slopes['slope'].apply(lambda x: categorize_slope_trend(x, threshold))
    pie = px.pie(slopes, names='Trend', title='Trends meetlocaties per geselecteerd waterlichaam', color='Trend', color_discrete_map={'Verbeterend ↗️': 'green', 'Verslechterend ↘️': 'red', 'Stabiel ➡️': 'grey'})
    return pie, slopes.sort_values('slope', ascending=False), threshold




def build_regression_map_figure(df_filtered: pd.DataFrame, slopes: pd.DataFrame):
    """
    Bouwt een OpenStreetMap-kaart met per locatie de regressietrend.
    Gebruikt de huidige sidebarselectie via df_filtered en koppelt de
    regressieresultaten op locatie_id aan lat/lon.
    """
    if slopes is None or slopes.empty:
        return None, 'Geen regressieresultaten beschikbaar om op de kaart te tonen.'

    required = {'locatie_id', 'lat', 'lon'}
    if df_filtered.empty or not required.issubset(df_filtered.columns):
        return None, 'Geen locatiecoördinaten beschikbaar voor de regressiekaart.'

    coord_cols = ['locatie_id', 'lat', 'lon']
    if 'Waterlichaam' in df_filtered.columns:
        coord_cols.append('Waterlichaam')

    coords = df_filtered[coord_cols].copy()
    coords['locatie_id'] = coords['locatie_id'].astype(str)
    coords['lat'] = as_numeric(coords['lat'])
    coords['lon'] = as_numeric(coords['lon'])
    coords = coords.dropna(subset=['locatie_id', 'lat', 'lon'])

    if coords.empty:
        return None, 'Geen geldige coördinaten gevonden voor de regressiekaart.'

    agg_spec = {'lat': ('lat', 'mean'), 'lon': ('lon', 'mean')}
    if 'Waterlichaam' in coords.columns:
        agg_spec['Waterlichaam'] = ('Waterlichaam', 'first')

    coords = coords.groupby('locatie_id', sort=False, observed=True).agg(**agg_spec).reset_index()

    map_df = slopes.copy()
    if 'locatie_id' not in map_df.columns or 'slope' not in map_df.columns:
        return None, 'Regressieresultaten missen locatie_id of slope.'

    map_df['locatie_id'] = map_df['locatie_id'].astype(str)
    map_df = map_df.merge(coords, on='locatie_id', how='left')
    map_df = map_df.dropna(subset=['lat', 'lon'])

    if map_df.empty:
        return None, 'Geen regressielocaties met geldige coördinaten gevonden.'

    if 'Trend' not in map_df.columns:
        map_df['Trend'] = 'Onbekend'
    if 'Waterlichaam' not in map_df.columns:
        map_df['Waterlichaam'] = 'Onbekend'
    if 'n_jaren' not in map_df.columns:
        map_df['n_jaren'] = np.nan

    map_df['abs_slope'] = pd.to_numeric(map_df['slope'], errors='coerce').abs().fillna(0)
    max_abs_slope = float(map_df['abs_slope'].max()) if not map_df.empty else 0.0
    if max_abs_slope <= 0:
        map_df['marker_size'] = 12
    else:
        map_df['marker_size'] = 8 + 16 * (map_df['abs_slope'] / max_abs_slope)

    center = {'lat': float(map_df['lat'].mean()), 'lon': float(map_df['lon'].mean())}

    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        color='Trend',
        size='marker_size',
        size_max=24,
        hover_name='locatie_id',
        hover_data={
            'Waterlichaam': True,
            'Trend': True,
            'slope': ':.4f',
            'n_jaren': ':.0f',
            'lat': False,
            'lon': False,
            'marker_size': False,
            'abs_slope': False,
        },
        color_discrete_map={
            'Verbeterend ↗️': 'green',
            'Verslechterend ↘️': 'red',
            'Stabiel ➡️': 'grey',
            'Onbekend': 'lightgrey',
        },
        mapbox_style='open-street-map',
        center=center,
        zoom=9,
        title='Regressietrend per meetlocatie',
        height=650,
    )
    fig.update_traces(marker=dict(opacity=0.85))
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(title='Trend', orientation='h', yanchor='bottom', y=0.01, xanchor='left', x=0.01),
    )
    return fig, None

def build_compare_figure(df_trend: pd.DataFrame, metric_label: str, year_start: int, year_end: int):
    compare = df_trend[df_trend['jaar'].isin([year_start, year_end])].copy()
    return None if compare.empty else px.bar(compare, x='locatie_id', y='waarde', color=compare['jaar'].astype(str), barmode='group', title=f'Vergelijking {year_start} vs {year_end} per locatie', labels={'waarde': metric_label, 'color': 'Jaar'})


def build_swipe_map_view(df_filtered: pd.DataFrame, metric: str, metric_label: str, year_left: int, year_right: int):
    if year_left == year_right:
        return None, None
    if metric == 'soort_count':
        base = df_filtered[['locatie_id', 'jaar', 'lat', 'lon', 'soort']].dropna(subset=['locatie_id', 'jaar', 'lat', 'lon']).copy(); src = base.groupby(['locatie_id', 'jaar'], sort=False, observed=True).agg(lat=('lat', 'first'), lon=('lon', 'first'), soort_count=('soort', 'nunique')).reset_index(); metric_col = 'soort_count'
    else:
        if metric not in df_filtered.columns:
            return None, None
        base = df_filtered[['locatie_id', 'jaar', 'lat', 'lon', metric]].dropna(subset=['locatie_id', 'jaar', 'lat', 'lon']).copy(); base[metric] = as_numeric(base[metric]); src = base.groupby(['locatie_id', 'jaar'], sort=False, observed=True).agg(lat=('lat', 'first'), lon=('lon', 'first'), **{metric: (metric, 'mean')}).reset_index(); metric_col = metric
    left = src[src['jaar'].eq(year_left)][['locatie_id', 'lat', 'lon', metric_col]].rename(columns={metric_col: 'value'}).copy(); right = src[src['jaar'].eq(year_right)][['locatie_id', 'lat', 'lon', metric_col]].rename(columns={metric_col: 'value'}).copy()
    gj_left, gj_right = df_to_geojson_points(left, 'value', 'locatie_id'), df_to_geojson_points(right, 'value', 'locatie_id')
    bounds_df = pd.concat([left, right], ignore_index=True).dropna(subset=['lat', 'lon']); bounds = [float(bounds_df['lon'].min()), float(bounds_df['lat'].min()), float(bounds_df['lon'].max()), float(bounds_df['lat'].max())] if len(bounds_df) >= 2 else None
    center_lat = float(src['lat'].mean()) if not src.empty else 52.5; center_lon = float(src['lon'].mean()) if not src.empty else 5.5
    values = as_numeric(src.get(metric_col, pd.Series(dtype='float')))
    min_val = float(np.nanmin(values.values)) if len(values) else 0.0; max_val = float(np.nanmax(values.values)) if len(values) else 1.0
    if not np.isfinite(min_val): min_val = 0.0
    if not np.isfinite(max_val) or max_val == min_val: max_val = min_val + 1.0
    map_html = render_swipe_map_html(geojson_left=gj_left, geojson_right=gj_right, year_left=year_left, year_right=year_right, metric_label=metric_label, min_val=min_val, max_val=max_val, center_lat=center_lat, center_lon=center_lon, zoom=9.0, height_px=650, bounds=bounds)
    return map_html, {'min': min_val, 'max': max_val, 'label': metric_label}


def build_heatmap_figure(df: pd.DataFrame, bodies: tuple[str, ...]):
    heat_base = df[df['Waterlichaam'].isin(bodies)].copy() if bodies else df.copy()
    cols = [c for c in ['type', 'Grootheid', 'soort', 'locatie_id', 'jaar', 'bedekking_pct'] if c in heat_base.columns]
    heat_base = heat_base[cols].copy() if cols else pd.DataFrame()
    species = heat_base[heat_base['type'].eq('Soort')].copy() if not heat_base.empty and 'type' in heat_base.columns else heat_base.copy()
    if not species.empty and 'Grootheid' in species.columns:
        species = species[species['Grootheid'].eq('BEDKG')].copy()
    if species.empty:
        return None, 'Geen soortdata gevonden voor de huidige selectie.'
    species['bedekking_pct'] = as_numeric(species['bedekking_pct']) if 'bedekking_pct' in species.columns else np.nan
    cells = species.dropna(subset=['soort', 'locatie_id', 'jaar']).groupby(['soort', 'locatie_id', 'jaar'], sort=False, observed=True)['bedekking_pct'].mean().reset_index()
    top_species = cells.groupby('soort', sort=False, observed=True)['bedekking_pct'].mean().sort_values(ascending=False).head(50).index
    heat = species[species['soort'].isin(top_species)].dropna(subset=['soort', 'jaar']).groupby(['soort', 'jaar'], sort=False, observed=True)['bedekking_pct'].mean().reset_index()
    if heat.empty:
        return None, 'Geen data beschikbaar voor de heatmap.'
    matrix = heat.pivot(index='soort', columns='jaar', values='bedekking_pct').fillna(0); zero_mask = matrix.eq(0); green = matrix.mask(zero_mask, other=np.nan)
    fig = px.imshow(green, color_continuous_scale='Greens', aspect='auto', title='Ontwikkeling bedekking (top 50 meest voorkomende soorten)', labels=dict(x='Jaar', y='Soort', color='Gem. Bedekking (%)'))
    fig.add_trace(go.Heatmap(z=zero_mask.astype(int).replace({0: np.nan}).values, x=matrix.columns, y=matrix.index, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(200,200,200,0.65)']], showscale=False, hoverinfo='skip'))
    ys, xs = np.where(zero_mask.values); fig.add_trace(go.Scatter(x=[matrix.columns[i] for i in xs], y=[matrix.index[i] for i in ys], mode='markers', marker=dict(symbol='x', size=6, color='rgba(120,120,120,0.55)'), showlegend=False, hoverinfo='skip'))
    fig.update_layout(height=1200, yaxis=dict(side='left'))
    return fig, None
