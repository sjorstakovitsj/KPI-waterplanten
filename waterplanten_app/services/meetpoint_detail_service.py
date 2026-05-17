from __future__ import annotations

import folium
import pandas as pd
import plotly.graph_objects as go
from waterplanten_app.config.settings import WMS_ATTRIBUTION, WMS_BASE_URL, WMS_LAYER_NAME

from waterplanten_app.core.chemistry import get_chemistry_location_points, load_chemistry_data
from waterplanten_app.core.data_access import load_data
from waterplanten_app.core.diagnostics import interpret_soil_state
from waterplanten_app.core.maps import add_chemistry_locations_to_map, build_bathymetry_legend_url, create_folium_base_map
from waterplanten_app.domain.contracts import DashboardFilters


def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')




def _map_has_named_layer(map_obj, layer_name: str) -> bool:
    """Controleer robuust of een Folium-kaartlaag al aanwezig is."""
    for child in getattr(map_obj, '_children', {}).values():
        if getattr(child, 'layer_name', None) == layer_name:
            return True
        options = getattr(child, 'options', None)
        if isinstance(options, dict) and options.get('layers') == layer_name:
            return True
    return False


def _ensure_osm_with_bathymetry(map_obj):
    """Voeg OpenStreetMap toe als baselaag en behoud bathymetrie als vaste overlay."""
    if map_obj is None:
        return map_obj

    if not _map_has_named_layer(map_obj, 'OpenStreetMap'):
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
            attr='&copy; OpenStreetMap contributors',
            overlay=False,
            control=False,
            show=True,
        ).add_to(map_obj)

    if not _map_has_named_layer(map_obj, 'Bathymetrie IJsselmeergebied'):
        folium.raster_layers.WmsTileLayer(
            url=WMS_BASE_URL,
            name='Bathymetrie IJsselmeergebied',
            layers=WMS_LAYER_NAME,
            fmt='image/png',
            transparent=True,
            version='1.3.0',
            attr=WMS_ATTRIBUTION,
            overlay=True,
            control=False,
            show=True,
        ).add_to(map_obj)

    return map_obj
def get_detail_base(filters: DashboardFilters) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return df
    out = df[df['Project'].isin(filters.projects)].copy() if filters.projects else df.copy()
    if filters.waterbodies:
        out = out[out['Waterlichaam'].isin(filters.waterbodies)].copy()
    return out


def get_chemistry_overlay() -> pd.DataFrame:
    return get_chemistry_location_points(df_chem=load_chemistry_data())


def build_location_overview(df_filtered: pd.DataFrame) -> pd.DataFrame:
    if df_filtered.empty:
        return pd.DataFrame(columns=['locatie_id', 'Waterlichaam', 'lat', 'lon', 'jaar_min', 'jaar_max', 'n_records'])
    return df_filtered.groupby(['locatie_id', 'Waterlichaam'], as_index=False).agg(lat=('lat', 'first'), lon=('lon', 'first'), jaar_min=('jaar', 'min'), jaar_max=('jaar', 'max'), n_records=('locatie_id', 'count')).dropna(subset=['lat', 'lon'])


def filter_locations_by_years(df_filtered: pd.DataFrame, year_range: tuple[int, int], meetjaren_range: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_filtered.empty:
        return df_filtered, pd.DataFrame(columns=['locatie_id', 'n_meetjaren'])
    years = _as_num(df_filtered['jaar'])
    scoped = df_filtered[years.between(int(year_range[0]), int(year_range[1]), inclusive='both')].copy()
    counts = scoped[['locatie_id', 'jaar']].dropna(subset=['locatie_id', 'jaar']).assign(jaar_num=lambda d: pd.to_numeric(d['jaar'], errors='coerce')).dropna(subset=['jaar_num']).drop_duplicates(subset=['locatie_id', 'jaar_num']).groupby('locatie_id', as_index=False)['jaar_num'].count().rename(columns={'jaar_num': 'n_meetjaren'})
    valid = counts.loc[counts['n_meetjaren'].between(int(meetjaren_range[0]), int(meetjaren_range[1]), inclusive='both'), 'locatie_id']
    return scoped[scoped['locatie_id'].isin(valid)].copy(), counts


def build_overview_map(locs_overview: pd.DataFrame, selected_loc: str | None, chem_points: pd.DataFrame):
    if locs_overview.empty:
        return None
    m = create_folium_base_map(float(locs_overview['lat'].mean()), float(locs_overview['lon'].mean()), zoom_start=10, control_scale=True, basemap='bathymetry')
    m = _ensure_osm_with_bathymetry(m)
    selected_loc = str(selected_loc or '')
    for row in locs_overview.itertuples(index=False):
        ymin = 'n.v.t.' if pd.isna(row.jaar_min) else int(row.jaar_min); ymax = 'n.v.t.' if pd.isna(row.jaar_max) else int(row.jaar_max)
        tooltip = f'<b>Meetpunt:</b> {row.locatie_id}<br><b>Waterlichaam:</b> {row.Waterlichaam}<br><b>Periode:</b> {ymin} - {ymax}<br><b>Aantal records:</b> {int(row.n_records)}'
        is_selected = str(row.locatie_id) == selected_loc
        folium.CircleMarker(location=[row.lat, row.lon], radius=8 if is_selected else 6, color='#000000', weight=3 if is_selected else 2, fill=True, fill_color='#000000', fill_opacity=0.12 if is_selected else 0.05, tooltip=tooltip, popup=str(row.locatie_id)).add_to(m)
    return add_chemistry_locations_to_map(m, chem_points)


def get_latest_location_summary(df_loc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    if df_loc.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    latest_year = df_loc['jaar'].dropna().max()
    if pd.isna(latest_year):
        return pd.DataFrame(), pd.DataFrame(), None
    latest = df_loc[df_loc['jaar'] == latest_year].copy()
    hist = df_loc[df_loc['type'] == 'Soort'][['soort', 'jaar']].dropna().drop_duplicates(subset=['soort', 'jaar']).sort_values(['soort', 'jaar'], ascending=[True, False])
    history = hist.groupby('soort', as_index=False)['jaar'].agg(list).rename(columns={'jaar': 'Gemeten in jaren'}) if not hist.empty else pd.DataFrame(columns=['soort', 'Gemeten in jaren'])
    if not history.empty:
        history['Gemeten in jaren'] = history['Gemeten in jaren'].apply(lambda yrs: ', '.join(map(str, yrs)))
    now_species = latest[latest['type'] == 'Soort'][['soort', 'bedekking_pct', 'groeivorm']].copy() if 'type' in latest.columns else pd.DataFrame(columns=['soort', 'bedekking_pct', 'groeivorm'])
    if not now_species.empty:
        now_species['bedekking_pct'] = pd.to_numeric(now_species['bedekking_pct'], errors='coerce').fillna(0.0)
        now_species = now_species.merge(history, on='soort', how='left').sort_values('bedekking_pct', ascending=False)
    return now_species, history, str(int(latest_year))


def build_location_trend_figure(df_loc: pd.DataFrame):
    if df_loc.empty:
        return None
    trend = df_loc.groupby('jaar', as_index=False).agg(totaal_bedekking_locatie=('totaal_bedekking_locatie', 'mean'), doorzicht_m=('doorzicht_m', 'mean'), diepte_m=('diepte_m', 'mean')).sort_values('jaar')
    if trend.empty:
        return None
    max_diepte = pd.to_numeric(trend['diepte_m'], errors='coerce').max(); y2_max = 3.0 if pd.isna(max_diepte) else max(3.0, float(max_diepte) * 1.2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend['jaar'], y=trend['diepte_m'], name='Waterdiepte (m)', fill='tozeroy', mode='none', fillcolor='rgba(200, 230, 255, 0.3)', yaxis='y2'))
    fig.add_trace(go.Bar(x=trend['jaar'], y=trend['totaal_bedekking_locatie'], name='Totale bedekking (%)', marker_color='rgba(34, 139, 34, 0.6)', yaxis='y'))
    fig.add_trace(go.Scatter(x=trend['jaar'], y=trend['doorzicht_m'], name='Doorzicht (m)', mode='lines+markers', line=dict(color='#1E90FF', width=3), marker=dict(size=8), yaxis='y2'))
    fig.update_layout(title='Interactie: vegetatie (staven) vs. waterkolom (lijn/vlak)', xaxis=dict(title='Jaar'), yaxis=dict(title=dict(text='Bedekking (%)', font=dict(color='#228B22')), tickfont=dict(color='#228B22'), range=[0, 105], side='left'), yaxis2=dict(title=dict(text='Meters (Doorzicht / Diepte)', font=dict(color='#1E90FF')), tickfont=dict(color='#1E90FF'), anchor='x', overlaying='y', side='right', range=[0, y2_max]), legend=dict(x=0.01, y=1.1, orientation='h'), hovermode='x unified', height=450)
    return fig


def build_kpis(df_loc: pd.DataFrame) -> dict:
    if df_loc.empty:
        return {'n_jaren': 0, 'mean_diepte': None, 'mean_doorzicht': None}
    return {'n_jaren': int(len(df_loc['jaar'].dropna().unique())), 'mean_diepte': float(pd.to_numeric(df_loc['diepte_m'], errors='coerce').mean()) if 'diepte_m' in df_loc.columns else None, 'mean_doorzicht': float(pd.to_numeric(df_loc['doorzicht_m'], errors='coerce').mean()) if 'doorzicht_m' in df_loc.columns else None}


def get_bathymetry_legend_url() -> str:
    return build_bathymetry_legend_url()


def get_soil_diagnosis(df_loc: pd.DataFrame) -> str:
    if df_loc.empty:
        return 'Geen data voor deze meetlocatie binnen de huidige filters.'
    latest_year = df_loc['jaar'].dropna().max()
    if pd.isna(latest_year):
        return 'Geen jaarinformatie beschikbaar voor deze locatie.'
    return interpret_soil_state(df_loc[df_loc['jaar'] == latest_year].copy())
