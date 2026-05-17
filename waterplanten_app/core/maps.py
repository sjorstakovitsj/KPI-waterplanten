from __future__ import annotations

"""Definitieve kaarthelperlaag.

Deze module is de enige bron voor:
- build_bathymetry_legend_url()
- add_bathymetry_wms()
- create_folium_base_map()
- create_pie_map()
- create_map()
- df_to_geojson_points()
- render_swipe_map_html()
- add_chemistry_locations_to_map()

De module importeert bewust niet uit services. Alleen settings en lage libraries/helpers.
"""

import json
import math
import urllib.parse
from html import escape

import folium
import pandas as pd

from waterplanten_app.config.settings import WMS_ATTRIBUTION, WMS_BASE_URL, WMS_LAYER_NAME


def _is_transparent_color(value) -> bool:
    text = str(value or '').strip().lower()
    if text in {'transparent', 'none', ''}:
        return True
    if text in {'#0000', '#00000000'}:
        return True
    if text.startswith('rgba('):
        try:
            alpha = float(text.rstrip(')').split(',')[-1].strip())
            return alpha <= 0.0
        except Exception:
            return False
    return False


def build_bathymetry_legend_url() -> str:
    """Legend URL voor dezelfde bathymetrie-WMS als in de kaartlagen."""
    params = {
        'SERVICE': 'WMS',
        'REQUEST': 'GetLegendGraphic',
        'VERSION': '1.0.0',
        'FORMAT': 'image/png',
        'LAYER': WMS_LAYER_NAME,
        'STYLE': '',
    }
    return f"{WMS_BASE_URL}?{urllib.parse.urlencode(params)}"


def add_bathymetry_wms(map_obj: folium.Map) -> folium.Map:
    """Voeg de RWS-bathymetrie als basislaag toe aan een Folium-kaart."""
    if map_obj is None:
        return map_obj
    folium.raster_layers.WmsTileLayer(
        url=WMS_BASE_URL,
        name='Bathymetrie IJsselmeergebied',
        layers=WMS_LAYER_NAME,
        fmt='image/png',
        transparent=True,
        version='1.3.0',
        attr=WMS_ATTRIBUTION,
        overlay=False,
        control=True,
        show=True,
    ).add_to(map_obj)
    return map_obj


def create_folium_base_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 10,
    control_scale: bool = True,
    basemap: str = 'default',
) -> folium.Map:
    """Maak een Folium-basiskaart.

    basemap='default' -> bestaand gedrag (OSM)
    basemap='bathymetry' -> geen OSM, wel RWS bathymetrie-WMS
    """
    use_bathymetry = str(basemap).strip().lower() == 'bathymetry'
    if use_bathymetry:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            control_scale=control_scale,
            tiles=None,
        )
        add_bathymetry_wms(m)
        return m
    return folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        control_scale=control_scale,
    )


def get_color_vegetation(value):
    """Rood (0%) -> Groen (100%)."""
    if value == 0:
        return '#d73027'
    elif value <= 5:
        return '#fc8d59'
    elif value <= 15:
        return '#fee08b'
    elif value <= 40:
        return '#d9ef8b'
    elif value <= 75:
        return '#91cf60'
    return '#1a9850'


def get_color_total_bedekking(value):
    """Specifieke kleurindeling voor totale bedekking."""
    if pd.isna(value):
        return 'gray'
    try:
        v = float(value)
    except Exception:
        return 'gray'
    if v <= 0:
        return '#808080'
    elif v < 1:
        return '#006400'
    elif v < 5:
        return '#2ca02c'
    elif v < 15:
        return '#ffd700'
    elif v < 25:
        return '#fdb462'
    elif v < 50:
        return '#ff7f0e'
    elif v < 75:
        return '#d95f02'
    return '#d73027'


def get_color_depth(value):
    """Lichtblauw (ondiep) -> Donkerblauw (diep)."""
    if pd.isna(value):
        return 'gray'
    elif value < 0.5:
        return '#eff3ff'
    elif value < 1.5:
        return '#bdd7e7'
    elif value < 2.5:
        return '#6baed6'
    elif value < 4.0:
        return '#3182bd'
    return '#08519c'


def get_color_transparency(value):
    """Bruin (weinig zicht) -> Groen (veel zicht)."""
    if pd.isna(value):
        return 'gray'
    elif value < 0.5:
        return '#8c510a'
    elif value < 1.0:
        return '#d8b365'
    elif value < 1.5:
        return '#f6e8c3'
    elif value < 2.0:
        return '#c7eae5'
    elif value < 3.0:
        return '#5ab4ac'
    return '#01665e'


def get_color_krw(score):
    """KRW-score 1-5: 1-2 groen, 3-4 oranje, 5 rood."""
    if pd.isna(score):
        return 'gray'
    try:
        s = float(score)
    except Exception:
        return 'gray'
    if s <= 2:
        return '#1a9850'
    elif s <= 4:
        return '#ff7f0e'
    return '#d73027'


def _polar_to_cart(cx, cy, r, angle_rad):
    return (cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad))


def _wedge_path(cx, cy, r, start_angle, end_angle):
    large_arc = 1 if (end_angle - start_angle) > math.pi else 0
    x1, y1 = _polar_to_cart(cx, cy, r, start_angle)
    x2, y2 = _polar_to_cart(cx, cy, r, end_angle)
    return f'M {cx:.2f},{cy:.2f} L {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f} Z'


def pie_svg(
    counts: dict,
    color_map: dict,
    order=None,
    size=30,
    border=1,
    border_color='#333',
    fixed_total=None,
    fill_gap=False,
    gap_color='transparent',
):
    """SVG pie chart helper voor pie markers op de kaart."""
    nonzero = [(k, float(v)) for k, v in counts.items() if v is not None and float(v) > 0]
    if not nonzero:
        r = (size / 2) - border
        cx = cy = size / 2
        return (
            f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#cccccc' stroke='{border_color}' stroke-width='{border}' />"
            f"</svg>"
        )

    r = (size / 2) - border
    cx = cy = size / 2
    denom = float(fixed_total) if fixed_total is not None else sum(v for _, v in nonzero)
    if denom <= 0:
        return (
            f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#cccccc' stroke='{border_color}' stroke-width='{border}' />"
            f"</svg>"
        )

    sum_vals = sum(v for _, v in nonzero)
    if fixed_total is not None:
        if (sum_vals / denom) >= 0.999 and len(nonzero) == 1:
            cat, _ = nonzero[0]
            fill = color_map.get(cat, '#999999')
            stroke_color = 'none' if _is_transparent_color(fill) else border_color
            stroke_width = 0 if _is_transparent_color(fill) else border
            return (
                f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"
                f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{fill}' stroke='{stroke_color}' stroke-width='{stroke_width}' />"
                f"</svg>"
            )
    else:
        if len(nonzero) == 1:
            cat, _ = nonzero[0]
            fill = color_map.get(cat, '#999999')
            stroke_color = 'none' if _is_transparent_color(fill) else border_color
            stroke_width = 0 if _is_transparent_color(fill) else border
            return (
                f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"
                f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{fill}' stroke='{stroke_color}' stroke-width='{stroke_width}' />"
                f"</svg>"
            )

    cats = order if order else list(counts.keys())
    start = -math.pi / 2
    paths = []
    visible_paths = 0
    if fixed_total is not None and fill_gap and gap_color != 'transparent':
        paths.append(f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{gap_color}' />")

    for cat in cats:
        val = float(counts.get(cat, 0) or 0)
        if val <= 0:
            continue
        frac = val / denom
        if frac <= 0:
            continue
        end = start + frac * 2 * math.pi
        color = color_map.get(cat, '#999999')
        d = _wedge_path(cx, cy, r, start, end)
        paths.append(f"<path d='{d}' fill='{color}' />")
        if not _is_transparent_color(color):
            visible_paths += 1
        start = end
        if fixed_total is not None and (start - (-math.pi / 2)) >= 2 * math.pi * 0.999:
            break

    final_stroke_color = border_color if visible_paths > 0 else 'none'
    final_stroke_width = border if visible_paths > 0 else 0
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"
        + ''.join(paths)
        + f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{final_stroke_color}' stroke-width='{final_stroke_width}' />"
        + '</svg>'
    )



def _normalize_locatie_key_for_lookup(value) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    if text.endswith('.0') and text[:-2].lstrip('+-').isdigit():
        text = text[:-2]
    return text


def _pie_counts_total(counts: dict) -> float:
    total = 0.0
    for value in (counts or {}).values():
        try:
            num = float(str(value).replace(',', '.').replace('%', '').strip())
        except Exception:
            num = 0.0
        if num > 0:
            total += num
    return total


def create_pie_map(
    df_locs: pd.DataFrame,
    counts_by_loc: dict,
    label: str,
    color_map: dict,
    order=None,
    size_px: int = 30,
    zoom_start: int = 10,
    fixed_total=None,
    fill_gap=False,
    gap_color='transparent',
    basemap: str = 'default',
    show_zero_measured: bool = False,
    zero_label: str = 'Geen aangetroffen bedekking (0%)',
):
    """Folium kaart met pie-chart markers (SVG via DivIcon) per locatie."""
    if df_locs['lat'].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat, center_lon = df_locs['lat'].mean(), df_locs['lon'].mean()

    m = create_folium_base_map(center_lat, center_lon, zoom_start=zoom_start, control_scale=True, basemap=basemap)

    for row in df_locs.dropna(subset=['lat', 'lon']).itertuples():
        loc_id = getattr(row, 'locatie_id')
        wb = getattr(row, 'Waterlichaam', '')
        counts = counts_by_loc.get(loc_id, counts_by_loc.get(str(loc_id), counts_by_loc.get(_normalize_locatie_key_for_lookup(loc_id), {})))
        if (not counts) and counts_by_loc:
            norm_loc = _normalize_locatie_key_for_lookup(loc_id)
            for key, value in counts_by_loc.items():
                if _normalize_locatie_key_for_lookup(key) == norm_loc:
                    counts = value
                    break
        if counts is None:
            counts = {}
        total_counts = _pie_counts_total(counts)
        if total_counts <= 0 and not show_zero_measured:
            continue
        svg = pie_svg(
            counts,
            color_map=color_map,
            order=order,
            size=size_px,
            fixed_total=fixed_total,
            fill_gap=fill_gap,
            gap_color=gap_color,
        )
        tooltip_keys = list(order) if order else list(counts.keys())
        if not tooltip_keys:
            tooltip_keys = list(color_map.keys())
        parts = []
        for k in tooltip_keys:
            try:
                raw_val = counts.get(k, 0) if isinstance(counts, dict) else 0
                num = float(str(raw_val).replace(',', '.').replace('%', '').strip())
            except Exception:
                num = 0.0
            if num % 1:
                val_txt = f"{num:.2f}"
            else:
                val_txt = str(int(num))
            parts.append(f"{escape(str(k))}: {val_txt}")
        if total_counts <= 0:
            dist_txt = zero_label
            if parts:
                dist_txt += '<br/>' + '<br/>'.join(parts)
        else:
            dist_txt = '<br/>'.join(parts) if parts else 'Geen data'
        diepte = getattr(row, 'diepte_m', float('nan'))
        doorzicht = getattr(row, 'doorzicht_m', float('nan'))
        diepte_txt = 'n.v.t.' if pd.isna(diepte) else f'{diepte:.2f} m'
        doorzicht_txt = 'n.v.t.' if pd.isna(doorzicht) else f'{doorzicht:.2f} m'
        tooltip_html = (
            f"<b>Locatie:</b> {escape(str(loc_id))}<br/>"
            f"<b>Water:</b> {escape(str(wb))}<br/>"
            f"<b>🌊 Diepte:</b> {escape(diepte_txt)}<br/>"
            f"<b>👁️ Doorzicht:</b> {escape(doorzicht_txt)}<br/>"
            f"<b>{escape(label)}:</b><br/>{dist_txt}"
        )
        icon = folium.DivIcon(
            html=f"""
            <div style='width:{size_px}px;height:{size_px}px;transform: translate(-50%, -50%);'>
            {svg}
            </div>
            """
        )
        folium.Marker(
            location=[getattr(row, 'lat'), getattr(row, 'lon')],
            icon=icon,
            tooltip=tooltip_html,
        ).add_to(m)
    return m


def create_map(dataframe, mode, label_veg='Vegetatie', value_style='vegetation', category_col=None, category_color_map=None, basemap: str = 'default'):
    """Genereert een Folium kaart."""
    if dataframe['lat'].isnull().all():
        center_lat, center_lon = 52.5, 5.5
    else:
        center_lat = dataframe['lat'].mean()
        center_lon = dataframe['lon'].mean()

    m = create_folium_base_map(center_lat, center_lon, zoom_start=10, control_scale=True, basemap=basemap)

    for row in dataframe.itertuples():
        radius = 5
        fill_opacity = 0.8

        if mode == 'Vegetatie':
            if value_style == 'categorical' and category_col:
                cat = getattr(row, category_col, None)
                color = (category_color_map or {}).get(cat, '#999999')
                main_line = f'<b>🌱 {label_veg}:</b> {cat}'
                radius = 6
            else:
                val = getattr(row, 'waarde_veg', 0.0)
                if value_style == 'krw':
                    color = get_color_krw(val)
                    main_line = f'<b>🌱 {label_veg}:</b> {val:.2f}'
                    radius = 6
                elif value_style == 'total_bedekking':
                    color = get_color_total_bedekking(val)
                    main_line = f'<b>🌱 {label_veg}:</b> {val:.1f}%'
                    radius = 4 + (min(val, 100) / 100 * 6) if val > 0 else 4
                else:
                    color = get_color_vegetation(val)
                    main_line = f'<b>🌱 {label_veg}:</b> {val:.1f}%'
                    radius = 4 + (min(val, 100) / 100 * 6) if val > 0 else 4
        elif mode == 'Diepte':
            val = getattr(row, 'diepte_m', float('nan'))
            color = get_color_depth(val)
            main_line = f'<b>🌊 Diepte:</b> {val:.2f} m'
        else:
            val = getattr(row, 'doorzicht_m', float('nan'))
            color = get_color_transparency(val)
            main_line = f'<b>👁️ Doorzicht:</b> {val:.2f} m'

        depth_line = f"<b>🌊 Diepte:</b> {getattr(row, 'diepte_m', float('nan')):.2f} m"
        trans_line = f"<b>👁️ Doorzicht:</b> {getattr(row, 'doorzicht_m', float('nan')):.2f} m"
        tooltip_html = (
            f"<b>Locatie:</b> {getattr(row, 'locatie_id', '')}<br>"
            f"<b>Water:</b> {getattr(row, 'Waterlichaam', '')}<br>"
            f"{main_line}<br>"
            f"{depth_line}<br>"
            f"{trans_line}"
        )

        if getattr(row, 'lat') is not None and getattr(row, 'lon') is not None:
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=radius,
                color='#333333',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                tooltip=tooltip_html,
            ).add_to(m)
    return m


def df_to_geojson_points(df: pd.DataFrame, value_col: str, id_col: str = 'locatie_id'):
    """Zet punten (lat/lon) om naar een GeoJSON FeatureCollection."""
    features = []
    for row in df.dropna(subset=['lat', 'lon']).itertuples(index=False):
        props = {
            'locatie_id': getattr(row, id_col),
            'value': float(getattr(row, value_col)) if getattr(row, value_col) is not None else None,
        }
        features.append(
            {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [float(row.lon), float(row.lat)]},
                'properties': props,
            }
        )
    return {'type': 'FeatureCollection', 'features': features}



def render_swipe_map_html(
    geojson_left: dict,
    geojson_right: dict,
    year_left: int,
    year_right: int,
    metric_label: str,
    min_val: float,
    max_val: float,
    center_lat: float,
    center_lon: float,
    zoom: float = 9.0,
    height_px: int = 650,
    bounds=None,
):
    """Rendert een swipe-map met dragbare divider/handle op de kaart zelf."""
    style_url = 'https://tiles.openfreemap.org/styles/liberty'
    left_json = json.dumps(geojson_left)
    right_json = json.dumps(geojson_right)
    bounds_json = 'null' if bounds is None else json.dumps(bounds)

    if max_val == min_val:
        max_val = min_val + 1e-6

    html_template = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <link href='https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css' rel='stylesheet' />
  <script src='https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js'></script>
  <style>
    body { margin: 0; padding: 0; }
    #wrap { position: relative; width: 100%; height: __HEIGHT_PX__px; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 10px rgba(0,0,0,0.08); background: #f7f7f7; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    #map_left { position: absolute; inset: 0; }
    #map_right { position: absolute; inset: 0; clip-path: inset(0 0 0 50%); }
    #divider { position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background: rgba(230,230,230,0.95); box-shadow: 0 0 0 1px rgba(0,0,0,0.08); z-index: 10; cursor: ew-resize; }
    #handle { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 16px; height: 120px; border-radius: 10px; background: rgba(255,255,255,0.95); border: 1px solid rgba(0,0,0,0.15); box-shadow: 0 4px 12px rgba(0,0,0,0.12); z-index: 11; cursor: ew-resize; }
    .year-label { position: absolute; top: 18px; font-size: 44px; font-weight: 700; color: rgba(0,0,0,0.78); text-shadow: 0 1px 0 rgba(255,255,255,0.6); z-index: 12; pointer-events: none; }
    #label_left { left: 30px; opacity: 0.40; }
    #label_right { right: 30px; opacity: 0.95; }
    #legend { position: absolute; left: 50%; bottom: 20px; transform: translateX(-50%); width: 520px; max-width: calc(100% - 40px); background: rgba(255,255,255,0.90); border: 1px solid rgba(0,0,0,0.12); border-radius: 12px; padding: 12px 14px; z-index: 12; backdrop-filter: blur(3px); }
    #legend .title { text-align: center; font-size: 22px; font-weight: 700; margin-bottom: 8px; }
    #legend .bar { height: 14px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.12); background: linear-gradient(90deg, #d73027 0%, #fee08b 50%, #1a9850 100%); }
    #legend .labels { display: flex; justify-content: space-between; margin-top: 8px; font-size: 16px; font-weight: 650; color: rgba(0,0,0,0.80); }
    #legend .sub { text-align: center; margin-top: 3px; font-size: 14px; color: rgba(0,0,0,0.60); font-weight: 600; }
  </style>
</head>
<body>
  <div id='wrap'>
    <div id='map_left'></div>
    <div id='map_right'></div>
    <div id='divider'></div>
    <div id='handle'></div>
    <div id='label_left' class='year-label'>__YEAR_LEFT__</div>
    <div id='label_right' class='year-label'>__YEAR_RIGHT__</div>
    <div id='legend'>
      <div class='title'>__METRIC_LABEL__</div>
      <div class='bar'></div>
      <div class='labels'><span>__MIN_VAL__</span><span>__MAX_VAL__</span></div>
      <div class='sub'>Laag → Hoog</div>
    </div>
  </div>

  <script>
    const styleUrl = __STYLE_URL__;
    const leftData = __LEFT_JSON__;
    const rightData = __RIGHT_JSON__;
    const minVal = __MIN_VAL_RAW__;
    const maxVal = __MAX_VAL_RAW__;
    const bounds = __BOUNDS_JSON__;

    const mapLeft = new maplibregl.Map({
      container: 'map_left',
      style: styleUrl,
      center: [__CENTER_LON__, __CENTER_LAT__],
      zoom: __ZOOM__,
      attributionControl: true
    });

    const mapRight = new maplibregl.Map({
      container: 'map_right',
      style: styleUrl,
      center: [__CENTER_LON__, __CENTER_LAT__],
      zoom: __ZOOM__,
      attributionControl: true,
      interactive: true
    });

    mapRight.scrollZoom.disable();
    mapRight.boxZoom.disable();
    mapRight.dragRotate.disable();
    mapRight.dragPan.disable();
    mapRight.keyboard.disable();
    mapRight.doubleClickZoom.disable();
    mapRight.touchZoomRotate.disable();

    function sync() {
      const c = mapLeft.getCenter();
      mapRight.jumpTo({
        center: c,
        zoom: mapLeft.getZoom(),
        bearing: mapLeft.getBearing(),
        pitch: mapLeft.getPitch()
      });
    }

    mapLeft.on('move', sync);
    mapLeft.on('moveend', sync);

    function applyBasemapGray(map) {
      const style = map.getStyle();
      const layers = (style && style.layers) ? style.layers : [];

      const GRAY_BG = '#e7e7e7';
      const GRAY_FILL = '#cdcdcd';
      const GRAY_LINE = '#9c9c9c';
      const GRAY_TEXT = '#666666';
      const GRAY_HALO = '#f2f2f2';

      layers.forEach((ly) => {
        if (!ly || !ly.id) return;
        try {
          switch (ly.type) {
            case 'background':
              map.setPaintProperty(ly.id, 'background-color', GRAY_BG);
              break;
            case 'fill':
              try { map.setPaintProperty(ly.id, 'fill-pattern', null); } catch (e) {}
              map.setPaintProperty(ly.id, 'fill-color', GRAY_FILL);
              try { map.setPaintProperty(ly.id, 'fill-outline-color', '#b0b0b0'); } catch (e) {}
              break;
            case 'fill-extrusion':
              map.setPaintProperty(ly.id, 'fill-extrusion-color', GRAY_FILL);
              break;
            case 'line':
              try { map.setPaintProperty(ly.id, 'line-pattern', null); } catch (e) {}
              map.setPaintProperty(ly.id, 'line-color', GRAY_LINE);
              break;
            case 'symbol':
              try { map.setPaintProperty(ly.id, 'text-color', GRAY_TEXT); } catch (e) {}
              try { map.setPaintProperty(ly.id, 'text-halo-color', GRAY_HALO); } catch (e) {}
              try { map.setPaintProperty(ly.id, 'icon-color', GRAY_TEXT); } catch (e) {}
              try { map.setPaintProperty(ly.id, 'text-opacity', 0.85); } catch (e) {}
              break;
            case 'circle':
              map.setPaintProperty(ly.id, 'circle-color', GRAY_LINE);
              try { map.setPaintProperty(ly.id, 'circle-opacity', 0.35); } catch (e) {}
              break;
            case 'heatmap':
              try { map.setPaintProperty(ly.id, 'heatmap-opacity', 0.25); } catch (e) {}
              break;
            case 'raster':
              try { map.setPaintProperty(ly.id, 'raster-saturation', -1); } catch (e) {}
              break;
            default:
              break;
          }
        } catch (e) {}
      });
    }

    function addPoints(map, sourceName, layerName, data, dim = false) {
      if (map.getSource(sourceName)) {
        map.getSource(sourceName).setData(data);
        return;
      }

      map.addSource(sourceName, {
        type: 'geojson',
        data: data
      });

      map.addLayer({
        id: layerName,
        type: 'circle',
        source: sourceName,
        paint: {
          'circle-radius': 6,
          'circle-stroke-color': dim ? 'rgba(40,40,40,0.35)' : 'rgba(40,40,40,0.75)',
          'circle-stroke-width': 1,
          'circle-opacity': dim ? 0.45 : 0.85,
          'circle-color': [
            'interpolate',
            ['linear'],
            ['get', 'value'],
            minVal,
            '#d73027',
            (minVal + maxVal) / 2.0,
            '#fee08b',
            maxVal,
            '#1a9850'
          ]
        }
      });
    }

    mapLeft.on('load', () => {
      applyBasemapGray(mapLeft);
      addPoints(mapLeft, 'leftPts', 'leftLayer', leftData, true);

      if (bounds && bounds.length === 4) {
        const sw = [bounds[0], bounds[1]];
        const ne = [bounds[2], bounds[3]];
        mapLeft.fitBounds([sw, ne], {
          padding: 70,
          maxZoom: 12,
          duration: 0
        });
      }

      const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      mapLeft.on('mousemove', 'leftLayer', (e) => {
        mapLeft.getCanvas().style.cursor = 'pointer';
        if (!e.features || !e.features.length) return;

        const p = e.features[0].properties;
        const v =
          p.value === null ||
          p.value === undefined ||
          p.value === '' ||
          isNaN(Number(p.value))
            ? 'n.v.t.'
            : Number(p.value).toFixed(1);

        popup
          .setLngLat(e.lngLat)
          .setHTML(
            '<b>Locatie:</b> ' +
            p.locatie_id +
            '<br/><b>Waarde:</b> ' +
            v
          )
          .addTo(mapLeft);
      });

      mapLeft.on('mouseleave', 'leftLayer', () => {
        mapLeft.getCanvas().style.cursor = '';
        popup.remove();
      });
    });

    mapRight.on('load', () => {
      addPoints(mapRight, 'rightPts', 'rightLayer', rightData, false);

      const popupR = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      mapRight.on('mousemove', 'rightLayer', (e) => {
        mapRight.getCanvas().style.cursor = 'pointer';
        if (!e.features || !e.features.length) return;

        const p = e.features[0].properties;
        const v =
          p.value === null ||
          p.value === undefined ||
          p.value === '' ||
          isNaN(Number(p.value))
            ? 'n.v.t.'
            : Number(p.value).toFixed(1);

        popupR
          .setLngLat(e.lngLat)
          .setHTML(
            '<b>Locatie:</b> ' +
            p.locatie_id +
            '<br/><b>Waarde (' + __YEAR_RIGHT_JS__ + '):</b> ' +
            v
          )
          .addTo(mapRight);
      });

      mapRight.on('mouseleave', 'rightLayer', () => {
        mapRight.getCanvas().style.cursor = '';
        popupR.remove();
      });
    });

    const wrap = document.getElementById('wrap');
    const mapRightDiv = document.getElementById('map_right');
    const divider = document.getElementById('divider');
    const handle = document.getElementById('handle');
    let isDragging = false;

    function setSwipe(p) {
      const pct = Math.max(0, Math.min(1, p));
      const x = pct * wrap.clientWidth;
      divider.style.left = x + 'px';
      handle.style.left = x + 'px';
      mapRightDiv.style.clipPath = 'inset(0 0 0 ' + (pct * 100).toFixed(2) + '%)';
    }

    function pointerToPct(clientX) {
      const rect = wrap.getBoundingClientRect();
      return (clientX - rect.left) / rect.width;
    }

    function onDown(e) {
      isDragging = true;
      const x = e.touches ? e.touches[0].clientX : e.clientX;
      setSwipe(pointerToPct(x));
      e.preventDefault();
    }

    function onMove(e) {
      if (!isDragging) return;
      const x = e.touches ? e.touches[0].clientX : e.clientX;
      setSwipe(pointerToPct(x));
      e.preventDefault();
    }

    function onUp() {
      isDragging = false;
    }

    divider.addEventListener('mousedown', onDown);
    handle.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    divider.addEventListener('touchstart', onDown, { passive: false });
    handle.addEventListener('touchstart', onDown, { passive: false });
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onUp);

    setSwipe(0.5);
  </script>
</body>
</html>
"""

    html_str = (
        html_template
        .replace('__HEIGHT_PX__', str(int(height_px)))
        .replace('__YEAR_LEFT__', escape(str(year_left)))
        .replace('__YEAR_RIGHT__', escape(str(year_right)))
        .replace('__METRIC_LABEL__', escape(str(metric_label)))
        .replace('__MIN_VAL__', f'{min_val:.1f}')
        .replace('__MAX_VAL__', f'{max_val:.1f}')
        .replace('__STYLE_URL__', json.dumps(style_url))
        .replace('__LEFT_JSON__', left_json)
        .replace('__RIGHT_JSON__', right_json)
        .replace('__MIN_VAL_RAW__', repr(float(min_val)))
        .replace('__MAX_VAL_RAW__', repr(float(max_val)))
        .replace('__BOUNDS_JSON__', bounds_json)
        .replace('__CENTER_LAT__', repr(float(center_lat)))
        .replace('__CENTER_LON__', repr(float(center_lon)))
        .replace('__ZOOM__', repr(float(zoom)))
        .replace('__YEAR_RIGHT_JS__', json.dumps(str(year_right)))
    )
    return html_str

def add_chemistry_locations_to_map(
    map_obj: folium.Map,
    df_points: pd.DataFrame | None,
    color: str = '#8e44ad',
    label: str = 'Chemische meetlocatie',
    radius: int = 8,
) -> folium.Map:
    """Voeg chemische meetlocaties toe als paarse ruitjes aan een Folium-kaart."""
    if map_obj is None or df_points is None or df_points.empty:
        return map_obj
    for row in df_points.dropna(subset=['chem_lat', 'chem_lon']).itertuples(index=False):
        loc_code = getattr(row, 'locatie_code', '')
        meetnaam = getattr(row, 'meetlocatie_naam', loc_code) or loc_code
        n_records = getattr(row, 'n_records', None)
        n_parameters = getattr(row, 'n_parameters', None)
        parts = [f"<b>{escape(label)}:</b> {escape(str(meetnaam))}"]
        if str(meetnaam) != str(loc_code) and str(loc_code).strip():
            parts.append(f"<b>Code:</b> {escape(str(loc_code))}")
        if n_records is not None and pd.notna(n_records):
            parts.append(f"<b>Records:</b> {int(n_records)}")
        if n_parameters is not None and pd.notna(n_parameters):
            parts.append(f"<b>Parameters:</b> {int(n_parameters)}")
        tooltip_html = '<br/>'.join(parts)
        folium.RegularPolygonMarker(
            location=[float(getattr(row, 'chem_lat')), float(getattr(row, 'chem_lon'))],
            number_of_sides=4,
            rotation=45,
            radius=radius,
            color='#5b2c6f',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=tooltip_html,
        ).add_to(map_obj)
    return map_obj


__all__ = [
    'WMS_BASE_URL',
    'WMS_LAYER_NAME',
    'WMS_ATTRIBUTION',
    'build_bathymetry_legend_url',
    'add_bathymetry_wms',
    'create_folium_base_map',
    'create_pie_map',
    'create_map',
    'df_to_geojson_points',
    'render_swipe_map_html',
    'add_chemistry_locations_to_map',
]
