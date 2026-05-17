from __future__ import annotations

"""UI-rendering voor ruimtelijke analyse (package-only)."""

import folium
import pandas as pd
from streamlit_folium import st_folium

from waterplanten_app.config.mappings import (
    CHEM_LOCATION_PREFERENCES,
    CHEM_MARKER_COLOR,
    GROEI_COLORS,
    KRW_COLORS,
    TROFIE_COLORS,
)
from waterplanten_app.config.settings import (
    SPATIAL_ANALYSIS_BASEMAP,
    SPATIAL_PIE_FILL_GAP,
    SPATIAL_PIE_FIXED_TOTAL,
    SPATIAL_PIE_GAP_COLOR,
    SPATIAL_PIE_SIZE_PX,
    SPATIAL_PIE_ZOOM_START,
    WMS_ATTRIBUTION,
    WMS_BASE_URL,
    WMS_LAYER_NAME,
)
from waterplanten_app.core.maps import *
from waterplanten_app.core.chemistry import get_chemistry_location_points
from waterplanten_app.core.diagnostics import get_color_absolute, get_color_diff
from waterplanten_app.domain.contracts import (
    ANALYSIS_LEVEL_GROUPS_AGGREGATIONS,
    ANALYSIS_LEVEL_INDIVIDUAL_SPECIES,
    COVERAGE_TYPE_GROEIVORMEN,
    COVERAGE_TYPE_KRW_SCORE,
    COVERAGE_TYPE_SOORTGROEPEN,
    COVERAGE_TYPE_TOTAL_BEDEKKING,
    COVERAGE_TYPE_TROFIENIVEAU,
    PIE_TYPES,
    PieMapInput,
    LAYER_MODE_DIEPTE,
    LAYER_MODE_DOORZICHT,
)
from waterplanten_app.services.spatial_pie_services import (
    _build_soortgroep_color_map,
    _clean_nested_counts,
    _filter_map_points_with_positive_counts,
    _prepare_krw_pie_inputs,
    _prepare_soortgroep_pie_inputs,
    _prepare_trofie_pie_inputs,
)






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
def _normalize_locatie_key_local(value) -> str:
    """Robuuste normalisatie van locatie-id's voor koppeling tussen punten en count-dicts."""
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    if text.endswith('.0') and text[:-2].lstrip('+-').isdigit():
        text = text[:-2]
    return text


def _counts_total_local(counts) -> float:
    total = 0.0
    if not isinstance(counts, dict):
        return 0.0
    for value in counts.values():
        try:
            num = float(str(value).replace(',', '.').replace('%', '').strip())
        except Exception:
            num = 0.0
        if num > 0:
            total += num
    return total


def _filter_map_points_with_positive_counts_safe(map_points: pd.DataFrame, counts_by_location: dict):
    """Veilige variant zonder afhankelijkheid van een _loc_norm-attribuut in itertuples().

    Pandas hernoemt kolommen die met '_' beginnen bij itertuples(), waardoor getattr(row, '_loc_norm')
    kan falen. Deze implementatie gebruikt expliciet de kolom 'locatie_id' en een genormaliseerde lookup.
    """
    if map_points is None or map_points.empty or not counts_by_location:
        return pd.DataFrame(columns=getattr(map_points, 'columns', [])), {}

    normalized_counts = {}
    for key, counts in (counts_by_location or {}).items():
        norm_key = _normalize_locatie_key_local(key)
        if not norm_key:
            continue
        if _counts_total_local(counts) > 0:
            normalized_counts[norm_key] = counts

    if not normalized_counts:
        return map_points.iloc[0:0].copy(), {}

    filtered = map_points.copy()
    if 'locatie_id' not in filtered.columns:
        return filtered.iloc[0:0].copy(), {}

    filtered['loc_norm'] = filtered['locatie_id'].map(_normalize_locatie_key_local)
    filtered = filtered[filtered['loc_norm'].isin(normalized_counts.keys())].copy()
    filtered = filtered.drop(columns=['loc_norm'], errors='ignore')
    return filtered, normalized_counts



def _first_non_null_local(series):
    for value in series:
        try:
            if pd.notna(value):
                return value
        except Exception:
            if value is not None:
                return value
    return None


def _supplement_groeivormen_zero_locations(
    map_points: pd.DataFrame,
    counts_by_location: dict,
    filtered_base_data: pd.DataFrame | None,
    categories,
):
    """Vul ontbrekende meetlocaties aan voor groeivormen, ook als de bedekking overal 0 is.

    result.map_points en/of counts_by_location kunnen upstream al gefilterd zijn op >0.
    Daarom gebruiken we filtered_base_data als bron voor *alle* gemeten locaties binnen de huidige selectie.
    Voor ontbrekende locaties maken we een nulverdeling aan zodat ze als grijze marker zichtbaar blijven.
    """
    if filtered_base_data is None or filtered_base_data.empty:
        return map_points, counts_by_location

    base = filtered_base_data.copy()
    if 'locatie_id' not in base.columns:
        return map_points, counts_by_location

    for col in ['Waterlichaam', 'lat', 'lon', 'diepte_m', 'doorzicht_m']:
        if col not in base.columns:
            base[col] = pd.NA

    base = base.dropna(subset=['lat', 'lon']).copy()
    if base.empty:
        return map_points, counts_by_location

    base['loc_norm'] = base['locatie_id'].map(_normalize_locatie_key_local)
    base = base[base['loc_norm'] != ''].copy()
    if base.empty:
        return map_points, counts_by_location

    measured_locations = (
        base.groupby('loc_norm', as_index=False)
        .agg(
            locatie_id=('locatie_id', _first_non_null_local),
            Waterlichaam=('Waterlichaam', _first_non_null_local),
            lat=('lat', 'mean'),
            lon=('lon', 'mean'),
            diepte_m=('diepte_m', 'mean'),
            doorzicht_m=('doorzicht_m', 'mean'),
        )
    )

    points = map_points.copy() if map_points is not None else pd.DataFrame()
    if not points.empty and 'locatie_id' in points.columns:
        points['loc_norm'] = points['locatie_id'].map(_normalize_locatie_key_local)
        existing_norms = set(points['loc_norm'].dropna().astype(str))
    else:
        existing_norms = set()

    counts_out = {}
    for key, value in (counts_by_location or {}).items():
        norm_key = _normalize_locatie_key_local(key)
        if norm_key:
            counts_out[norm_key] = value if isinstance(value, dict) else {}

    missing = measured_locations[~measured_locations['loc_norm'].isin(existing_norms)].copy()
    if missing.empty:
        if 'loc_norm' in points.columns:
            points = points.drop(columns=['loc_norm'], errors='ignore')
        return points, counts_out

    zero_template = {str(cat): 0.0 for cat in list(categories or [])}
    for norm_key in missing['loc_norm']:
        counts_out.setdefault(norm_key, zero_template.copy())

    missing = missing.drop(columns=['loc_norm'], errors='ignore')

    if points.empty:
        points = missing.copy()
    else:
        for col in missing.columns:
            if col not in points.columns:
                points[col] = pd.NA
        for col in points.columns:
            if col not in missing.columns:
                missing[col] = pd.NA
        missing = missing[points.columns]
        points = pd.concat([points, missing], ignore_index=True)

    points = points.drop(columns=['loc_norm'], errors='ignore')
    return points, counts_out

def _pie_map(result, coverage: str, filtered_base_data: pd.DataFrame | None = None):
    """
    Bouwt de juiste taartdiagramkaart op basis van het gevraagde coverage-type.
    """
    counts = result.metadata.get("counts_by_location", {})
    dist = result.metadata.get("distribution_by_location")

    if coverage == COVERAGE_TYPE_GROEIVORMEN:
        filtered_points, filtered_counts = _filter_map_points_with_positive_counts_safe(
            result.map_points,
            _clean_nested_counts(counts),
        )
        filtered_points, filtered_counts = _supplement_groeivormen_zero_locations(
            filtered_points,
            filtered_counts,
            filtered_base_data,
            tuple(GROEI_COLORS),
        )
        pie_input = PieMapInput(
            map_points=filtered_points,
            counts_by_location=filtered_counts,
            label="Groeivormen (% bedekking)",
            order=tuple(GROEI_COLORS),
            color_map=GROEI_COLORS,
        )
        return create_pie_map(
            pie_input.map_points,
            counts_by_loc=pie_input.counts_by_location,
            label=pie_input.label,
            color_map=pie_input.color_map,
            order=list(pie_input.order),
            size_px=SPATIAL_PIE_SIZE_PX,
            zoom_start=SPATIAL_PIE_ZOOM_START,
            fixed_total=SPATIAL_PIE_FIXED_TOTAL,
            fill_gap=SPATIAL_PIE_FILL_GAP,
            gap_color=SPATIAL_PIE_GAP_COLOR,
            basemap=SPATIAL_ANALYSIS_BASEMAP,
            show_zero_measured=True,
            zero_label='Geen aangetroffen bedekking (0%)',
        )

    if coverage == COVERAGE_TYPE_TROFIENIVEAU:
        filtered_points, filtered_counts = _prepare_trofie_pie_inputs(result)
        pie_input = PieMapInput(
            map_points=filtered_points,
            counts_by_location=filtered_counts,
            label="Trofieniveau (naar rato)",
            order=tuple(TROFIE_COLORS),
            color_map=TROFIE_COLORS,
        )
        return create_pie_map(
            pie_input.map_points,
            counts_by_loc=pie_input.counts_by_location,
            label=pie_input.label,
            color_map=pie_input.color_map,
            order=list(pie_input.order),
            size_px=SPATIAL_PIE_SIZE_PX,
            zoom_start=SPATIAL_PIE_ZOOM_START,
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    if coverage == COVERAGE_TYPE_KRW_SCORE:
        filtered_points, filtered_counts = _prepare_krw_pie_inputs(result)
        pie_input = PieMapInput(
            map_points=filtered_points,
            counts_by_location=filtered_counts,
            label="KRW-score (naar rato)",
            order=tuple(KRW_COLORS),
            color_map=KRW_COLORS,
        )
        return create_pie_map(
            pie_input.map_points,
            counts_by_loc=pie_input.counts_by_location,
            label=pie_input.label,
            color_map=pie_input.color_map,
            order=list(pie_input.order),
            size_px=SPATIAL_PIE_SIZE_PX,
            zoom_start=SPATIAL_PIE_ZOOM_START,
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    color_map, ordered = _build_soortgroep_color_map(dist)
    filtered_points, filtered_counts = _prepare_soortgroep_pie_inputs(result)
    pie_input = PieMapInput(
        map_points=filtered_points,
        counts_by_location=filtered_counts,
        label="Soortgroepen (% bedekking)",
        order=tuple(ordered),
        color_map=color_map,
    )
    return create_pie_map(
        pie_input.map_points,
        counts_by_loc=pie_input.counts_by_location,
        label=pie_input.label,
        color_map=pie_input.color_map,
        order=list(pie_input.order),
        size_px=SPATIAL_PIE_SIZE_PX,
        zoom_start=SPATIAL_PIE_ZOOM_START,
        fixed_total=SPATIAL_PIE_FIXED_TOTAL,
        fill_gap=SPATIAL_PIE_FILL_GAP,
        gap_color=SPATIAL_PIE_GAP_COLOR,
        basemap=SPATIAL_ANALYSIS_BASEMAP,
    )


def render_spatial_map(result, chem_points, filtered_base_data: pd.DataFrame | None = None) -> None:
    """
    Render de ruimtelijke analysekaart voor de huidige selectie.
    """
    coverage = result.coverage_type
    layer_mode = result.layer_mode
    analysis = result.analysis_level

    if layer_mode in [LAYER_MODE_DIEPTE, LAYER_MODE_DOORZICHT]:
        map_obj = create_map(
            result.map_points.assign(waarde_veg=0.0),
            layer_mode,
            label_veg=coverage,
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    elif analysis == ANALYSIS_LEVEL_GROUPS_AGGREGATIONS and coverage in PIE_TYPES:
        map_obj = _pie_map(result, coverage, filtered_base_data=filtered_base_data)

    elif analysis == ANALYSIS_LEVEL_INDIVIDUAL_SPECIES:
        map_obj = create_map(
            result.map_points,
            "Vegetatie",
            label_veg=coverage,
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    elif coverage == COVERAGE_TYPE_TOTAL_BEDEKKING:
        map_obj = create_map(
            result.map_points,
            "Vegetatie",
            label_veg="totale bedekking",
            value_style="total_bedekking",
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    elif coverage == COVERAGE_TYPE_KRW_SCORE:
        map_obj = create_map(
            result.map_points,
            "Vegetatie",
            label_veg="KRW score",
            value_style="krw",
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    elif coverage == COVERAGE_TYPE_TROFIENIVEAU:
        trofie_series = result.map_points.get("trofieniveau_loc")
        if trofie_series is None:
            trofie_series = pd.Series("Onbekend", index=result.map_points.index)
        else:
            trofie_series = trofie_series.fillna("Onbekend")

        map_obj = create_map(
            result.map_points.assign(trofieniveau_loc=trofie_series),
            mode="Vegetatie",
            label_veg="Trofieniveau (dominant)",
            value_style="categorical",
            category_col="trofieniveau_loc",
            category_color_map={
                "oligotroof": "#2ca02c",
                "mesotroof": "#1f77b4",
                "eutroof": "#ff7f0e",
                "sterk eutroof": "#d62728",
                "brak": "#ffd700",
                "marien": "#8c510a",
                "kroos": "#7f7f7f",
                "Onbekend": "transparent",
            "Geen match": "transparent",
            },
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    else:
        map_obj = create_map(
            result.map_points.assign(waarde_veg=0.0),
            "Vegetatie",
            label_veg=coverage,
            basemap=SPATIAL_ANALYSIS_BASEMAP,
        )

    map_obj = _ensure_osm_with_bathymetry(map_obj)
    st_folium(
        add_chemistry_locations_to_map(map_obj, chem_points),
        height=600,
        width=None,
    )


__all__ = [
    "CHEM_LOCATION_PREFERENCES",
    "CHEM_MARKER_COLOR",
    "WMS_ATTRIBUTION",
    "WMS_BASE_URL",
    "WMS_LAYER_NAME",
    "get_chemistry_location_points",
    "get_color_absolute",
    "get_color_diff",
    "render_spatial_map",
]