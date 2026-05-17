
from __future__ import annotations

import pandas as pd

from waterplanten_app.domain.contracts import (
    COVERAGE_TYPE_GROEIVORMEN,
    COVERAGE_TYPE_KRW_SCORE,
    COVERAGE_TYPE_SOORTGROEPEN,
    COVERAGE_TYPE_TOTAL_BEDEKKING,
    COVERAGE_TYPE_TROFIENIVEAU,
    DashboardFilters,
    LegendItem,
    LAYER_MODE_DIEPTE,
    PieMapInput,
    SpatialPageState,
)
from waterplanten_app.services.spatial_service import build_spatial_result

try:
    from waterplanten_app.config.mappings import (
        GROEI_COLORS,
        KRW_COLORS,
        PIE_TYPES,
        SOORTGROEP_ORDER,
        TOTAL_BEDEKKING_LEGEND,
        TROFIE_COLORS,
    )
    from waterplanten_app.core.data_access import load_chemistry_data, load_data
    from waterplanten_app.core.taxonomy import get_sorted_species_list
    from waterplanten_app.repositories.spatial_repository import get_chemistry_location_points
    from waterplanten_app.services.spatial_pie_services import (
        _build_soortgroep_color_map,
        _clean_nested_counts,
        _filter_map_points_with_positive_counts,
        _prepare_krw_pie_inputs,
        _prepare_soortgroep_pie_inputs,
        _prepare_trofie_pie_inputs,
    )
except ImportError:
    from mappings import GROEI_COLORS, KRW_COLORS, PIE_TYPES, SOORTGROEP_ORDER, TOTAL_BEDEKKING_LEGEND, TROFIE_COLORS
    from data_access import load_chemistry_data, load_data
    from taxonomy import get_sorted_species_list
    from spatial_repository import get_chemistry_location_points
    from spatial_pie_services import (
        _build_soortgroep_color_map,
        _clean_nested_counts,
        _filter_map_points_with_positive_counts,
        _prepare_krw_pie_inputs,
        _prepare_soortgroep_pie_inputs,
        _prepare_trofie_pie_inputs,
    )




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

def load_spatial_base_data() -> pd.DataFrame:
    """Laad de basisdataset voor de pagina Ruimtelijke analyse."""
    return load_data()



def prepare_spatial_filter_context(df: pd.DataFrame, year: int, projects: tuple[str, ...]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Valideer filters en geef gefilterde basisdata + soortopties terug."""
    if df is None or df.empty:
        return pd.DataFrame(), tuple()
    filtered = df[(df['jaar'].astype(str) == str(year)) & (df['Project'].isin(projects))].copy()
    if filtered.empty:
        return filtered, tuple()
    species_options = tuple(get_sorted_species_list(filtered))
    return filtered, species_options



def _build_legend_spec(result) -> tuple[str | None, tuple[LegendItem, ...], str | None]:
    layer_mode = result.layer_mode
    coverage = result.coverage_type

    if layer_mode == LAYER_MODE_DIEPTE:
        return None, tuple(), 'Legenda: Lichtblauw (Ondiep) → Donkerblauw (Diep)'
    if layer_mode not in {'Vegetatie', 'Doorzicht'} and layer_mode != 'Vegetatie':
        return None, tuple(), 'Legenda: Bruin (Troebel) → Groen (Helder)'
    if layer_mode == 'Doorzicht':
        return None, tuple(), 'Legenda: Bruin (Troebel) → Groen (Helder)'

    if coverage == COVERAGE_TYPE_TOTAL_BEDEKKING:
        return (
            'Legenda totale bedekking',
            tuple(LegendItem(label=label, color=color) for label, color in TOTAL_BEDEKKING_LEGEND),
            'Deze specifieke kleurschaal geldt alleen voor de kaartmarkeringen van totale bedekking.',
        )
    if coverage == COVERAGE_TYPE_GROEIVORMEN:
        return (
            'Legenda groeivormen',
            tuple(LegendItem(label=label, color=color) for label, color in GROEI_COLORS.items()),
            'De taartdiagrammen behouden de huidige opvulling; de kleuren hieronder tonen de categorieën in de kaart.',
        )
    if coverage == COVERAGE_TYPE_TROFIENIVEAU:
        return (
            'Legenda trofieniveau',
            tuple(LegendItem(label=label, color=color) for label, color in TROFIE_COLORS.items()),
            'Vaste kleurschakering voor trofieniveaus op de kaart.',
        )
    if coverage == COVERAGE_TYPE_KRW_SCORE:
        return (
            'Legenda KRW-score',
            tuple(LegendItem(label=label, color=color) for label, color in KRW_COLORS.items()),
            'Vaste kleurschakering voor de KRW-score op de kaart.',
        )
    if coverage == COVERAGE_TYPE_SOORTGROEPEN:
        color_map, _ = _build_soortgroep_color_map(result.metadata.get('distribution_by_location'))
        items = tuple(LegendItem(label=label, color=color_map[label]) for label in SOORTGROEP_ORDER if label in color_map)
        return (
            'Legenda soortgroepen',
            items,
            'De taartdiagrammen behouden de huidige opvulling; de kleuren hieronder tonen de soortgroepen in de kaart.',
        )
    return None, tuple(), None



def _prepare_pie_input(result, filtered_base_data: pd.DataFrame | None = None) -> PieMapInput | None:
    coverage = result.coverage_type
    counts = result.metadata.get('counts_by_location', {})
    dist = result.metadata.get('distribution_by_location')

    if coverage == COVERAGE_TYPE_GROEIVORMEN:
        filtered_points, filtered_counts = _filter_map_points_with_positive_counts_safe(result.map_points, _clean_nested_counts(counts))
        filtered_points, filtered_counts = _supplement_groeivormen_zero_locations(filtered_points, filtered_counts, filtered_base_data, tuple(GROEI_COLORS))
        return PieMapInput(map_points=filtered_points, counts_by_location=filtered_counts, label='Groeivormen (% bedekking)', order=tuple(GROEI_COLORS), color_map=GROEI_COLORS)
    if coverage == COVERAGE_TYPE_TROFIENIVEAU:
        filtered_points, filtered_counts = _prepare_trofie_pie_inputs(result)
        return PieMapInput(map_points=filtered_points, counts_by_location=filtered_counts, label='Trofieniveau (naar rato)', order=tuple(TROFIE_COLORS), color_map=TROFIE_COLORS)
    if coverage == COVERAGE_TYPE_KRW_SCORE:
        filtered_points, filtered_counts = _prepare_krw_pie_inputs(result)
        return PieMapInput(map_points=filtered_points, counts_by_location=filtered_counts, label='KRW-score (naar rato)', order=tuple(KRW_COLORS), color_map=KRW_COLORS)
    if coverage == COVERAGE_TYPE_SOORTGROEPEN:
        color_map, ordered = _build_soortgroep_color_map(dist)
        filtered_points, filtered_counts = _prepare_soortgroep_pie_inputs(result)
        return PieMapInput(map_points=filtered_points, counts_by_location=filtered_counts, label='Soortgroepen (% bedekking)', order=tuple(ordered), color_map=color_map)
    return None



def build_spatial_page_state(df: pd.DataFrame, year: int, projects: tuple[str, ...], analysis: str, coverage: str, layer: str) -> SpatialPageState:
    """Orkestreer de paginaflow voor Ruimtelijke analyse en geef een compacte viewmodel/state terug.

    Taken:
    1. filters valideren,
    2. basisdata ophalen/filteren,
    3. chemistry-points ophalen,
    4. build_spatial_result(...) uitvoeren,
    5. pie-inputs / legend specs voorbereiden,
    6. een compacte SpatialPageState teruggeven.
    """
    filtered_base_data, species_options = prepare_spatial_filter_context(df, year, projects)
    filters = DashboardFilters(year=year, projects=projects, analysis_level=analysis, coverage_type=coverage, layer_mode=layer)
    result = build_spatial_result(filters, analysis, coverage, layer)
    chem_points = get_chemistry_location_points(df_chem=load_chemistry_data())
    legend_title, legend_items, legend_note = _build_legend_spec(result)
    pie_input = _prepare_pie_input(result, filtered_base_data=filtered_base_data) if coverage in PIE_TYPES else None
    metadata = {
        'n_filtered_rows': int(len(filtered_base_data)),
        'has_pie_input': pie_input is not None,
        'pie_dimension': coverage if coverage in PIE_TYPES else None,
    }
    return SpatialPageState(
        result=result,
        chem_points=chem_points,
        selected_filters=filters,
        filtered_base_data=filtered_base_data,
        species_options=species_options,
        legend_title=legend_title,
        legend_items=legend_items,
        legend_note=legend_note,
        pie_input=pie_input,
        metadata=metadata,
    )


__all__ = [
    'load_spatial_base_data',
    'prepare_spatial_filter_context',
    'build_spatial_page_state',
]
