
from __future__ import annotations

import pandas as pd

try:
    from waterplanten_app.config.mappings import SOORTGROEP_ORDER, SOORTGROEP_PALETTE
    from waterplanten_app.core.data_access import normalize_locatie_key, to_positive_number
    from waterplanten_app.core.taxonomy import normalize_krw_category
except ImportError:
    from mappings import SOORTGROEP_ORDER, SOORTGROEP_PALETTE
    from data_access import normalize_locatie_key, to_positive_number
    from taxonomy import normalize_krw_category


def _build_soortgroep_color_map(dist) -> tuple[dict[str, str], list[str]]:
    """Bouw een kleurmapping en presentatievolgorde voor soortgroepen op basis van aanwezige categorieën."""
    present = [] if dist is None or getattr(dist, 'empty', True) else [c for c in dist['categorie'].dropna().astype(str).unique().tolist()]
    ordered_present = [c for c in SOORTGROEP_ORDER if c in present]
    remaining = [c for c in present if c not in ordered_present]
    ordered = ordered_present + remaining

    color_map: dict[str, str] = {}
    for idx, category in enumerate(SOORTGROEP_ORDER):
        if idx < len(SOORTGROEP_PALETTE):
            color_map[category] = SOORTGROEP_PALETTE[idx]

    extra_palette = SOORTGROEP_PALETTE[len(SOORTGROEP_ORDER):] + SOORTGROEP_PALETTE[:len(SOORTGROEP_ORDER)]
    for idx, category in enumerate(remaining):
        color_map[category] = extra_palette[idx % len(extra_palette)]

    return color_map, ordered



def _clean_nested_counts(counts_by_loc: dict | None) -> dict:
    cleaned: dict = {}
    for loc, counts in (counts_by_loc or {}).items():
        if not isinstance(counts, dict):
            continue
        loc_counts = {}
        total = 0.0
        for category, value in counts.items():
            num = to_positive_number(value)
            if num > 0:
                loc_counts[str(category)] = num
                total += num
        if total > 0:
            cleaned[loc] = loc_counts
    return cleaned



def _derive_counts_from_distribution(dist) -> dict:
    if dist is None or getattr(dist, 'empty', True):
        return {}
    df = dist.copy()
    loc_col = next((c for c in ['locatie_id', 'locatie', 'location_id', 'location'] if c in df.columns), None)
    cat_col = next((c for c in ['categorie', 'category', 'groep', 'group'] if c in df.columns), None)
    if loc_col is None or cat_col is None:
        return {}
    val_col = next((c for c in ['waarde', 'value', 'aandeel', 'percentage', 'pct', 'count', 'records', 'n', 'bedekking_pct'] if c in df.columns), None)
    if val_col is None:
        grouped = df.dropna(subset=[loc_col, cat_col]).groupby([loc_col, cat_col], observed=True).size().reset_index(name='__n__')
        val_col = '__n__'
    else:
        df[val_col] = df[val_col].apply(to_positive_number)
        grouped = df.dropna(subset=[loc_col, cat_col]).groupby([loc_col, cat_col], observed=True)[val_col].sum().reset_index()
    counts: dict = {}
    for row in grouped.itertuples(index=False):
        loc = str(getattr(row, loc_col))
        cat = str(getattr(row, cat_col))
        val = to_positive_number(getattr(row, val_col))
        if val > 0:
            counts.setdefault(loc, {})[cat] = counts.setdefault(loc, {}).get(cat, 0.0) + val
    return counts



def _filter_map_points_with_positive_counts(map_points, counts_by_loc: dict) -> tuple:
    """Houd alleen locaties over met ten minste één positieve categorie-waarde.

    Matcht locatie-ids robuust op genormaliseerde stringvormen, zodat verschillen zoals
    123 vs '123' vs '123.0' of extra spaties geen kaartpunten laten verdwijnen.

    Belangrijk: gebruik géén getattr(row, '_loc_norm') op itertuples()-rijen; Pandas kan
    kolomnamen met een leidende underscore hernoemen. Daarom werken we hieronder expliciet
    met DataFrame-kolommen.
    """
    if map_points is None or getattr(map_points, 'empty', True):
        return map_points, {}
    if 'locatie_id' not in map_points.columns:
        return map_points.iloc[0:0].copy(), {}

    positive_by_norm: dict[str, dict] = {}
    for loc, counts in (counts_by_loc or {}).items():
        if not isinstance(counts, dict):
            continue
        total = sum(to_positive_number(v) for v in counts.values())
        norm_loc = normalize_locatie_key(loc)
        if total > 0 and norm_loc:
            positive_by_norm[norm_loc] = counts

    if not positive_by_norm:
        return map_points.iloc[0:0].copy(), {}

    tmp = map_points.copy()
    tmp['loc_norm'] = tmp['locatie_id'].map(normalize_locatie_key)
    filtered_points = tmp[tmp['loc_norm'].isin(set(positive_by_norm.keys()))].copy()
    if filtered_points.empty:
        return map_points.iloc[0:0].copy(), {}

    filtered_counts = {}
    unique_pairs = filtered_points[['locatie_id', 'loc_norm']].drop_duplicates()
    for _, pair in unique_pairs.iterrows():
        raw_loc = pair['locatie_id']
        norm_loc = pair['loc_norm']
        if norm_loc in positive_by_norm:
            filtered_counts[raw_loc] = positive_by_norm[norm_loc]
            filtered_counts[str(raw_loc)] = positive_by_norm[norm_loc]
            filtered_counts[normalize_locatie_key(raw_loc)] = positive_by_norm[norm_loc]

    return filtered_points.drop(columns=['loc_norm'], errors='ignore'), filtered_counts



def _prepare_trofie_pie_inputs(result):
    raw_counts = result.metadata.get('counts_by_location', {})
    counts = _clean_nested_counts(raw_counts)
    if not counts:
        counts = _derive_counts_from_distribution(result.metadata.get('distribution_by_location'))
        counts = _clean_nested_counts(counts)
    filtered_points, filtered_counts = _filter_map_points_with_positive_counts(result.map_points, counts)
    return filtered_points, filtered_counts



def _clean_nested_counts_krw(counts_by_loc: dict | None) -> dict:
    cleaned: dict = {}
    for loc, counts in (counts_by_loc or {}).items():
        if not isinstance(counts, dict):
            continue
        loc_counts: dict[str, float] = {}
        total = 0.0
        for category, value in counts.items():
            norm = normalize_krw_category(category)
            if not norm:
                continue
            num = to_positive_number(value)
            if num > 0:
                loc_counts[norm] = loc_counts.get(norm, 0.0) + num
                total += num
        if total > 0:
            cleaned[loc] = loc_counts
    return cleaned



def _derive_krw_counts_from_distribution(dist) -> dict:
    if dist is None or getattr(dist, 'empty', True):
        return {}
    df = dist.copy()
    loc_col = next((c for c in ['locatie_id', 'locatie', 'location_id', 'location'] if c in df.columns), None)
    cat_col = next((c for c in ['categorie', 'category', 'groep', 'group'] if c in df.columns), None)
    if loc_col is None or cat_col is None:
        return {}
    val_col = next((c for c in ['waarde', 'value', 'aandeel', 'percentage', 'pct', 'count', 'records', 'n', 'bedekking_pct'] if c in df.columns), None)
    if val_col is None:
        grouped = df.dropna(subset=[loc_col, cat_col]).groupby([loc_col, cat_col], observed=True).size().reset_index(name='__n__')
        val_col = '__n__'
    else:
        df[val_col] = df[val_col].apply(to_positive_number)
        grouped = df.dropna(subset=[loc_col, cat_col]).groupby([loc_col, cat_col], observed=True)[val_col].sum().reset_index()
    counts: dict = {}
    for row in grouped.itertuples(index=False):
        loc = str(getattr(row, loc_col))
        norm = normalize_krw_category(getattr(row, cat_col))
        if not norm:
            continue
        val = to_positive_number(getattr(row, val_col))
        if val > 0:
            counts.setdefault(loc, {})[norm] = counts.setdefault(loc, {}).get(norm, 0.0) + val
    return counts



def _prepare_krw_pie_inputs(result):
    raw_counts = result.metadata.get('counts_by_location', {})
    counts = _clean_nested_counts_krw(raw_counts)
    if not counts:
        counts = _derive_krw_counts_from_distribution(result.metadata.get('distribution_by_location'))
        counts = _clean_nested_counts_krw(counts)
    filtered_points, filtered_counts = _filter_map_points_with_positive_counts(result.map_points, counts)
    return filtered_points, filtered_counts



def _prepare_soortgroep_pie_inputs(result):
    raw_counts = result.metadata.get('counts_by_location', {})
    counts = _clean_nested_counts(raw_counts)
    if not counts:
        counts = _derive_counts_from_distribution(result.metadata.get('distribution_by_location'))
        counts = _clean_nested_counts(counts)
    filtered_points, filtered_counts = _filter_map_points_with_positive_counts(result.map_points, counts)
    return filtered_points, filtered_counts


__all__ = [
    '_build_soortgroep_color_map',
    '_clean_nested_counts',
    '_derive_counts_from_distribution',
    '_filter_map_points_with_positive_counts',
    '_prepare_trofie_pie_inputs',
    '_clean_nested_counts_krw',
    '_derive_krw_counts_from_distribution',
    '_prepare_krw_pie_inputs',
    '_prepare_soortgroep_pie_inputs',
]
