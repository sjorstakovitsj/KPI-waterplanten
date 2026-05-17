from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterable

import streamlit as st


# -------------------------------
# Robuuste imports / fallbacks
# -------------------------------

def _try_import_constants() -> dict[str, Any]:
    """Laad configuratie/contracten zo robuust mogelijk zonder bestaande functionaliteit te verliezen."""
    ns: dict[str, Any] = {}

    import_attempts = [
        (
            'waterplanten_app.config.mappings',
            ['GROEI_COLORS', 'KRW_COLORS', 'SOORTGROEP_ORDER', 'TOTAL_BEDEKKING_LEGEND', 'TROFIE_COLORS'],
        ),
        (
            'mappings',
            ['GROEI_COLORS', 'KRW_COLORS', 'SOORTGROEP_ORDER', 'TOTAL_BEDEKKING_LEGEND', 'TROFIE_COLORS'],
        ),
    ]
    for module_name, names in import_attempts:
        try:
            module = __import__(module_name, fromlist=names)
            for name in names:
                ns[name] = getattr(module, name)
            break
        except Exception:
            continue

    contract_attempts = [
        (
            'waterplanten_app.domain.contracts',
            [
                'ANALYSIS_LEVEL_INDIVIDUAL_SPECIES',
                'COVERAGE_TYPE_GROEIVORMEN',
                'COVERAGE_TYPE_KRW_SCORE',
                'COVERAGE_TYPE_SOORTGROEPEN',
                'COVERAGE_TYPE_TOTAL_BEDEKKING',
                'COVERAGE_TYPE_TROFIENIVEAU',
                'LAYER_MODE_DIEPTE',
                'LAYER_MODE_VEGETATIE',
                'LegendItem',
            ],
        ),
        (
            'contracts',
            [
                'ANALYSIS_LEVEL_INDIVIDUAL_SPECIES',
                'COVERAGE_TYPE_GROEIVORMEN',
                'COVERAGE_TYPE_KRW_SCORE',
                'COVERAGE_TYPE_SOORTGROEPEN',
                'COVERAGE_TYPE_TOTAL_BEDEKKING',
                'COVERAGE_TYPE_TROFIENIVEAU',
                'LAYER_MODE_DIEPTE',
                'LAYER_MODE_VEGETATIE',
                'LegendItem',
            ],
        ),
    ]
    for module_name, names in contract_attempts:
        try:
            module = __import__(module_name, fromlist=names)
            for name in names:
                ns[name] = getattr(module, name)
            break
        except Exception:
            continue

    pie_attempts = [
        ('waterplanten_app.services.spatial_pie_services', ['_build_soortgroep_color_map']),
        ('spatial_pie_services', ['_build_soortgroep_color_map']),
    ]
    for module_name, names in pie_attempts:
        try:
            module = __import__(module_name, fromlist=names)
            for name in names:
                ns[name] = getattr(module, name)
            break
        except Exception:
            continue

    return ns


_IMPORTED = _try_import_constants()


@dataclass(frozen=True)
class _FallbackLegendItem:
    label: str
    color: str


LegendItem = _IMPORTED.get('LegendItem', _FallbackLegendItem)

ANALYSIS_LEVEL_INDIVIDUAL_SPECIES = _IMPORTED.get('ANALYSIS_LEVEL_INDIVIDUAL_SPECIES', 'Individuele soort')
COVERAGE_TYPE_GROEIVORMEN = _IMPORTED.get('COVERAGE_TYPE_GROEIVORMEN', 'Groeivormen')
COVERAGE_TYPE_KRW_SCORE = _IMPORTED.get('COVERAGE_TYPE_KRW_SCORE', 'KRW score')
COVERAGE_TYPE_SOORTGROEPEN = _IMPORTED.get('COVERAGE_TYPE_SOORTGROEPEN', 'Soortgroepen')
COVERAGE_TYPE_TOTAL_BEDEKKING = _IMPORTED.get('COVERAGE_TYPE_TOTAL_BEDEKKING', 'Totale bedekking')
COVERAGE_TYPE_TROFIENIVEAU = _IMPORTED.get('COVERAGE_TYPE_TROFIENIVEAU', 'Trofieniveau')
LAYER_MODE_DIEPTE = _IMPORTED.get('LAYER_MODE_DIEPTE', 'Diepte')
LAYER_MODE_VEGETATIE = _IMPORTED.get('LAYER_MODE_VEGETATIE', 'Vegetatie')

GROEI_COLORS = _IMPORTED.get(
    'GROEI_COLORS',
    {
        'ondergedoken': '#1f77b4',
        'drijvend': '#17becf',
        'emergent': '#2ca02c',
        'kroos': '#7f7f7f',
    },
)
KRW_COLORS = _IMPORTED.get(
    'KRW_COLORS',
    {
        'slecht': '#d62728',
        'ontoereikend': '#ff7f0e',
        'matig': '#ffd700',
        'goed': '#2ca02c',
    },
)
SOORTGROEP_ORDER = _IMPORTED.get('SOORTGROEP_ORDER', tuple())
TOTAL_BEDEKKING_LEGEND = _IMPORTED.get(
    'TOTAL_BEDEKKING_LEGEND',
    [
        LegendItem('0–5%', '#f7fbff'),
        LegendItem('5–25%', '#c6dbef'),
        LegendItem('25–50%', '#6baed6'),
        LegendItem('50–75%', '#3182bd'),
        LegendItem('75–100%', '#08519c'),
    ],
)
TROFIE_COLORS = _IMPORTED.get(
    'TROFIE_COLORS',
    {
        'oligotroof': '#2ca02c',
        'mesotroof': '#1f77b4',
        'eutroof': '#ff7f0e',
        'sterk eutroof': '#d62728',
        'brak': '#ffd700',
        'marien': '#8c510a',
        'kroos': '#7f7f7f',
        'Onbekend': '#999999',
    },
)


if '_build_soortgroep_color_map' in _IMPORTED:
    _build_soortgroep_color_map = _IMPORTED['_build_soortgroep_color_map']
else:
    def _build_soortgroep_color_map(distribution_by_location):
        palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ]
        labels: list[str] = []
        if isinstance(distribution_by_location, dict):
            seen: set[str] = set()
            for nested in distribution_by_location.values():
                if isinstance(nested, dict):
                    for key in nested:
                        label = str(key)
                        if label not in seen:
                            labels.append(label)
                            seen.add(label)
        ordered = [label for label in SOORTGROEP_ORDER if label in labels] + [label for label in labels if label not in SOORTGROEP_ORDER]
        color_map = {label: palette[i % len(palette)] for i, label in enumerate(ordered)}
        return color_map, ordered


# -------------------------------
# Render helpers
# -------------------------------

def _coerce_legend_item(item: Any) -> LegendItem:
    if isinstance(item, LegendItem):
        return item
    if isinstance(item, tuple) and len(item) >= 2:
        return LegendItem(label=str(item[0]), color=str(item[1]))
    if hasattr(item, 'label') and hasattr(item, 'color'):
        return LegendItem(label=str(item.label), color=str(item.color))
    return LegendItem(label=str(item), color='#999999')


def render_legend_card(title: str, items: Iterable[LegendItem | tuple[str, str] | Any], note: str | None = None) -> None:
    normalized_items = [_coerce_legend_item(item) for item in items]
    items_html = ''.join(
        f"<div class='legend-item'><span class='legend-swatch' style='background:{escape(item.color)};'></span><span>{escape(item.label)}</span></div>"
        for item in normalized_items
    )
    note_html = f"<div class='legend-note'>{escape(note)}</div>" if note else ''
    html = (
        "<style>"
        ".legend-card {border: 1px solid rgba(49, 51, 63, 0.16); border-radius: 12px; padding: 14px 16px; margin: 0.35rem 0 0.75rem 0; background: rgba(250, 250, 250, 0.65);}"
        ".legend-title {font-weight: 700; margin-bottom: 0.7rem; font-size: 1.03rem;}"
        ".legend-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); column-gap: 1.4rem; row-gap: 0.55rem;}"
        ".legend-item {display: flex; align-items: center; gap: 0.55rem; min-height: 1.65rem;}"
        ".legend-swatch {width: 16px; height: 16px; border-radius: 3px; border: 1px solid rgba(49, 51, 63, 0.45); box-sizing: border-box; flex: 0 0 16px;}"
        ".legend-note {margin-top: 0.75rem; color: rgba(49, 51, 63, 0.80); font-size: 0.95rem;}"
        "</style>"
        f"<div class='legend-card'><div class='legend-title'>{escape(title)}</div><div class='legend-grid'>{items_html}</div>{note_html}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_spatial_legend(result) -> None:
    """Behoudt de bestaande legenda-functionaliteit, maar faalt niet hard als velden missen."""
    layer_mode = getattr(result, 'layer_mode', 'Onbekende kaartlaag')
    analysis_level = getattr(result, 'analysis_level', None)
    coverage = getattr(result, 'coverage_type', 'Onbekend')
    metadata = getattr(result, 'metadata', {}) or {}

    st.subheader(f'Kaartweergave: {layer_mode}')

    if layer_mode == LAYER_MODE_VEGETATIE:
        if analysis_level == ANALYSIS_LEVEL_INDIVIDUAL_SPECIES:
            st.info(f'Je bekijkt de verspreiding van de soort: **{coverage}**')
        elif coverage != COVERAGE_TYPE_TOTAL_BEDEKKING:
            st.info(f'Je bekijkt de verspreiding van de groep: **{coverage}**')

        if coverage == COVERAGE_TYPE_TOTAL_BEDEKKING:
            render_legend_card(
                'Legenda totale bedekking',
                TOTAL_BEDEKKING_LEGEND,
                'Deze specifieke kleurschaal geldt alleen voor de kaartmarkeringen van totale bedekking.',
            )
        elif coverage == COVERAGE_TYPE_GROEIVORMEN:
            render_legend_card(
                'Legenda groeivormen',
                list(GROEI_COLORS.items()),
                'De taartdiagrammen behouden de huidige opvulling; de kleuren hieronder tonen de categorieën in de kaart.',
            )
        elif coverage == COVERAGE_TYPE_TROFIENIVEAU:
            render_legend_card(
                'Legenda trofieniveau',
                list(TROFIE_COLORS.items()),
                'Vaste kleurschakering voor trofieniveaus op de kaart.',
            )
        elif coverage == COVERAGE_TYPE_KRW_SCORE:
            render_legend_card(
                'Legenda KRW-score',
                list(KRW_COLORS.items()),
                'Vaste kleurschakering voor de KRW-score op de kaart.',
            )
        elif coverage == COVERAGE_TYPE_SOORTGROEPEN:
            color_map, ordered = _build_soortgroep_color_map(metadata.get('distribution_by_location'))
            if ordered:
                items = [(label, color_map[label]) for label in ordered if label in color_map]
            else:
                items = [(label, color_map[label]) for label in SOORTGROEP_ORDER if label in color_map]
            render_legend_card(
                'Legenda soortgroepen',
                items,
                'De taartdiagrammen behouden de huidige opvulling; de kleuren hieronder tonen de soortgroepen in de kaart.',
            )
    elif layer_mode == LAYER_MODE_DIEPTE:
        st.caption('Legenda: Lichtblauw (Ondiep) → Donkerblauw (Diep)')
    else:
        st.caption('Legenda: Bruin (Troebel) → Groen (Helder)')

    st.caption('Paarse ruitjes op de kaart geven de locaties van chemische metingen weer.')


def render_bathymetry_legend(url: str | None) -> None:
    with st.expander('Legenda bathymetrie', expanded=False):
        if url:
            st.markdown(
                f'<img src="{escape(url)}" alt="Legenda bathymetrie" style="max-width:360px; width:100%; height:auto;"/>',
                unsafe_allow_html=True,
            )
        else:
            st.info('Geen bathymetrie-legenda beschikbaar.')
        st.caption('Legenda uit de WMS-kaartservice van Rijkswaterstaat.')


def render_gradient_legend(label: str, min_val: float, max_val: float) -> None:
    st.markdown(
        f"""
Legenda ({escape(label)}):
{min_val:.1f} (Laag) → {max_val:.1f} (Hoog)
""",
        unsafe_allow_html=True,
    )


__all__ = [
    'render_legend_card',
    'render_spatial_legend',
    'render_bathymetry_legend',
    'render_gradient_legend',
]
