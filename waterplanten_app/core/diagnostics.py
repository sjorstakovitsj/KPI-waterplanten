from __future__ import annotations

"""Definitieve diagnose-/KPI-helperlaag.

Deze module is de enige bron voor:
- interpret_soil_state()
- categorize_slope_trend()
- calculate_kpi()
- get_location_metric_mean()
- get_color_absolute()
- get_color_diff()

Services horen deze helpers direct uit `waterplanten_app.core.diagnostics`
te importeren. Deze module importeert bewust niet uit pipelines of services.
"""

import pandas as pd


def interpret_soil_state(df_loc: pd.DataFrame) -> str:
    """Genereert automatische tekstinterpretatie van bodemconditie."""
    if df_loc.empty:
        return 'Geen data beschikbaar.'

    total_cover = df_loc['totaal_bedekking_locatie'].mean() if 'totaal_bedekking_locatie' in df_loc.columns else None
    modes = df_loc['groeivorm'].mode() if 'groeivorm' in df_loc.columns else pd.Series(dtype='object')
    dom_type = modes.iloc[0] if not modes.empty else 'Onbekend'

    text = '**Automatische Interpretatie:**\n'
    if pd.isna(total_cover):
        text += 'Geen bedekkingsgegevens beschikbaar.\n'
    elif total_cover < 5:
        text += '⚠️ **Zeer kale bodem** (<5% bedekking).\n'
    elif dom_type == 'Ondergedoken':
        text += f'✅ Goede ontwikkeling (**{total_cover:.0f}%**).\n'
    elif dom_type == 'Drijvend':
        text += f'⚠️ Veel drijfbladplanten (**{total_cover:.0f}%**).\n'
    elif dom_type == 'Draadalgen':
        text += '❌ Dominantie van draadalgen wijst op verstoring.\n'
    return text


def categorize_slope_trend(val, threshold):
    """Bepaalt de trendcategorie op basis van een drempelwaarde."""
    if val > threshold:
        return 'Verbeterend ↗️'
    if val < -threshold:
        return 'Verslechterend ↘️'
    return 'Stabiel ➡️'


def get_location_metric_mean(dataframe: pd.DataFrame, metric_col: str):
    """Gemiddelde van een locatie-parameter via unieke CollectieReferentie."""
    if dataframe.empty or metric_col not in dataframe.columns:
        return 0.0
    if 'CollectieReferentie' not in dataframe.columns:
        return dataframe[metric_col].mean()
    per_sample = dataframe.groupby('CollectieReferentie')[metric_col].first()
    return per_sample.mean()


def calculate_kpi(curr_df: pd.DataFrame, prev_df: pd.DataFrame, metric_col: str, is_loc_metric: bool = False):
    if curr_df.empty:
        return 0.0, 0.0
    if is_loc_metric:
        curr_val = get_location_metric_mean(curr_df, metric_col)
        prev_val = get_location_metric_mean(prev_df, metric_col) if not prev_df.empty else curr_val
    else:
        curr_val = curr_df[metric_col].mean() if metric_col in curr_df.columns else 0.0
        prev_val = prev_df[metric_col].mean() if (not prev_df.empty and metric_col in prev_df.columns) else curr_val
    delta = curr_val - prev_val
    return curr_val, delta


def get_color_absolute(val, min_v, max_v):
    """Geeft RGB kleur terug van rood (laag) naar groen (hoog)."""
    if pd.isna(val):
        return [200, 200, 200, 100]
    norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
    norm = max(0, min(1, norm))
    r = int(255 * (1 - norm))
    g = int(255 * norm)
    b = 0
    return [r, g, b, 200]


def get_color_diff(val):
    """Geeft rood (verslechtering), grijs (stabiel), groen (verbetering)."""
    threshold = 0.5
    if val < -threshold:
        return [255, 0, 0, 200]
    if val > threshold:
        return [0, 255, 0, 200]
    return [128, 128, 128, 100]


__all__ = [
    'interpret_soil_state',
    'categorize_slope_trend',
    'get_location_metric_mean',
    'calculate_kpi',
    'get_color_absolute',
    'get_color_diff',
]
