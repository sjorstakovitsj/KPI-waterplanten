
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_overview_table(df: pd.DataFrame) -> None:
    st.dataframe(
        df[["Waterlichaam", "Bedekking (Totaal %)", "Gem. Doorzicht (m)", "Gem. Diepte (m)", "Soortenrijkdom"]],
        width='stretch',
        hide_index=True,
        column_config={
            "Bedekking (Totaal %)": st.column_config.ProgressColumn(format='%.1f%%', min_value=0, max_value=100),
            "Gem. Doorzicht (m)": st.column_config.NumberColumn(format='%.2f m'),
            "Gem. Diepte (m)": st.column_config.NumberColumn(format='%.2f m'),
            "Soortenrijkdom": st.column_config.NumberColumn(format='%d soorten'),
        },
    )


def render_species_history(now_species: pd.DataFrame, species_history: pd.DataFrame, latest_year: str | None) -> None:
    if not now_species.empty:
        st.dataframe(now_species.style.background_gradient(subset=['bedekking_pct'], cmap='Greens'), width='stretch', hide_index=True)
    else:
        st.info(f"Geen specifieke soorten geregistreerd in {latest_year}." if latest_year else 'Geen soorten beschikbaar.')
    if not species_history.empty:
        st.write('Historisch aangetroffen soorten (niet in laatste jaar):')
        st.dataframe(species_history, width='stretch', hide_index=True)


def render_spatial_table(result) -> None:
    with st.expander(f'Toon data voor {result.coverage_type}'):
        df_table = result.location_table.copy()
        if 'trofieniveau_loc_selected' in df_table.columns and 'trofieniveau_loc' not in df_table.columns:
            df_table = df_table.rename(columns={'trofieniveau_loc_selected': 'trofieniveau_loc'})

        sort_col = 'waarde_veg' if 'waarde_veg' in df_table.columns else ('krw_score_loc' if 'krw_score_loc' in df_table.columns else None)
        if sort_col:
            df_table = df_table.sort_values(sort_col, ascending=(sort_col == 'krw_score_loc'))

        cols = [c for c in ['locatie_id', 'Waterlichaam', 'waarde_veg', 'krw_score_loc', 'trofieniveau_loc', 'diepte_m', 'doorzicht_m'] if c in df_table.columns]

        if result.coverage_type == 'Trofieniveau':
            waarde_cfg = st.column_config.TextColumn('waarde_veg')
        else:
            waarde_cfg = st.column_config.NumberColumn(
                'waarde_veg' if result.coverage_type == 'KRW score' else f'{result.coverage_type} (%)',
                format='%.2f' if result.coverage_type == 'KRW score' else '%.1f%%',
            )

        st.dataframe(
            df_table[cols],
            width='stretch',
            column_config={
                'waarde_veg': waarde_cfg,
                'krw_score_loc': st.column_config.NumberColumn('KRW score (locatie)', format='%.2f'),
                'trofieniveau_loc': st.column_config.TextColumn('Trofieniveau (dominant)'),
                'diepte_m': st.column_config.NumberColumn('Diepte (m)', format='%.2f'),
                'doorzicht_m': st.column_config.NumberColumn('Doorzicht (m)', format='%.2f'),
            },
        )


__all__ = [
    'render_overview_table',
    'render_species_history',
    'render_spatial_table',
]
