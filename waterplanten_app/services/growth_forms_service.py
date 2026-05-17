from __future__ import annotations

import pandas as pd
import plotly.express as px

from waterplanten_app.config.mappings import RWS_GROEIVORM_CODES
from waterplanten_app.core.data_access import load_data
from waterplanten_app.core.diagnostics import interpret_soil_state
from waterplanten_app.core.taxonomy import add_species_group_columns
from waterplanten_app.domain.contracts import DashboardFilters

GROWTH_ORDER = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]
GROWTH_COLORS = {"Ondergedoken": "#2ca02c", "Drijvend": "#1f77b4", "Emergent": "#ff7f0e", "Draadalgen": "#d62728", "FLAB": "#7f7f7f", "Kroos": "#bcbd22"}
PREFERRED_ORDERS = {
    "KRW score": ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)", "Geen match"],
    "Trofieniveau": ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos", "Onbekend", "Geen match"],
    "Soortgroepen": ["chariden", "iseotiden", "parvopotamiden", "magnopotamiden", "myriophylliden", "vallisneriiden", "elodeiden", "stratiotiden", "pepliden", "batrachiiden", "nymphaeiden", "haptofyten", "Overig / Individueel", "Geen match"],
}


def _filtered_df(filters: DashboardFilters) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return df
    out = df[df['Project'].isin(filters.projects)].copy() if filters.projects else df.copy()
    if filters.waterbodies:
        out = out[out['Waterlichaam'].isin(filters.waterbodies)].copy()
    return out


def _year_total_cover_mean(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame(columns=['jaar', 'waarde'])
    samples = df_in.groupby(['jaar', 'CollectieReferentie'], as_index=False)['totaal_bedekking_locatie'].first()
    return samples.groupby('jaar', as_index=False)['totaal_bedekking_locatie'].mean().rename(columns={'totaal_bedekking_locatie': 'waarde'}).sort_values('jaar')


def _year_total_cover_sum(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame(columns=['jaar', 'totaal_jaar'])
    samples = df_in.groupby(['jaar', 'CollectieReferentie'], as_index=False)['totaal_bedekking_locatie'].first()
    return samples.groupby('jaar', as_index=False)['totaal_bedekking_locatie'].sum().rename(columns={'totaal_bedekking_locatie': 'totaal_jaar'}).sort_values('jaar')


def _compute_growth_trend(df_in: pd.DataFrame):
    df_codes = df_in[df_in['soort'].isin(RWS_GROEIVORM_CODES)].copy()
    if not df_codes.empty:
        out = df_codes.groupby(['jaar', 'groeivorm'], as_index=False)['bedekking_pct'].mean().rename(columns={'bedekking_pct': 'waarde'})
        return out, 'Methode: brondata Aquadesk (gemiddelde bedekking per groeivorm en jaar)'
    df_species = df_in[(~df_in['soort'].isin(RWS_GROEIVORM_CODES)) & (df_in['type'] != 'Groep')].copy()
    if df_species.empty:
        return pd.DataFrame(columns=['jaar', 'groeivorm', 'waarde']), 'Geen data beschikbaar voor groeivorm-analyse.'
    out = df_species.groupby(['jaar', 'groeivorm'], as_index=False)['bedekking_pct'].sum().rename(columns={'bedekking_pct': 'waarde'})
    return out, 'Methode: fallback op soortrecords (som bedekking per groeivorm en jaar)'


def _compute_fraction_trend(df_in: pd.DataFrame, trend_mode: str):
    df_species = df_in[(df_in['type'] == 'Soort') & (~df_in['soort'].isin(RWS_GROEIVORM_CODES))].copy()
    if df_species.empty:
        return pd.DataFrame(columns=['jaar', 'categorie', 'fractie']), ''
    denominator = _year_total_cover_sum(df_in)
    if trend_mode == 'Soortgroepen':
        df_h = add_species_group_columns(df_species)
        if 'is_kenmerkende_soort_n2000' in df_h.columns:
            df_h = df_h[~df_h['is_kenmerkende_soort_n2000'].fillna(False)].copy()
        if df_h.empty:
            return pd.DataFrame(columns=['jaar', 'categorie', 'fractie']), ''
        category_col = 'soortgroep_weergave' if 'soortgroep_weergave' in df_h.columns else 'soortgroep'
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ''), category_col] = 'Geen match'
        df_h['waarde_num'] = pd.to_numeric(df_h.get('bedekkingsgraad_proc', df_h['bedekking_pct']), errors='coerce').fillna(0).clip(lower=0)
        caption = 'Aggregatie: som bedekking per soortgroep, gedeeld door de totale bedekking (WATPTN) per jaar.'
    elif trend_mode == 'Trofieniveau':
        df_h = df_species[df_species['Grootheid'].astype(str).eq('BEDKG')].copy() if 'Grootheid' in df_species.columns else df_species.copy()
        if df_h.empty:
            return pd.DataFrame(columns=['jaar', 'categorie', 'fractie']), ''
        category_col = 'categorie'
        df_h[category_col] = df_h.get('trofisch_niveau_weergave', df_h.get('trofisch_niveau', pd.Series(index=df_h.index, dtype='object'))).astype(object)
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ''), category_col] = 'Geen match'
        df_h['waarde_num'] = pd.to_numeric(df_h['bedekking_pct'], errors='coerce').fillna(0).clip(lower=0)
        caption = 'Aggregatie: som bedekking per trofieniveau, gedeeld door de totale bedekking (WATPTN) per jaar.'
    elif trend_mode == 'KRW score':
        df_h = df_species[df_species['Grootheid'].astype(str).eq('BEDKG')].copy() if 'Grootheid' in df_species.columns else df_species.copy()
        if df_h.empty:
            return pd.DataFrame(columns=['jaar', 'categorie', 'fractie']), ''
        fallback = pd.cut(pd.to_numeric(df_h['krw_score'], errors='coerce'), bins=[0, 2, 4, 5], labels=['Gunstig (1-2)', 'Neutraal (3-4)', 'Ongewenst (5)'], include_lowest=True)
        category_col = 'categorie'
        df_h[category_col] = df_h.get('krw_class_weergave', df_h.get('krw_class', fallback)).astype(object)
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ''), category_col] = 'Geen match'
        df_h['waarde_num'] = pd.to_numeric(df_h['bedekking_pct'], errors='coerce').fillna(0).clip(lower=0)
        caption = 'Aggregatie: som bedekking per KRW-klasse, gedeeld door de totale bedekking (WATPTN) per jaar.'
    else:
        df_h = add_species_group_columns(df_species)
        if 'Grootheid' in df_h.columns:
            df_h = df_h[df_h['Grootheid'].astype(str).eq('AANWZHD')].copy()
        elif 'is_kenmerkende_soort_n2000' in df_h.columns:
            df_h = df_h[df_h['is_kenmerkende_soort_n2000'].fillna(False)].copy()
        if df_h.empty:
            return pd.DataFrame(columns=['jaar', 'categorie', 'fractie']), ''
        category_col = 'categorie'
        df_h[category_col] = df_h.get('kenmerkende_soort_n2000_weergave', df_h.get('soort_display', df_h['soort'])).astype(object)
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ''), category_col] = 'Geen match'
        df_h['waarde_num'] = pd.to_numeric(df_h.get('bedekkingsgraad_proc', df_h['bedekking_pct']), errors='coerce').fillna(0).clip(lower=0)
        if float(df_h['waarde_num'].sum()) <= 0:
            df_num = df_h.groupby(['jaar', category_col], as_index=False).size().rename(columns={'size': 'waarde_num'})
            df_den = df_h.groupby('jaar', as_index=False).size().rename(columns={'size': 'totaal_jaar'})
            out = df_num.merge(df_den, on='jaar', how='left')
            out['fractie'] = 0.0
            mask = out['totaal_jaar'].notna() & (out['totaal_jaar'] > 0)
            out.loc[mask, 'fractie'] = out.loc[mask, 'waarde_num'] / out.loc[mask, 'totaal_jaar']
            out = out[['jaar', category_col, 'fractie']].rename(columns={category_col: 'categorie'}).sort_values(['jaar', 'categorie'])
            return out, 'Aggregatie: aandeel per kenmerkende soort binnen alle N2000-waarnemingen per jaar (fallback op recordaantallen).'
        caption = 'Aggregatie: som waarde per kenmerkende soort (N2000), gedeeld door de totale bedekking (WATPTN) per jaar.'
    df_num = df_h.groupby(['jaar', category_col], as_index=False)['waarde_num'].sum()
    out = df_num.merge(denominator, on='jaar', how='left')
    out['fractie'] = 0.0
    mask = out['totaal_jaar'].notna() & (out['totaal_jaar'] > 0)
    out.loc[mask, 'fractie'] = out.loc[mask, 'waarde_num'] / out.loc[mask, 'totaal_jaar']
    out = out[['jaar', category_col, 'fractie']].rename(columns={category_col: 'categorie'}).sort_values(['jaar', 'categorie'])
    return out, caption


def build_trend_figure(filters: DashboardFilters, trend_mode: str):
    df = _filtered_df(filters)
    if df.empty:
        return None, 'Geen data gevonden voor de huidige selectie.', ''
    if trend_mode == 'Totale bedekking':
        trend = _year_total_cover_mean(df)
        if trend.empty:
            return None, 'Geen data beschikbaar voor totale bedekking over de jaren.', ''
        fig = px.line(trend, x='jaar', y='waarde', markers=True, title='Trend totale bedekking over de jaren', labels={'jaar': 'Jaar', 'waarde': 'Gem. totale bedekking (%)'})
        fig.update_layout(height=420)
        return fig, None, 'Aggregatie: gemiddelde totale bedekking per monstername (CollectieReferentie) per jaar.'
    if trend_mode == 'Groeivormen':
        trend, caption = _compute_growth_trend(df)
        if trend.empty:
            return None, 'Geen data beschikbaar voor groeivormen over de jaren.', ''
        fig = px.area(trend, x='jaar', y='waarde', color='groeivorm', category_orders={'groeivorm': GROWTH_ORDER}, color_discrete_map=GROWTH_COLORS, title='Trend in groeivormen over de jaren', labels={'jaar': 'Jaar', 'waarde': 'Bedekking (%)', 'groeivorm': 'Groeivorm'})
        fig.update_layout(height=420, yaxis_title='Bedekking (%)')
        return fig, None, caption
    trend, caption = _compute_fraction_trend(df, trend_mode)
    if trend.empty:
        return None, f'Geen data beschikbaar voor {trend_mode.lower()} over de jaren.', ''
    order = PREFERRED_ORDERS.get(trend_mode)
    if trend_mode == 'Kenmerkende soorten (N2000)':
        cats = [x for x in trend['categorie'].dropna().astype(str).unique().tolist() if x != 'Geen match']
        order = sorted(cats, key=str.lower) + (['Geen match'] if 'Geen match' in set(trend['categorie'].astype(str)) else [])
    fig = px.area(trend, x='jaar', y='fractie', color='categorie', category_orders={'categorie': order} if order else None, title=f'Trend in {trend_mode.lower()} over de jaren', labels={'jaar': 'Jaar', 'fractie': 'Fractie van totale bedekking', 'categorie': trend_mode}, color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=420, yaxis=dict(range=[0, 1], tickformat='.0%'))
    extra = 'Bron trofieniveau-indeling: Verhofstad et al. (2025) – Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang. https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf' if trend_mode == 'Trofieniveau' else ''
    full_caption = caption + ('\n' + extra if extra else '')
    return fig, None, full_caption


def build_species_group_view(filters: DashboardFilters):
    df = _filtered_df(filters)
    raw = df[(~df['soort'].isin(RWS_GROEIVORM_CODES)) & (df['type'] != 'Groep')].copy()
    if raw.empty:
        return None, None, 'Geen soort-specifieke data gevonden (alleen groepscodes aanwezig?).'
    mapped = add_species_group_columns(raw)
    trend = mapped.groupby(['jaar', 'soortgroep'], as_index=False)['bedekkingsgraad_proc'].sum()
    totals = mapped.groupby(['jaar', 'CollectieReferentie'], as_index=False)['totaal_bedekking_locatie'].first().rename(columns={'totaal_bedekking_locatie': 'sample_total'})
    year_totals = totals.groupby('jaar', as_index=False)['sample_total'].sum().rename(columns={'sample_total': 'totaal_bedekking_jaar'})
    trend = trend.merge(year_totals, on='jaar', how='left')
    trend['fractie_tov_totaal'] = 0.0
    mask = trend['totaal_bedekking_jaar'].notna() & (trend['totaal_bedekking_jaar'] > 0)
    trend.loc[mask, 'fractie_tov_totaal'] = trend.loc[mask, 'bedekkingsgraad_proc'] / trend.loc[mask, 'totaal_bedekking_jaar']
    fig = px.bar(trend, x='jaar', y='fractie_tov_totaal', color='soortgroep', title='Samenstelling soortgroepen t.o.v. totale bedekking (WATPTN)', labels={'fractie_tov_totaal': 'Fractie van totale bedekking', 'jaar': 'Jaar', 'soortgroep': 'Groep'}, color_discrete_sequence=px.colors.qualitative.Safe, height=500)
    fig.update_layout(yaxis=dict(range=[0, 1]))
    overig = mapped[mapped['soortgroep'] == 'Overig / Individueel']
    missing = None if overig.empty else overig.groupby('soort', as_index=False).agg(Aantal_Metingen=('bedekkingsgraad_proc', 'count'), Max_Bedekking=('bedekkingsgraad_proc', 'max')).sort_values('Max_Bedekking', ascending=False)
    return fig, missing, None


def get_soil_locations(filters: DashboardFilters) -> list[str]:
    df = _filtered_df(filters)
    if filters.year is None or df.empty:
        return []
    return sorted(df[df['jaar'] == int(filters.year)]['locatie_id'].dropna().astype(str).unique().tolist())


def get_soil_diagnosis(filters: DashboardFilters, location_id: str) -> str:
    df = _filtered_df(filters)
    if filters.year is None or df.empty or not location_id:
        return 'Selecteer een jaar met beschikbare data voor de diagnose.'
    sample = df[(df['jaar'] == int(filters.year)) & (df['locatie_id'].astype(str) == str(location_id))].copy()
    return interpret_soil_state(sample) if not sample.empty else 'Geen diagnose mogelijk voor deze selectie.'
