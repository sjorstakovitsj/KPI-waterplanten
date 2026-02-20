import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils import load_data, interpret_soil_state, add_species_group_columns, RWS_GROEIVORM_CODES

st.set_page_config(layout="wide", page_title="Groeivormen & Bodem")
st.title("🌱 Groeivormen en soortgroepen")
st.markdown("Analyse van vegetaties. Boven: functionele groeivormen. Onder: taxonomische soortgroepen.")

# --- 1. DATA INLADEN ---
df_raw = load_data()
if df_raw.empty:
    st.error("Geen data geladen. Controleer het bronbestand.")
    st.stop()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("Filters")

all_years = sorted(df_raw["jaar"].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Selecteer meetjaar", all_years)

all_projects = sorted(df_raw["Project"].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)",
    options=all_projects,
    default=all_projects,
)

# Waterlichaam filter beperkt tot gekozen projecten (zoals in jouw code)
available_bodies = sorted(df_raw[df_raw["Project"].isin(selected_projects)]["Waterlichaam"].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam / waterlichamen",
    options=available_bodies,
    default=available_bodies,
)

# --- 3. FILTER TOEPASSEN ---
df_filtered = df_raw[
    (df_raw["Project"].isin(selected_projects)) &
    (df_raw["Waterlichaam"].isin(selected_bodies))
].copy()

if df_filtered.empty:
    st.warning("Geen data gevonden voor de huidige selectie.")
    st.stop()

# --- 4. GROEIVORMEN (BOVENSTE GRAFIEKEN) ---
df_rws_codes = df_filtered[df_filtered["soort"].isin(RWS_GROEIVORM_CODES)].copy()

use_aggregated_species = False
df_trend_growth = pd.DataFrame()

if not df_rws_codes.empty:
    # SCENARIO 1: RWS-codes aanwezig -> mean bedekking per groeivorm per jaar
    df_trend_growth = (
        df_rws_codes
        .groupby(["jaar", "groeivorm"], as_index=False)["bedekking_pct"]
        .mean()
    )
    source_label = "Bron: ruwe data Aquadesk"
    df_radar_source = df_rws_codes[df_rws_codes["jaar"] == selected_year].copy()
else:
    # SCENARIO 2: geen RWS-codes -> som van soorten (geen 'Groep' en geen RWS codes)
    df_species_only = df_filtered[
        (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES)) &
        (df_filtered["type"] != "Groep")
    ].copy()

    if df_species_only.empty:
        st.error("Geen data beschikbaar voor groeivorm-analyse (noch codes, noch soorten).")
        st.stop()

    use_aggregated_species = True
    source_label = "Berekend: Som van soorten"

    df_trend_growth = (
        df_species_only
        .groupby(["jaar", "groeivorm"], as_index=False)["bedekking_pct"]
        .sum()
    )
    df_radar_source = df_species_only[df_species_only["jaar"] == selected_year].copy()

# --- 5. VISUALISATIE: GROEIVORMEN & RADAR ---
c1, c2 = st.columns([2, 1])

GROWTH_ORDER = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]

with c1:
    st.subheader("Trend in groeivormen")
    st.caption(f"Methode: {source_label}")

    if not df_trend_growth.empty:
        fig_area = px.area(
            df_trend_growth,
            x="jaar",
            y="bedekking_pct",
            color="groeivorm",
            category_orders={"groeivorm": GROWTH_ORDER},
            color_discrete_map={
                "Ondergedoken": "#2ca02c",
                "Drijvend": "#1f77b4",
                "Emergent": "#ff7f0e",
                "Draadalgen": "#d62728",
                "FLAB": "#7f7f7f",
                "Kroos": "#bcbd22",
            },
        )
        fig_area.update_layout(yaxis_title="Bedekking (%)", xaxis_title="Jaar", height=400)
        st.plotly_chart(fig_area, use_container_width=True)

with c2:
    st.subheader(f"Profiel {selected_year}")

    radar_mode = st.selectbox(
        "Kies spingrafiek voor",
        ["Groeivormen", "Soortgroepen", "Trofieniveau", "KRW score"],
        index=0,
        key="radar_mode_choice",
    )

    def _normalize_series(s: pd.Series) -> pd.Series:
        s = s.fillna(0.0)
        total = float(s.sum())
        return (s / total) if total > 0 else s

    current_dist = pd.Series(dtype=float)
    categories = []
    ref_vals = None

    if radar_mode == "Groeivormen":
        if df_radar_source.empty:
            st.info(f"Geen data beschikbaar voor radarplot in {selected_year}")
            categories = GROWTH_ORDER
            ref_vals = [0] * len(categories)
        else:
            if use_aggregated_species:
                current_dist = df_radar_source.groupby("groeivorm")["bedekking_pct"].sum()
            else:
                current_dist = df_radar_source.groupby("groeivorm")["bedekking_pct"].mean()

            current_dist = _normalize_series(current_dist)
            categories = GROWTH_ORDER

            ref_dict = {"Ondergedoken": 0.6, "Drijvend": 0.2, "Emergent": 0.15, "Draadalgen": 0.05}
            ref_vals = [ref_dict.get(c, 0) for c in categories]

    elif radar_mode == "Soortgroepen":
        df_species_raw = df_filtered[~df_filtered["soort"].isin(RWS_GROEIVORM_CODES)].copy()
        df_species_raw = df_species_raw[df_species_raw["type"] != "Groep"].copy()
        df_species_raw = df_species_raw[df_species_raw["jaar"] == selected_year].copy()

        if df_species_raw.empty:
            st.info(f"Geen soortdata beschikbaar voor soortgroepen in {selected_year}.")
        else:
            df_species_mapped = add_species_group_columns(df_species_raw)
            current_dist = df_species_mapped.groupby("soortgroep")["bedekkingsgraad_proc"].sum()
            current_dist = _normalize_series(current_dist)
            categories = list(current_dist.sort_values(ascending=False).index)

    elif radar_mode == "Trofieniveau":
        df_h = df_filtered[(df_filtered["type"] == "Soort") & (df_filtered["jaar"] == selected_year)].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()
        df_h = df_h.dropna(subset=["trofisch_niveau"])

        if df_h.empty:
            st.info(f"Geen data beschikbaar voor trofieniveau in {selected_year}.")
        else:
            df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)
            current_dist = df_h.groupby("trofisch_niveau")["bedekking_num"].sum()
            current_dist = _normalize_series(current_dist)

            preferred = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos"]
            keep = [x for x in preferred if x in current_dist.index]
            rest = [x for x in current_dist.sort_values(ascending=False).index if x not in keep]
            categories = keep + rest

    else:  # "KRW score"
        df_h = df_filtered[(df_filtered["type"] == "Soort") & (df_filtered["jaar"] == selected_year)].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()

        if df_h.empty:
            st.info(f"Geen data beschikbaar voor KRW score in {selected_year}.")
        else:
            if "krw_class" in df_h.columns and df_h["krw_class"].notna().any():
                df_h["krw_cat"] = df_h["krw_class"]
            else:
                df_h["krw_cat"] = pd.cut(
                    pd.to_numeric(df_h["krw_score"], errors="coerce"),
                    bins=[0, 2, 4, 5],
                    labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                    include_lowest=True
                )

            df_h = df_h.dropna(subset=["krw_cat"])
            df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

            current_dist = df_h.groupby("krw_cat")["bedekking_num"].sum()
            current_dist = _normalize_series(current_dist)

            order = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"]
            categories = [x for x in order if x in current_dist.index]

    # Plot radar
    if len(categories) > 0 and not current_dist.empty:
        r_vals = [float(current_dist.get(c, 0.0)) for c in categories]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=categories,
            fill="toself",
            name=f"Data {selected_year}",
            hovertemplate="<b>%{theta}</b><br>Aandeel: %{r:.0%}<extra></extra>",
        ))

        if radar_mode == "Groeivormen" and ref_vals is not None:
            fig_radar.add_trace(go.Scatterpolar(
                r=ref_vals,
                theta=categories,
                fill="toself",
                name="Referentie",
                line=dict(dash="dot"),
                hovertemplate="<b>%{theta}</b><br>Referentie: %{r:.0%}<extra></extra>",
            ))
        else:
            st.caption("ℹ️ Geen referentieprofiel beschikbaar voor deze parameter (alleen voor groeivormen).")

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info(f"Geen data beschikbaar voor radarplot in {selected_year}")

    if radar_mode == "Trofieniveau":
        st.caption(
            "Bron trofieniveau-indeling: Verhofstad et al. (2025) – Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang. "
            "https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf"
        )

    with st.expander("ℹ️ Hoe lees ik deze spingrafiek?"):
        st.markdown(
            """
**Wat wordt er weergegeven?**  
Deze spingrafiek toont de *relatieve verdeling* van de gekozen categorieën.

**Referentielijn (alleen groeivormen):**  
Gestippelde lijn = theoretisch streefbeeld voor een helder systeem:
- Ondergedoken dominant (60%)
- Drijvend/emergent beperkt (15–20%)
- Draadalgen max (5%)
"""
        )

st.divider()

# --- 6. SOORTGROEPEN (ONDERSTE GRAFIEK) ---
st.subheader("🌿 Samenstelling soortgroepen (relatief)")

with st.expander("ℹ️ Hoe komt deze grafiek tot stand?"):
    st.markdown(
        """
Per jaar: de bijdrage van soortgroepen aan de totale bedekking (WATPTN).
- Filter: groeivormcodes en `type='Groep'` eruit
- Indeling: `soortgroep` + numerieke bedekking `bedekkingsgraad_proc`
- Teller: som bedekking per jaar × soortgroep
- Noemer: som WATPTN per jaar (1× per CollectieReferentie)
- Fractie: teller/noemer
"""
    )

df_species_raw = df_filtered[~df_filtered["soort"].isin(RWS_GROEIVORM_CODES)].copy()
df_species_raw = df_species_raw[df_species_raw["type"] != "Groep"].copy()

if df_species_raw.empty:
    st.info("Geen soort-specifieke data gevonden (alleen groepscodes aanwezig?).")
else:
    df_species_mapped = add_species_group_columns(df_species_raw)

    # Teller: som bedekking per jaar/soortgroep
    df_trend_species = (
        df_species_mapped.groupby(["jaar", "soortgroep"], as_index=False)["bedekkingsgraad_proc"]
        .sum()
    )

    # Noemer: totale bedekking per jaar (WATPTN) 1× per CollectieReferentie
    df_year_totals = (
        df_species_mapped.groupby(["jaar", "CollectieReferentie"], as_index=False)["totaal_bedekking_locatie"]
        .first()
        .rename(columns={"totaal_bedekking_locatie": "totaal_bedekking_jaar_sample"})
    )
    year_total_cover = (
        df_year_totals.groupby("jaar", as_index=False)["totaal_bedekking_jaar_sample"]
        .sum()
        .rename(columns={"totaal_bedekking_jaar_sample": "totaal_bedekking_jaar"})
    )

    # Vectorized fractie (geen apply)
    df_trend_species = df_trend_species.merge(year_total_cover, on="jaar", how="left")
    df_trend_species["fractie_tov_totaal"] = 0.0
    mask = df_trend_species["totaal_bedekking_jaar"].notna() & (df_trend_species["totaal_bedekking_jaar"] > 0)
    df_trend_species.loc[mask, "fractie_tov_totaal"] = (
        df_trend_species.loc[mask, "bedekkingsgraad_proc"] / df_trend_species.loc[mask, "totaal_bedekking_jaar"]
    )

    fig_stack = px.bar(
        df_trend_species,
        x="jaar",
        y="fractie_tov_totaal",
        color="soortgroep",
        title="Samenstelling soortgroepen t.o.v. totale bedekking (WATPTN)",
        labels={
            "fractie_tov_totaal": "Fractie van totale bedekking",
            "jaar": "Jaar",
            "soortgroep": "Groep",
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        height=500,
    )
    fig_stack.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_stack, use_container_width=True)

    with st.expander("🔍 Analyse 'overig / individueel' (soorten die nog niet zijn ingedeeld)"):
        df_overig = df_species_mapped[df_species_mapped["soortgroep"] == "Overig / Individueel"]
        if not df_overig.empty:
            missing_stats = (
                df_overig.groupby("soort", as_index=False)
                .agg(
                    Aantal_Metingen=("bedekkingsgraad_proc", "count"),
                    Max_Bedekking=("bedekkingsgraad_proc", "max"),
                )
                .sort_values("Max_Bedekking", ascending=False)
            )
            st.dataframe(missing_stats, use_container_width=True)
        else:
            st.success("Alle aangetroffen soorten zijn succesvol ingedeeld in een groep!")

with st.expander("ℹ️ Toelichting op de soortgroepen"):
    st.write(
        "Hieronder vind je een beschrijving van de verschillende ecologische soortgroepen die in de grafiek worden getoond. "
        "Bron: waterplanten en waterkwaliteit, van Geest, G. et al."
    )
    tab1, tab2 = st.tabs(["Wortelend in sediment", "Overigen/mossen en vrijzwevende groeivormen"])

    with tab1:
        st.markdown("(... jouw bestaande tekst ongewijzigd ...)")
    with tab2:
        st.markdown("(... jouw bestaande tekst ongewijzigd ...)")

# --- 7. BODEMDIAGNOSE ---
st.divider()
st.subheader("🕵️ Bodemdiagnose")

df_year_locs = df_filtered[df_filtered["jaar"] == selected_year].copy()
available_locs = sorted(df_year_locs["locatie_id"].dropna().unique()) if not df_year_locs.empty else []

if not available_locs:
    st.write("Selecteer een jaar met beschikbare data voor de diagnose.")
else:
    c_loc, c_txt = st.columns([1, 2])
    with c_loc:
        selected_loc = st.selectbox("Selecteer specifieke locatie voor diagnose", available_locs)

    with c_txt:
        if selected_loc:
            df_sample = df_year_locs[df_year_locs["locatie_id"] == selected_loc]
            interpretation = interpret_soil_state(df_sample)
            st.markdown(f"**Diagnose voor {selected_loc} ({selected_year}):**")
            st.markdown(interpretation)