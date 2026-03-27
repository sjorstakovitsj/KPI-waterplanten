import streamlit as st
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

# --- 4. TRENDS OVER DE JAREN ---
@st.cache_data(show_spinner=False)
def _year_total_cover_mean(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame(columns=["jaar", "waarde"])
    df_samples = (
        df_in.groupby(["jaar", "CollectieReferentie"], as_index=False)["totaal_bedekking_locatie"]
        .first()
    )
    out = (
        df_samples.groupby("jaar", as_index=False)["totaal_bedekking_locatie"]
        .mean()
        .rename(columns={"totaal_bedekking_locatie": "waarde"})
    )
    return out.sort_values("jaar")


@st.cache_data(show_spinner=False)
def _year_total_cover_sum(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame(columns=["jaar", "totaal_jaar"])
    df_samples = (
        df_in.groupby(["jaar", "CollectieReferentie"], as_index=False)["totaal_bedekking_locatie"]
        .first()
    )
    out = (
        df_samples.groupby("jaar", as_index=False)["totaal_bedekking_locatie"]
        .sum()
        .rename(columns={"totaal_bedekking_locatie": "totaal_jaar"})
    )
    return out.sort_values("jaar")


@st.cache_data(show_spinner=False)
def _compute_growth_trend(df_in: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df_rws_codes = df_in[df_in["soort"].isin(RWS_GROEIVORM_CODES)].copy()
    if not df_rws_codes.empty:
        out = (
            df_rws_codes
            .groupby(["jaar", "groeivorm"], as_index=False)["bedekking_pct"]
            .mean()
        )
        out = out.rename(columns={"bedekking_pct": "waarde"})
        return out, "Methode: brondata Aquadesk (gemiddelde bedekking per groeivorm en jaar)"

    df_species_only = df_in[
        (~df_in["soort"].isin(RWS_GROEIVORM_CODES)) &
        (df_in["type"] != "Groep")
    ].copy()
    if df_species_only.empty:
        return pd.DataFrame(columns=["jaar", "groeivorm", "waarde"]), "Geen data beschikbaar voor groeivorm-analyse."

    out = (
        df_species_only
        .groupby(["jaar", "groeivorm"], as_index=False)["bedekking_pct"]
        .sum()
        .rename(columns={"bedekking_pct": "waarde"})
    )
    return out, "Methode: fallback op soortrecords (som bedekking per groeivorm en jaar)"


@st.cache_data(show_spinner=False)
def _compute_fraction_trend(df_in: pd.DataFrame, trend_mode: str) -> tuple[pd.DataFrame, str]:
    if df_in.empty:
        return pd.DataFrame(columns=["jaar", "categorie", "fractie"]), ""

    df_species = df_in[
        (df_in["type"] == "Soort") &
        (~df_in["soort"].isin(RWS_GROEIVORM_CODES))
    ].copy()
    if df_species.empty:
        return pd.DataFrame(columns=["jaar", "categorie", "fractie"]), ""

    category_col = "categorie"
    denominator = _year_total_cover_sum(df_in)
    caption = ""

    if trend_mode == "Soortgroepen":
        df_h = add_species_group_columns(df_species)
        if "is_kenmerkende_soort_n2000" in df_h.columns:
            df_h = df_h[~df_h["is_kenmerkende_soort_n2000"].fillna(False)].copy()
        if df_h.empty:
            return pd.DataFrame(columns=["jaar", "categorie", "fractie"]), ""
        df_h[category_col] = df_h["soortgroep_weergave"] if "soortgroep_weergave" in df_h.columns else df_h["soortgroep"]
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ""), category_col] = "Geen match"
        df_h["waarde_num"] = pd.to_numeric(df_h.get("bedekkingsgraad_proc", df_h["bedekking_pct"]), errors="coerce").fillna(0).clip(lower=0)
        caption = "Aggregatie: som bedekking per soortgroep, gedeeld door de totale bedekking (WATPTN) per jaar."

    elif trend_mode == "Trofieniveau":
        df_h = df_species.copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()
        if df_h.empty:
            return pd.DataFrame(columns=["jaar", "categorie", "fractie"]), ""
        if "trofisch_niveau_weergave" in df_h.columns:
            df_h[category_col] = df_h["trofisch_niveau_weergave"].astype(object)
        else:
            df_h[category_col] = df_h.get("trofisch_niveau", pd.Series(index=df_h.index, dtype="object")).astype(object)
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ""), category_col] = "Geen match"
        df_h["waarde_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)
        caption = "Aggregatie: som bedekking per trofieniveau, gedeeld door de totale bedekking (WATPTN) per jaar."

    elif trend_mode == "KRW score":
        df_h = df_species.copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()
        if df_h.empty:
            return pd.DataFrame(columns=["jaar", "categorie", "fractie"]), ""
        if "krw_class_weergave" in df_h.columns:
            df_h[category_col] = df_h["krw_class_weergave"].astype(object)
        elif "krw_class" in df_h.columns and df_h["krw_class"].notna().any():
            df_h[category_col] = df_h["krw_class"].astype(object)
        else:
            df_h[category_col] = pd.cut(
                pd.to_numeric(df_h["krw_score"], errors="coerce"),
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True,
            ).astype(object)
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ""), category_col] = "Geen match"
        df_h["waarde_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)
        caption = "Aggregatie: som bedekking per KRW-klasse, gedeeld door de totale bedekking (WATPTN) per jaar."

    else:  # Kenmerkende soorten (N2000)
        df_h = add_species_group_columns(df_species)
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "AANWZHD"].copy()
        elif "is_kenmerkende_soort_n2000" in df_h.columns:
            df_h = df_h[df_h["is_kenmerkende_soort_n2000"].fillna(False)].copy()
        if df_h.empty:
            return pd.DataFrame(columns=["jaar", "categorie", "fractie"]), ""
        if "kenmerkende_soort_n2000_weergave" in df_h.columns:
            df_h[category_col] = df_h["kenmerkende_soort_n2000_weergave"].astype(object)
        elif "soort_display" in df_h.columns:
            df_h[category_col] = df_h["soort_display"].astype(object)
        else:
            df_h[category_col] = df_h["soort"].astype(object)
        df_h.loc[df_h[category_col].isna() | (df_h[category_col].astype(str).str.strip() == ""), category_col] = "Geen match"
        df_h["waarde_num"] = pd.to_numeric(df_h.get("bedekkingsgraad_proc", df_h["bedekking_pct"]), errors="coerce").fillna(0).clip(lower=0)

        # Als N2000-records geen zinvolle bedekkingswaarden bevatten, val terug op records per jaar.
        if float(df_h["waarde_num"].sum()) <= 0:
            df_num = df_h.groupby(["jaar", category_col], as_index=False).size().rename(columns={"size": "waarde_num"})
            df_den = df_h.groupby("jaar", as_index=False).size().rename(columns={"size": "totaal_jaar"})
            out = df_num.merge(df_den, on="jaar", how="left")
            out["fractie"] = 0.0
            mask = out["totaal_jaar"].notna() & (out["totaal_jaar"] > 0)
            out.loc[mask, "fractie"] = out.loc[mask, "waarde_num"] / out.loc[mask, "totaal_jaar"]
            out = out.rename(columns={category_col: "categorie"}).sort_values(["jaar", "categorie"])
            caption = "Aggregatie: aandeel per kenmerkende soort binnen alle N2000-waarnemingen per jaar (fallback op recordaantallen)."
            return out[["jaar", "categorie", "fractie"]], caption

        caption = "Aggregatie: som waarde per kenmerkende soort (N2000), gedeeld door de totale bedekking (WATPTN) per jaar."

    df_num = df_h.groupby(["jaar", category_col], as_index=False)["waarde_num"].sum()
    out = df_num.merge(denominator, on="jaar", how="left")
    out["fractie"] = 0.0
    mask = out["totaal_jaar"].notna() & (out["totaal_jaar"] > 0)
    out.loc[mask, "fractie"] = out.loc[mask, "waarde_num"] / out.loc[mask, "totaal_jaar"]
    out = out.rename(columns={category_col: "categorie"}).sort_values(["jaar", "categorie"])
    return out[["jaar", "categorie", "fractie"]], caption


st.subheader("Trend over de jaren")
trend_mode = st.selectbox(
    "Kies trendweergave",
    [
        "Groeivormen",
        "Totale bedekking",
        "KRW score",
        "Trofieniveau",
        "Soortgroepen",
        "Kenmerkende soorten (N2000)",
    ],
    index=0,
    key="trend_mode_choice",
)

GROWTH_ORDER = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]
GROWTH_COLOR_MAP = {
    "Ondergedoken": "#2ca02c",
    "Drijvend": "#1f77b4",
    "Emergent": "#ff7f0e",
    "Draadalgen": "#d62728",
    "FLAB": "#7f7f7f",
    "Kroos": "#bcbd22",
}

if trend_mode == "Totale bedekking":
    df_total_trend = _year_total_cover_mean(df_filtered)
    if df_total_trend.empty:
        st.info("Geen data beschikbaar voor totale bedekking over de jaren.")
    else:
        fig_trend = px.line(
            df_total_trend,
            x="jaar",
            y="waarde",
            markers=True,
            title="Trend totale bedekking over de jaren",
            labels={"jaar": "Jaar", "waarde": "Gem. totale bedekking (%)"},
        )
        fig_trend.update_layout(height=420)
        st.plotly_chart(fig_trend, width='stretch')
        st.caption("Aggregatie: gemiddelde totale bedekking per monstername (CollectieReferentie) per jaar.")

elif trend_mode == "Groeivormen":
    df_trend_growth, growth_caption = _compute_growth_trend(df_filtered)
    if df_trend_growth.empty:
        st.info("Geen data beschikbaar voor groeivormen over de jaren.")
    else:
        fig_trend = px.area(
            df_trend_growth,
            x="jaar",
            y="waarde",
            color="groeivorm",
            category_orders={"groeivorm": GROWTH_ORDER},
            color_discrete_map=GROWTH_COLOR_MAP,
            title="Trend in groeivormen over de jaren",
            labels={"jaar": "Jaar", "waarde": "Bedekking (%)", "groeivorm": "Groeivorm"},
        )
        fig_trend.update_layout(height=420, yaxis_title="Bedekking (%)")
        st.plotly_chart(fig_trend, width='stretch')
        st.caption(growth_caption)

else:
    df_trend_cat, cat_caption = _compute_fraction_trend(df_filtered, trend_mode)
    if df_trend_cat.empty:
        st.info(f"Geen data beschikbaar voor {trend_mode.lower()} over de jaren.")
    else:
        preferred_orders = {
            "KRW score": ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)", "Geen match"],
            "Trofieniveau": ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos", "Onbekend", "Geen match"],
            "Soortgroepen": [
                "chariden", "iseotiden", "parvopotamiden", "magnopotamiden", "myriophylliden",
                "vallisneriiden", "elodeiden", "stratiotiden", "pepliden", "batrachiiden",
                "nymphaeiden", "haptofyten", "Overig / Individueel", "Geen match",
            ],
        }
        category_order = preferred_orders.get(trend_mode)
        if trend_mode == "Kenmerkende soorten (N2000)":
            ordered = sorted([x for x in df_trend_cat["categorie"].dropna().unique() if x != "Geen match"], key=lambda x: str(x).lower())
            if "Geen match" in set(df_trend_cat["categorie"].dropna().astype(str)):
                ordered.append("Geen match")
            category_order = ordered

        fig_trend = px.area(
            df_trend_cat,
            x="jaar",
            y="fractie",
            color="categorie",
            category_orders={"categorie": category_order} if category_order else None,
            title=f"Trend in {trend_mode.lower()} over de jaren",
            labels={"jaar": "Jaar", "fractie": "Fractie van totale bedekking", "categorie": trend_mode},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_trend.update_layout(height=420, yaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(fig_trend, width='stretch')
        st.caption(cat_caption)
        if trend_mode == "Trofieniveau":
            st.caption(
                "Bron trofieniveau-indeling: Verhofstad et al. (2025) – Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang. "
                "https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf"
            )

st.divider()

# --- 6. SOORTGROEPEN (ONDERSTE GRAFIEK) ---
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
    st.plotly_chart(fig_stack, width='stretch')

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
            st.dataframe(missing_stats, width='stretch')
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