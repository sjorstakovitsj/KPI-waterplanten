# 1_Overzicht.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data, add_species_group_columns, calculate_kpi

st.set_page_config(page_title="Waterplanten Monitor", layout="wide")
st.title("🌱 Waterplanten dashboard IJsselmeergebied")
st.markdown("Gemiddelden van geselecteerd meetjaar.")

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
RWS_GROEIVORM_CODES = ["FLAB", "KROOS", "SUBMSPTN", "DRAADAGN", "DRIJFBPTN", "EMSPTN", "WATPTN"]

# -----------------------------------------------------------------------------
# CACHED HELPERS (sneller bij UI-wijzigingen)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _filter_projects(df: pd.DataFrame, selected_projects: tuple) -> pd.DataFrame:
    # select only needed columns? (we keep all for downstream plots)
    return df[df["Project"].isin(selected_projects)].copy()

@st.cache_data(show_spinner=False)
def _match_prev_by_waterbody(df_filtered: pd.DataFrame, selected_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized versie van:
    - per Waterlichaam: vorige jaar < selected_year zoeken
    - df_prev_matched en df_year_matched teruggeven
    """
    df_year = df_filtered[df_filtered["jaar"] == selected_year].copy()
    if df_year.empty:
        return df_year, pd.DataFrame(columns=df_filtered.columns)

    # Waterlichamen die in geselecteerde jaar voorkomen
    current_wbs = df_year["Waterlichaam"].dropna().unique()
    df_curr_wb = df_filtered[df_filtered["Waterlichaam"].isin(current_wbs)].copy()

    # Bepaal per Waterlichaam het max jaar < selected_year
    df_prev_years = (
        df_curr_wb[df_curr_wb["jaar"] < selected_year]
        .groupby("Waterlichaam", as_index=False)["jaar"]
        .max()
        .rename(columns={"jaar": "prev_jaar"})
    )
    if df_prev_years.empty:
        # Geen vorige metingen beschikbaar
        return df_year.iloc[0:0].copy(), pd.DataFrame(columns=df_filtered.columns)

    # Join om vorige meetdataset te maken
    df_prev_matched = df_curr_wb.merge(
        df_prev_years,
        on="Waterlichaam",
        how="inner"
    )
    df_prev_matched = df_prev_matched[df_prev_matched["jaar"] == df_prev_matched["prev_jaar"]].drop(columns=["prev_jaar"])

    # Alleen waterlichamen waar ook echt een vorige meting bestaat
    matched_wbs = df_prev_years["Waterlichaam"].unique()
    df_year_matched = df_year[df_year["Waterlichaam"].isin(matched_wbs)].copy()

    return df_year_matched, df_prev_matched

@st.cache_data(show_spinner=False)
def _overview_per_waterbody(df_year: pd.DataFrame) -> pd.DataFrame:
    if df_year.empty:
        return pd.DataFrame(columns=["Waterlichaam", "Bedekking (Totaal %)", "Gem. Diepte (m)", "Gem. Doorzicht (m)", "Soortenrijkdom"])

    # 1) Unieke waarde per sample
    df_samples = (
        df_year.groupby(["Waterlichaam", "CollectieReferentie"], as_index=False)
        .agg({
            "totaal_bedekking_locatie": "first",
            "diepte_m": "first",
            "doorzicht_m": "first"
        })
    )

    # 2) Mean per waterlichaam
    df_water_stats = (
        df_samples.groupby("Waterlichaam", as_index=False)
        .agg({
            "totaal_bedekking_locatie": "mean",
            "diepte_m": "mean",
            "doorzicht_m": "mean"
        })
    )

    # 3) Soortenrijkdom: unieke soorten per waterlichaam (alleen type=Soort)
    df_species_only = df_year[df_year["type"] == "Soort"]
    df_richness = df_species_only.groupby("Waterlichaam", as_index=False)["soort"].nunique()
    df_richness = df_richness.rename(columns={"soort": "Soortenrijkdom"})

    overview_df = df_water_stats.merge(df_richness, on="Waterlichaam", how="left")
    overview_df = overview_df.rename(columns={
        "totaal_bedekking_locatie": "Bedekking (Totaal %)",
        "diepte_m": "Gem. Diepte (m)",
        "doorzicht_m": "Gem. Doorzicht (m)",
    })
    overview_df["Soortenrijkdom"] = overview_df["Soortenrijkdom"].fillna(0).astype(int)
    return overview_df

@st.cache_data(show_spinner=False)
def _trend_cover(df_trend_base: pd.DataFrame) -> pd.DataFrame:
    # Dedup per sample -> mean per jaar per waterlichaam
    df_samples = (
        df_trend_base.groupby(["jaar", "Waterlichaam", "CollectieReferentie"], as_index=False)
        .agg({"totaal_bedekking_locatie": "first"})
    )
    return (
        df_samples.groupby(["jaar", "Waterlichaam"], as_index=False)["totaal_bedekking_locatie"]
        .mean()
    )

@st.cache_data(show_spinner=False)
def _trend_forms(df_trend_base: pd.DataFrame) -> pd.DataFrame:
    df_forms = df_trend_base[df_trend_base["type"] == "Groep"].copy()
    if df_forms.empty:
        return df_forms

    df_form_sample = (
        df_forms.groupby(["jaar", "CollectieReferentie", "groeivorm"], as_index=False)["bedekking_pct"]
        .sum()
    )
    df_form_trend = (
        df_form_sample.groupby(["jaar", "groeivorm"], as_index=False)["bedekking_pct"]
        .mean()
    )
    return df_form_trend

@st.cache_data(show_spinner=False)
def _trend_speciesgroups_fraction(df_trend_base: pd.DataFrame) -> pd.DataFrame:
    # Alleen echte soorten (geen groeivormcodes en geen type Groep)
    df_species_raw = df_trend_base[~df_trend_base["soort"].isin(RWS_GROEIVORM_CODES)].copy()
    df_species_raw = df_species_raw[df_species_raw["type"] != "Groep"]
    if df_species_raw.empty:
        return pd.DataFrame()

    # Mapping + bedekkingsgraad
    df_mapped = add_species_group_columns(df_species_raw)

    # Teller: som bedekking per jaar en soortgroep
    df_num = (
        df_mapped.groupby(["jaar", "soortgroep"], as_index=False)["bedekkingsgraad_proc"]
        .sum()
    )

    # Noemer: totale bedekking per jaar (WATPTN) 1x per sample
    df_den = (
        df_mapped.groupby(["jaar", "CollectieReferentie"], as_index=False)
        .agg({"totaal_bedekking_locatie": "first"})
    )
    df_den = (
        df_den.groupby("jaar", as_index=False)["totaal_bedekking_locatie"]
        .sum()
        .rename(columns={"totaal_bedekking_locatie": "totaal_bedekking_jaar"})
    )

    # Merge + fractie
    out = df_num.merge(df_den, on="jaar", how="left")
    out["fractie_tov_totaal"] = 0.0
    mask = out["totaal_bedekking_jaar"].notna() & (out["totaal_bedekking_jaar"] > 0)
    out.loc[mask, "fractie_tov_totaal"] = out.loc[mask, "bedekkingsgraad_proc"] / out.loc[mask, "totaal_bedekking_jaar"]
    return out

@st.cache_data(show_spinner=False)
def _speciesgroup_counts(df_individual_species: pd.DataFrame) -> pd.DataFrame:
    """
    Verdeling waarnemingen per soortgroep (op basis van individuele soorten in df_individual_species).
    Verwacht: kolommen 'soort' en (via utils) mapping naar 'soortgroep'.
    """
    if df_individual_species.empty:
        return pd.DataFrame(columns=["Soortgroep", "Aantal waarnemingen"])

    df_mapped = add_species_group_columns(df_individual_species.copy())
    if "soortgroep" not in df_mapped.columns:
        return pd.DataFrame(columns=["Soortgroep", "Aantal waarnemingen"])

    s = df_mapped["soortgroep"].dropna()
    if s.empty:
        return pd.DataFrame(columns=["Soortgroep", "Aantal waarnemingen"])

    out = s.value_counts().rename_axis("Soortgroep").reset_index(name="Aantal waarnemingen")
    return out

# -----------------------------------------------------------------------------
# DATA LOAD
# -----------------------------------------------------------------------------
df = load_data()

st.sidebar.header("Algemene filters")
if df.empty:
    st.error("Geen data geladen. Controleer utils.py en het bronbestand.")
    st.stop()

# 1) Jaar
all_years = sorted(df["jaar"].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Selecteer meetjaar", all_years)

# 2) Project
all_projects = sorted(df["Project"].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)",
    options=all_projects,
    default=all_projects
)

df_filtered = _filter_projects(df, tuple(selected_projects))

# Match huidige vs vorige meting per waterlichaam (sneller)
df_year_matched, df_prev_matched = _match_prev_by_waterbody(df_filtered, int(selected_year))

# Voor onderdelen die ook de “huidige jaar” selectie zonder match willen:
df_year = df_filtered[df_filtered["jaar"] == selected_year].copy()

# -----------------------------------------------------------------------------
# 1. KPI’s
# -----------------------------------------------------------------------------
avg_bedekking, d_bedekking = calculate_kpi(df_year_matched, df_prev_matched, "totaal_bedekking_locatie", is_loc_metric=True)
avg_doorzicht, d_doorzicht = calculate_kpi(df_year_matched, df_prev_matched, "doorzicht_m", is_loc_metric=True)
avg_diepte, d_diepte = calculate_kpi(df_year_matched, df_prev_matched, "diepte_m", is_loc_metric=True)

# soortenrijkdom (alleen individuele soorten)
df_year_species = df_year_matched[(df_year_matched["type"] == "Soort") & (~df_year_matched["soort"].isin(RWS_GROEIVORM_CODES))]
df_prev_species = df_prev_matched[(df_prev_matched["type"] == "Soort") & (~df_prev_matched["soort"].isin(RWS_GROEIVORM_CODES))]
n_soorten = df_year_species["soort"].nunique()
prev_soorten = df_prev_species["soort"].nunique() if not df_prev_species.empty else n_soorten
d_soorten = n_soorten - prev_soorten

# -----------------------------------------------------------------------------
# Pie charts: samenstelling waarnemingen
# -----------------------------------------------------------------------------
st.subheader("🥧 Samenstelling waarnemingen (individuele soorten)")
df_ind = df_year[(df_year["type"] == "Soort") & (~df_year["soort"].isin(RWS_GROEIVORM_CODES))].copy()

# >>> AANGEPAST: 3 kolommen i.p.v. 2
c_pie1, c_pie2, c_pie3 = st.columns(3)

with c_pie1:
    st.markdown("**Verdeling waarnemingen per KRW-score**")
    if "krw_class" not in df_ind.columns and "krw_score" not in df_ind.columns:
        st.info("KRW-score is nog niet beschikbaar in de dataset (controleer verrijking in utils.py).")
    else:
        if "krw_class" in df_ind.columns:
            s = df_ind["krw_class"].dropna()
        else:
            s = pd.cut(
                df_ind["krw_score"],
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True,
            ).dropna()

        if s.empty:
            st.info("Geen KRW-scores beschikbaar voor de huidige selectie.")
        else:
            pie_df = s.value_counts().rename_axis("KRW-klasse").reset_index(name="Aantal waarnemingen")
            color_map = {
                "Gunstig (1-2)": "#2ca02c",
                "Neutraal (3-4)": "#ff7f0e",
                "Ongewenst (5)": "#d62728",
            }
            fig_krw = px.pie(
                pie_df,
                names="KRW-klasse",
                values="Aantal waarnemingen",
                color="KRW-klasse",
                color_discrete_map=color_map,
                hole=0.35,
            )
            fig_krw.update_traces(textposition="inside", textinfo="percent+label")
            fig_krw.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_krw, use_container_width=True)

with c_pie2:
    st.markdown("**Verdeling waarnemingen per trofieniveau**")
    if "trofisch_niveau" not in df_ind.columns:
        st.info("Trofieniveau is nog niet beschikbaar in de dataset (controleer verrijking in utils.py).")
    else:
        t = df_ind["trofisch_niveau"].dropna()
        if t.empty:
            st.info("Geen trofieniveaus beschikbaar voor de huidige selectie.")
        else:
            pie_df = t.value_counts().rename_axis("Trofieniveau").reset_index(name="Aantal waarnemingen")
            fig_trofie = px.pie(pie_df, names="Trofieniveau", values="Aantal waarnemingen", hole=0.35)
            fig_trofie.update_traces(textposition="inside", textinfo="percent+label")
            fig_trofie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_trofie, use_container_width=True)

    with st.expander("ℹ️ Toelichting"):
        st.markdown(
            """
De indeling van soorten naar trofieniveau is gebaseerd op:
**Verhofstad et al. (2025)** – *Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang*.
🔗 [Download het rapport](https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf)
"""
        )

# >>> NIEUW: derde taartdiagram voor soortgroep
with c_pie3:
    st.markdown("**Verdeling waarnemingen per soortgroep**")
    pie_df = _speciesgroup_counts(df_ind)

    if pie_df.empty:
        st.info("Geen soortgroep-indeling beschikbaar voor de huidige selectie (check mapping in utils.add_species_group_columns).")
    else:
        fig_group = px.pie(
            pie_df,
            names="Soortgroep",
            values="Aantal waarnemingen",
            hole=0.35,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_group.update_traces(textposition="inside", textinfo="percent+label")
        fig_group.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_group, use_container_width=True)

# KPI-metrics (blijven gelijk)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("gem. totale bedekking", f"{avg_bedekking:.1f}%", f"{d_bedekking:.1f}%")
with c2:
    st.metric("gem. doorzicht", f"{avg_doorzicht:.2f} m", f"{d_doorzicht:.2f} m")
with c3:
    st.metric("gem. diepte", f"{avg_diepte:.2f} m", f"{d_diepte:.2f} m")
with c4:
    st.metric("gem. soortenrijkdom", n_soorten, d_soorten)

st.divider()

# -----------------------------------------------------------------------------
# 2. Detailoverzicht per waterlichaam
# -----------------------------------------------------------------------------
st.subheader(f"📊 Opsomming per waterlichaam ({selected_year})")
overview_df = _overview_per_waterbody(df_year)

if overview_df.empty:
    st.info("Geen data beschikbaar voor de huidige filters.")
else:
    st.dataframe(
        overview_df[["Waterlichaam", "Bedekking (Totaal %)", "Gem. Doorzicht (m)", "Gem. Diepte (m)", "Soortenrijkdom"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Bedekking (Totaal %)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Gem. Doorzicht (m)": st.column_config.NumberColumn(format="%.2f m"),
            "Gem. Diepte (m)": st.column_config.NumberColumn(format="%.2f m"),
            "Soortenrijkdom": st.column_config.NumberColumn(format="%d soorten"),
        }
    )

st.divider()