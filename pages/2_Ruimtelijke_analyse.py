# 2_Ruimtelijke_analyse.py
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

from utils import (
    load_data,
    add_species_group_columns,
    create_pie_map,
    create_map,
    get_sorted_species_list,
    EXCLUDED_SPECIES_CODES,
    RWS_GROEIVORM_CODES,
)

st.set_page_config(layout="wide", page_title="Ruimtelijke analyse")
st.title("🗺️ Ruimtelijke analyse")
st.markdown("Vergelijk de vegetatieontwikkeling met diepte en doorzicht.")

# -----------------------------------------------------------------------------
# CONSTANTS (consistent met UI-opties)
# -----------------------------------------------------------------------------
PIE_TYPES = ["KRW score", "Trofieniveau", "Groeivormen", "soortgroepen"]


# -----------------------------------------------------------------------------
# SESSION STATE CACHE (sneller bij intensieve interacties dan st.cache_data)
# -----------------------------------------------------------------------------
def _cache() -> dict:
    if "_ra_cache" not in st.session_state:
        st.session_state["_ra_cache"] = {}
    return st.session_state["_ra_cache"]


def _ck(prefix: str, *parts):
    # compacte cache key
    return (prefix,) + tuple(parts)


def _cache_get(key):
    return _cache().get(key)


def _cache_set(key, value):
    _cache()[key] = value


# -----------------------------------------------------------------------------
# VECTORIZED HELPERS
# -----------------------------------------------------------------------------
def _scale_rows_to_total(pivot: pd.DataFrame, total: float = 100.0) -> pd.DataFrame:
    """Vectorized: schaal rijen naar max 'total' behoudend verdeling."""
    if pivot.empty:
        return pivot
    row_sum = pivot.sum(axis=1).astype(float)
    factor = np.where(row_sum > 0, np.minimum(1.0, total / row_sum), 1.0)
    return pivot.mul(pd.Series(factor, index=pivot.index), axis=0)


def _pivot_to_counts_by_loc(pivot: pd.DataFrame) -> dict:
    """Fast {locatie_id: {categorie: waarde}}."""
    if pivot.empty:
        return {}
    return pivot.to_dict(orient="index")


def _weighted_mean(df: pd.DataFrame, group_col: str, value_col: str, weight_col: str, out_col: str) -> pd.DataFrame:
    """Vectorized weighted mean per group: sum(v*w)/sum(w)."""
    if df.empty:
        return pd.DataFrame(columns=[group_col, out_col])

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    v = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    tmp = pd.DataFrame({group_col: df[group_col].values, "num": (v * w).values, "w": w.values})
    agg = tmp.groupby(group_col, as_index=False).agg(num_sum=("num", "sum"), w_sum=("w", "sum"))
    agg[out_col] = np.where(agg["w_sum"] > 0, agg["num_sum"] / agg["w_sum"], np.nan)
    return agg[[group_col, out_col]]


def _dominant_category(df: pd.DataFrame, group_col: str, cat_col: str, weight_col: str, out_col: str) -> pd.DataFrame:
    """Dominante categorie per group obv hoogste som gewicht (vectorized)."""
    if df.empty:
        return pd.DataFrame(columns=[group_col, out_col])

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp = pd.DataFrame({group_col: df[group_col].values, cat_col: df[cat_col].values, "w": w.values})
    tmp = tmp.dropna(subset=[cat_col])
    if tmp.empty:
        return pd.DataFrame(columns=[group_col, out_col])

    s = tmp.groupby([group_col, cat_col], as_index=False)["w"].sum()
    s = s.sort_values([group_col, "w"], ascending=[True, False])
    dom = s.drop_duplicates(subset=[group_col], keep="first").rename(columns={cat_col: out_col})
    return dom[[group_col, out_col]]


def _safe_numeric(s: pd.Series, fallback: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fallback)


# -----------------------------------------------------------------------------
# PRECOMPUTE: alles wat “intensief” is, 1x per filter-combinatie
# -----------------------------------------------------------------------------
def _precompute_for_filter(df_filtered: pd.DataFrame, filter_key: tuple) -> dict:
    """
    Precompute & cache per filter_key:
      - df_locs (lat/lon + mean diepte/doorzicht)
      - df_groups (type=Groep) met bedekking_num
      - df_species_base (type=Soort, excl. EXCLUDED_SPECIES_CODES) voor records/pies
      - trof_counts pivot (records)
      - krw_counts pivot (records)
      - groeivormen pivot (% bedekking) scaled to 100
      - krw_score_loc (weighted)
      - trofieniveau_loc (dominant, weighted)
      - df_species_groups + pivot_soortgroepen (% bedekking) scaled to 100  [heavy but needed for “soortgroepen”]
    """
    key = _ck("precomp", *filter_key)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # --- df_locs ---
    df_locs = (
        df_filtered.groupby(["locatie_id", "Waterlichaam"], as_index=False)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            diepte_m=("diepte_m", "mean"),
            doorzicht_m=("doorzicht_m", "mean"),
        )
    )

    # --- Groeivormen (type Groep) ---
    df_groups = df_filtered[df_filtered["type"] == "Groep"][["locatie_id", "groeivorm", "bedekking_pct"]].copy()
    df_groups["bedekking_num"] = _safe_numeric(df_groups["bedekking_pct"], 0.0).clip(lower=0.0)

    pivot_groeivormen = (
        df_groups.groupby(["locatie_id", "groeivorm"])["bedekking_num"]
        .sum()
        .unstack(fill_value=0.0)
    )
    pivot_groeivormen = _scale_rows_to_total(pivot_groeivormen, total=100.0)
    counts_groeivormen = _pivot_to_counts_by_loc(pivot_groeivormen)

    # --- Species base (voor records: trofie / krw pies) ---
    df_species_base = df_filtered[
        (df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(EXCLUDED_SPECIES_CODES))
    ][["locatie_id", "soort", "bedekking_pct", "krw_score", "krw_class", "trofisch_niveau"]].copy()

    # Trofieniveau counts (records)
    df_trof_counts = df_species_base.dropna(subset=["trofisch_niveau"])[["locatie_id", "trofisch_niveau"]]
    if df_trof_counts.empty:
        counts_trof = {}
    else:
        pivot_trof = df_trof_counts.groupby(["locatie_id", "trofisch_niveau"]).size().unstack(fill_value=0)
        counts_trof = _pivot_to_counts_by_loc(pivot_trof)

    # KRW counts (records)
    df_krw_counts = df_species_base.copy()
    if "krw_class" in df_krw_counts.columns and df_krw_counts["krw_class"].notna().any():
        df_krw_counts["krw_cat"] = df_krw_counts["krw_class"]
    else:
        df_krw_counts["krw_cat"] = pd.cut(
            _safe_numeric(df_krw_counts["krw_score"], np.nan),
            bins=[0, 2, 4, 5],
            labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
            include_lowest=True,
        )
    df_krw_counts = df_krw_counts.dropna(subset=["krw_cat"])[["locatie_id", "krw_cat"]]
    if df_krw_counts.empty:
        counts_krw = {}
    else:
        pivot_krw = df_krw_counts.groupby(["locatie_id", "krw_cat"]).size().unstack(fill_value=0)
        counts_krw = _pivot_to_counts_by_loc(pivot_krw)

    # --- Extra kolommen voor tabel: KRW score loc (weighted), trofie dominant loc ---
    df_species_for_loc = df_filtered[
        (df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES))
    ][["locatie_id", "krw_score", "trofisch_niveau", "bedekking_pct"]].copy()

    # gewogen krw
    df_krw_loc = df_species_for_loc.dropna(subset=["krw_score"])
    krw_loc = _weighted_mean(df_krw_loc, "locatie_id", "krw_score", "bedekking_pct", "krw_score_loc")

    # dominant trofie
    df_trof_loc = df_species_for_loc.dropna(subset=["trofisch_niveau"])
    trof_loc = _dominant_category(df_trof_loc, "locatie_id", "trofisch_niveau", "bedekking_pct", "trofieniveau_loc")

    # --- Soortgroepen (heavy) ---
    # reduce before mapping: keep only relevant columns for add_species_group_columns
    df_species_raw = df_filtered[
        (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES)) & (df_filtered["type"] != "Groep")
    ][["locatie_id", "soort", "type", "Grootheid", "bedekking_pct", "waarde_bedekking", "totaal_bedekking_locatie"]].copy()

    df_species_groups = add_species_group_columns(df_species_raw)
    df_species_groups["bedekking_num"] = _safe_numeric(df_species_groups["bedekkingsgraad_proc"], 0.0).clip(lower=0.0)

    pivot_soortgroepen = (
        df_species_groups.groupby(["locatie_id", "soortgroep"])["bedekking_num"]
        .sum()
        .unstack(fill_value=0.0)
    )
    pivot_soortgroepen = _scale_rows_to_total(pivot_soortgroepen, total=100.0)
    counts_soortgroepen = _pivot_to_counts_by_loc(pivot_soortgroepen)

    precomp = {
        "df_locs": df_locs,

        "counts_groeivormen": counts_groeivormen,
        "counts_trof": counts_trof,
        "counts_krw": counts_krw,
        "counts_soortgroepen": counts_soortgroepen,

        "krw_loc": krw_loc,
        "trof_loc": trof_loc,
    }

    _cache_set(key, precomp)
    return precomp


# -----------------------------------------------------------------------------
# DATA LOAD + FILTERS
# -----------------------------------------------------------------------------
df = load_data()

st.sidebar.header("Filters")
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

all_years = sorted(df["jaar"].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Selecteer jaar", all_years)

all_projects = sorted(df["Project"].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)",
    options=all_projects,
    default=all_projects,
)

df_filtered = df[(df["jaar"] == selected_year) & (df["Project"].isin(selected_projects))].copy()
if df_filtered.empty:
    st.warning("Geen data gevonden voor deze selectie.")
    st.stop()

filter_key = (int(selected_year), tuple(selected_projects))

st.sidebar.markdown("---")
st.sidebar.header("Kaartinstellingen")

analysis_level = st.sidebar.radio(
    "Kies analyseniveau",
    options=["groepen & aggregaties", "individuele soorten"],
)

if analysis_level == "groepen & aggregaties":
    selected_coverage_type = st.sidebar.selectbox(
        "selecteer groep",
        options=["totale bedekking", "Groeivormen", "Trofieniveau", "KRW score", "soortgroepen"],
    )
else:
    # lijst is niet super duur maar we cachen hem per filter_key
    key = _ck("species_list", *filter_key)
    species_list = _cache_get(key)
    if species_list is None:
        species_list = get_sorted_species_list(df_filtered)
        _cache_set(key, species_list)
    if not species_list:
        st.sidebar.warning("Geen individuele soorten gevonden in deze selectie.")
        st.stop()
    selected_coverage_type = st.sidebar.selectbox("selecteer soort", options=species_list)

layer_mode = st.sidebar.radio("kies kaartlaag", options=["Vegetatie", "Diepte", "Doorzicht"])

# -----------------------------------------------------------------------------
# PRECOMPUTE alleen als we een intensief pad gebruiken:
# - alle pie-lagen
# - KRW score numeric (weighted)
# - Trofieniveau (dominant) map/tabel
# -----------------------------------------------------------------------------
needs_precomp = (
    (layer_mode == "Vegetatie" and analysis_level == "groepen & aggregaties" and selected_coverage_type in PIE_TYPES)
    or (layer_mode == "Vegetatie" and analysis_level == "groepen & aggregaties" and selected_coverage_type in ["KRW score", "Trofieniveau"])
    # tabel extra kolommen gebruiken we altijd -> precomp helpt ook, maar is zwaar (soortgroepen mapping).
    # Daarom: we precomputen pas als gebruiker een intensieve optie kiest,
    # maar we bouwen wel een lichte tabelbasis zonder soortgroepen mapping als het niet nodig is.
)

precomp = None
df_locs = None

if needs_precomp:
    with st.spinner("Intensieve analyse wordt voorbereid (1× per selectie)..."):
        precomp = _precompute_for_filter(df_filtered, filter_key)
    df_locs = precomp["df_locs"]
else:
    # light locs only
    locs_key = _ck("locs_light", *filter_key)
    df_locs = _cache_get(locs_key)
    if df_locs is None:
        df_locs = (
            df_filtered.groupby(["locatie_id", "Waterlichaam"], as_index=False)
            .agg(
                lat=("lat", "first"),
                lon=("lon", "first"),
                diepte_m=("diepte_m", "mean"),
                doorzicht_m=("doorzicht_m", "mean"),
            )
        )
        _cache_set(locs_key, df_locs)

# -----------------------------------------------------------------------------
# INFO / LEGENDA
# -----------------------------------------------------------------------------
st.subheader(f"Kaartweergave: {layer_mode}")

if layer_mode == "Vegetatie":
    if analysis_level == "individuele soorten":  # fix case mismatch
        st.info(f"Je bekijkt de verspreiding van de soort: **{selected_coverage_type}**")
    elif selected_coverage_type != "totale bedekking":
        st.info(f"Je bekijkt de verspreiding van de groep: **{selected_coverage_type}**")

if layer_mode == "Vegetatie":
    st.caption("Legenda: Rood (0%) → Geel → Donkergroen (Hoge bedekking)")
elif layer_mode == "Diepte":
    st.caption("Legenda: Lichtblauw (Ondiep) → Donkerblauw (Diep)")
else:
    st.caption("Legenda: Bruin (Troebel) → Groen (Helder)")

# -----------------------------------------------------------------------------
# MAP DATA + RENDER
# -----------------------------------------------------------------------------
map_obj = None

if layer_mode in ["Diepte", "Doorzicht"]:
    # direct (geen heavy work)
    df_map_data = df_locs.copy()
    if "waarde_veg" not in df_map_data.columns:
        df_map_data["waarde_veg"] = 0.0
    map_obj = create_map(df_map_data, layer_mode, label_veg=selected_coverage_type)

else:
    # Vegetatie mode
    if analysis_level == "groepen & aggregaties" and selected_coverage_type in PIE_TYPES:
        # PIES: alles uit precomp
        df_locs_for_map = df_locs.copy()

        if selected_coverage_type == "Groeivormen":
            counts_by_loc = precomp["counts_groeivormen"]

            color_map = {
                "Ondergedoken": "#2ca02c",
                "Emergent": "#ffd700",
                "Draadalgen": "#c2a5cf",
                "Drijvend": "#ff7f0e",
                "FLAB": "#d62728",
                "Kroos": "#8c510a",
            }
            order = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]

            map_obj = create_pie_map(
                df_locs_for_map,
                counts_by_loc=counts_by_loc,
                label="Groeivormen (% bedekking)",
                color_map=color_map,
                order=order,
                size_px=30,
                zoom_start=10,
                fixed_total=100,
                fill_gap=True,
                gap_color="transparent",
            )

        elif selected_coverage_type == "Trofieniveau":
            counts_by_loc = precomp["counts_trof"]

            color_map = {
                "oligotroof": "#2ca02c",
                "mesotroof": "#1f77b4",
                "eutroof": "#ff7f0e",
                "sterk eutroof": "#d62728",
                "brak": "#ffd700",
                "marien": "#8c510a",
                "kroos": "#7f7f7f",
                "Onbekend": "#999999",
            }
            order = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos", "Onbekend"]

            map_obj = create_pie_map(
                df_locs_for_map,
                counts_by_loc=counts_by_loc,
                label="Trofieniveau (records)",
                color_map=color_map,
                order=order,
                size_px=30,
                zoom_start=10,
            )

        elif selected_coverage_type == "KRW score":
            counts_by_loc = precomp["counts_krw"]

            color_map = {
                "Gunstig (1-2)": "#2ca02c",
                "Neutraal (3-4)": "#ff7f0e",
                "Ongewenst (5)": "#d62728",
            }
            order = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"]

            map_obj = create_pie_map(
                df_locs_for_map,
                counts_by_loc=counts_by_loc,
                label="KRW-score (records)",
                color_map=color_map,
                order=order,
                size_px=30,
                zoom_start=10,
            )

        else:  # soortgroepen
            counts_by_loc = precomp["counts_soortgroepen"]

            color_map = {
                "chariden": "#1b9e77",
                "iseotiden": "#7570b3",
                "parvopotamiden": "#d95f02",
                "magnopotamiden": "#66a61e",
                "myriophylliden": "#e7298a",
                "vallisneriiden": "#e6ab02",
                "elodeiden": "#a6761d",
                "stratiotiden": "#1f78b4",
                "pepliden": "#b2df8a",
                "batrachiiden": "#fb9a99",
                "nymphaeiden": "#cab2d6",
                "haptofyten": "#fdbf6f",
                "Kenmerkende soort (N2000)": "#000000",
                "Overig / Individueel": "#999999",
            }
            order = [
                "chariden", "iseotiden", "parvopotamiden", "magnopotamiden", "myriophylliden",
                "vallisneriiden", "elodeiden", "stratiotiden", "pepliden", "batrachiiden",
                "nymphaeiden", "haptofyten", "Kenmerkende soort (N2000)", "Overig / Individueel"
            ]

            map_obj = create_pie_map(
                df_locs_for_map,
                counts_by_loc=counts_by_loc,
                label="Soortgroepen (% bedekking)",
                color_map=color_map,
                order=order,
                size_px=30,
                zoom_start=10,
                fixed_total=100,
                fill_gap=True,
                gap_color="transparent",
            )

    else:
        # Numerieke vegetatiekaart of individuele soort
        df_map_data = df_locs.copy()

        if analysis_level == "individuele soorten":
            df_sub = df_filtered[df_filtered["soort"] == selected_coverage_type][["locatie_id", "bedekking_pct"]].copy()
            df_veg = (
                df_sub.groupby("locatie_id", as_index=False)["bedekking_pct"]
                .mean()
                .rename(columns={"bedekking_pct": "waarde_veg"})
            )
            df_map_data = df_map_data.merge(df_veg, on="locatie_id", how="left")
            df_map_data["waarde_veg"] = _safe_numeric(df_map_data["waarde_veg"], 0.0)
            map_obj = create_map(df_map_data, "Vegetatie", label_veg=selected_coverage_type)

        else:
            if selected_coverage_type == "totale bedekking":
                df_veg = (
                    df_filtered.groupby("locatie_id", as_index=False)["totaal_bedekking_locatie"]
                    .mean()
                    .rename(columns={"totaal_bedekking_locatie": "waarde_veg"})
                )
                df_map_data = df_map_data.merge(df_veg, on="locatie_id", how="left")
                df_map_data["waarde_veg"] = _safe_numeric(df_map_data["waarde_veg"], 0.0)
                map_obj = create_map(df_map_data, "Vegetatie", label_veg="totale bedekking")

            elif selected_coverage_type == "KRW score":
                # Weighted KRW per locatie: uit precomp als die er is, anders snel berekenen
                if precomp is None:
                    # fallback (zou zelden gebeuren)
                    df_sub = df_filtered[
                        (df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(EXCLUDED_SPECIES_CODES))
                    ][["locatie_id", "krw_score", "bedekking_pct"]].dropna(subset=["krw_score"])
                    df_veg = _weighted_mean(df_sub, "locatie_id", "krw_score", "bedekking_pct", "waarde_veg")
                else:
                    df_veg = precomp["krw_loc"].rename(columns={"krw_score_loc": "waarde_veg"})

                df_map_data = df_map_data.merge(df_veg, on="locatie_id", how="left")
                df_map_data["waarde_veg"] = _safe_numeric(df_map_data["waarde_veg"], 0.0)
                map_obj = create_map(df_map_data, "Vegetatie", label_veg="KRW score", value_style="krw")

            elif selected_coverage_type == "Trofieniveau":
                # Dominant trofie per locatie (categorisch) uit precomp
                if precomp is None:
                    df_sub = df_filtered[
                        (df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES))
                    ][["locatie_id", "trofisch_niveau", "bedekking_pct"]].dropna(subset=["trofisch_niveau"])
                    dom = _dominant_category(df_sub, "locatie_id", "trofisch_niveau", "bedekking_pct", "trofieniveau_loc")
                else:
                    dom = precomp["trof_loc"]

                df_map_data = df_map_data.merge(dom, on="locatie_id", how="left")
                df_map_data["trofieniveau_loc"] = df_map_data["trofieniveau_loc"].fillna("Onbekend")

                cat_colors = {
                    "oligotroof": "#2ca02c",
                    "mesotroof": "#1f77b4",
                    "eutroof": "#ff7f0e",
                    "sterk eutroof": "#d62728",
                    "brak": "#ffd700",
                    "marien": "#8c510a",
                    "kroos": "#7f7f7f",
                    "Onbekend": "#999999",
                }

                # create_map ondersteunt categorical rendering (in utils)
                map_obj = create_map(
                    df_map_data,
                    mode="Vegetatie",
                    label_veg="Trofieniveau (dominant)",
                    value_style="categorical",
                    category_col="trofieniveau_loc",
                    category_color_map=cat_colors,
                )
            else:
                # fallback
                df_map_data["waarde_veg"] = 0.0
                map_obj = create_map(df_map_data, "Vegetatie", label_veg=selected_coverage_type)

# Render map
st_folium(map_obj, height=600, width=None)

# -----------------------------------------------------------------------------
# Toelichting Trofieniveau
# -----------------------------------------------------------------------------
if analysis_level == "groepen & aggregaties" and selected_coverage_type == "Trofieniveau":
    with st.expander("ℹ️ Toelichting trofieniveau"):
        st.markdown(
            """
De indeling van soorten naar trofieniveau is gebaseerd op:
**Verhofstad et al. (2025)** – *Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang*.
🔗 https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf
"""
        )

# -----------------------------------------------------------------------------
# TABEL (met extra kolommen)
# -----------------------------------------------------------------------------
st.divider()
with st.expander(f"Toon data voor {selected_coverage_type}"):
    df_table = df_locs.copy()
    # voeg waarde_veg toe indien aanwezig
    if "df_map_data" in locals() and "waarde_veg" in df_map_data.columns:
        df_table = df_table.merge(df_map_data[["locatie_id", "waarde_veg"]], on="locatie_id", how="left")
    else:
        df_table["waarde_veg"] = np.nan

    # extra loc cols: gebruik precomp als beschikbaar, anders light compute
    if precomp is not None:
        df_table = df_table.merge(precomp["krw_loc"], on="locatie_id", how="left").merge(precomp["trof_loc"], on="locatie_id", how="left")
    else:
        # light compute (zonder soortgroepen mapping)
        df_species_for_loc = df_filtered[
            (df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES))
        ][["locatie_id", "krw_score", "trofisch_niveau", "bedekking_pct"]].copy()

        krw_loc = _weighted_mean(df_species_for_loc.dropna(subset=["krw_score"]), "locatie_id", "krw_score", "bedekking_pct", "krw_score_loc")
        trof_loc = _dominant_category(df_species_for_loc.dropna(subset=["trofisch_niveau"]), "locatie_id", "trofisch_niveau", "bedekking_pct", "trofieniveau_loc")

        df_table = df_table.merge(krw_loc, on="locatie_id", how="left").merge(trof_loc, on="locatie_id", how="left")

    df_display = df_table.copy()

    # sortering
    if "waarde_veg" in df_display.columns:
        df_display = df_display.sort_values(by="waarde_veg", ascending=False)
    elif "krw_score_loc" in df_display.columns:
        df_display = df_display.sort_values(by="krw_score_loc", ascending=True)

    cols = ["locatie_id", "Waterlichaam", "waarde_veg", "krw_score_loc", "trofieniveau_loc", "diepte_m", "doorzicht_m"]
    cols = [c for c in cols if c in df_display.columns]

    # formattering
    if selected_coverage_type == "KRW score":
        waarde_cfg = st.column_config.NumberColumn("waarde_veg (KRW)", format="%.2f")
    elif selected_coverage_type == "Trofieniveau":
        waarde_cfg = st.column_config.TextColumn("waarde_veg")
    else:
        waarde_cfg = st.column_config.NumberColumn(f"{selected_coverage_type} (%)", format="%.1f%%")

    st.dataframe(
        df_display[cols],
        use_container_width=True,
        column_config={
            "waarde_veg": waarde_cfg,
            "krw_score_loc": st.column_config.NumberColumn("KRW score (locatie)", format="%.2f"),
            "trofieniveau_loc": st.column_config.TextColumn("Trofieniveau (dominant)"),
            "diepte_m": st.column_config.NumberColumn("Diepte (m)", format="%.2f"),
            "doorzicht_m": st.column_config.NumberColumn("Doorzicht (m)", format="%.2f"),
        },
    )