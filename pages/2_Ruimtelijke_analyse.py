# 2_Ruimtelijke_analyse.py
import streamlit as st
import pandas as pd
import numpy as np
import html
from streamlit_folium import st_folium

from utils import (
    load_data,
    add_species_group_columns,
    create_pie_map,
    create_map,
    get_sorted_species_list,
    get_species_group_mapping,
    EXCLUDED_SPECIES_CODES,
    RWS_GROEIVORM_CODES,
    build_bathymetry_legend_url,
)

st.set_page_config(layout="wide", page_title="Ruimtelijke analyse")
st.title("🗺️ Ruimtelijke analyse")
st.markdown("Vergelijk de vegetatieontwikkeling met diepte en doorzicht.")

# -----------------------------------------------------------------------------
# CONSTANTS (consistent met UI-opties)
# -----------------------------------------------------------------------------
PIE_TYPES = ["KRW score", "Trofieniveau", "Groeivormen", "soortgroepen"]


VEGETATION_LEGEND_ITEMS = [
    ("0%", "#d73027"),
    ("1–5%", "#fc8d59"),
    ("6–15%", "#fee08b"),
    ("16–40%", "#d9ef8b"),
    ("41–75%", "#91cf60"),
    ("> 75%", "#1a9850"),
]
TOTAL_BED_COVER_LEGEND_ITEMS = [
    ("0%", "#808080"),
    ("0.01–1%", "#006400"),
    ("1–5%", "#2ca02c"),
    ("5–15%", "#ffd700"),
    ("15–25%", "#fdb462"),
    ("25–50%", "#ff7f0e"),
    ("50–75%", "#d95f02"),
    ("75–100%", "#d73027"),
]
GROEIVORM_COLOR_MAP = {
    "Ondergedoken": "#2ca02c",
    "Emergent": "#ffd700",
    "Draadalgen": "#c2a5cf",
    "Drijvend": "#ff7f0e",
    "FLAB": "#d62728",
    "Kroos": "#8c510a",
}
GROEIVORM_ORDER = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]
TROFIENIVEAU_COLOR_MAP = {"oligotroof": "#1b9e77","mesotroof": "#d95f02","eutroof": "#7570b3","sterk eutroof": "#e7298a","brak": "#66a61e","marien": "#e6ab02","kroos": "#a6761d","Onbekend": "#666666","Geen match": "#bbbbbb"}
TROFIENIVEAU_ORDER = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos", "Onbekend", "Geen match"]
KRW_COLOR_MAP = {
    "Gunstig (1-2)": "#2ca02c",
    "Neutraal (3-4)": "#ff7f0e",
    "Ongewenst (5)": "#d62728",
    "Geen match": "#9e9e9e",
}
KRW_ORDER = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)", "Geen match"]
def _ordered_unique(values) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        v = str(value).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        ordered.append(v)
    return ordered


_SOORTGROEPEN_BASE_ORDER = _ordered_unique(get_species_group_mapping().values())
SOORTGROEPEN_ORDER = _SOORTGROEPEN_BASE_ORDER + ["Overig / Individueel", "Geen match"]
_SOORTGROEPEN_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    "#a6761d", "#1f78b4", "#b2df8a", "#fb9a99", "#cab2d6", "#fdbf6f",
    "#6a3d9a", "#ff7f00", "#b15928", "#17becf",
]
SOORTGROEPEN_COLOR_MAP = {
    group: _SOORTGROEPEN_PALETTE[i % len(_SOORTGROEPEN_PALETTE)]
    for i, group in enumerate(_SOORTGROEPEN_BASE_ORDER)
}
SOORTGROEPEN_COLOR_MAP.update({
    "Overig / Individueel": "#FFD700",
    "Geen match": "#bbbbbb",
})


def _render_html_legend(title: str, entries, note: str | None = None, columns: int = 3):
    if not entries:
        return

    safe_columns = max(1, int(columns))
    items_html = "".join(
        (
            f'<div style="display:flex; align-items:center; gap:0.5rem; min-width:0;">'
            f'<span style="display:inline-block; width:16px; height:16px; border-radius:3px; '
            f'border:1px solid #666; background:{color}; flex:0 0 16px;"></span>'
            f'<span style="font-size:0.95rem; line-height:1.2;">{html.escape(str(label))}</span>'
            f'</div>'
        )
        for label, color in entries
    )

    note_html = ""
    if note:
        note_html = (
            f'<div style="margin-top:0.45rem; color:#666; font-size:0.85rem;">'
            f'{html.escape(str(note))}'
            f'</div>'
        )

    legend_html = (
        f'<div style="margin:0.35rem 0 0.8rem 0; padding:0.8rem 0.95rem; '
        f'border:1px solid #d9d9d9; border-radius:0.6rem; background:#fafafa;">'
        f'<div style="font-weight:600; margin-bottom:0.55rem;">{html.escape(str(title))}</div>'
        f'<div style="display:grid; grid-template-columns:repeat({safe_columns}, minmax(0, 1fr)); '
        f'gap:0.45rem 0.9rem;">{items_html}</div>'
        f'{note_html}'
        f'</div>'
    )

    st.markdown(legend_html, unsafe_allow_html=True)


def render_active_map_legend(layer_mode: str, analysis_level: str, selected_coverage_type: str):
    if layer_mode != "Vegetatie":
        return

    if analysis_level == "individuele soorten":
        _render_html_legend(
            "Legenda vegetatiebedekking",
            VEGETATION_LEGEND_ITEMS,
            note="Zelfde kleurschaal als de markeringen op de kaart.",
            columns=3,
        )
        return

    if selected_coverage_type == "totale bedekking":
        _render_html_legend(
            "Legenda totale bedekking",
            TOTAL_BED_COVER_LEGEND_ITEMS,
            note="Deze specifieke kleurschaal geldt alleen voor de kaartmarkeringen van totale bedekking.",
            columns=4,
        )
    elif selected_coverage_type == "Groeivormen":
        _render_html_legend(
            "Legenda groeivormen",
            [(label, GROEIVORM_COLOR_MAP[label]) for label in GROEIVORM_ORDER],
            note="De taartdiagrammen behouden de huidige opvulling; de kleuren hieronder tonen de categorieën in de kaart.",
            columns=3,
        )
    elif selected_coverage_type == "Trofieniveau":
        _render_html_legend(
            "Legenda trofieniveau",
            [(label, TROFIENIVEAU_COLOR_MAP[label]) for label in TROFIENIVEAU_ORDER],
            note="Vaste kleurschakering voor trofieniveaus op de kaart.",
            columns=3,
        )
    elif selected_coverage_type == "KRW score":
        _render_html_legend(
            "Legenda KRW-score",
            [(label, KRW_COLOR_MAP[label]) for label in KRW_ORDER],
            note="Vaste kleurschakering voor de KRW-score op de kaart.",
            columns=3,
        )
    elif selected_coverage_type == "soortgroepen":
        _render_html_legend(
            "Legenda soortgroepen",
            [(label, SOORTGROEPEN_COLOR_MAP[label]) for label in SOORTGROEPEN_ORDER],
            note="De taartdiagrammen behouden de huidige opvulling; de kleuren hieronder tonen de soortgroepen in de kaart.",
            columns=2,
        )


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


def _ensure_nomatch_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maak ruimtelijke analyse robuust voor oudere datasets zonder display-kolommen."""
    if df is None or df.empty:
        return df
    df = df.copy()

    if "trofisch_niveau" not in df.columns:
        df["trofisch_niveau"] = np.nan
    if "trofisch_niveau_weergave" not in df.columns:
        df["trofisch_niveau_weergave"] = np.where(
            df["trofisch_niveau"].notna() & (df["trofisch_niveau"].astype(str).str.strip() != ""),
            df["trofisch_niveau"].astype(str),
            "Geen match",
        )

    if "krw_score" not in df.columns:
        df["krw_score"] = np.nan
    if "krw_class" not in df.columns:
        df["krw_class"] = pd.cut(
            pd.to_numeric(df["krw_score"], errors="coerce"),
            bins=[0, 2, 4, 5],
            labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
            include_lowest=True,
        )
    if "krw_class_weergave" not in df.columns:
        df["krw_class_weergave"] = df["krw_class"].astype(object)
        df.loc[df["krw_class_weergave"].isna(), "krw_class_weergave"] = "Geen match"

    return df


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
    species_base_cols = ["locatie_id", "soort", "bedekking_pct", "krw_score", "krw_class", "krw_class_weergave", "trofisch_niveau", "trofisch_niveau_weergave"]
    df_species_base = df_filtered[(df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(EXCLUDED_SPECIES_CODES))].copy()
    for col in species_base_cols:
        if col not in df_species_base.columns:
            df_species_base[col] = np.nan
    df_species_base = df_species_base[species_base_cols].copy()

    # Trofieniveau counts (records)
    trof_col = "trofisch_niveau_weergave" if "trofisch_niveau_weergave" in df_species_base.columns else "trofisch_niveau"
    df_trof_counts = df_species_base[["locatie_id", trof_col]].copy()
    df_trof_counts[trof_col] = df_trof_counts[trof_col].fillna("Geen match")
    if df_trof_counts.empty:
        counts_trof = {}
    else:
        pivot_trof = df_trof_counts.groupby(["locatie_id", trof_col]).size().unstack(fill_value=0)
        counts_trof = _pivot_to_counts_by_loc(pivot_trof)

    # KRW counts (records)
    df_krw_counts = df_species_base.copy()
    if "krw_class_weergave" in df_krw_counts.columns:
        df_krw_counts["krw_cat"] = df_krw_counts["krw_class_weergave"].fillna("Geen match")
    elif "krw_class" in df_krw_counts.columns and df_krw_counts["krw_class"].notna().any():
        df_krw_counts["krw_cat"] = df_krw_counts["krw_class"].astype(object).where(df_krw_counts["krw_class"].notna(), "Geen match")
    else:
        df_krw_counts["krw_cat"] = pd.cut(
            _safe_numeric(df_krw_counts["krw_score"], np.nan),
            bins=[0, 2, 4, 5],
            labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
            include_lowest=True,
        ).astype(object)
        df_krw_counts.loc[df_krw_counts["krw_cat"].isna(), "krw_cat"] = "Geen match"
    df_krw_counts = df_krw_counts[["locatie_id", "krw_cat"]]
    if df_krw_counts.empty:
        counts_krw = {}
    else:
        pivot_krw = df_krw_counts.groupby(["locatie_id", "krw_cat"]).size().unstack(fill_value=0)
        counts_krw = _pivot_to_counts_by_loc(pivot_krw)

    # --- Extra kolommen voor tabel: KRW score loc (weighted), trofie dominant loc ---
    df_species_for_loc = df_filtered[
        (df_filtered["type"] == "Soort") & (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES))
    ][["locatie_id", "krw_score", "trofisch_niveau", "trofisch_niveau_weergave", "bedekking_pct"]].copy()

    # gewogen krw
    df_krw_loc = df_species_for_loc.dropna(subset=["krw_score"])
    krw_loc = _weighted_mean(df_krw_loc, "locatie_id", "krw_score", "bedekking_pct", "krw_score_loc")

    # dominant trofie
    trof_loc_col = "trofisch_niveau_weergave" if "trofisch_niveau_weergave" in df_species_for_loc.columns else "trofisch_niveau"
    df_species_for_loc[trof_loc_col] = df_species_for_loc[trof_loc_col].fillna("Geen match")
    df_trof_loc = df_species_for_loc.dropna(subset=[trof_loc_col])
    trof_loc = _dominant_category(df_trof_loc, "locatie_id", trof_loc_col, "bedekking_pct", "trofieniveau_loc")

    # --- Soortgroepen (heavy) ---
    # reduce before mapping: keep only relevant columns for add_species_group_columns
    df_species_raw = df_filtered[
        (~df_filtered["soort"].isin(RWS_GROEIVORM_CODES)) & (df_filtered["type"] != "Groep")
    ][["locatie_id", "soort", "type", "Grootheid", "bedekking_pct", "waarde_bedekking", "totaal_bedekking_locatie"]].copy()

    df_species_groups = add_species_group_columns(df_species_raw)
    # Kenmerkende soorten (N2000) expliciet uitsluiten van soortgroepen,
    # zodat deze als aparte entiteit behandeld kunnen worden (conform 1_Overzicht.py).
    if "is_kenmerkende_soort_n2000" in df_species_groups.columns:
        df_species_groups = df_species_groups[~df_species_groups["is_kenmerkende_soort_n2000"].fillna(False)].copy()
    df_species_groups["bedekking_num"] = _safe_numeric(df_species_groups["bedekkingsgraad_proc"], 0.0).clip(lower=0.0)

    pivot_soortgroepen = (
        df_species_groups.groupby(["locatie_id", "soortgroep_weergave" if "soortgroep_weergave" in df_species_groups.columns else "soortgroep"])["bedekking_num"]
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
df = _ensure_nomatch_display_columns(df)

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
df_filtered = _ensure_nomatch_display_columns(df_filtered)
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
    render_active_map_legend(layer_mode, analysis_level, selected_coverage_type)
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
    map_obj = create_map(df_map_data, layer_mode, label_veg=selected_coverage_type, basemap="bathymetry")

else:
    # Vegetatie mode
    if analysis_level == "groepen & aggregaties" and selected_coverage_type in PIE_TYPES:
        # PIES: alles uit precomp
        df_locs_for_map = df_locs.copy()

        if selected_coverage_type == "Groeivormen":
            counts_by_loc = precomp["counts_groeivormen"]

            color_map = GROEIVORM_COLOR_MAP
            order = GROEIVORM_ORDER

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
                basemap="bathymetry",
            )

        elif selected_coverage_type == "Trofieniveau":
            counts_by_loc = precomp["counts_trof"]

            color_map = TROFIENIVEAU_COLOR_MAP
            order = TROFIENIVEAU_ORDER

            map_obj = create_pie_map(
                df_locs_for_map,
                counts_by_loc=counts_by_loc,
                label="Trofieniveau (records)",
                color_map=color_map,
                order=order,
                size_px=30,
                zoom_start=10,
                basemap="bathymetry",
            )

        elif selected_coverage_type == "KRW score":
            counts_by_loc = precomp["counts_krw"]

            color_map = KRW_COLOR_MAP
            order = KRW_ORDER

            map_obj = create_pie_map(
                df_locs_for_map,
                counts_by_loc=counts_by_loc,
                label="KRW-score (records)",
                color_map=color_map,
                order=order,
                size_px=30,
                zoom_start=10,
                basemap="bathymetry",
            )

        else:  # soortgroepen
            counts_by_loc = precomp["counts_soortgroepen"]

            color_map = SOORTGROEPEN_COLOR_MAP
            order = SOORTGROEPEN_ORDER

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
                basemap="bathymetry",
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
            map_obj = create_map(df_map_data, "Vegetatie", label_veg=selected_coverage_type, basemap="bathymetry")

        else:
            if selected_coverage_type == "totale bedekking":
                df_veg = (
                    df_filtered.groupby("locatie_id", as_index=False)["totaal_bedekking_locatie"]
                    .mean()
                    .rename(columns={"totaal_bedekking_locatie": "waarde_veg"})
                )
                df_map_data = df_map_data.merge(df_veg, on="locatie_id", how="left")
                df_map_data["waarde_veg"] = _safe_numeric(df_map_data["waarde_veg"], 0.0)
                map_obj = create_map(df_map_data, "Vegetatie", label_veg="totale bedekking", value_style="total_bedekking", basemap="bathymetry")

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
                map_obj = create_map(df_map_data, "Vegetatie", label_veg="KRW score", value_style="krw", basemap="bathymetry")

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
                map_obj = create_map(df_map_data, "Vegetatie", label_veg=selected_coverage_type, basemap="bathymetry")

# Render map
st_folium(map_obj, height=600, width=None)

legend_url = build_bathymetry_legend_url()
with st.expander("Legenda bathymetrie", expanded=False):
    st.markdown(
        f'<img src="{legend_url}" alt="Legenda bathymetrie" style="max-width:360px; width:100%; height:auto;"/>',
        unsafe_allow_html=True,
    )
    st.caption("Legenda uit de WMS-kaartservice van Rijkswaterstaat.")

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
        ][["locatie_id", "krw_score", "trofisch_niveau", "trofisch_niveau_weergave", "bedekking_pct"]].copy()

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
        width='stretch',
        column_config={
            "waarde_veg": waarde_cfg,
            "krw_score_loc": st.column_config.NumberColumn("KRW score (locatie)", format="%.2f"),
            "trofieniveau_loc": st.column_config.TextColumn("Trofieniveau (dominant)"),
            "diepte_m": st.column_config.NumberColumn("Diepte (m)", format="%.2f"),
            "doorzicht_m": st.column_config.NumberColumn("Doorzicht (m)", format="%.2f"),
        },
    )