# 3_Ecologische_indices.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data, add_species_group_columns

st.set_page_config(layout="wide")
st.title("🌿 Ecologische indices")

# -----------------------------------------------------------------------------
# Helpers: caching op intensieve stappen (per selectie), zonder functieverlies
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _filter_base(df: pd.DataFrame, projects: tuple, bodies: tuple) -> pd.DataFrame:
    """Filter op project + waterlichaam (basis voor alle analyses)."""
    out = df[df["Project"].isin(projects)]
    out = out[out["Waterlichaam"].isin(bodies)]
    return out.copy()

@st.cache_data(show_spinner=False)
def _species_only(df_base: pd.DataFrame) -> pd.DataFrame:
    """Alleen type=Soort + numeric kolommen alvast casten voor snellere aggregaties."""
    df_s = df_base[df_base["type"] == "Soort"].copy()
    # numeric casts (1x) -> voorkomt herhaalde coercions in groupby’s
    for col in ["doorzicht_m", "diepte_m", "bedekking_pct", "krw_score"]:
        if col in df_s.columns:
            df_s[col] = pd.to_numeric(df_s[col], errors="coerce")
    return df_s

@st.cache_data(show_spinner=False)
def _bubble_yearly(df_species: pd.DataFrame) -> pd.DataFrame:
    """
    Stap 1 bubble: per (soort, jaar) jaarlijkse gemiddelden.
    Dit is de zware groupby en is goed te cachen.
    """
    if df_species.empty:
        return pd.DataFrame(columns=["soort", "jaar", "doorzicht_m", "bedekking_pct", "diepte_m"])

    df_bubble = (
        df_species.groupby(["soort", "jaar"], as_index=False)
        .agg(
            doorzicht_m=("doorzicht_m", "mean"),
            bedekking_pct=("bedekking_pct", "mean"),
            diepte_m=("diepte_m", "mean"),
        )
    )
    return df_bubble

@st.cache_data(show_spinner=False)
def _bubble_period_means(df_bubble: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    """
    Stap 2 bubble: filter jaar-range + mean-of-means per soort.
    (Bewust hetzelfde gedrag als in jouw toelichting: mean van jaarlijkse gemiddelden.)
    """
    if df_bubble.empty:
        return pd.DataFrame(columns=["soort", "doorzicht_m", "bedekking_pct", "diepte_m", "doorzicht_diepte_ratio"])

    df_rng = df_bubble[(df_bubble["jaar"] >= year_min) & (df_bubble["jaar"] <= year_max)]
    if df_rng.empty:
        return pd.DataFrame(columns=["soort", "doorzicht_m", "bedekking_pct", "diepte_m", "doorzicht_diepte_ratio"])

    df_plot = (
        df_rng.groupby("soort", as_index=False)
        .agg(
            doorzicht_m=("doorzicht_m", "mean"),
            bedekking_pct=("bedekking_pct", "mean"),
            diepte_m=("diepte_m", "mean"),
        )
    )

    # ratio doorzicht/diepte
    diepte_safe = df_plot["diepte_m"].astype(float).copy()
    diepte_safe = diepte_safe.where(diepte_safe > 0, np.nan)
    df_plot["doorzicht_diepte_ratio"] = df_plot["doorzicht_m"] / diepte_safe

    # verwijderen waar ratio niet kan
    df_plot = df_plot.dropna(subset=["doorzicht_diepte_ratio"])

    # bubble-size fix (Plotly: size > 0)
    df_plot["diepte_m"] = df_plot["diepte_m"].fillna(0.1)
    df_plot.loc[df_plot["diepte_m"] <= 0, "diepte_m"] = 0.1

    return df_plot

@st.cache_data(show_spinner=False)
def _heatmap_matrix(
    df_base: pd.DataFrame,
    heatmap_param: str,
    heatmap_basis: str,
    normalize_year: bool
) -> tuple[pd.DataFrame, str, str]:
    """
    Bouw heatmap matrix categorie x jaar, met keuze:
    - Records (count)
    - Bedekking-gewogen (sum)
    - Normaliseren per jaar naar 100%
    Retourneert: (matrix, category_column_name, value_label)
    """
    if df_base.empty:
        return pd.DataFrame(), "", ""

    # brondata per param
    if heatmap_param == "Groeivormen":
        df_h = df_base[df_base["type"] == "Groep"].copy()
        df_h = df_h.dropna(subset=["groeivorm", "jaar"])
        cat_col = "groeivorm"
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Soortgroepen":
        df_species = df_base[df_base["type"] == "Soort"].copy()
        # als Grootheid bestaat, neem BEDKG mee (compat met je oude intentie)
        if "Grootheid" in df_species.columns:
            df_species = df_species[df_species["Grootheid"] == "BEDKG"].copy()

        # zwaar: mapping -> alleen uitvoeren in dit pad, en cache via deze functie
        df_h = add_species_group_columns(df_species)
        df_h = df_h.dropna(subset=["soortgroep", "jaar"])
        cat_col = "soortgroep"
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekkingsgraad_proc"], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Trofieniveau":
        df_h = df_base[df_base["type"] == "Soort"].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()
        df_h = df_h.dropna(subset=["trofisch_niveau", "jaar"])
        cat_col = "trofisch_niveau"
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    else:  # "KRW score"
        df_h = df_base[df_base["type"] == "Soort"].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()

        if "krw_class" in df_h.columns and df_h["krw_class"].notna().any():
            df_h["krw_cat"] = df_h["krw_class"]
        else:
            df_h["krw_cat"] = pd.cut(
                pd.to_numeric(df_h.get("krw_score"), errors="coerce"),
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True,
            )
        df_h = df_h.dropna(subset=["krw_cat", "jaar"])
        cat_col = "krw_cat"
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    if df_h.empty:
        return pd.DataFrame(), cat_col, ""

    # aggregatie
    if heatmap_basis.startswith("Records"):
        df_agg = df_h.groupby([cat_col, "jaar"]).size().reset_index(name="waarde")
        value_label = "Aandeel (%)" if normalize_year else "Aantal records"
    else:
        df_agg = df_h.groupby([cat_col, "jaar"])["bedekking_num"].sum().reset_index(name="waarde")
        value_label = "Aandeel (%)" if normalize_year else "Som bedekking"

    heat = df_agg.pivot(index=cat_col, columns="jaar", values="waarde").fillna(0)

    # normaliseren per jaar
    if normalize_year:
        col_sums = heat.sum(axis=0).replace(0, np.nan)
        heat = heat.div(col_sums, axis=1).fillna(0) * 100

    # optionele ordering voor bekende categorieën (zoals in je oude code)
    if heatmap_param == "KRW score":
        order = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"]
        heat = heat.reindex([x for x in order if x in heat.index])

    elif heatmap_param == "Trofieniveau":
        order = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos"]
        keep = [x for x in order if x in heat.index]
        rest = [x for x in heat.index if x not in keep]
        heat = heat.reindex(keep + rest)

    elif heatmap_param == "Groeivormen":
        order = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]
        keep = [x for x in order if x in heat.index]
        rest = [x for x in heat.index if x not in keep]
        heat = heat.reindex(keep + rest)

    return heat, cat_col, value_label


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
df = load_data()
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Selectie filters")

all_projects = sorted(df["Project"].dropna().unique())
selected_projects = st.sidebar.multiselect(
    "Selecteer project(en)",
    options=all_projects,
    default=all_projects,
)

df_project = df[df["Project"].isin(selected_projects)]
all_bodies = sorted(df_project["Waterlichaam"].dropna().unique())
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam",
    options=all_bodies,
    default=all_bodies,
)

df_filtered_base = _filter_base(df, tuple(selected_projects), tuple(selected_bodies))

# alleen soorten (gecached + numeric)
df_species_only = _species_only(df_filtered_base)

# -----------------------------------------------------------------------------
# BUBBLE PLOT
# -----------------------------------------------------------------------------
st.subheader("Relatie doorzicht vs bedekking")

with st.expander("ℹ️ Hoe komt deze bubble plot tot stand? (toelichting)"):
    st.markdown(
        """
### Wat stelt één bubble voor?
Elke **bubble staat voor één individuele plantensoort** (de wetenschappelijke naam in `soort`).

### Welke data gaat er de plot in?
Na jouw selectie op **project** en **waterlichaam** worden alleen records met `type == 'Soort'` meegenomen.

### Stap 1 — Aggregatie per soort × jaar
Per (soort, jaar) worden gemiddelden berekend voor:
- `doorzicht_m`, `bedekking_pct`, `diepte_m`

### Stap 2 — Selectie van periode + aggregatie per soort
De gekozen periode wordt gefilterd en daarna wordt per soort het gemiddelde genomen van de jaarlijkse gemiddelden (“mean-of-means”).

### Assen
- y-as: gemiddelde bedekking (%)
- bubble size: gemiddelde diepte (m)
- x-as: `doorzicht_m / diepte_m` (ratio)
"""
    )

df_bubble = _bubble_yearly(df_species_only)

if df_bubble.empty or df_bubble["jaar"].dropna().empty:
    st.warning("Geen data beschikbaar voor bubbleplot na filtering.")
else:
    min_year = int(df_bubble["jaar"].min())
    max_year = int(df_bubble["jaar"].max())

    sel_years = st.slider(
        "Selecteer periode",
        min_year,
        max_year,
        [min_year, max_year],
        key="ecol_bubble_period",
    )

    df_bubble_plot = _bubble_period_means(df_bubble, int(sel_years[0]), int(sel_years[1]))

    if df_bubble_plot.empty:
        st.warning("Geen data gevonden voor deze filtercombinatie.")
    else:
        fig_bubble = px.scatter(
            df_bubble_plot,
            x="doorzicht_diepte_ratio",
            y="bedekking_pct",
            size="diepte_m",
            hover_name="soort",
            size_max=40,
            title=f"Ecologische indices ({sel_years[0]} - {sel_years[1]})",
            labels={
                "doorzicht_diepte_ratio": "gem. doorzicht / gem. diepte (-)",
                "bedekking_pct": "gem. bedekking (%)",
                "diepte_m": "gem. diepte (m)",
            },
        )

        # zones + lijnen (zelfde intentie als jouw code)
        fig_bubble.add_vrect(
            x0=0.6, x1=0.8,
            fillcolor="rgba(46, 204, 113, 0.16)",
            line_width=0,
            layer="below",
            annotation_text="OK (0.6–0.8)",
            annotation_position="top left",
        )
        fig_bubble.add_vrect(
            x0=0.8, x1=1.0,
            fillcolor="rgba(39, 174, 96, 0.24)",
            line_width=0,
            layer="below",
            annotation_text="Ideaal (≥0.8)",
            annotation_position="top left",
        )
        fig_bubble.add_vline(
            x=0.6,
            line_width=2,
            line_dash="dot",
            line_color="rgba(255, 165, 0, 0.85)",
            annotation_text="Min 0.6",
            annotation_position="top left",
        )
        fig_bubble.add_vline(
            x=0.8,
            line_width=2,
            line_dash="dot",
            line_color="rgba(0, 100, 0, 0.90)",
            annotation_text="Streef 0.8",
            annotation_position="top left",
        )

        st.plotly_chart(fig_bubble, use_container_width=True)

# -----------------------------------------------------------------------------
# HEATMAP
# -----------------------------------------------------------------------------
st.subheader("📊 Verdeling per jaar (heatmap)")

with st.expander("ℹ️ Uitleg: hoe wordt deze heatmap berekend?", expanded=False):
    st.markdown(
        """
De heatmap toont per **jaar** hoe de **verdeling** eruitziet van een gekozen parameter:
- Trofieniveau
- Groeivormen
- Soortgroepen
- KRW score

Je kunt kiezen:
- **Records** (aantal regels)
- **Bedekking-gewogen** (som van bedekking)
En optioneel normaliseren per jaar (0–100% verdeling).
"""
    )

heatmap_param = st.selectbox("Kies parameter voor heatmap", ["Trofieniveau", "Groeivormen", "Soortgroepen", "KRW score"])
heatmap_basis = st.radio(
    "Bereken verdeling op basis van",
    ["Records (aantal waarnemingen)", "Bedekking-gewogen (som bedekking)"],
    index=1,
    horizontal=True,
    key="heatmap_basis_choice",
)
normalize_year = st.checkbox("Normaliseer per jaar (100% verdeling)", value=True)

years = sorted(df_filtered_base["jaar"].dropna().unique())
if not years:
    st.info("Geen jaren beschikbaar voor heatmap.")
else:
    heat_matrix, cat_col, value_label = _heatmap_matrix(df_filtered_base, heatmap_param, heatmap_basis, normalize_year)

    if heat_matrix.empty:
        st.info("Geen data beschikbaar voor deze heatmap-keuze (na filters).")
    else:
        fig = px.imshow(
            heat_matrix,
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(x="Jaar", y=heatmap_param, color=value_label),
            title=f"Heatmap {heatmap_param} per jaar" + (" (genormaliseerd)" if normalize_year else ""),
        )

        # tekst in cellen
        if normalize_year:
            text_matrix = heat_matrix.round(1).astype(str) + "%"
        else:
            if heatmap_basis.startswith("Records"):
                text_matrix = heat_matrix.round(0).astype(int).astype(str)
            else:
                text_matrix = heat_matrix.round(1).astype(str)

        fig.update_traces(
            text=text_matrix.values,
            texttemplate="%{text}",
            textfont=dict(color="white", size=12),
        )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="show", height=650, yaxis=dict(side="left"))
        st.plotly_chart(fig, use_container_width=True)

        if heatmap_param == "Trofieniveau":
            st.caption(
                "Bron trofieniveau-indeling: Verhofstad et al. (2025) – Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang. "
                "https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf"
            )