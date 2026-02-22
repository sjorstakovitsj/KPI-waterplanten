# -*- coding: utf-8 -*-
"""5_Tijd_trendontwikkeling.py

Aanpassingen t.o.v. vorige versie:
- Compatibel met nieuwe utils.load_data() (Parquet/DuckDB): defensieve type-casting (jaar/metrics) zodat object/strings uit Parquet geen issues geven.
- Performance:
  * Vectorized lineaire regressie (slope) per locatie i.p.v. Python-loop + np.polyfit.
  * Minder kopieën en gerichtere kolomselecties bij groupby.
  * sort=False/observed=True waar zinvol.

Functioneel gedrag/UX is gelijk gehouden.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html

from utils import (
    load_data,
    categorize_slope_trend,
    get_sorted_species_list,
    create_map,  # nog in gebruik in andere pages; hier niet direct, maar laten staan
    df_to_geojson_points,
    render_swipe_map_html,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_numeric(s: pd.Series) -> pd.Series:
    """Robuuste numeric casting (strings met komma/tekens) -> float."""
    if s is None:
        return pd.Series(dtype="float")
    if pd.api.types.is_numeric_dtype(s):
        return s
    # veel voorkomende formats in exports: '12,3', '<1', '>5'
    out = (
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(out, errors="coerce")


def _ensure_numeric_cols(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Zet opgegeven kolommen om naar numeriek (kopie alleen als nodig)."""
    df = df_in
    for c in cols:
        if c in df.columns:
            # alleen kopiëren als we echt gaan schrijven
            if df is df_in:
                df = df.copy()
            df[c] = _as_numeric(df[c])
    return df


def _compute_trend_table(df_filtered: pd.DataFrame, selected_metric: str) -> pd.DataFrame:
    """Aggregatie per (locatie_id, jaar) -> waarde."""
    if df_filtered.empty:
        return pd.DataFrame(columns=["locatie_id", "jaar", "waarde"])

    if selected_metric == "soort_count":
        # unieke soorten per locatie-jaar
        base = df_filtered[["locatie_id", "jaar", "soort"]].dropna(subset=["locatie_id", "jaar"])
        out = (
            base.groupby(["locatie_id", "jaar"], sort=False, observed=True)["soort"]
            .nunique()
            .reset_index(name="waarde")
        )
        return out

    # gemiddelde van gekozen metriek per locatie-jaar
    want_cols = ["locatie_id", "jaar", selected_metric]
    base = df_filtered[want_cols].dropna(subset=["locatie_id", "jaar"])
    out = (
        base.groupby(["locatie_id", "jaar"], sort=False, observed=True)[selected_metric]
        .mean()
        .reset_index(name="waarde")
    )
    return out


def _compute_slopes_vectorized(df_trend: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Vectorized slope (OLS) per locatie.

    slope = (n*sum(xy) - sum(x)sum(y)) / (n*sum(xx) - sum(x)^2)

    Equivalent aan np.polyfit(x,y,1)[0] voor lineaire regressie.
    """
    if df_trend.empty:
        return pd.DataFrame(columns=["locatie_id", "slope", "n_jaren"])

    # Maak numeriek (defensief)
    x = _as_numeric(df_trend["jaar"]).astype(float)
    y = _as_numeric(df_trend["waarde"]).astype(float)

    tmp = df_trend[["locatie_id"]].copy()
    tmp["x"] = x
    tmp["y"] = y
    tmp = tmp.dropna(subset=["locatie_id", "x", "y"])
    if tmp.empty:
        return pd.DataFrame(columns=["locatie_id", "slope", "n_jaren"])

    tmp["xy"] = tmp["x"] * tmp["y"]
    tmp["xx"] = tmp["x"] * tmp["x"]

    agg = (
        tmp.groupby("locatie_id", sort=False, observed=True)
        .agg(
            n=("x", "count"),
            sum_x=("x", "sum"),
            sum_y=("y", "sum"),
            sum_xy=("xy", "sum"),
            sum_xx=("xx", "sum"),
        )
        .reset_index()
    )

    # Filter op minimaal aantal meetjaren
    agg = agg[agg["n"] >= min_years].copy()
    if agg.empty:
        return pd.DataFrame(columns=["locatie_id", "slope", "n_jaren"])

    denom = (agg["n"] * agg["sum_xx"]) - (agg["sum_x"] ** 2)
    numer = (agg["n"] * agg["sum_xy"]) - (agg["sum_x"] * agg["sum_y"])

    # Vermijd deling door nul
    agg["slope"] = np.where(denom != 0, numer / denom, np.nan)

    out = agg[["locatie_id", "slope", "n"]].rename(columns={"n": "n_jaren"})
    out = out.dropna(subset=["slope"])
    return out


def _trend_cover(df_in: pd.DataFrame) -> pd.DataFrame:
    required = {"Waterlichaam", "jaar", "locatie_id", "bedekking_pct"}
    if df_in.empty or not required.issubset(df_in.columns):
        return pd.DataFrame(columns=["Waterlichaam", "jaar", "totaal_bedekking_locatie"])

    # Filter op soort + BEDKG waar mogelijk (zonder join, maar direct op df_in)
    dfx = df_in.copy()

    if "type" in dfx.columns:
        dfx = dfx.loc[dfx["type"] == "Soort"].copy()

    if "Grootheid" in dfx.columns:
        dfx = dfx.loc[dfx["Grootheid"] == "BEDKG"].copy()

    dfx = dfx[["Waterlichaam", "jaar", "locatie_id", "bedekking_pct"]].copy()
    dfx["jaar"] = pd.to_numeric(dfx["jaar"], errors="coerce")
    dfx["bedekking_pct"] = _as_numeric(dfx["bedekking_pct"])
    dfx = dfx.dropna(subset=["Waterlichaam", "jaar", "locatie_id", "bedekking_pct"])

    if dfx.empty:
        return pd.DataFrame(columns=["Waterlichaam", "jaar", "totaal_bedekking_locatie"])

    dfx["jaar"] = dfx["jaar"].astype(int)

    per_loc = (
        dfx.groupby(["Waterlichaam", "jaar", "locatie_id"], sort=False, observed=True)["bedekking_pct"]
        .sum()
        .reset_index(name="totaal_bedekking_locatie")
    )

    out = (
        per_loc.groupby(["Waterlichaam", "jaar"], sort=False, observed=True)["totaal_bedekking_locatie"]
        .mean()
        .reset_index()
        .sort_values(["Waterlichaam", "jaar"])
    )

    return out

# -----------------------------------------------------------------------------
# Streamlit pagina
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("📈 Tijd- en trendanalyse")

# -------------------------------------------------------------
# Data laden
# -------------------------------------------------------------
df = load_data()

# -------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------
st.sidebar.header("Selectiefilters")
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# 1) Waterlicha(a)m(en)
all_bodies = sorted(df["Waterlichaam"].dropna().unique()) if "Waterlichaam" in df.columns else []
selected_bodies = st.sidebar.multiselect(
    "Selecteer waterlichaam / waterlichamen",
    options=all_bodies,
    default=all_bodies[:1] if all_bodies else [],
)

# 2) Individuele soort
species_options = ["— Alle soorten —"] + get_sorted_species_list(df)
selected_species = st.sidebar.selectbox(
    "Selecteer individuele soort (optioneel)",
    species_options,
    index=0,
)
species_is_selected = bool(selected_species and selected_species != "— Alle soorten —")

# 2b) Plantgroep filter – alleen zinvol als GEEN individuele soort gekozen is
st.sidebar.subheader("Extra kaartfilter")
if species_is_selected:
    st.sidebar.caption("Plantgroep-filter is uitgeschakeld bij een gekozen soort.")
    show_ondergedoken = st.sidebar.checkbox("Onder­gedoken waterplanten", value=False, disabled=True)
    show_chariden = st.sidebar.checkbox("Chariden (kranswieren)", value=False, disabled=True)
else:
    show_ondergedoken = st.sidebar.checkbox("Onder­gedoken waterplanten", value=False)
    show_chariden = st.sidebar.checkbox("Chariden (kranswieren)", value=False)

# 3) Metriek-keuze (als soort is gekozen, beperken we tot bedekking)
if species_is_selected:
    metric_options = {
        "Bedekkingsgraad (%)": "bedekking_pct",
    }
else:
    metric_options = {
        "Bedekkingsgraad (%)": "bedekking_pct",
        "Doorzicht (m)": "doorzicht_m",
        "Diepte (m)": "diepte_m",
        "Soortenrijkdom": "soort_count",
    }
selected_metric_label = st.sidebar.selectbox("Kies analysevariabele", list(metric_options.keys()))
selected_metric = metric_options[selected_metric_label]

# -------------------------------------------------------------
# Data filteren
# -------------------------------------------------------------

# Waterlichaam filter
if selected_bodies and "Waterlichaam" in df.columns:
    df_filtered = df.loc[df["Waterlichaam"].isin(selected_bodies)].copy()
else:
    df_filtered = df.copy()

# Defensieve numeric casting (parquet/duckdb kan object/strings opleveren)
# NB: 'soort_count' bestaat niet in df_filtered; die wordt later geaggregeerd.
metric_cols_to_cast = [c for c in ["bedekking_pct", "doorzicht_m", "diepte_m", "lat", "lon", "jaar"] if c in df_filtered.columns]
df_filtered = _ensure_numeric_cols(df_filtered, metric_cols_to_cast)

# Soortfilter – alleen echte soorten met bedekking (BEDKG)
if species_is_selected:
    mask = df_filtered.get("soort", pd.Series(False, index=df_filtered.index)) == selected_species
    if "type" in df_filtered.columns:
        mask &= df_filtered["type"] == "Soort"
    if "Grootheid" in df_filtered.columns:
        mask &= df_filtered["Grootheid"] == "BEDKG"
    df_filtered = df_filtered.loc[mask].copy()

# -------------------------------------------------------------
# Extra filter: plantgroepen (ondergedoken / chariden)
# Alleen toepassen als GEEN individuele soort is gekozen
# -------------------------------------------------------------
if not species_is_selected and (show_ondergedoken or show_chariden):
    # Chariden herkennen op basis van (Latijnse) genusnamen in kolom 'soort'
    chara_genera = ("Chara", "Nitella", "Nitellopsis", "Tolypella")
    soort_ser = df_filtered.get("soort", pd.Series("", index=df_filtered.index)).fillna("").astype(str).str.strip()
    mask_chariden = soort_ser.str.startswith(chara_genera)

    # Ondergedoken waterplanten via groeivorm
    groeivorm_ser = df_filtered.get("groeivorm", pd.Series("", index=df_filtered.index)).fillna("").astype(str)
    mask_ondergedoken = groeivorm_ser == "Ondergedoken"

    mask = pd.Series(False, index=df_filtered.index)
    if show_ondergedoken:
        mask |= mask_ondergedoken
    if show_chariden:
        mask |= mask_chariden
    df_filtered = df_filtered.loc[mask].copy()

if df_filtered.empty:
    st.warning("Geen data beschikbaar voor deze selectie.")
    st.stop()

# -------------------------------------------------------------
# Voorbereiden trend-data (aggregatie per jaar/locatie)
# -------------------------------------------------------------

# Zorg dat jaar numeriek is (Int/float) voor groepering
if "jaar" in df_filtered.columns:
    df_filtered = df_filtered.dropna(subset=["jaar"]).copy()

# Trendtable
# (let op: voor soort_count aggregeren we op basis van 'soort')
df_trend = _compute_trend_table(df_filtered, selected_metric)

# ---------------------------------------------------------------------------
# 0) Kaart (Swipe Map)
# ---------------------------------------------------------------------------

st.subheader("Kaart – resultaten voor een gekozen jaar")

if "jaar" not in df_filtered.columns:
    st.info("Er zijn geen jaartallen beschikbaar om op de kaart te tonen.")
else:
    years_available = (
        pd.to_numeric(df_filtered["jaar"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    years_available = sorted(years_available)

    if not years_available:
        st.info("Er zijn geen jaartallen beschikbaar om op de kaart te tonen.")
    else:
        # --- 1) Bepaal defaults: twee meest recente jaren binnen de selectie ---
        # Bron voor kaart: per locatie-jaar een waarde + lat/lon
        if selected_metric == "soort_count":
            cols = ["locatie_id", "jaar", "lat", "lon", "soort"]
            base = df_filtered[cols].dropna(subset=["locatie_id", "jaar", "lat", "lon"]).copy()
            df_map_source = (
                base.groupby(["locatie_id", "jaar"], sort=False, observed=True)
                .agg(lat=("lat", "first"), lon=("lon", "first"), soort_count=("soort", "nunique"))
                .reset_index()
            )
        else:
            # selected_metric kan ontbreken in subset; dan leeg
            if selected_metric not in df_filtered.columns:
                df_map_source = pd.DataFrame(columns=["locatie_id", "jaar", "lat", "lon", selected_metric])
            else:
                cols = ["locatie_id", "jaar", "lat", "lon", selected_metric]
                base = df_filtered[cols].dropna(subset=["locatie_id", "jaar", "lat", "lon"]).copy()
                base[selected_metric] = _as_numeric(base[selected_metric])
                df_map_source = (
                    base.groupby(["locatie_id", "jaar"], sort=False, observed=True)
                    .agg(lat=("lat", "first"), lon=("lon", "first"), **{selected_metric: (selected_metric, "mean")})
                    .reset_index()
                )

        if len(years_available) >= 2:
            default_left = years_available[-2]
            default_right = years_available[-1]
        else:
            default_left = years_available[0]
            default_right = years_available[0]

        # --- 2) Maak een context-id uniek voor selectie ---
        bodies_key = "|".join(sorted(selected_bodies)) if selected_bodies else "ALL"
        species_key = selected_species if species_is_selected else "ALLSPECIES"
        context_id = f"bodies={bodies_key}__metric={selected_metric}__species={species_key}__pg={int(show_ondergedoken)}{int(show_chariden)}"

        # --- 3) Init/restore per context ---
        if "swipe_years_by_context" not in st.session_state:
            st.session_state["swipe_years_by_context"] = {}
        if context_id not in st.session_state["swipe_years_by_context"]:
            st.session_state["swipe_years_by_context"][context_id] = {"left": default_left, "right": default_right}

        # --- 4) Valideer jaren bij veranderde selectie ---
        stored_left = st.session_state["swipe_years_by_context"][context_id]["left"]
        stored_right = st.session_state["swipe_years_by_context"][context_id]["right"]
        if stored_left not in years_available:
            stored_left = default_left
        if stored_right not in years_available:
            stored_right = default_right
        if stored_left == stored_right and len(years_available) >= 2:
            stored_left = years_available[-2]
            stored_right = years_available[-1]

        # --- 5) UI: selectboxen ---
        c_left, c_right = st.columns([1, 1])
        with c_left:
            year_left = st.selectbox(
                "Jaar links",
                years_available,
                index=years_available.index(stored_left),
                key=f"swipe_year_left__{context_id}",
            )
        with c_right:
            year_right = st.selectbox(
                "Jaar rechts",
                years_available,
                index=years_available.index(stored_right),
                key=f"swipe_year_right__{context_id}",
            )

        # --- 6) Persist ---
        st.session_state["swipe_years_by_context"][context_id]["left"] = year_left
        st.session_state["swipe_years_by_context"][context_id]["right"] = year_right

        if year_left == year_right:
            st.warning("Kies twee verschillende jaren voor de swipe‑vergelijking.")
        else:
            # selecteer waardekolom voor links/rechts
            metric_col = selected_metric
            if selected_metric == "soort_count":
                metric_col = "soort_count"

            df_left = df_map_source.loc[df_map_source["jaar"] == year_left, ["locatie_id", "lat", "lon", metric_col]].copy()
            df_right = df_map_source.loc[df_map_source["jaar"] == year_right, ["locatie_id", "lat", "lon", metric_col]].copy()

            df_left = df_left.rename(columns={metric_col: "value"})
            df_right = df_right.rename(columns={metric_col: "value"})

            # GeoJSON
            gj_left = df_to_geojson_points(df_left, value_col="value", id_col="locatie_id")
            gj_right = df_to_geojson_points(df_right, value_col="value", id_col="locatie_id")

            # bounds voor autozoom
            df_bounds = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["lat", "lon"])
            if len(df_bounds) >= 2:
                bounds = [
                    float(df_bounds["lon"].min()),
                    float(df_bounds["lat"].min()),
                    float(df_bounds["lon"].max()),
                    float(df_bounds["lat"].max()),
                ]
            else:
                bounds = None

            # center/scale
            center_lat = float(df_map_source["lat"].mean()) if not df_map_source.empty else 52.5
            center_lon = float(df_map_source["lon"].mean()) if not df_map_source.empty else 5.5

            v_ser = _as_numeric(df_map_source.get(metric_col, pd.Series(dtype="float")))
            min_val = float(np.nanmin(v_ser.values)) if len(v_ser) else 0.0
            max_val = float(np.nanmax(v_ser.values)) if len(v_ser) else 1.0
            if not np.isfinite(min_val):
                min_val = 0.0
            if not np.isfinite(max_val) or max_val == min_val:
                max_val = min_val + 1.0

            swipe_html = render_swipe_map_html(
                geojson_left=gj_left,
                geojson_right=gj_right,
                year_left=year_left,
                year_right=year_right,
                metric_label=selected_metric_label,
                min_val=min_val,
                max_val=max_val,
                center_lat=center_lat,
                center_lon=center_lon,
                zoom=9.0,
                height_px=650,
                bounds=bounds,
            )
            html(swipe_html, height=670)

            # Eenvoudige legenda
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;background-color: rgba(255,255,255,0.85);
                padding:10px;border-radius:6px;border:1px solid #ddd;margin-top:6px;">
                <div style="margin-right:10px;"><strong>Legenda ({selected_metric_label}):</strong></div>
                <div style="background: linear-gradient(90deg, red, green); width: 160px; height: 14px; border: 1px solid #ccc;"></div>
                <div style="margin-left:10px; font-size: 0.9em;">
                {min_val:.1f} (Laag) → {max_val:.1f} (Hoog)
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# (Nieuw) 0B) Basale trendanalyse – c1 grafiek (Totale bedekking)
# -----------------------------------------------------------------------------
st.subheader("📈 Basale trendanalyse – Totale bedekking")

df_trend_cover = _trend_cover(df_filtered)

if df_trend_cover.empty:
    st.info("Geen (soort-)bedekkingsdata (BEDKG) gevonden binnen de huidige sidebar-filters.")
else:
    title_extra = f" – {selected_species}" if species_is_selected else ""
    fig_cover = px.line(
        df_trend_cover,
        x="jaar",
        y="totaal_bedekking_locatie",
        color="Waterlichaam",
        markers=True,
        title=f"Trend totale bedekking (%) per waterlichaam{title_extra}",
        labels={
            "jaar": "Jaar",
            "totaal_bedekking_locatie": "Totale bedekking (%)",
            "Waterlichaam": "Waterlichaam",
        },
    )
    fig_cover.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_cover, use_container_width=True)



# -------------------------------------------------------------
# 1) Tijdreeks trendlijnen
# -------------------------------------------------------------

st.subheader(f"Verloop {selected_metric_label} door de jaren heen")
if species_is_selected and selected_metric == "bedekking_pct":
    st.caption(f"Weergave voor soort: **{selected_species}**")

fig_line = px.line(
    df_trend,
    x="jaar",
    y="waarde",
    color="locatie_id",
    markers=True,
    title="Trendontwikkeling per meetpunt in geselecteerde wateren",
)
st.plotly_chart(fig_line, use_container_width=True)

# -------------------------------------------------------------
# 2) Regressieanalyse – verbetert of verslechtert de toestand?
# -------------------------------------------------------------

st.subheader("Regressieanalyse: verbetert of verslechtert de toestand?")
st.markdown("Analyse per meetpunt over de beschikbare jaren.")

with st.expander("ℹ️ Hoe interpreteer ik de hellingwaarde?", expanded=False):
    st.markdown(
        """
        De **helling** (richtingscoëfficiënt) is een getal dat de gemiddelde verandering per meetjaar aangeeft op basis van lineaire regressie.

        * **Positief getal (+):** De waarde stijgt gemiddeld elk meetjaar.
        * **Voorbeeld:** Een slope van `5.0` bij bedekkingsgraad betekent dat de bedekking gemiddeld met 5% per meetjaar toeneemt.
        * **Negatief getal (-):** De waarde daalt gemiddeld elk meetjaar.
        * **Nul (0):** Er is geen stijgende of dalende trend (stabiel).

        Drempelwaarden voor trendclassificatie:
        * **Diepte**/**Doorzicht**: ±0,1 per jaar
        * **Soortenrijkdom**: ±0,4 per jaar
        * **Bedekkingsgraad**: ±1,0 per jaar

        *Let op: hieronder worden alleen locaties getoond met **minimaal 5 meetjaren**.*
        """
    )

MIN_JAREN = 5

df_slopes = _compute_slopes_vectorized(df_trend, min_years=MIN_JAREN)

if not df_slopes.empty:
    # Thresholds per metriek
    if selected_metric in ["diepte_m", "doorzicht_m"]:
        threshold = 0.1
    elif selected_metric == "soort_count":
        threshold = 0.4
    else:
        threshold = 1.0

    df_slopes["Trend"] = df_slopes["slope"].apply(lambda x: categorize_slope_trend(x, threshold))

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_pie = px.pie(
            df_slopes,
            names="Trend",
            title="Trends meetlocaties per geselecteerd waterlichaam",
            color="Trend",
            color_discrete_map={
                "Verbeterend ↗️": "green",
                "Verslechterend ↘️": "red",
                "Stabiel ➡️": "grey",
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.write(f"**Detailtabel (meetlocaties met ≥ {MIN_JAREN} jaar data)**")
        st.dataframe(
            df_slopes.sort_values("slope", ascending=False)
            .style.background_gradient(subset=["slope"], cmap="RdYlGn")
            .format({"slope": "{:.4f}", "n_jaren": "{:.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.warning(
        f"Geen locaties gevonden met minimaal {MIN_JAREN} jaar aan meetgegevens in de huidige selectie."
    )

# -------------------------------------------------------------
# 4) Voor/na vergelijking (balken) – default 2 meest recente jaren
# -------------------------------------------------------------

st.divider()
st.subheader("Vergelijking versus een historisch meetjaar")

available_years = (
    pd.to_numeric(df_filtered.get("jaar", pd.Series(dtype="float")), errors="coerce")
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)
available_years = sorted(available_years)

if len(available_years) >= 2:
    c_year1, c_year2 = st.columns(2)

    default_start = int(available_years[-2])
    default_end = int(available_years[-1])

    year_start = c_year1.selectbox(
        "Referentiejaar",
        available_years,
        index=available_years.index(default_start),
        key="bar_ref_year",
    )
    year_end = c_year2.selectbox(
        "Vergelijkingsjaar",
        available_years,
        index=available_years.index(default_end),
        key="bar_cmp_year",
    )

    df_compare = df_trend.loc[df_trend["jaar"].isin([year_start, year_end])].copy()

    fig_bar = px.bar(
        df_compare,
        x="locatie_id",
        y="waarde",
        color=df_compare["jaar"].astype(str),
        barmode="group",
        title=f"Vergelijking {year_start} vs {year_end} per locatie",
        labels={"waarde": selected_metric_label, "color": "Jaar"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Niet genoeg jaren aan data voor een vergelijking.")

# -------------------------------------------------------------
# 5) TOP 50 HEATMAP
# -------------------------------------------------------------

st.divider()
st.subheader("Soortenaanwezigheid heatmap (top 50)")

# Basis voor heatmap: alleen filter op waterlichamen (niet op selected_species)
if selected_bodies and "Waterlichaam" in df.columns:
    df_heat_base = df.loc[df["Waterlichaam"].isin(selected_bodies)].copy()
else:
    df_heat_base = df.copy()

# Alleen relevante kolommen (scheelt geheugen/CPU)
heat_cols = [c for c in ["type", "Grootheid", "soort", "locatie_id", "jaar", "bedekking_pct"] if c in df_heat_base.columns]
df_heat_base = df_heat_base[heat_cols].copy() if heat_cols else pd.DataFrame()

# Alleen individuele soorten en bij voorkeur alleen bedekking (BEDKG)
if not df_heat_base.empty and "type" in df_heat_base.columns:
    df_species_only = df_heat_base.loc[df_heat_base["type"] == "Soort"].copy()
else:
    df_species_only = df_heat_base.copy()

if not df_species_only.empty and "Grootheid" in df_species_only.columns:
    df_species_only = df_species_only.loc[df_species_only["Grootheid"] == "BEDKG"].copy()

if "bedekking_pct" in df_species_only.columns:
    df_species_only["bedekking_pct"] = _as_numeric(df_species_only["bedekking_pct"])  # compat parquet/duckdb

if df_species_only.empty:
    st.info("Geen soortdata gevonden voor de huidige selectie.")
else:
    # --- Gewogen TOP 50: elk (locatie_id, jaar) telt even zwaar ---
    # 1) per soort-locatie-jaar gemiddelde bedekking
    df_cells = (
        df_species_only.dropna(subset=["soort", "locatie_id", "jaar"])            .groupby(["soort", "locatie_id", "jaar"], sort=False, observed=True)["bedekking_pct"]
            .mean()
            .reset_index()
    )

    # 2) score per soort: gemiddelde over unieke locatie-jaar cellen
    weighted_score = (
        df_cells.groupby("soort", sort=False, observed=True)["bedekking_pct"]
        .mean()
        .sort_values(ascending=False)
    )

    top_species = weighted_score.head(50).index

    # 3) Filter op top 50 en aggregeer per soort-jaar
    df_top_heatmap = df_species_only.loc[df_species_only["soort"].isin(top_species)].copy()

    heatmap_data = (
        df_top_heatmap.dropna(subset=["soort", "jaar"])            .groupby(["soort", "jaar"], sort=False, observed=True)["bedekking_pct"]
            .mean()
            .reset_index()
    )

    if heatmap_data.empty:
        st.warning("Geen data beschikbaar voor de heatmap.")
    else:
        heatmap_matrix = (
            heatmap_data.pivot(index="soort", columns="jaar", values="bedekking_pct")
            .fillna(0)
        )

        # Maskers
        zero_mask = heatmap_matrix == 0
        matrix_green = heatmap_matrix.mask(zero_mask, other=np.nan)

        fig_heat = px.imshow(
            matrix_green,
            color_continuous_scale="Greens",
            aspect="auto",
            title="Ontwikkeling bedekking (top 50 meest voorkomende soorten)",
            labels=dict(x="Jaar", y="Soort", color="Gem. Bedekking (%)"),
        )

        # Grijze overlay voor 0-cellen
        matrix_zero = zero_mask.astype(int).replace({0: np.nan})
        fig_heat.add_trace(
            go.Heatmap(
                z=matrix_zero.values,
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(200,200,200,0.65)"]],
                showscale=False,
                hoverinfo="skip",
            )
        )

        # X-markers bovenop 0-cellen
        ys, xs = np.where(zero_mask.values)
        x_vals = [heatmap_matrix.columns[i] for i in xs]
        y_vals = [heatmap_matrix.index[i] for i in ys]
        fig_heat.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(symbol="x", size=6, color="rgba(120,120,120,0.55)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig_heat.update_layout(height=1200, yaxis=dict(side="left"))
        st.plotly_chart(fig_heat, use_container_width=True)
