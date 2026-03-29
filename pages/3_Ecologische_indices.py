# 3_Ecologische_indices.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import ( 
 load_data, 
 add_species_group_columns, 
 load_chemistry_data, 
 load_ecology_timeseries_data_filtered, 
 load_filtered_ecology_base, 
 get_bubble_yearly_filtered, 
 get_available_chemistry_locations, 
 get_available_chemistry_parameter_labels, 
 get_preferred_chemistry_locations, 
 get_chem_ecology_timeseries,
 summarize_chemistry_period_average,
 CHEM_PARAM_SUGGESTIONS,
 SEASON_ORDER,
)
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
def _ensure_nomatch_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maak ecologische indices robuust voor datasets zonder display-kolommen voor no-matchs."""
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

    if "is_kenmerkende_soort_n2000" not in df.columns:
        if "Grootheid" in df.columns:
            df["is_kenmerkende_soort_n2000"] = df["Grootheid"].astype(str).eq("AANWZHD")
        else:
            df["is_kenmerkende_soort_n2000"] = False

    if "kenmerkende_soort_n2000_weergave" not in df.columns:
        display_source = df["soort_display"] if "soort_display" in df.columns else (df["soort"] if "soort" in df.columns else pd.Series("", index=df.index))
        df["kenmerkende_soort_n2000_weergave"] = np.where(
            df["is_kenmerkende_soort_n2000"].fillna(False),
            display_source.astype(str),
            "Geen match",
        )

    return df


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

    Inclusief soorten zonder match voor trofieniveau, soortgroep en KRW score,
    conform de ruimtelijke analyse.

    Retourneert: (matrix, category_column_name, value_label)
    """
    if df_base.empty:
        return pd.DataFrame(), "", ""

    # brondata per param
    if heatmap_param == "Groeivormen":
        df_h = df_base[df_base["type"] == "Groep"].copy()
        if "groeivorm" not in df_h.columns:
            df_h["groeivorm"] = np.nan
        df_h = df_h.dropna(subset=["groeivorm", "jaar"])
        cat_col = "groeivorm"
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Soortgroepen":
        df_species = df_base[df_base["type"] == "Soort"].copy()
        # als Grootheid bestaat, neem BEDKG mee (compat met je oude intentie)
        if "Grootheid" in df_species.columns:
            df_species = df_species[df_species["Grootheid"] == "BEDKG"].copy()

        # mapping behouden, maar no-matchs expliciet zichtbaar maken
        df_h = add_species_group_columns(df_species)
        if "soortgroep_weergave" in df_h.columns:
            df_h["soortgroep_cat"] = df_h["soortgroep_weergave"].astype(object)
            df_h.loc[df_h["soortgroep_cat"].isna() | (df_h["soortgroep_cat"].astype(str).str.strip() == ""), "soortgroep_cat"] = "Geen match"
        else:
            if "soortgroep" not in df_h.columns:
                df_h["soortgroep"] = np.nan
            df_h["soortgroep_cat"] = df_h["soortgroep"].astype(object)
            df_h.loc[df_h["soortgroep_cat"].isna() | (df_h["soortgroep_cat"].astype(str).str.strip() == ""), "soortgroep_cat"] = "Geen match"

        df_h = df_h.dropna(subset=["jaar"])
        cat_col = "soortgroep_cat"
        bedekking_source = "bedekkingsgraad_proc" if "bedekkingsgraad_proc" in df_h.columns else "bedekking_pct"
        df_h["bedekking_num"] = pd.to_numeric(df_h[bedekking_source], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Kenmerkende soorten (N2000)":
        df_species = df_base[df_base["type"] == "Soort"].copy()
        df_h = add_species_group_columns(df_species)

        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "AANWZHD"].copy()
        else:
            df_h = df_h[df_h["is_kenmerkende_soort_n2000"].fillna(False)].copy()

        if "kenmerkende_soort_n2000_weergave" in df_h.columns:
            df_h["n2000_cat"] = df_h["kenmerkende_soort_n2000_weergave"].astype(object)
        elif "soort_display" in df_h.columns:
            df_h["n2000_cat"] = df_h["soort_display"].astype(object)
        else:
            df_h["n2000_cat"] = df_h["soort"].astype(object)

        df_h.loc[
            df_h["n2000_cat"].isna() | (df_h["n2000_cat"].astype(str).str.strip() == ""),
            "n2000_cat",
        ] = "Geen match"
        df_h = df_h.dropna(subset=["jaar"])
        cat_col = "n2000_cat"
        bedekking_source = "bedekkingsgraad_proc" if "bedekkingsgraad_proc" in df_h.columns else "bedekking_pct"
        df_h["bedekking_num"] = pd.to_numeric(df_h[bedekking_source], errors="coerce").fillna(0).clip(lower=0)

    elif heatmap_param == "Trofieniveau":
        df_h = df_base[df_base["type"] == "Soort"].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()

        if "trofisch_niveau_weergave" in df_h.columns:
            df_h["trofie_cat"] = df_h["trofisch_niveau_weergave"].astype(object)
        else:
            if "trofisch_niveau" not in df_h.columns:
                df_h["trofisch_niveau"] = np.nan
            df_h["trofie_cat"] = df_h["trofisch_niveau"].astype(object)

        df_h.loc[df_h["trofie_cat"].isna() | (df_h["trofie_cat"].astype(str).str.strip() == ""), "trofie_cat"] = "Geen match"
        df_h = df_h.dropna(subset=["jaar"])
        cat_col = "trofie_cat"
        df_h["bedekking_num"] = pd.to_numeric(df_h["bedekking_pct"], errors="coerce").fillna(0).clip(lower=0)

    else:  # "KRW score"
        df_h = df_base[df_base["type"] == "Soort"].copy()
        if "Grootheid" in df_h.columns:
            df_h = df_h[df_h["Grootheid"] == "BEDKG"].copy()

        if "krw_class_weergave" in df_h.columns:
            df_h["krw_cat"] = df_h["krw_class_weergave"].astype(object)
            df_h.loc[df_h["krw_cat"].isna() | (df_h["krw_cat"].astype(str).str.strip() == ""), "krw_cat"] = "Geen match"
        elif "krw_class" in df_h.columns and df_h["krw_class"].notna().any():
            df_h["krw_cat"] = df_h["krw_class"].astype(object)
            df_h.loc[df_h["krw_cat"].isna() | (df_h["krw_cat"].astype(str).str.strip() == ""), "krw_cat"] = "Geen match"
        else:
            df_h["krw_cat"] = pd.cut(
                pd.to_numeric(df_h.get("krw_score"), errors="coerce"),
                bins=[0, 2, 4, 5],
                labels=["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)"],
                include_lowest=True,
            ).astype(object)
            df_h.loc[df_h["krw_cat"].isna(), "krw_cat"] = "Geen match"

        df_h = df_h.dropna(subset=["jaar"])
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

    # optionele ordering voor bekende categorieën (zoals in je oude code + ruimtelijke analyse)
    if heatmap_param == "KRW score":
        order = ["Gunstig (1-2)", "Neutraal (3-4)", "Ongewenst (5)", "Geen match"]
        keep = [x for x in order if x in heat.index]
        rest = [x for x in heat.index if x not in keep]
        heat = heat.reindex(keep + rest)
    elif heatmap_param == "Trofieniveau":
        order = ["oligotroof", "mesotroof", "eutroof", "sterk eutroof", "brak", "marien", "kroos", "Onbekend", "Geen match"]
        keep = [x for x in order if x in heat.index]
        rest = [x for x in heat.index if x not in keep]
        heat = heat.reindex(keep + rest)
    elif heatmap_param == "Groeivormen":
        order = ["Ondergedoken", "Drijvend", "Emergent", "Draadalgen", "Kroos", "FLAB"]
        keep = [x for x in order if x in heat.index]
        rest = [x for x in heat.index if x not in keep]
        heat = heat.reindex(keep + rest)
    elif heatmap_param == "Soortgroepen":
        order = [
            "chariden", "iseotiden", "parvopotamiden", "magnopotamiden", "myriophylliden",
            "vallisneriiden", "elodeiden", "stratiotiden", "pepliden", "batrachiiden",
            "nymphaeiden", "haptofyten", "Kenmerkende soort (N2000)", "Overig / Individueel", "Geen match",
        ]
        keep = [x for x in order if x in heat.index]
        rest = [x for x in heat.index if x not in keep]
        heat = heat.reindex(keep + rest)

    elif heatmap_param == "Kenmerkende soorten (N2000)":
        ordered = sorted([x for x in heat.index if x != "Geen match"], key=lambda x: str(x).lower())
        if "Geen match" in heat.index:
            ordered.append("Geen match")
        heat = heat.reindex(ordered)

    return heat, cat_col, value_label


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
df = load_data()
if df.empty:
    st.error("Geen data geladen.")
    st.stop()

# Zorg dat no-match display-kolommen aanwezig zijn (zelfde principe als 2_Ruimtelijke_analyse.py)
df = _ensure_nomatch_display_columns(df)

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

df_filtered_base = load_filtered_ecology_base(tuple(selected_projects), tuple(selected_bodies))

# alleen soorten (gecached + numeric)
df_species_only = _species_only(df_filtered_base)

# -----------------------------------------------------------------------------
# GEDEELDE JAARSLIDER (chemie vs ecologie + bubble plot)
# -----------------------------------------------------------------------------
df_bubble = get_bubble_yearly_filtered(tuple(selected_projects), tuple(selected_bodies))
shared_years = sorted(
    pd.to_numeric(df_bubble.get("jaar"), errors="coerce").dropna().astype(int).unique().tolist()
) if not df_bubble.empty else []

if not shared_years:
    st.warning("Geen jaardata beschikbaar voor de gedeelde periode-selectie.")
    shared_period = None
else:
    shared_min_year = int(min(shared_years))
    shared_max_year = int(max(shared_years))
    shared_period = st.slider(
        "Selecteer periode (voor chemie vs ecologie én bubble plot)",
        shared_min_year,
        shared_max_year,
        [shared_min_year, shared_max_year],
        key="ecol_shared_period",
    )

# -----------------------------------------------------------------------------
# CHEMIE VS ECOLOGIE (DUBBELE Y-AS)
# -----------------------------------------------------------------------------
st.subheader("🧪 Chemie vs ecologische indices")
st.caption("Deze grafiek staat boven de bubble plot en gebruikt dezelfde periode-selectie als de bubble plot. Chemische metingen worden standaard inclusief niet-definitieve records getoond en de chemische lijnen behouden markers. Voor performance wordt waar mogelijk DuckDB/parquet gebruikt voor de ecologische basis- en bubbledata.")
with st.expander("ℹ️ Uitleg: vergelijking chemische stoffen met ecologische bedekking / index", expanded=False):
    st.markdown(
        """
Deze interactieve grafiek combineert **jaartallen op de x-as** met een **dubbele Y-as**:
- **Linker Y-as:** bedekkingsgraad / index
- **Rechter Y-as:** concentratie t/m 5 geselecteerde chemische stoffen

Je kunt hiermee bijvoorbeeld nagaan of veranderingen in **stikstof, fosfor of koolstof** samenvallen met veranderingen in:
- totale bedekking
- soortgroepen
- trofieniveau
- KRW-score
- kenmerkende soorten (N2000)
- groeivormen

**Seizoensfilter**
- Voorjaar = maart t/m mei
- Zomer = juni t/m augustus
- Najaar = september t/m november
- Winter = december t/m februari

De chemische reeksen worden als **gemiddelde van de geselecteerde seizoenen** getoond voor de gekozen locatie. Als je bijvoorbeeld alleen **Zomer** kiest en de jaarslider op **2015–2020** zet, dan worden per stof de **zomergemiddelden per jaar** getoond én daarnaast het **gemiddelde van die zomers over 2015–2020** samengevat.
        """
    )

df_eco_timeseries = load_ecology_timeseries_data_filtered(tuple(selected_projects), tuple(selected_bodies))
df_chem = load_chemistry_data()

if df_eco_timeseries.empty:
    st.warning("Ecologische tijdreeksdata kon niet worden voorbereid voor de chemie-koppeling.")
elif df_chem.empty:
    st.warning("Chemische data kon niet worden geladen voor de dubbele Y-as grafiek.")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Chemie vs bedekking")

    chem_locations = get_available_chemistry_locations(df_chem)
    chem_location_options, default_loc, chem_location_required = get_preferred_chemistry_locations(
        tuple(selected_bodies),
        chem_locations,
    )
    if not chem_location_options and chem_locations:
        chem_location_options = chem_locations
    selected_chem_location = st.sidebar.selectbox(
        "Meetlocatie chemie",
        options=chem_location_options,
        index=(chem_location_options.index(default_loc) if (default_loc in chem_location_options) else None),
        key="chem_vs_eco_location",
        placeholder="Kies een meetlocatie",
    ) if chem_location_options else None
    if chem_location_required and selected_chem_location is None:
        st.sidebar.info("Kies een meetlocatie chemie voor het geselecteerde waterlichaam / de geselecteerde waterlichamen.")

    chem_labels = get_available_chemistry_parameter_labels(df_chem)
    preferred_labels = []
    for code in CHEM_PARAM_SUGGESTIONS:
        preferred_labels.extend([x for x in chem_labels if str(x).startswith(f"{code} ") or str(x) == code or str(x).startswith(f"{code}—") or str(x).startswith(f"{code} —")])
    preferred_labels = [x for x in preferred_labels if x in chem_labels]
    default_chems = preferred_labels[:2] if preferred_labels else chem_labels[:1]
    selected_chems = st.sidebar.multiselect(
        "Selecteer stof(fen) (max. 5)",
        options=chem_labels,
        default=default_chems,
        key="chem_vs_eco_params",
        max_selections=5,
    )
    if len(selected_chems) > 5:
        selected_chems = selected_chems[:5]
        st.sidebar.warning("Maximaal 5 stoffen tegelijk worden getoond; de selectie is teruggebracht naar de eerste 5.")

    left_metric_dual = st.sidebar.selectbox(
        "Linker Y-as",
        [
            "Totale bedekking",
            "Soortgroep",
            "Trofieniveau",
            "KRW score",
            "Kenmerkende soort (N2000)",
            "Groeivormen",
        ],
        index=0,
        key="chem_vs_eco_left_metric",
    )
    left_display_mode = st.sidebar.radio(
        "Weergave linker Y-as",
        ["Lijnen", "Kolommen", "Gestapeld gebied"],
        index=1,
        key="chem_vs_eco_left_display_mode",
    )
    selected_seasons = st.sidebar.multiselect(
        "Seizoensgemiddelden chemie",
        options=SEASON_ORDER,
        default=SEASON_ORDER,
        key="chem_vs_eco_seasons",
    )
    definitive_only = False
    show_chem_markers = True

    krw_mode = "index"
    n2000_mode = "records"
    if left_metric_dual == "KRW score":
        krw_mode = st.sidebar.radio(
            "KRW-score tonen als",
            ["index", "klassen"],
            index=0,
            horizontal=True,
            key="chem_vs_eco_krw_mode",
        )
    if left_metric_dual == "Kenmerkende soort (N2000)":
        n2000_mode = st.sidebar.radio(
            "Kenmerkende soort (N2000) tonen als",
            ["records", "soorten"],
            index=0,
            horizontal=True,
            key="chem_vs_eco_n2000_mode",
        )

    top_n_dual = None
    if left_metric_dual in {"Soortgroep", "Trofieniveau", "Groeivormen", "Kenmerkende soort (N2000)"}:
        top_n_dual = st.sidebar.slider(
            "Max. aantal ecologische series links",
            min_value=1,
            max_value=12,
            value=6,
            key="chem_vs_eco_top_n",
        )

    eco_mode = "default"
    if left_metric_dual == "KRW score":
        eco_mode = krw_mode
    elif left_metric_dual == "Kenmerkende soort (N2000)":
        eco_mode = n2000_mode

    eco_year_dual, chem_year_dual, common_years_dual = get_chem_ecology_timeseries(
        df_eco=df_eco_timeseries,
        df_chem=df_chem,
        project_sel=tuple(selected_projects),
        body_sel=tuple(selected_bodies),
        ecology_metric=left_metric_dual,
        chemistry_labels=tuple(selected_chems),
        chemistry_location=selected_chem_location,
        ecology_mode=eco_mode,
        definitive_only=definitive_only,
        seasons=tuple(selected_seasons),
        top_n=top_n_dual,
    )

    if chem_location_required and selected_chem_location is None:
        st.info("Kies eerst een meetlocatie chemie voor het geselecteerde waterlichaam / de geselecteerde waterlichamen.")
    elif not selected_chems:
        st.info("Selecteer minimaal één chemische stof om de dubbele Y-as grafiek te tonen.")
    elif eco_year_dual.empty:
        st.info("Geen ecologische data beschikbaar voor deze combinatie van filters en seizoenen.")
    elif chem_year_dual.empty:
        st.info("Geen chemische data beschikbaar voor deze combinatie van filters en seizoenen.")
    elif not common_years_dual:
        st.info("Geen overlappende jaren tussen ecologie en chemie voor deze filtercombinatie.")
    elif shared_period is None:
        st.info("Geen gedeelde periode beschikbaar voor chemie vs ecologie.")
    else:
        eco_year_dual = eco_year_dual[(eco_year_dual["jaar"] >= int(shared_period[0])) & (eco_year_dual["jaar"] <= int(shared_period[1]))].copy()
        chem_year_dual = chem_year_dual[(chem_year_dual["jaar"] >= int(shared_period[0])) & (chem_year_dual["jaar"] <= int(shared_period[1]))].copy()
        if eco_year_dual.empty or chem_year_dual.empty:
            st.info("Geen chemie- of ecologiedata binnen de gekozen gedeelde periode.")
        else:
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

            eco_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
            ]
            chem_palette = [
                "#111111", "#4c4c4c", "#7f7f7f", "#555555", "#999999",
            ]
            chem_dashes = ["dash", "dot", "dashdot", "longdash", "solid"]

            eco_series = sorted(eco_year_dual["serie"].dropna().astype(str).unique().tolist())
            eco_colors = {s: eco_palette[i % len(eco_palette)] for i, s in enumerate(eco_series)}
            for serie in eco_series:
                d = eco_year_dual[eco_year_dual["serie"] == serie].sort_values("jaar")
                color = eco_colors[serie]
                if left_display_mode == "Kolommen":
                    fig_dual.add_trace(
                        go.Bar(
                            x=d["jaar"],
                            y=d["waarde"],
                            name=serie,
                            marker_color=color,
                            opacity=0.82,
                        ),
                        secondary_y=False,
                    )
                elif left_display_mode == "Gestapeld gebied":
                    fig_dual.add_trace(
                        go.Scatter(
                            x=d["jaar"],
                            y=d["waarde"],
                            name=serie,
                            mode="lines",
                            line=dict(color=color, width=1.5),
                            stackgroup="one",
                        ),
                        secondary_y=False,
                    )
                else:
                    fig_dual.add_trace(
                        go.Scatter(
                            x=d["jaar"],
                            y=d["waarde"],
                            name=serie,
                            mode="lines+markers",
                            line=dict(color=color, width=2),
                            marker=dict(size=6),
                        ),
                        secondary_y=False,
                    )

            chem_series_order = [x for x in selected_chems if x in chem_year_dual["serie"].unique().tolist()]
            for idx, serie in enumerate(chem_series_order):
                d = chem_year_dual[chem_year_dual["serie"] == serie].sort_values("jaar")
                if d.empty:
                    continue
                unit = ""
                if "eenheid_omschrijving" in d.columns and not d["eenheid_omschrijving"].dropna().empty:
                    unit = str(d["eenheid_omschrijving"].dropna().iloc[0])
                display_name = f"{serie} ({unit})" if unit else serie
                fig_dual.add_trace(
                    go.Scatter(
                        x=d["jaar"],
                        y=d["chem_value"],
                        name=display_name,
                        mode="lines+markers" if show_chem_markers else "lines",
                        line=dict(color=chem_palette[idx % len(chem_palette)], width=3, dash=chem_dashes[idx % len(chem_dashes)]),
                        marker=dict(size=7, symbol="diamond"),
                    ),
                    secondary_y=True,
                )

            if left_display_mode == "Kolommen":
                fig_dual.update_layout(barmode="group")

            left_axis_title = left_metric_dual
            if left_metric_dual == "KRW score" and krw_mode == "index":
                left_axis_title = "Gemiddelde KRW-score"
            elif left_metric_dual == "Kenmerkende soort (N2000)" and n2000_mode == "records":
                left_axis_title = "Aantal aanwezigheidsrecords"
            else:
                left_axis_title = f"{left_metric_dual} / bedekking"

            chem_units = [u for u in chem_year_dual["eenheid_omschrijving"].dropna().astype(str).unique().tolist() if u]
            right_axis_title = "Concentratie"
            if len(chem_units) == 1:
                right_axis_title = f"Concentratie ({chem_units[0]})"
            elif len(chem_units) > 1:
                right_axis_title = "Concentratie (eenheidsafhankelijk)"

            season_label = ", ".join(selected_seasons) if selected_seasons else "alle seizoenen"
            fig_dual.update_layout(
                title=(
                    f"Chemie vs {left_metric_dual}<br>"
                    f"<sup>Ecologie: {', '.join(selected_projects) if selected_projects else 'geen project'} | "
                    f"{', '.join(selected_bodies) if selected_bodies else 'geen waterlichaam'} — "
                    f"Chemie: {selected_chem_location or 'geen locatie'} — Seizoen: {season_label}</sup>"
                ),
                height=760,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(
                    title="Jaar",
                    tickmode="linear",
                    rangeslider=dict(visible=True),
                ),
            )
            fig_dual.update_yaxes(title_text=left_axis_title, secondary_y=False)
            fig_dual.update_yaxes(title_text=right_axis_title, secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)
            st.caption(
                f"Seizoenen werken hier alleen op de chemische data; de ecologische bedekking blijft jaarrond zichtbaar. "
                f"Chemische lijnen tonen per jaar het gemiddelde van de geselecteerde seizoenen. "
                f"Voor periode {int(shared_period[0])}–{int(shared_period[1])} en seizoen(en) {season_label} "
                f"wordt hieronder ook per stof het gemiddelde over de gekozen periode samengevat."
            )

            chem_period_summary = summarize_chemistry_period_average(
                chem_year_dual,
                year_min=int(shared_period[0]),
                year_max=int(shared_period[1]),
            )
            if not chem_period_summary.empty:
                st.markdown("**Gemiddelde chemische concentratie over de gekozen periode en seizoenen**")
                st.dataframe(
                    chem_period_summary,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Jaarreeksen achter de chemie-vs-ecologie grafiek", expanded=False):
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**Ecologische reeksen (linker Y-as)**")
                    st.dataframe(
                        eco_year_dual.sort_values(["jaar", "serie"]).rename(columns={"serie": "reeks", "waarde": "linker_y"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                with col_right:
                    st.markdown("**Chemische reeksen (rechter Y-as)**")
                    st.dataframe(
                        chem_year_dual.sort_values(["jaar", "serie"]).rename(columns={"serie": "stof", "chem_value": "rechter_y"}),
                        use_container_width=True,
                        hide_index=True,
                    )


# -----------------------------------------------------------------------------
# BUBBLE PLOT
# -----------------------------------------------------------------------------
st.subheader("Relatie doorzicht vs bedekking")
with st.expander("ℹ️ Hoe komt deze bubble plot tot stand? (toelichting)"):
    st.caption("De periode van deze bubble plot wordt gestuurd door de gedeelde jaarslider bovenaan de pagina, dezelfde als voor de chemie-vs-ecologie grafiek.")
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

if df_bubble.empty or df_bubble["jaar"].dropna().empty or shared_period is None:
    st.warning("Geen data beschikbaar voor bubbleplot na filtering.")
else:
    df_bubble_plot = _bubble_period_means(df_bubble, int(shared_period[0]), int(shared_period[1]))
    if df_bubble_plot.empty:
        st.warning("Geen data gevonden voor deze filtercombinatie of periode.")
    else:
        fig_bubble = px.scatter(
            df_bubble_plot,
            x="doorzicht_diepte_ratio",
            y="bedekking_pct",
            size="diepte_m",
            hover_name="soort",
            size_max=40,
            title=f"Ecologische indices ({shared_period[0]} - {shared_period[1]})",
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
        st.plotly_chart(fig_bubble, width='stretch')

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
- Kenmerkende soorten (N2000)

Je kunt kiezen:
- **Records** (aantal regels)
- **Bedekking-gewogen** (som van bedekking)

En optioneel normaliseren per jaar (0–100% verdeling).

Voor **Trofieniveau**, **Soortgroepen**, **KRW score** en **Kenmerkende soorten (N2000)** worden ook records zonder match expliciet getoond als **Geen match**.
"""
    )

heatmap_param = st.selectbox("Kies parameter voor heatmap", ["Trofieniveau", "Groeivormen", "Soortgroepen", "KRW score", "Kenmerkende soorten (N2000)"])
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
        st.plotly_chart(fig, width='stretch')

        if heatmap_param == "Trofieniveau":
            st.caption(
                "Bron trofieniveau-indeling: Verhofstad et al. (2025) – Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang. "
                "https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf"
            )
