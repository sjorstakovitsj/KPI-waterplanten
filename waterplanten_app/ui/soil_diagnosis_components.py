from __future__ import annotations
from collections.abc import Sequence
import altair as alt
import pandas as pd
import streamlit as st

from waterplanten_app.config.soil_diagnosis_config import (
    DIAGNOSIS_COMPANION_CATEGORY_KEYS,
    DIAGNOSIS_USE_EXISTING_CATEGORY,
    DIAGNOSIS_VALUE_CLASS_BREAKS,
    PARAMETER_EXPLAINERS,
    TREND_BOOTSTRAP_MODE_DEFAULT,
    TREND_BOOTSTRAP_MODE_LABELS,
    TREND_FORMAL_ANALYSIS_DEFAULT,
    WEIGHTING_COVER,
    WEIGHTING_PRESENCE,
)
from waterplanten_app.domain.soil_diagnosis import SoilDiagnosisResult, WaterbodyDiagnosisResult

ANALYSIS_WATERBODY = "Waterlichaam"
ANALYSIS_LOCATION = "Detail meetlocatie"


def render_analysis_level_selector() -> str:
    return st.radio(
        "Weergave",
        [ANALYSIS_WATERBODY, ANALYSIS_TRENDS, ANALYSIS_LOCATION],
        horizontal=True,
        key="soil_analysis_level",
    )


def render_weighting_selector() -> str:
    label = st.radio(
        "Berekeningswijze",
        ["Aanwezigheid + indicatorkracht", "Bedekking + indicatorkracht"],
        horizontal=True,
        key="soil_weighting",
    )
    return WEIGHTING_PRESENCE if label.startswith("Aanwezigheid") else WEIGHTING_COVER


def render_location_selector(locations: Sequence[str]) -> str | None:
    locations = sorted({str(value) for value in locations if str(value).strip()})
    if not locations:
        st.info("Geen meetlocaties beschikbaar voor de gekozen filters.")
        return None
    return st.selectbox("Selecteer meetlocatie", locations, key="soil_location")


def _format_coverage(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    if column in out:
        out[column] = pd.to_numeric(out[column], errors="coerce").map(lambda x: f"{x:.0%}" if pd.notna(x) else "")
    return out



def _classify_diagnosis_value(parameter_key: str, value: object) -> str | None:
    """Vertaal elke numerieke diagnose- of spreidingswaarde naar een klasse."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    value = float(numeric)

    if parameter_key == "ph":
        if value < 6.5:
            return "Zuur water"
        if value <= 7.5:
            return "Neutraal water"
        return "Basisch water"

    if parameter_key == "salinity":
        if value <= 2:
            return "Zeer ionenarm water"
        if value <= 4:
            return "Ionenarm water"
        if value <= 9:
            return "Matig ionenrijk water"
        if value <= 15:
            return "Ionenrijk water"
        return "Zeer ionenrijk water"

    if parameter_key == "alkalinity":
        if value < 0.1:
            return "Ongebufferd (zuur) water"
        if value < 0.5:
            return "Zeer zacht water"
        if value < 1:
            return "Zacht water"
        if value < 2:
            return "Zacht tot matig hard water"
        if value <= 4:
            return "Hard water"
        return "Zeer hard water"

    if parameter_key == "redox":
        if value < -175:
            return "Zeer reductief"
        if value < -100:
            return "Reductief"
        if value < 0:
            return "Matig reductief"
        if value <= 100:
            return "Matig oxiderend"
        return "Oxiderend"

    if parameter_key == "carbonate":
        if value < 0.1:
            return "Zeer laag; geen afzonderlijke klasse"
        if value < 1:
            return "Zeer carbonaatarm"
        if value < 20:
            return "Carbonaatarm"
        if value < 200:
            return "Matig carbonaatrijk"
        if value <= 360:
            return "Carbonaatrijk"
        return "Zeer carbonaatrijk"

    if parameter_key in {"perc_bicarb", "perc_cl", "perc_so4"}:
        if value < -2:
            return "Zeer sterke negatieve associatie"
        if value < -1:
            return "Sterke negatieve associatie"
        if value < -0.25:
            return "Zwakke negatieve associatie"
        if value < 0.25:
            return "Geen duidelijke associatie"
        if value < 1:
            return "Zwakke positieve associatie"
        if value < 2:
            return "Sterke positieve associatie"
        return "Zeer sterke positieve associatie"

    for upper_bound, label in DIAGNOSIS_VALUE_CLASS_BREAKS.get(parameter_key, ()):
        if value < upper_bound:
            return label
    return None


def _salinity_context(value: object, base_label: str) -> str:
    """Voeg waar mogelijk ook zoet, brak of zout toe aan de ionenrijkdomklasse."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return base_label
    numeric = float(numeric)
    if numeric > 850:
        salinity_label = "zout water"
    elif numeric > 16:
        salinity_label = "brak water"
    elif numeric < 16:
        salinity_label = "zoet water"
    else:
        salinity_label = "grenswaarde zoet/brak"
    return f"{base_label} ({salinity_label})"


def _interpret_diagnosis_table(
    frame: pd.DataFrame,
    value_columns: str | tuple[str, ...],
) -> pd.DataFrame:
    """Vervang centrale waarde en spreidingsmaten door hun eigen interpretatie."""
    if frame.empty or "parameter_key" not in frame.columns:
        return frame

    columns = (value_columns,) if isinstance(value_columns, str) else value_columns
    columns = tuple(column for column in columns if column in frame.columns)
    if not columns:
        return frame

    output = frame.copy()
    companion_categories = {
        str(row["parameter_key"]): str(row.get("Categorie", ""))
        for _, row in output.iterrows()
        if str(row.get("Categorie", "")).strip()
    }

    for column in columns:
        interpreted_values: list[object] = []
        for _, row in output.iterrows():
            parameter_key = str(row["parameter_key"])
            original_value = row[column]

            companion_key = DIAGNOSIS_COMPANION_CATEGORY_KEYS.get(parameter_key)
            if companion_key:
                companion = companion_categories.get(companion_key, "").strip()
                if companion and companion not in {"Geen informatie", "Numerieke bio-indicatie"}:
                    interpreted_values.append(companion)
                    continue

            label = _classify_diagnosis_value(parameter_key, original_value)
            if label is not None:
                if parameter_key == "salinity":
                    label = _salinity_context(original_value, label)
                interpreted_values.append(label)
                continue

            if parameter_key in DIAGNOSIS_USE_EXISTING_CATEGORY:
                category = str(row.get("Categorie", "")).strip()
                interpreted_values.append(category or original_value)
                continue

            interpreted_values.append(original_value)

        output[column] = interpreted_values

    return output


def render_waterbody_diagnosis(result: WaterbodyDiagnosisResult) -> None:
    year = f" ({result.filters.year})" if result.filters.year is not None else ""
    st.subheader(f"Interpretatie bodemtoestand: {result.waterbody}{year}")
    st.info(result.summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meetlocaties", result.metadata.get("location_count", 0))
    c2.metric("Mediane profieldekking", f"{result.metadata.get('median_profile_coverage', 0):.0%}")
    c3.metric("Beoordeelde kenmerken", result.metadata.get("parameter_count", 0))
    c4.metric("Sterke aanwijzingen", result.metadata.get("strong_parameter_count", 0))

    st.subheader("Interpretatie per parameter")
    raw_frame = result.diagnoses_frame()

    central_value_column = next(
        (
            column
            for column in (
                "Interpretatie bodemtoestand",
                "Interpretatie waterlichaam",
                "Diagnose waterlichaam",
            )
            if column in raw_frame.columns
        ),
        None,
    )

    interpretation_columns = tuple(
        column
        for column in (
            central_value_column,
            "P25",
            "P75",
            "Minimum",
            "Maximum",
        )
        if column is not None and column in raw_frame.columns
    )
    frame = _interpret_diagnosis_table(raw_frame, interpretation_columns)

    if frame.empty:
        st.warning("Onvoldoende informatie voor een interpretatie bodemtoestand.")
    else:
        rename_columns = {}
        if central_value_column is not None:
            rename_columns[central_value_column] = "Interpretatie bodemtoestand"
        frame = frame.rename(columns=rename_columns)
        frame = frame.drop(
            columns=["parameter_key", "Categorie"],
            errors="ignore",
        )
        frame = _format_coverage(frame, "Locatiedekking")

        preferred_order = [
            "Karakteristiek",
            "Interpretatie bodemtoestand",
            "Aantal locaties",
            "Locaties met voldoende informatie",
            "Bewijskracht",
            "P25",
            "P75",
            "Minimum",
            "Maximum",
            "Locatiedekking",
            "Eenheid",
            "Toelichting",
        ]
        ordered_columns = [column for column in preferred_order if column in frame.columns]
        remaining_columns = [column for column in frame.columns if column not in ordered_columns]
        frame = frame[ordered_columns + remaining_columns]
        st.dataframe(frame, width="stretch", hide_index=True)

    st.subheader("Vergelijking tussen meetlocaties")
    overview = _format_coverage(result.location_overview, "Profieldekking")
    if overview.empty:
        st.info("Geen locatieoverzicht beschikbaar.")
    else:
        st.dataframe(overview, width="stretch", hide_index=True)

    with st.expander("Locatie interpretatie per parameter"):
        detail = result.location_diagnoses.copy()
        if detail.empty:
            st.info("Geen locatie interpretatie mogelijk.")
        else:
            labels = detail[["parameter_key", "parameter_label"]].drop_duplicates()
            options = dict(zip(labels["parameter_label"], labels["parameter_key"]))
            selected = st.selectbox("Selecteer parameter", list(options), key="waterbody_parameter")
            view = detail[detail["parameter_key"].eq(options[selected])].copy()
            view["profile_coverage"] = view["profile_coverage"].map(lambda x: f"{x:.0%}")
            st.dataframe(view.drop(columns=["parameter_key"], errors="ignore"), width="stretch", hide_index=True)


def render_soil_diagnosis(result: SoilDiagnosisResult) -> None:
    year = f" ({result.filters.year})" if result.filters.year is not None else ""
    st.subheader(f"Locatie interpretatie: {result.location_id}{year}")
    st.info(result.summary)
    frame = _interpret_diagnosis_table(
        result.diagnoses_frame(),
        "Waarde",
    ).drop(columns=["parameter_key"], errors="ignore")
    frame = _format_coverage(frame, "Profieldekking")
    st.dataframe(frame, width="stretch", hide_index=True)
    with st.expander("Onderbouwing met soorten"):
        st.dataframe(result.species_evidence, width="stretch", hide_index=True)
    with st.expander(f"Soorten zonder profiel ({len(result.unmatched_species)})"):
        st.dataframe(result.unmatched_species, width="stretch", hide_index=True)


ANALYSIS_TRENDS = "Trends door de tijd"
TREND_MODE_LABELS = {
    "Alle beschikbare locaties": "all",
    "Alleen terugkerende locaties": "recurring",
}


def render_trend_controls(parameters: Sequence[tuple[str, str]]) -> tuple[str | None, str, int, bool, str, bool]:
    """Bundel trendwidgets in een formulier; reken pas na verzenden."""
    if not parameters:
        st.info("Er zijn geen trendparameters beschikbaar.")
        return None, "all", 3, False, TREND_BOOTSTRAP_MODE_DEFAULT, False
    labels={label:key for key,label in parameters}
    modes={"Alle beschikbare locaties":"all","Alleen terugkerende locaties":"recurring"}
    boot=dict(TREND_BOOTSTRAP_MODE_LABELS)
    default_label=next((k for k,v in boot.items() if v==TREND_BOOTSTRAP_MODE_DEFAULT),next(iter(boot)))
    with st.form("soil_trend_analysis_form",clear_on_submit=False):
        c1,c2,c3=st.columns([2,2,1])
        with c1: selected=st.selectbox("Trendparameter",list(labels),key="soil_trend_parameter")
        with c2: mode_label=st.radio("Trendbasis",list(modes),key="soil_trend_mode")
        with c3: minimum_years=st.number_input("Minimum jaren",2,20,3,1,key="soil_trend_minimum_years")
        formal=st.toggle("Voer formele trend- en gevoeligheidsanalyse uit",value=TREND_FORMAL_ANALYSIS_DEFAULT,key="soil_trend_formal_analysis")
        bootstrap_label=st.selectbox("Rekennauwkeurigheid",list(boot),index=list(boot).index(default_label),disabled=not formal,key="soil_trend_bootstrap_mode")
        submitted=st.form_submit_button("Bereken trend",type="primary",use_container_width=True)
    return labels[selected],modes[mode_label],int(minimum_years),bool(formal),boot[bootstrap_label],bool(submitted)



def _render_compact_annual_boxplots(result) -> bool:
    """Toon uitsluitend meetjaren en markeer hiaten met een gestippelde scheiding."""
    annual = result.annual_diagnoses.copy()
    required = {"jaar", "central_value", "p25", "p75", "minimum", "maximum"}
    if annual.empty or not required.issubset(annual.columns):
        return False

    annual["jaar"] = pd.to_numeric(annual["jaar"], errors="coerce")
    for column in ("central_value", "p25", "p75", "minimum", "maximum"):
        annual[column] = pd.to_numeric(annual[column], errors="coerce")
    annual = annual.dropna(subset=["jaar", "central_value", "p25", "p75"]).sort_values("jaar")
    if annual.empty:
        return False

    annual["jaar"] = annual["jaar"].astype(int)
    annual["jaar_label"] = annual["jaar"].astype(str)
    measured_years = annual["jaar"].tolist()
    year_order = annual["jaar_label"].tolist()

    y_min = float(annual["minimum"].min()) if annual["minimum"].notna().any() else float(annual["p25"].min())
    y_max = float(annual["maximum"].max()) if annual["maximum"].notna().any() else float(annual["p75"].max())
    padding = max((y_max - y_min) * 0.05, 0.5)

    x = alt.X(
        "jaar_label:O",
        title="Jaar",
        sort=year_order,
        scale=alt.Scale(domain=year_order, paddingInner=0.48, paddingOuter=0.2),
        axis=alt.Axis(labelAngle=0),
    )
    tooltip = [
        alt.Tooltip("jaar:O", title="Meetjaar"),
        alt.Tooltip("minimum:Q", title="Minimum", format=".3f"),
        alt.Tooltip("p25:Q", title="P25", format=".3f"),
        alt.Tooltip("central_value:Q", title="Mediaan", format=".3f"),
        alt.Tooltip("p75:Q", title="P75", format=".3f"),
        alt.Tooltip("maximum:Q", title="Maximum", format=".3f"),
        alt.Tooltip("location_count:Q", title="Aantal locaties"),
    ]

    extreme_line = alt.Chart(annual).mark_rule(color="#7f8c8d", strokeWidth=1).encode(
        x=x,
        y=alt.Y(
            "minimum:Q",
            title=result.parameter_label,
            scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
        ),
        y2="maximum:Q",
        tooltip=tooltip,
    )
    box = alt.Chart(annual).mark_bar(
        size=36,
        color="#64a9df",
        opacity=0.65,
        stroke="#1261a0",
        strokeWidth=1.5,
    ).encode(x=x, y="p25:Q", y2="p75:Q", tooltip=tooltip)
    lower_cap = alt.Chart(annual).mark_tick(
        color="#1261a0", thickness=2, size=40
    ).encode(x=x, y="p25:Q")
    upper_cap = alt.Chart(annual).mark_tick(
        color="#1261a0", thickness=2, size=40
    ).encode(x=x, y="p75:Q")
    median = alt.Chart(annual).mark_tick(
        color="#d62728", thickness=3, size=36
    ).encode(x=x, y="central_value:Q", tooltip=tooltip)
    minimum_points = alt.Chart(annual).mark_point(
        color="#7f8c8d", size=28, filled=True
    ).encode(x=x, y="minimum:Q")
    maximum_points = alt.Chart(annual).mark_point(
        color="#7f8c8d", size=28, filled=True
    ).encode(x=x, y="maximum:Q")

    layers = [extreme_line, box, lower_cap, upper_cap, median, minimum_points, maximum_points]
    gaps = []
    for previous_year, next_year in zip(measured_years, measured_years[1:]):
        if next_year - previous_year > 1:
            first_missing = previous_year + 1
            last_missing = next_year - 1
            missing_label = str(first_missing) if first_missing == last_missing else f"{first_missing}-{last_missing}"
            gaps.append({
                "jaar_label": str(next_year),
                "vorig_meetjaar": previous_year,
                "volgend_meetjaar": next_year,
                "ontbrekende_jaren": missing_label,
                "y_min": y_min - padding,
                "y_max": y_max + padding,
            })

    if gaps:
        gap_frame = pd.DataFrame(gaps)
        separators = alt.Chart(gap_frame).mark_rule(
            color="#7f7f7f",
            strokeDash=[5, 5],
            strokeWidth=1.5,
        ).encode(
            x=alt.X(
                "jaar_label:O",
                sort=year_order,
                scale=alt.Scale(domain=year_order, paddingInner=0.48, paddingOuter=0.2),
            ),
            xOffset=alt.value(-28),
            y="y_min:Q",
            y2="y_max:Q",
            tooltip=[
                alt.Tooltip("vorig_meetjaar:O", title="Vorig meetjaar"),
                alt.Tooltip("volgend_meetjaar:O", title="Volgend meetjaar"),
                alt.Tooltip("ontbrekende_jaren:N", title="Geen data in"),
            ],
        )
        layers.insert(0, separators)

    chart = alt.layer(*layers).properties(height=420).configure_view(strokeWidth=0)
    st.altair_chart(chart, width="stretch")

    if gaps:
        st.caption(
            "Alleen meetjaren staan op de x-as. Box: P25-P75; rood: mediaan; "
            "grijze punten: minimum en maximum. Gestippelde lijnen markeren "
            "een onderbreking van één of meer jaren zonder meetdata."
        )
    else:
        st.caption(
            "Alleen meetjaren staan op de x-as. Box: P25-P75; rood: mediaan; "
            "grijze punten: minimum en maximum."
        )
    return True




def render_parameter_explainer(result) -> None:
    """Toon altijd het referentiekader dat hoort bij de geselecteerde parameter."""
    explainer = PARAMETER_EXPLAINERS.get(str(result.parameter_key))
    if not explainer:
        return

    with st.expander(explainer["title"], expanded=True):
        st.info(explainer["intro"])
        st.markdown(explainer["body"])
        st.caption(
            "De grenzen ondersteunen de interpretatie van de bio-indicatie. "
            "De weergegeven waarde is niet gelijk aan een directe meting."
        )


def render_trend_interpretation(result) -> None:
    """Toon de inhoudelijke duiding als compacte, scanbare resultaatblokken."""
    interpretation = getattr(result, "interpretation", None)
    if interpretation is None:
        st.info(
            "Voor deze selectie is nog geen uitgebreide resultaatduiding beschikbaar. "
            "Bereken de trend opnieuw nadat de bijgewerkte service is geïnstalleerd."
        )
        return

    st.subheader("Duiding van de resultaten")
    st.markdown(f"### {interpretation.headline}")

    temporal_direction = str(result.metadata.get("temporal_direction", "onvoldoende"))
    spatial_variability = str(result.metadata.get("spatial_variability", "onvoldoende"))
    sampling_quality = str(result.metadata.get("sampling_quality", "onvoldoende"))
    uncertainty_level = str(result.metadata.get("uncertainty_level", "beschrijvend"))

    direction_labels = {
        "stijgend": "Stijgend patroon",
        "dalend": "Dalend patroon",
        "stabiel": "Overwegend stabiel",
        "onvoldoende": "Onvoldoende informatie",
    }
    variability_labels = {
        "klein": "Kleine ruimtelijke spreiding",
        "matig": "Matige ruimtelijke spreiding",
        "groot": "Grote ruimtelijke spreiding",
        "onvoldoende": "Spreiding niet te beoordelen",
    }
    sampling_labels = {
        "voldoende": "Voldoende bemonsteringsbasis",
        "beperkt": "Beperkte bemonsteringsbasis",
        "onvoldoende": "Bemonstering niet te beoordelen",
    }
    uncertainty_labels = {
        "ondersteund": "Richting ondersteund",
        "onzeker": "Richting onzeker",
        "beschrijvend": "Alleen beschrijvend",
        "onvoldoende": "Onzekerheid niet te bepalen",
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tijdspatroon", direction_labels.get(temporal_direction, temporal_direction))
    c2.metric("Verschillen locaties", variability_labels.get(spatial_variability, spatial_variability))
    c3.metric("Bemonstering", sampling_labels.get(sampling_quality, sampling_quality))
    c4.metric("Zekerheid", uncertainty_labels.get(uncertainty_level, uncertainty_level))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Ontwikkeling door de tijd")
        st.write(interpretation.temporal_pattern)
        st.markdown("#### Verschillen tussen locaties")
        st.write(interpretation.spatial_pattern)
    with right:
        st.markdown("#### Ecologische betekenis")
        st.write(interpretation.ecological_meaning)
        st.markdown("#### Bemonstering en representativiteit")
        st.write(interpretation.sampling_context)

    if uncertainty_level in {"onzeker", "onvoldoende"}:
        st.warning(interpretation.uncertainty)
    elif uncertainty_level == "ondersteund":
        st.success(interpretation.uncertainty)
    else:
        st.info(interpretation.uncertainty)

    with st.expander("Beperkingen en interpretatievoorwaarden"):
        for limitation in interpretation.limitations:
            st.markdown(f"- {limitation}")
        st.markdown(
            "- Vergelijk de bio-indicatie waar mogelijk met directe water- en bodemmetingen."
        )


def render_waterbody_trends(result) -> None:
    """Render jaarlijkse medianen, ruimtelijke bandbreedte en locatiedekking."""
    st.subheader(f"Trend: {result.parameter_label} in {result.waterbody}")
    st.info(result.summary)
    render_parameter_explainer(result)
    render_trend_interpretation(result)
    render_formal_trend_assessment(result)
    render_trend_sensitivity(result)
    metadata = result.metadata
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beschikbare jaren", metadata.get("year_count", 0))
    c2.metric("Locaties in analyse", metadata.get("available_location_count", 0))
    c3.metric("Terugkerende locaties", metadata.get("recurring_location_count", 0))
    change = metadata.get("absolute_change")
    c4.metric("Verschil eerste-laatste jaar", f"{change:.3f}" if isinstance(change, (int, float)) else "n.v.t.")

    annual = result.annual_diagnoses.copy()
    if annual.empty:
        st.warning("Voor deze selectie zijn geen jaarlijkse interpretaties beschikbaar.")
        return

    st.subheader("Jaarlijkse verdeling van locatiewaarden")
    if not _render_compact_annual_boxplots(result):
        st.info(
            "Voor deze parameter zijn geen numerieke jaarlijkse waarden beschikbaar "
            "voor een boxplot. Bekijk de jaarlijkse tabel hieronder."
        )

    st.subheader("Jaarlijkse onderbouwing")
    table = annual.rename(columns={
        "jaar": "Jaar",
        "central_value": "Mediane indicatie",
        "category": "Categorie",
        "p25": "P25",
        "p75": "P75",
        "minimum": "Minimum",
        "maximum": "Maximum",
        "location_count": "Aantal locaties",
        "sufficient_location_count": "Locaties met voldoende informatie",
        "location_coverage": "Locatiedekking",
        "evidence_strength": "Bewijskracht",
        "unit": "Eenheid",
    })
    if "Locatiedekking" in table:
        table["Locatiedekking"] = table["Locatiedekking"].map(
            lambda value: f"{value:.0%}" if pd.notna(value) else ""
        )
    visible = [
        "Jaar", "Mediane indicatie", "Eenheid", "Categorie", "P25", "P75",
        "Minimum", "Maximum", "Aantal locaties",
        "Locaties met voldoende informatie", "Locatiedekking", "Bewijskracht",
    ]
    st.dataframe(table[[c for c in visible if c in table]], width="stretch", hide_index=True)

    with st.expander("Interpretaties per locatie en jaar"):
        detail = result.location_year_diagnoses.copy()
        if detail.empty:
            st.info("Geen locatie-jaarresultaten beschikbaar.")
        else:
            detail = detail.sort_values(["jaar", "location_id"])
            detail["profile_coverage"] = detail["profile_coverage"].map(
                lambda value: f"{value:.0%}" if pd.notna(value) else ""
            )
            st.dataframe(detail, width="stretch", hide_index=True)


def render_formal_trend_assessment(result) -> None:
    """Toon de fase-3-trendraming, onzekerheid en datakwaliteit."""
    estimate = getattr(result, "formal_trend", None)
    quality = getattr(result, "quality", None)

    st.subheader("Formele trendbeoordeling")
    if estimate is None or quality is None:
        st.info("Er is geen formele trendbeoordeling beschikbaar voor deze selectie.")
        return

    slope_text = (
        f"{estimate.slope_per_year:.4g} per jaar"
        if estimate.slope_per_year is not None
        else "n.v.t."
    )
    interval_text = (
        f"{estimate.lower_interval:.4g} tot {estimate.upper_interval:.4g}"
        if estimate.lower_interval is not None and estimate.upper_interval is not None
        else "n.v.t."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trendclassificatie", estimate.classification)
    c2.metric("Robuuste helling", slope_text)
    c3.metric(f"{estimate.confidence_level:.0%}-interval", interval_text)
    c4.metric("Datakwaliteit", quality.status)

    st.caption(
        f"Methode: {estimate.method}. Gebruikte jaren: {estimate.year_count}. "
        f"Voltooide bootstrapherhalingen: {estimate.bootstrap_iterations}."
    )

    quality_frame = pd.DataFrame(
        [
            {
                "Kwaliteitsaspect": "Aantal jaren",
                "Waarde": quality.year_count,
            },
            {
                "Kwaliteitsaspect": "Minimum locaties in een jaar",
                "Waarde": quality.minimum_locations_per_year,
            },
            {
                "Kwaliteitsaspect": "Mediaan locaties per jaar",
                "Waarde": round(quality.median_locations_per_year, 1),
            },
            {
                "Kwaliteitsaspect": "Laagste locatiedekking",
                "Waarde": f"{quality.minimum_profile_coverage:.0%}",
            },
            {
                "Kwaliteitsaspect": "Aandeel terugkerende locaties",
                "Waarde": f"{quality.recurring_location_share:.0%}",
            },
            {
                "Kwaliteitsaspect": "Kwaliteitsscore",
                "Waarde": f"{quality.score}/4",
            },
        ]
    )
    st.dataframe(quality_frame, width="stretch", hide_index=True)

    if quality.warnings:
        for warning in quality.warnings:
            st.warning(warning)
    else:
        st.success("Er zijn geen kwaliteitswaarschuwingen voor deze trendselectie.")


def render_trend_sensitivity(result) -> None:
    """Toon de gevoeligheid voor locatiebasis en soortweging."""
    sensitivity = getattr(result, "sensitivity", None)
    if sensitivity is None:
        return

    st.subheader("Gevoeligheidsanalyse")
    if sensitivity.conclusion_stable:
        st.success(sensitivity.explanation)
    else:
        st.warning(sensitivity.explanation)

    scenarios = sensitivity.scenarios.copy()
    if scenarios.empty:
        st.info("Er zijn geen scenarioresultaten beschikbaar.")
        return

    labels = {
        "all_presence": "Alle locaties, aanwezigheid",
        "all_cover": "Alle locaties, bedekking",
        "recurring_presence": "Terugkerende locaties, aanwezigheid",
        "recurring_cover": "Terugkerende locaties, bedekking",
        "all": "Alle locaties",
        "recurring": "Terugkerende locaties",
        "presence": "Aanwezigheid",
        "cover": "Bedekking",
    }
    for column in ("scenario", "location_mode", "weighting"):
        if column in scenarios:
            scenarios[column] = scenarios[column].map(
                lambda value: labels.get(str(value), str(value))
            )

    scenarios = scenarios.rename(
        columns={
            "scenario": "Scenario",
            "location_mode": "Locatiebasis",
            "weighting": "Soortweging",
            "slope_per_year": "Helling per jaar",
            "lower_interval": "Ondergrens interval",
            "upper_interval": "Bovengrens interval",
            "classification": "Trendclassificatie",
            "quality_status": "Datakwaliteit",
            "year_count": "Aantal jaren",
            "minimum_locations_per_year": "Minimum locaties per jaar",
        }
    )
    st.dataframe(scenarios, width="stretch", hide_index=True)
