from __future__ import annotations
from collections.abc import Sequence
import base64
import io
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

from waterplanten_app.config.soil_diagnosis_config import (
    DIAGNOSIS_COMPANION_CATEGORY_KEYS,
    DIAGNOSIS_USE_EXISTING_CATEGORY,
    DIAGNOSIS_VALUE_CLASS_BREAKS,
    PARAMETER_EXPLAINERS,
    SOIL_MAP_CLASS_COLORS,
    SOIL_MAP_DEFAULT_PARAMETER_KEY,
    SOIL_MAP_HEIGHT,
    SOIL_MAP_OSM_ATTRIBUTION,
    SOIL_MAP_OSM_TILE_URL,
    SOIL_MAP_POINT_RADIUS,
    SOIL_MAP_UNKNOWN_COLOR,
    SOIL_TREND_ICON_CONTENT_SIZE,
    SOIL_TREND_ICON_MAX_PIXELS,
    SOIL_TREND_ICON_MIN_PIXELS,
    SOIL_TREND_ICON_PLOT_SIZE,
    SOIL_TREND_ICON_TILE_SIZE,
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



@st.cache_resource(show_spinner=False)
def _rd_to_wgs84_transformer():
    """Maak de RD-WGS84-transformer pas aan wanneer de kaart deze nodig heeft."""
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def _normalise_map_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Gebruik WGS84 indien aanwezig en zet anders Nederlandse RD om naar WGS84."""
    output = frame.copy()
    for column in ("latitude", "longitude", "rd_x", "rd_y"):
        if column not in output:
            output[column] = pd.NA
        output[column] = pd.to_numeric(output[column], errors="coerce")

    valid_wgs84 = output["latitude"].between(-90, 90) & output["longitude"].between(-180, 180)
    use_rd = ~valid_wgs84 & output["rd_x"].notna() & output["rd_y"].notna()
    if use_rd.any():
        longitudes, latitudes = _rd_to_wgs84_transformer().transform(
            output.loc[use_rd, "rd_x"].to_numpy(),
            output.loc[use_rd, "rd_y"].to_numpy(),
        )
        output.loc[use_rd, "longitude"] = longitudes
        output.loc[use_rd, "latitude"] = latitudes

    valid = output["latitude"].between(-90, 90) & output["longitude"].between(-180, 180)
    return output.loc[valid].copy()


def _map_color(interpretation: object) -> list[int]:
    """Koppel een interpretatie aan een consistente, kleurenblindvriendelijke kleur."""
    text = str(interpretation).strip()
    direct = SOIL_MAP_CLASS_COLORS.get(text)
    if direct is not None:
        return list(direct)
    # Saliniteit bevat aanvullend '(zoet water)' of '(brak water)'.
    for label, color in SOIL_MAP_CLASS_COLORS.items():
        if text.startswith(label):
            return list(color)
    return list(SOIL_MAP_UNKNOWN_COLOR)


def _hex_color(rgba: list[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgba[:3])


def _render_map_legend(frame: pd.DataFrame) -> None:
    rows = (
        frame[["interpretation", "map_color"]]
        .drop_duplicates("interpretation")
        .sort_values("interpretation")
    )
    items = []
    for row in rows.itertuples():
        color = _hex_color(list(row.map_color))
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin:0 16px 8px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'border:1px solid #444;margin-right:6px;"></span>{row.interpretation}</span>'
        )
    st.markdown("**Legenda**<br>" + "".join(items), unsafe_allow_html=True)


def render_waterbody_soil_map(result: WaterbodyDiagnosisResult) -> None:
    """Render een OSM-puntenkaart met één geselecteerde bodemtoestandparameter."""
    try:
        import pydeck as pdk
    except ImportError:
        st.warning(
            "De kaartmodule is niet beschikbaar. Installeer pydeck en pyproj; "
            "de overige bodemdiagnosefuncties blijven wel bruikbaar."
        )
        return

    source = getattr(result, "map_locations", pd.DataFrame()).copy()
    if source.empty:
        st.info(
            "De kaart kan niet worden weergegeven omdat nog geen kaartgegevens voor "
            "de geselecteerde waterlichaamdiagnose beschikbaar zijn."
        )
        return

    required = {"parameter_key", "parameter_label", "location_id", "interpretation"}
    if not required.issubset(source.columns):
        st.warning("De kaartdataset mist noodzakelijke diagnosevelden.")
        return

    labels = (
        source[["parameter_key", "parameter_label"]]
        .drop_duplicates()
        .sort_values("parameter_label")
    )
    label_to_key = dict(zip(labels["parameter_label"], labels["parameter_key"]))
    key_to_label = dict(zip(labels["parameter_key"], labels["parameter_label"]))
    default_key = str(
        getattr(result, "map_metadata", {}).get(
            "default_parameter_key", SOIL_MAP_DEFAULT_PARAMETER_KEY
        )
    )
    if default_key not in key_to_label:
        default_key = str(labels["parameter_key"].iloc[0])
    option_labels = list(label_to_key)
    default_index = option_labels.index(key_to_label[default_key])

    left, right = st.columns([2, 1])
    with left:
        selected_label = st.selectbox(
            "Karakteristiek op de kaart",
            option_labels,
            index=default_index,
            key="soil_waterbody_map_parameter",
        )
    with right:
        sufficient_only = st.toggle(
            "Alleen voldoende informatie",
            value=False,
            key="soil_waterbody_map_sufficient_only",
            help="Verberg locaties met onvoldoende informatie of zonder bruikbare interpretatie.",
        )

    parameter_key = str(label_to_key[selected_label])
    frame = source.loc[source["parameter_key"].astype(str).eq(parameter_key)].copy()
    if sufficient_only and "evidence_strength" in frame:
        frame = frame.loc[
            ~frame["evidence_strength"].astype(str).isin(
                ["Geen informatie", "Onvoldoende informatie"]
            )
        ].copy()

    frame = _normalise_map_coordinates(frame)
    if frame.empty:
        metadata = getattr(result, "map_metadata", {})
        coordinate_metadata = metadata.get("coordinate_metadata", {})
        st.warning(
            "Voor de geselecteerde locaties zijn geen bruikbare WGS84- of "
            "Rijksdriehoekscoördinaten gevonden. Herkende bronkolommen: "
            f"{coordinate_metadata or 'geen'}."
        )
        return

    frame["interpretation"] = frame["interpretation"].fillna("Onvoldoende informatie").astype(str)
    frame["map_color"] = frame["interpretation"].map(_map_color)
    frame["value_display"] = frame["value"].map(
        lambda value: f"{value:.3f}" if isinstance(value, (int, float)) and pd.notna(value) else str(value)
    )
    frame["coverage_display"] = pd.to_numeric(
        frame.get("profile_coverage", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).map(lambda value: f"{value:.0%}" if pd.notna(value) else "Onbekend")
    frame["species_display"] = pd.to_numeric(
        frame.get("species_count", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).map(lambda value: str(int(value)) if pd.notna(value) else "Onbekend")

    center_latitude = float(frame["latitude"].mean())
    center_longitude = float(frame["longitude"].mean())
    longitude_span = float(frame["longitude"].max() - frame["longitude"].min())
    latitude_span = float(frame["latitude"].max() - frame["latitude"].min())
    span = max(longitude_span, latitude_span)
    zoom = 11 if span < 0.08 else 10 if span < 0.18 else 9 if span < 0.4 else 8

    tile_layer = pdk.Layer(
        "TileLayer",
        data=SOIL_MAP_OSM_TILE_URL,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
    )
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        get_position="[longitude, latitude]",
        get_fill_color="map_color",
        get_line_color=[35, 35, 35, 230],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        get_radius=SOIL_MAP_POINT_RADIUS,
        radius_min_pixels=6,
        radius_max_pixels=16,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<b>Locatie:</b> {location_id}<br/>"
            "<b>Karakteristiek:</b> {parameter_label}<br/>"
            "<b>Interpretatie:</b> {interpretation}<br/>"
            "<b>Indicatieve waarde:</b> {value_display}<br/>"
            "<b>Bewijskracht:</b> {evidence_strength}<br/>"
            "<b>Profieldekking:</b> {coverage_display}<br/>"
            "<b>Indicatieve soorten:</b> {species_display}"
        ),
        "style": {
            "backgroundColor": "white",
            "color": "#1f2937",
            "fontSize": "13px",
            "border": "1px solid #9ca3af",
        },
    }
    deck = pdk.Deck(
        layers=[tile_layer, point_layer],
        initial_view_state=pdk.ViewState(
            latitude=center_latitude,
            longitude=center_longitude,
            zoom=zoom,
            pitch=0,
            bearing=0,
        ),
        tooltip=tooltip,
        map_style=None,
    )
    st.pydeck_chart(deck, height=SOIL_MAP_HEIGHT, use_container_width=True)
    _render_map_legend(frame)

    total_locations = int(frame["location_id"].nunique())
    sufficient_locations = int(
        frame.loc[
            ~frame["evidence_strength"].astype(str).isin(
                ["Geen informatie", "Onvoldoende informatie"]
            ),
            "location_id",
        ].nunique()
    ) if "evidence_strength" in frame else total_locations
    modal_interpretation = frame["interpretation"].mode()
    main_interpretation = (
        str(modal_interpretation.iloc[0]) if not modal_interpretation.empty else "Onbekend"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Locaties op kaart", total_locations)
    m2.metric("Voldoende informatie", sufficient_locations)
    m3.metric("Meest voorkomende interpretatie", main_interpretation)
    st.caption(
        f"Ondergrond: {SOIL_MAP_OSM_ATTRIBUTION}. De punten tonen bio-indicaties "
        "op meetlocaties en geen directe chemische water- of bodemmetingen. "
        "Tussen meetlocaties is niet geïnterpoleerd."
    )



def render_waterbody_diagnosis(result: WaterbodyDiagnosisResult) -> None:
    year = f" ({result.filters.year})" if result.filters.year is not None else ""
    st.subheader(f"Interpretatie bodemtoestand: {result.waterbody}{year}")
    st.info(result.summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meetlocaties", result.metadata.get("location_count", 0))
    c2.metric("Mediane profieldekking", f"{result.metadata.get('median_profile_coverage', 0):.0%}")
    c3.metric("Beoordeelde kenmerken", result.metadata.get("parameter_count", 0))
    c4.metric("Sterke aanwijzingen", result.metadata.get("strong_parameter_count", 0))

    st.subheader("Ruimtelijke interpretatie bodemtoestand")
    render_waterbody_soil_map(result)

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



def _selected_pydeck_objects(state: object) -> list[dict[str, object]]:
    """Lees geselecteerde objecten robuust uit verschillende Streamlit-statevormen."""
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    objects = getattr(selection, "objects", None)
    if objects is None and isinstance(selection, dict):
        objects = selection.get("objects", {})
    if not isinstance(objects, dict):
        return []
    selected: list[dict[str, object]] = []
    for layer_objects in objects.values():
        if isinstance(layer_objects, list):
            selected.extend(
                item for item in layer_objects if isinstance(item, dict)
            )
    return selected


def _select_detail_location_from_map() -> None:
    """Werk de bestaande meetlocatieselector bij vóór de pagina opnieuw rendert."""
    state = st.session_state.get("soil_detail_location_map")
    selected = _selected_pydeck_objects(state)
    if selected and selected[0].get("location_id") is not None:
        st.session_state["soil_location"] = str(selected[0]["location_id"])


def render_location_navigation_map(
    result: SoilDiagnosisResult,
    waterbody_result: WaterbodyDiagnosisResult | None,
) -> None:
    """Toon onder Detail meetlocatie een aanklikbare OSM-kaart van alle locaties."""
    if waterbody_result is None:
        st.info("Geen waterlichaamkaart beschikbaar voor locatienavigatie.")
        return
    source = getattr(waterbody_result, "map_locations", pd.DataFrame()).copy()
    if source.empty or "location_id" not in source.columns:
        st.info("Voor dit waterlichaam zijn geen kaartlocaties beschikbaar.")
        return

    coordinate_columns = [
        column for column in ("latitude", "longitude", "rd_x", "rd_y")
        if column in source.columns
    ]
    frame = source[["location_id", *coordinate_columns]].copy()
    frame = frame.drop_duplicates(subset=["location_id"]).reset_index(drop=True)
    frame = _normalise_map_coordinates(frame)
    if frame.empty:
        st.warning("Voor de meetlocaties zijn geen bruikbare kaartcoördinaten gevonden.")
        return

    try:
        import pydeck as pdk
    except ImportError:
        st.warning("Installeer pydeck en pyproj om de locatienavigatiekaart te tonen.")
        return

    current_location = str(result.location_id)
    frame["is_selected"] = frame["location_id"].astype(str).eq(current_location)
    frame["fill_color"] = frame["is_selected"].map(
        {True: [220, 38, 38, 245], False: [37, 99, 235, 210]}
    )
    frame["line_color"] = frame["is_selected"].map(
        {True: [127, 29, 29, 255], False: [30, 64, 175, 235]}
    )
    frame["point_radius"] = frame["is_selected"].map(
        {True: SOIL_MAP_POINT_RADIUS * 1.5, False: SOIL_MAP_POINT_RADIUS}
    )
    frame["selected_label"] = frame["is_selected"].map(
        {True: "Huidige meetlocatie", False: "Klik om dit detail te openen"}
    )

    longitude_span = float(frame["longitude"].max() - frame["longitude"].min())
    latitude_span = float(frame["latitude"].max() - frame["latitude"].min())
    span = max(longitude_span, latitude_span)
    zoom = 11 if span < 0.08 else 10 if span < 0.18 else 9 if span < 0.4 else 8

    tile_layer = pdk.Layer(
        "TileLayer",
        id="soil-detail-osm",
        data=SOIL_MAP_OSM_TILE_URL,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
    )
    location_layer = pdk.Layer(
        "ScatterplotLayer",
        id="soil-detail-locations",
        data=frame,
        get_position="[longitude, latitude]",
        get_fill_color="fill_color",
        get_line_color="line_color",
        get_radius="point_radius",
        radius_min_pixels=7,
        radius_max_pixels=18,
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<b>Meetlocatie:</b> {location_id}<br/>"
            "{selected_label}"
        ),
        "style": {
            "backgroundColor": "white",
            "color": "#1f2937",
            "fontSize": "13px",
            "border": "1px solid #9ca3af",
        },
    }
    deck = pdk.Deck(
        layers=[tile_layer, location_layer],
        initial_view_state=pdk.ViewState(
            latitude=float(frame["latitude"].mean()),
            longitude=float(frame["longitude"].mean()),
            zoom=zoom,
            pitch=0,
            bearing=0,
        ),
        tooltip=tooltip,
        map_style=None,
    )
    st.pydeck_chart(
        deck,
        height=SOIL_MAP_HEIGHT,
        width="stretch",
        selection_mode="single-object",
        on_select=_select_detail_location_from_map,
        key="soil_detail_location_map",
    )
    st.caption(
        "Klik op een blauwe meetlocatie om het detail van die locatie te openen. "
        "De huidige meetlocatie is rood gemarkeerd."
    )

def render_soil_diagnosis(
    result: SoilDiagnosisResult,
    waterbody_result: WaterbodyDiagnosisResult | None = None,
) -> None:
    year = f" ({result.filters.year})" if result.filters.year is not None else ""
    st.subheader(f"Locatie interpretatie: {result.location_id}{year}")
    st.info(result.summary)
    st.subheader("Meetlocaties op de kaart")
    render_location_navigation_map(result, waterbody_result)
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


_TREND_ARROW_SYMBOLS = {
    "Overtuigend stijgend": "↑",
    "Mogelijk stijgend": "↗",
    "Stabiel": "→",
    "Mogelijk dalend": "↘",
    "Overtuigend dalend": "↓",
    "Onvoldoende gegevens": "·",
}


def _trend_arrow_symbol(classification: object) -> str:
    text = str(classification).strip()
    if text in _TREND_ARROW_SYMBOLS:
        return _TREND_ARROW_SYMBOLS[text]
    lowered = text.lower()
    if "stijgend" in lowered:
        return "↑" if "overtuigend" in lowered else "↗"
    if "dalend" in lowered:
        return "↓" if "overtuigend" in lowered else "↘"
    if "stabiel" in lowered or "geen trend" in lowered:
        return "→"
    return "·"


def _render_trend_arrow_legend(frame: pd.DataFrame) -> None:
    present = set(frame["classification"].dropna().astype(str))
    items = []
    for label, symbol in _TREND_ARROW_SYMBOLS.items():
        if label in present:
            items.append(
                '<span style="display:inline-flex;align-items:center;margin:0 18px 8px 0;">'
                f'<b style="font-size:25px;width:25px;text-align:center;margin-right:5px;">{symbol}</b>'
                f'{label}</span>'
            )
    if items:
        st.markdown("**Legenda lokale trendrichting**<br>" + "".join(items), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _trend_icon_definitions() -> dict[str, dict[str, object]] | None:
    """Maak per trendklasse een IconLayer-object van het bijbehorende PNG-bestand."""
    from PIL import Image

    asset_directory = Path(__file__).resolve().parents[1] / "assets" / "map"
    filenames = {
        "up": "trend_up.png",
        "stable": "trend_stable.png",
        "down": "trend_down.png",
        "unknown": "trend_unknown.png",
    }
    missing = [
        filename
        for filename in filenames.values()
        if not (asset_directory / filename).is_file()
    ]
    if missing:
        return None

    definitions: dict[str, dict[str, object]] = {}
    for key, filename in filenames.items():
        with Image.open(asset_directory / filename) as source:
            icon = source.convert("RGBA")
            alpha_bbox = icon.getchannel("A").getbbox()
            if alpha_bbox is not None:
                icon = icon.crop(alpha_bbox)
            icon.thumbnail(
                (SOIL_TREND_ICON_CONTENT_SIZE, SOIL_TREND_ICON_CONTENT_SIZE),
                Image.Resampling.LANCZOS,
            )

            buffer = io.BytesIO()
            icon.save(buffer, format="PNG", optimize=True)

        definitions[key] = {
            "url": "data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "width": icon.width,
            "height": icon.height,
            "anchorX": icon.width // 2,
            "anchorY": icon.height // 2,
        }

    return definitions


def _trend_icon_name(classification: object) -> str:
    text = str(classification).lower()
    if "stijgend" in text:
        return "up"
    if "dalend" in text:
        return "down"
    if "stabiel" in text or "geen trend" in text:
        return "stable"
    return "unknown"


def render_waterbody_trend_map(result) -> None:
    """Toon lokale trends met iconen uit assets/map op een OSM-ondergrond."""
    maps = dict(getattr(result, "trend_maps", {}) or {})
    fallback = getattr(result, "trend_map", None)
    if not maps and fallback is not None:
        maps = {str(result.parameter_key): fallback}
    if not maps:
        st.info("Voor dit resultaat is nog geen lokale trendkaart beschikbaar. Bereken de trend opnieuw.")
        return

    labels = {key: str(getattr(value, "parameter_label", key)) for key, value in maps.items()}
    label_to_key = {label: key for key, label in labels.items()}
    options = sorted(label_to_key)
    default = labels.get(str(result.parameter_key), options[0])
    selected = st.selectbox(
        "Parameter op de trendkaart",
        options,
        index=options.index(default) if default in options else 0,
        key="soil_trend_map_parameter",
        help="Wisselen gebruikt vooraf berekende kaartresultaten en start geen nieuwe bootstrap.",
    )

    trend_map = maps[label_to_key[selected]]
    frame = getattr(trend_map, "locations", pd.DataFrame()).copy()
    if frame.empty:
        st.info("Voor deze parameter zijn onvoldoende locatie-jaarresultaten.")
        return

    try:
        import pydeck as pdk
    except ImportError:
        st.warning("Installeer pydeck en pyproj om de trendkaart weer te geven.")
        return

    icon_definitions = _trend_icon_definitions()
    if icon_definitions is None:
        st.warning(
            "De trendkaart mist een of meer bestanden in waterplanten_app/assets/map: "
            "trend_up.png, trend_down.png, trend_stable.png of trend_unknown.png."
        )
        return
    frame = _normalise_map_coordinates(frame)
    if frame.empty:
        st.warning("Voor de trendresultaten zijn geen bruikbare coördinaten gevonden.")
        return

    for column, default_value in {
        "classification": "Onvoldoende gegevens",
        "location_id": "Onbekend",
        "parameter_label": selected,
        "latest_interpretation": "Onbekend",
    }.items():
        if column not in frame:
            frame[column] = default_value
        frame[column] = frame[column].fillna(default_value).astype(str)

    frame["icon_name"] = frame["classification"].map(_trend_icon_name)
    frame["icon_data"] = frame["icon_name"].map(
        lambda key: icon_definitions.get(str(key), icon_definitions["unknown"])
    )
    supported = frame.get(
        "statistically_supported", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    # Houd de accessor op 1. De uiteindelijke grootte wordt uitsluitend door
    # size_scale en de harde pixelgrenzen van de IconLayer bepaald.
    frame["icon_size"] = 1.0

    frame["tau_display"] = pd.to_numeric(
        frame.get("tau", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    ).map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "Niet berekend"
    )
    frame["p_display"] = pd.to_numeric(
        frame.get("p_value", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    ).map(
        lambda value: f"{value:.3g}" if pd.notna(value) else "Niet berekend"
    )
    frame["slope_display"] = pd.to_numeric(
        frame.get(
            "sen_slope_per_year", pd.Series(index=frame.index, dtype=float)
        ),
        errors="coerce",
    ).map(lambda value: f"{value:+.4g} per jaar" if pd.notna(value) else "Niet berekend")
    frame["supported_display"] = supported.map({True: "Ja", False: "Nee"})
    frame["years_display"] = pd.to_numeric(
        frame.get("year_count", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).map(
        lambda value: str(int(value)) if pd.notna(value) else "Onbekend"
    )

    span = max(
        float(frame["longitude"].max() - frame["longitude"].min()),
        float(frame["latitude"].max() - frame["latitude"].min()),
    )
    zoom = 11 if span < 0.08 else 10 if span < 0.18 else 9 if span < 0.4 else 8
    tile_layer = pdk.Layer(
        "TileLayer",
        data=SOIL_MAP_OSM_TILE_URL,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
    )
    icon_layer = pdk.Layer(
        "IconLayer",
        data=frame,
        get_position="[longitude, latitude]",
        get_icon="icon_data",
        get_size="icon_size",
        size_scale=SOIL_TREND_ICON_PLOT_SIZE,
        size_units="pixels",
        size_min_pixels=SOIL_TREND_ICON_MIN_PIXELS,
        size_max_pixels=SOIL_TREND_ICON_MAX_PIXELS,
        billboard=True,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<b>Locatie:</b> {location_id}<br/>"
            "<b>Parameter:</b> {parameter_label}<br/>"
            "<b>Trendklasse:</b> {classification}<br/>"
            "<b>Kendall's Tau:</b> {tau_display}<br/>"
            "<b>p-waarde:</b> {p_display}<br/>"
            "<b>Statistisch ondersteund:</b> {supported_display}<br/>"
            "<b>Sen-helling:</b> {slope_display}<br/>"
            "<b>Meetjaren:</b> {years_display}<br/>"
            "<b>Laatste interpretatie:</b> {latest_interpretation}"
        )
    }
    deck = pdk.Deck(
        layers=[tile_layer, icon_layer],
        initial_view_state=pdk.ViewState(
            latitude=float(frame["latitude"].mean()),
            longitude=float(frame["longitude"].mean()),
            zoom=zoom,
        ),
        tooltip=tooltip,
        map_style=None,
    )
    st.pydeck_chart(deck, height=SOIL_MAP_HEIGHT, use_container_width=True)

    class_labels = (
        ("Sterk stijgend", "up"),
        ("Stijgend", "up"),
        ("Weinig consistente richting", "stable"),
        ("Dalend", "down"),
        ("Sterk dalend", "down"),
        ("Onvoldoende gegevens", "unknown"),
    )
    present = set(frame["classification"].astype(str))
    file_by_key = {
        "up": "trend_up.png",
        "down": "trend_down.png",
        "stable": "trend_stable.png",
        "unknown": "trend_unknown.png",
    }
    legend_items = [
        f"{file_by_key[key]}: {label}"
        for label, key in class_labels
        if label in present
    ]
    if legend_items:
        st.caption("Legenda: " + " | ".join(legend_items))

    metadata = dict(getattr(trend_map, "metadata", {}))
    dominant = frame["classification"].mode()
    dominant_class = str(dominant.iloc[0]) if not dominant.empty else "Onbekend"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locaties met trendraming", int(metadata.get("tested_location_count", frame["location_id"].nunique())))
    c2.metric("Statistisch ondersteund", int(metadata.get("statistically_supported_count", 0)))
    c3.metric("Onvoldoende meetjaren", int(metadata.get("insufficient_location_count", 0)))
    c4.metric("Meest voorkomende trendklasse", dominant_class)
    st.caption(
        f"Ondergrond: {SOIL_MAP_OSM_ATTRIBUTION}. De pictogramrichting toont de lokale "
        "trend; grotere pictogrammen wijzen op een sterkere en/of statistisch "
        "ondersteunde trend. Tussen locaties is niet geïnterpoleerd."
    )


def render_waterbody_trends(result) -> None:
    """Render jaarlijkse medianen, ruimtelijke bandbreedte en locatiedekking."""
    st.subheader(f"Trend: {result.parameter_label} in {result.waterbody}")
    st.info(result.summary)
    render_parameter_explainer(result)
    st.subheader("Lokale trends op de kaart")
    render_waterbody_trend_map(result)
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
