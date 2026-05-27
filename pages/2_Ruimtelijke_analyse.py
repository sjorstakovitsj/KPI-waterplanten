import streamlit as st

from waterplanten_app.domain.contracts import (
    ANALYSIS_LEVEL_GROUPS_AGGREGATIONS,
    ANALYSIS_LEVEL_OPTIONS,
    COVERAGE_TYPE_OPTIONS,
    COVERAGE_TYPE_TROFIENIVEAU,
    LAYER_MODE_OPTIONS,
)
from waterplanten_app.ui.filters import select_projects, select_year
from waterplanten_app.ui.legends import render_spatial_legend
from waterplanten_app.ui.maps import render_spatial_map
from waterplanten_app.ui.tables import render_spatial_table
from waterplanten_app.pipelines.spatial_page_pipeline import (
    build_spatial_page_state,
    load_spatial_base_data,
    prepare_spatial_filter_context,
)

st.set_page_config(layout="wide", page_title="Ruimtelijke analyse")
st.title("🗺️ Ruimtelijke analyse")
st.markdown("Vergelijk de vegetatieontwikkeling met diepte en doorzicht.")


def main() -> None:
    df = load_spatial_base_data()
    st.sidebar.header("Filters")

    if df.empty:
        st.error("Geen data geladen.")
        st.stop()

    year = select_year(df, "Selecteer jaar")
    projects = select_projects(df)
    filtered, species_options = prepare_spatial_filter_context(df, year, projects)

    if filtered.empty:
        st.warning("Geen data gevonden voor deze selectie.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("Kaartinstellingen")

    analysis = st.sidebar.radio("Kies analyseniveau", ANALYSIS_LEVEL_OPTIONS)
    coverage = st.sidebar.selectbox(
        "selecteer groep" if analysis == ANALYSIS_LEVEL_GROUPS_AGGREGATIONS else "selecteer soort",
        COVERAGE_TYPE_OPTIONS if analysis == ANALYSIS_LEVEL_GROUPS_AGGREGATIONS else list(species_options),
    )
    layer = st.sidebar.radio("kies kaartlaag", LAYER_MODE_OPTIONS)

    page_state = build_spatial_page_state(
        df=df,
        year=year,
        projects=projects,
        analysis=analysis,
        coverage=coverage,
        layer=layer,
    )

    result = page_state.result
    chem_points = page_state.chem_points

    render_spatial_legend(result)
    render_spatial_map(result, chem_points, filtered_base_data=page_state.filtered_base_data)

    if (
        result.analysis_level == ANALYSIS_LEVEL_GROUPS_AGGREGATIONS
        and result.coverage_type == COVERAGE_TYPE_TROFIENIVEAU
    ):
        with st.expander("ℹ️ Toelichting trofieniveau"):
            st.markdown(
                """De indeling van soorten naar trofieniveau is gebaseerd op:

**Verhofstad et al. (2025)** - *Waterplanten in Nederland: Regionaal herstel, landelijke achteruitgang*.

https://www.floron.nl/Portals/1/Downloads/Publicaties/VerhofstadETAL2025_DLN_Waterplanten_in_Nederland_Regionaal_herstel_Landelijke_achteruitgang.pdf"""
            )

    st.divider()
    render_spatial_table(result)


if __name__ == "__main__":
    main()