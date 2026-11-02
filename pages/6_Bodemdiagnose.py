from __future__ import annotations
import streamlit as st
import pydeck as pdk
from pyproj import Transformer
from utils import load_data
from waterplanten_app.config.soil_diagnosis_config import SOIL_DIAGNOSIS_SESSION_PREFIX
from waterplanten_app.domain.contracts import DashboardFilters
from waterplanten_app.services.soil_diagnosis_service import get_available_locations,get_available_trend_parameters,get_soil_diagnosis,get_waterbody_diagnosis,get_waterbody_trends
from waterplanten_app.ui.filters import select_projects,select_waterbodies,select_year
from waterplanten_app.ui.soil_diagnosis_components import ANALYSIS_LOCATION,ANALYSIS_TRENDS,render_analysis_level_selector,render_location_selector,render_soil_diagnosis,render_trend_controls,render_waterbody_diagnosis,render_waterbody_trends,render_weighting_selector


def _filters_key(filters): return (filters.year,tuple(filters.projects),tuple(filters.waterbodies))
def _result_key(kind,*parts): return f"{SOIL_DIAGNOSIS_SESSION_PREFIX}:{kind}:{hash(parts)}"
def _remember(key,factory):
    if key not in st.session_state:
        with st.spinner("Afgeleide bodemtoestand wordt berekend..."): st.session_state[key]=factory()
    return st.session_state[key]

st.set_page_config(layout="wide",page_title="Bodemdiagnose")
st.title("🧭 Afgeleide bodemtoestand")
st.warning("De resultaten zijn ecologische aanwijzingen en geen vervanging voor directe metingen.")
data=load_data()
if data.empty: st.error("Geen vegetatiegegevens geladen."); st.stop()
st.sidebar.header("Filters")
projects=select_projects(data,default=("KRW","N2000")); waterbodies=select_waterbodies(data,projects,default=("Gooimeer",)); year=select_year(data,"Selecteer meetjaar",default=2024)
filters=DashboardFilters(year=year,projects=projects,waterbodies=waterbodies)
if len(waterbodies)!=1: st.info("Selecteer precies één waterlichaam."); st.stop()
analysis=render_analysis_level_selector(); weighting=render_weighting_selector(); filter_key=_filters_key(filters)
try:
    if analysis==ANALYSIS_LOCATION:
        locations=_remember(_result_key("locations",filter_key),lambda:get_available_locations(filters))
        location=render_location_selector(locations)
        if location is None: st.stop()
        result=_remember(_result_key("location",filter_key,location,weighting),lambda:get_soil_diagnosis(filters,location,weighting))
        waterbody_result=_remember(
            _result_key("waterbody",filter_key,weighting),
            lambda:get_waterbody_diagnosis(filters,weighting),
        )
        render_soil_diagnosis(result,waterbody_result)
    elif analysis==ANALYSIS_TRENDS:
        st.caption("Kies de instellingen en klik op ‘Bereken trend’. Formele bootstrapanalyse is optioneel.")
        parameter,mode,min_years,formal,bootstrap,submitted=render_trend_controls(get_available_trend_parameters())
        if parameter is None: st.stop()
        key=_result_key("trend",tuple(projects),tuple(waterbodies),parameter,weighting,mode,min_years,formal,bootstrap)
        if submitted:
            with st.spinner("Trendanalyse wordt berekend..."):
                st.session_state[key]=get_waterbody_trends(filters,parameter,weighting,mode,min_years,include_formal_analysis=formal,bootstrap_mode=bootstrap)
        if key in st.session_state: render_waterbody_trends(st.session_state[key])
        else: st.info("Klik op ‘Bereken trend’. Zonder formele analyse verschijnt snel de beschrijvende trend.")
    else:
        result=_remember(_result_key("waterbody",filter_key,weighting),lambda:get_waterbody_diagnosis(filters,weighting))
        render_waterbody_diagnosis(result)
except (FileNotFoundError,ValueError,RuntimeError) as exc:
    st.error(f"De bodemdiagnose kon niet worden uitgevoerd: {exc}")
