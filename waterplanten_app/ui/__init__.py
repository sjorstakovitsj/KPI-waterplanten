"""UI-componenten voor de Streamlit-app."""

from .filters import select_projects, select_year
from .metrics import *
from .legends import render_bathymetry_legend, render_spatial_legend
from .maps import render_spatial_map
from .tables import render_spatial_table
from .charts import *

__all__ = [
    "select_projects",
    "select_year",
    "render_bathymetry_legend",
    "render_spatial_legend",
    "render_spatial_map",
    "render_spatial_table",
]