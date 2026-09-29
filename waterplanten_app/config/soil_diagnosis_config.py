"""Configuratie voor de soortgebaseerde bodem- en waterdiagnose."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from waterplanten_app.config.settings import (
    SOIL_PROFILES_METADATA,
    SOIL_PROFILES_PARQUET,
    SOIL_PROFILES_PIPELINE_VERSION,
    SOIL_PROFILES_SOURCE,
)
SOIL_PROFILES_VIEW: Final[str] = "gold_soil_species_profiles"

WEIGHTING_PRESENCE: Final[str] = "presence"
WEIGHTING_COVER: Final[str] = "cover"
WEIGHTING_OPTIONS: Final[tuple[str, ...]] = (WEIGHTING_PRESENCE, WEIGHTING_COVER)

MIN_SPECIES_FOR_INDICATION: Final[int] = 3
MIN_SPECIES_FOR_STRONG_INDICATION: Final[int] = 5
MIN_PROFILE_COVERAGE_STRONG: Final[float] = 0.70
MAX_WEIGHTED_CV_STRONG: Final[float] = 0.25

INDICATOR_STRENGTH_LABELS: Final[dict[int, str]] = {
    1: "Zwak",
    2: "Matig",
    3: "Sterk",
}

REQUIRED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "soortnaam",
    "ind_pH", "gg_pH", "min_pH", "max_pH",
    "ind_sal", "gg_sal", "min_sal", "max_sal",
    "perc_bicarb", "perc_cl", "perc_so4",
    "ind_alk", "gg_alk", "min_alk", "max_alk",
    "ind_bodem", "bodemtype",
    "ind_os_bodem", "gg_os_bodem", "os_bodem",
    "ind_redox", "gg_redox", "redox",
    "ind_carb", "gg_carb", "carb_bodem",
)

INDICATOR_COLUMNS: Final[tuple[str, ...]] = (
    "ind_pH", "ind_sal", "ind_alk", "ind_bodem",
    "ind_os_bodem", "ind_redox", "ind_carb",
)
ANION_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "perc_bicarb", "perc_cl", "perc_so4",
)
NUMERIC_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_SOURCE_COLUMNS
    if column not in {"soortnaam", "bodemtype", "os_bodem", "redox", "carb_bodem"}
)


@dataclass(frozen=True)
class NumericParameterSpec:
    key: str
    label: str
    indicator_column: str
    mean_column: str
    minimum_column: str | None = None
    maximum_column: str | None = None
    unit: str = ""


NUMERIC_PARAMETER_SPECS: Final[tuple[NumericParameterSpec, ...]] = (
    NumericParameterSpec("ph", "pH waterlaag", "ind_pH", "gg_pH", "min_pH", "max_pH", "pH"),
    NumericParameterSpec("salinity", "Saliniteit waterlaag", "ind_sal", "gg_sal", "min_sal", "max_sal"),
    NumericParameterSpec("alkalinity", "Alkaliniteit waterlaag", "ind_alk", "gg_alk", "min_alk", "max_alk"),
    NumericParameterSpec("organic_matter", "Organisch-stofgehalte bodem", "ind_os_bodem", "gg_os_bodem"),
    NumericParameterSpec("redox", "Redoxpotentiaal bodem", "ind_redox", "gg_redox", unit="mV"),
    NumericParameterSpec("carbonate", "Carbonaatgehalte bodem", "ind_carb", "gg_carb"),
)

CATEGORY_PARAMETER_LABELS: Final[dict[str, str]] = {
    "bodemtype": "Granulaire bodemsamenstelling",
    "os_bodem": "Organisch-stofklasse bodem",
    "redox": "Redoxklasse bodem",
    "carb_bodem": "Carbonaatklasse bodem",
}

ANION_LABELS: Final[dict[str, str]] = {
    "perc_bicarb": "Hoge bicarbonaatgehalten",
    "perc_cl": "Hoge chloridegehalten",
    "perc_so4": "Hoge sulfaatgehalten",
}

ANION_CLASS_BREAKS: Final[tuple[tuple[float, float, str], ...]] = (
    (-3.000001, -2.0, "Zeer sterke negatieve associatie"),
    (-2.0, -1.0, "Sterke negatieve associatie"),
    (-1.0, -0.25, "Zwakke negatieve associatie"),
    (-0.25, 0.25, "Geen duidelijke associatie"),
    (0.25, 1.0, "Zwakke positieve associatie"),
    (1.0, 2.0, "Sterke positieve associatie"),
    (2.0, 3.000001, "Zeer sterke positieve associatie"),
)

MIN_FORMAL_TREND_YEARS: Final[int]=5
MIN_LOCATIONS_PER_YEAR: Final[int]=3
MIN_ANNUAL_PROFILE_COVERAGE: Final[float]=0.50
MIN_RECURRING_LOCATION_SHARE: Final[float]=0.40
BOOTSTRAP_ITERATIONS: Final[int]=1000
BOOTSTRAP_CONFIDENCE_LEVEL: Final[float]=0.95
BOOTSTRAP_RANDOM_SEED: Final[int]=42
TREND_STABLE_SLOPE_TOLERANCE: Final[float]=1e-9


# Interactieve uitvoering
TREND_FORMAL_ANALYSIS_DEFAULT: Final[bool] = False
TREND_BOOTSTRAP_MODE_DEFAULT: Final[str] = "fast"
TREND_BOOTSTRAP_MODE_LABELS: Final[dict[str, str]] = {
    "Snel (200 herhalingen)": "fast",
    "Standaard (1.000 herhalingen)": "standard",
    "Uitgebreid (5.000 herhalingen)": "extended",
}
SOIL_DIAGNOSIS_SESSION_PREFIX: Final[str] = "soil_diagnosis_result"

# ============================================================================
# PARAMETERAFHANKELIJKE UITLEG VOOR DE GEBRUIKERSINTERFACE
# ============================================================================
# Deze teksten zijn verklarende referentiekaders. De berekende indicatiewaarde blijft
# een bio-indicatie op basis van soorten en is geen directe chemische meting.
PARAMETER_EXPLAINERS: Final[dict[str, dict[str, str]]] = {
    "salinity": {
        "title": "Hoe interpreteer ik saliniteit van de waterlaag?",
        "intro": (
            "Gebruik de saliniteitswaarde als bio-indicatieve aanwijzing. De waarde kan "
            "zowel naar zoutgehalte als naar algemene ionenrijkdom worden geduid."
        ),
        "body": """
**Indeling naar zoutgehalte**

- Zoet water: **< 16 mmol/l**
- Brak water: **> 16 mmol/l**
- Zout water: **> 850 mmol/l**

**Indeling naar ionenrijkdom**

- Zeer ionenarm water: **0 tot 2 mmol/l**
- Ionenarm water: **2 tot 4 mmol/l**
- Matig ionenrijk water: **4 tot 9 mmol/l**
- Ionenrijk water: **9 tot 15 mmol/l**
- Zeer ionenrijk water: **> 15 mmol/l**

De twee indelingen beschrijven verschillende aspecten. Een waarde kan daarom tegelijk
binnen een zoutgehalteklasse en een ionenrijkdomklasse vallen.
""",
    },
    "alkalinity": {
        "title": "Hoe interpreteer ik alkaliniteit van de waterlaag?",
        "intro": (
            "Alkaliniteit beschrijft het bufferend vermogen van de waterlaag. "
            "Hogere waarden wijzen op een groter vermogen om verzuring te neutraliseren."
        ),
        "body": """
- Ongebufferd, zuur water: **< 0,1**
- Zeer zacht water: **0,1 tot 0,5**
- Zacht water: **0,5 tot 1**
- Zacht tot matig hard water: **1 tot 2**
- Hard water: **2 tot 4**
- Zeer hard water: **> 4**
""",
    },
    "redox": {
        "title": "Hoe interpreteer ik de redoxpotentiaal van de bodem?",
        "intro": (
            "De redoxpotentiaal geeft een aanwijzing voor reducerende of oxiderende "
            "bodemomstandigheden. De waarden worden uitgedrukt in mV."
        ),
        "body": """
- Zeer reductief: **< -175 mV**
- Reductief: **-175 tot -100 mV**
- Matig reductief: **-100 tot 0 mV**
- Matig oxiderend: **0 tot 100 mV**
- Oxiderend: **> 100 mV**
""",
    },
    "carbonate": {
        "title": "Hoe interpreteer ik het carbonaatgehalte van de bodem?",
        "intro": "De carbonaatindicatie beschrijft de associatie met carbonaatarme of carbonaatrijke bodems.",
        "body": """
- Zeer carbonaatarm: **0,1 tot 1**
- Carbonaatarm: **1 tot 20**
- Matig carbonaatrijk: **20 tot 200**
- Carbonaatrijk: **200 tot 360**
- Zeer carbonaatrijk: **> 360**

Voor waarden **< 0,1** is in dit referentiekader geen afzonderlijke klasse opgegeven.
""",
    },
    "perc_bicarb": {
        "title": "Hoe interpreteer ik de associatie met hoge bicarbonaat- of carbonaatgehalten?",
        "intro": "Deze score is een soortassociatie op een schaal van -3 tot +3 en is geen gemeten concentratie.",
        "body": """
- **+3**: sterke voorkeur voor dominantie van dit specifieke anion in de waterkolom.
- **-3**: sterke voorkeur voor het tegenovergestelde, dus omstandigheden waarin dit anion niet dominant is.
- Waarden rond **0**: geen duidelijke voorkeur voor of afkeer van dominantie van dit anion.
""",
    },
    "perc_cl": {
        "title": "Hoe interpreteer ik de associatie met hoge chloridegehalten?",
        "intro": "Deze score is een soortassociatie op een schaal van -3 tot +3 en is geen gemeten chlorideconcentratie.",
        "body": """
- **+3**: sterke voorkeur voor dominantie van chloride in de waterkolom.
- **-3**: sterke voorkeur voor het tegenovergestelde, dus omstandigheden waarin chloride niet dominant is.
- Waarden rond **0**: geen duidelijke voorkeur voor of afkeer van chloridedominantie.
""",
    },
    "perc_so4": {
        "title": "Hoe interpreteer ik de associatie met hoge sulfaatgehalten?",
        "intro": "Deze score is een soortassociatie op een schaal van -3 tot +3 en is geen gemeten sulfaatconcentratie.",
        "body": """
- **+3**: sterke voorkeur voor dominantie van sulfaat in de waterkolom.
- **-3**: sterke voorkeur voor het tegenovergestelde, dus omstandigheden waarin sulfaat niet dominant is.
- Waarden rond **0**: geen duidelijke voorkeur voor of afkeer van sulfaatdominantie.
""",
    },
}

# ============================================================================
# TABELINTERPRETATIES VOOR NUMERIEKE DIAGNOSEWAARDEN
# ============================================================================
# Grenzen worden in oplopende volgorde toegepast als: waarde < bovengrens.
DIAGNOSIS_VALUE_CLASS_BREAKS: Final[dict[str, tuple[tuple[float, str], ...]]] = {
    "salinity": (
        (2.0, "Zeer ionenarm water"),
        (4.0, "Ionenarm water"),
        (9.0, "Matig ionenrijk water"),
        (15.0, "Ionenrijk water"),
        (float("inf"), "Zeer ionenrijk water"),
    ),
    "alkalinity": (
        (0.1, "Ongebufferd (zuur) water"),
        (0.5, "Zeer zacht water"),
        (1.0, "Zacht water"),
        (2.0, "Zacht tot matig hard water"),
        (4.0, "Hard water"),
        (float("inf"), "Zeer hard water"),
    ),
    "redox": (
        (-175.0, "Zeer reductief"),
        (-100.0, "Reductief"),
        (0.0, "Matig reductief"),
        (100.0, "Matig oxiderend"),
        (float("inf"), "Oxiderend"),
    ),
    "carbonate": (
        (0.1, "Zeer laag; geen afzonderlijke klasse in het referentiekader"),
        (1.0, "Zeer carbonaatarm"),
        (20.0, "Carbonaatarm"),
        (200.0, "Matig carbonaatrijk"),
        (360.0, "Carbonaatrijk"),
        (float("inf"), "Zeer carbonaatrijk"),
    ),
}

# Parameters waarvoor de bestaande categorische diagnose de numerieke waarde vervangt.
DIAGNOSIS_COMPANION_CATEGORY_KEYS: Final[dict[str, str]] = {
    "organic_matter": "os_bodem",
}

# Anionscores hebben al een categorie via ANION_CLASS_BREAKS.
DIAGNOSIS_USE_EXISTING_CATEGORY: Final[tuple[str, ...]] = (
    "perc_bicarb",
    "perc_cl",
    "perc_so4",
)
