from __future__ import annotations

"""Bron- en mappingcompatibiliteit voor oudere imports.

Deze module blijft bewust dun: alle echte configuratie zit in `settings.py` en
`mappings.py`. `sources.py` bestaat alleen als overgangslaag voor bestaande imports.
"""

from waterplanten_app.config.settings import *
from waterplanten_app.config.mappings import *
