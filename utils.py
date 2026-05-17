from __future__ import annotations

"""Package-brede utility re-exports voor de waterplanten-app."""

from waterplanten_app.core.data_access import *
from waterplanten_app.core.taxonomy import *
from waterplanten_app.core.chemistry import *
from waterplanten_app.core.maps import *
from waterplanten_app.core.diagnostics import *

# Optionele legacy helpers die nog niet gemigreerd zijn.
# Dit blijft package-only: de import blijft binnen waterplanten_app.core.
try:
    from waterplanten_app.core.legacy_utils_unmigrated import *
except Exception:
    pass

__all__ = [name for name in globals() if not name.startswith("_")]