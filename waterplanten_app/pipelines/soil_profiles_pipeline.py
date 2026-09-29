"""Efficiënte pipeline van soort_randvoorwaarden.csv naar gevalideerd Parquet."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from waterplanten_app.config.settings import (
    SOIL_PROFILES_METADATA,
    SOIL_PROFILES_PARQUET,
    SOIL_PROFILES_PIPELINE_VERSION,
    SOIL_PROFILES_SOURCE,
)
from waterplanten_app.core.duckdb_runtime import (
    _path_fingerprint,
    _write_parquet_from_pandas,
)
from waterplanten_app.core.soil_profile_validation import (
    SoilProfileValidationReport,
    load_and_validate_soil_profiles,
)


def _read_metadata(metadata_path: Path) -> dict[str, Any] | None:
    """Lees kleine pipeline-metadata; een onleesbaar bestand geldt als cachemiss."""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _metadata_is_current(
    source_path: Path,
    parquet_path: Path,
    metadata_path: Path,
) -> bool:
    """Controleer actualiteit zonder CSV of Parquet te openen."""
    if not source_path.exists() or not parquet_path.exists() or not metadata_path.exists():
        return False

    payload = _read_metadata(metadata_path)
    if payload is None:
        return False

    source_fingerprint = list(_path_fingerprint(source_path))
    parquet_fingerprint = list(_path_fingerprint(parquet_path))
    return (
        payload.get("pipeline_version") == SOIL_PROFILES_PIPELINE_VERSION
        and payload.get("source_fingerprint") == source_fingerprint
        and payload.get("parquet_fingerprint") == parquet_fingerprint
    )


def _metadata_payload(
    source_path: Path,
    parquet_path: Path,
    report: SoilProfileValidationReport,
) -> dict[str, Any]:
    return {
        "pipeline_version": SOIL_PROFILES_PIPELINE_VERSION,
        "source": str(source_path.resolve()),
        "source_fingerprint": list(_path_fingerprint(source_path)),
        "parquet_fingerprint": list(_path_fingerprint(parquet_path)),
        "row_count": int(report.row_count),
        "profile_count": int(report.profile_count),
        "warning_count": int(report.warning_count),
        "warnings": list(report.warnings),
    }


def _write_artifacts(
    frame,
    report: SoilProfileValidationReport,
    source_path: Path,
    parquet_path: Path,
    metadata_path: Path,
) -> Path:
    """Schrijf Parquet en metadata atomair vanuit reeds gevalideerde data."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_parquet = parquet_path.with_name(parquet_path.name + ".tmp")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")

    try:
        _write_parquet_from_pandas(frame, temporary_parquet)
        temporary_parquet.replace(parquet_path)

        payload = _metadata_payload(source_path, parquet_path, report)
        temporary_metadata.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_parquet.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)

    # Alleen na een echte herbouw moeten bestaande DuckDB-views opnieuw worden
    # geregistreerd. De lokale import voorkomt een circulaire module-import.
    try:
        from waterplanten_app.pipelines.gold_views import clear_gold_view_state

        clear_gold_view_state()
    except (ImportError, AttributeError):
        pass

    return parquet_path


def ensure_soil_profiles_parquet(
    source_path: Path = SOIL_PROFILES_SOURCE,
    parquet_path: Path = SOIL_PROFILES_PARQUET,
    metadata_path: Path = SOIL_PROFILES_METADATA,
    force: bool = False,
) -> Path:
    """Retourneer direct de cache of bouw deze één keer opnieuw op."""
    source_path = Path(source_path)
    parquet_path = Path(parquet_path)
    metadata_path = Path(metadata_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Bronbestand niet gevonden: {source_path}")

    if not force and _metadata_is_current(source_path, parquet_path, metadata_path):
        return parquet_path

    frame, report = load_and_validate_soil_profiles(source_path)
    return _write_artifacts(
        frame=frame,
        report=report,
        source_path=source_path,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )


def build_soil_profiles(
    source_path: Path = SOIL_PROFILES_SOURCE,
    parquet_path: Path = SOIL_PROFILES_PARQUET,
    metadata_path: Path = SOIL_PROFILES_METADATA,
    force: bool = True,
) -> tuple[Path, SoilProfileValidationReport]:
    """Expliciete build zonder dubbele CSV-inlezing of dubbele validatie."""
    source_path = Path(source_path)
    parquet_path = Path(parquet_path)
    metadata_path = Path(metadata_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Bronbestand niet gevonden: {source_path}")

    if not force and _metadata_is_current(source_path, parquet_path, metadata_path):
        payload = _read_metadata(metadata_path) or {}
        report = SoilProfileValidationReport(
            row_count=int(payload.get("row_count", 0)),
            profile_count=int(payload.get("profile_count", 0)),
            warning_count=int(payload.get("warning_count", 0)),
            warnings=tuple(payload.get("warnings", ())),
        )
        return parquet_path, report

    frame, report = load_and_validate_soil_profiles(source_path)
    path = _write_artifacts(
        frame=frame,
        report=report,
        source_path=source_path,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )
    return path, report
