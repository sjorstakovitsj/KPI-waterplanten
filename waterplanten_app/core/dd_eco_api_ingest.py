from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import re

import pandas as pd

from waterplanten_app.config.settings import DD_API_MEAS_PARQUET, DD_API_REFERENCES_PARQUET
from waterplanten_app.core.dd_eco_api_client import fetch_observations, fetch_references

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None

RAW_CONTRACT_COLUMNS = [
    'MetingDatumTijd',
    'MeetObject',
    'Projecten',
    'GeografieDatum',
    'GeografieVorm',
    'CollectieReferentie',
    'Parameter',
    'Grootheid',
    'WaardeGemeten',
    'EenheidGemeten',
]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pq is not None and pa is not None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path.as_posix(), compression='zstd')
    else:
        df.to_parquet(path.as_posix(), index=False)


def _flatten(record: Any, prefix: str = '') -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(record, dict):
        for key, value in record.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten(value, name))
            elif isinstance(value, list):
                flat[name] = value
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(_flatten(item, f'{name}[{idx}]'))
                    else:
                        flat[f'{name}[{idx}]'] = item
            else:
                flat[name] = value
    return flat


def _pick(flat: Dict[str, Any], candidates: Iterable[str], default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in flat and flat[candidate] not in (None, ''):
            return flat[candidate]
    return default


_NUM_RE = re.compile(r'(-?\d+(?:[\.,]\d+)?)')


def _extract_lon_lat_from_any(value: Any) -> tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None, None
    text = str(value).strip()
    if not text:
        return None, None
    nums = _NUM_RE.findall(text)
    if len(nums) < 2:
        return None, None
    try:
        a = float(nums[0].replace(',', '.'))
        b = float(nums[1].replace(',', '.'))
    except Exception:
        return None, None
    if 3.0 <= a <= 8.5 and 50.0 <= b <= 54.8:
        return a, b
    if 50.0 <= a <= 54.8 and 3.0 <= b <= 8.5:
        return b, a
    return None, None


def _geometry_to_epsg_wkt(flat: Dict[str, Any]) -> tuple[str, str]:
    epsg = str(_pick(flat, [
        'geometry.crs.properties.name', 'geometry.crs.name', 'crs.properties.name', 'crs.name',
        'coordinateReferenceSystem', 'coordinateSystem', 'epsg', 'EPSG',
    ], default='EPSG:4258') or 'EPSG:4258')
    wkt = _pick(flat, ['geometry.wkt', 'shape.wkt', 'location.wkt', 'geometryWkt', 'wkt'])
    if wkt:
        return epsg, str(wkt)
    lon, lat = None, None
    for key in ['geometry.coordinates', 'shape.coordinates', 'location.coordinates']:
        if key in flat:
            lon, lat = _extract_lon_lat_from_any(flat[key])
            if lon is not None and lat is not None:
                break
    if lon is None or lat is None:
        lon = _pick(flat, ['longitude', 'lon', 'location.longitude', 'geometry.longitude'])
        lat = _pick(flat, ['latitude', 'lat', 'location.latitude', 'geometry.latitude'])
        lon, lat = _extract_lon_lat_from_any(f'{lon} {lat}')
    if lon is not None and lat is not None:
        return epsg, f'POINT ({lon} {lat})'
    return epsg, ''


def _normalize_grootheid(raw_value: Any, parameter_hint: Any = None) -> str:
    text = str(raw_value or '').strip().lower()
    hint = str(parameter_hint or '').strip().lower()
    combo = f'{text} {hint}'.strip()
    if any(x in combo for x in ['bedkg', 'bedekking', 'coverage']):
        return 'BEDKG'
    if any(x in combo for x in ['aanwzhd', 'aanwezig', 'presence']):
        return 'AANWZHD'
    if any(x in combo for x in ['diepte', 'depth']):
        return 'DIEPTE'
    if any(x in combo for x in ['zicht', 'doorzicht', 'secchi', 'transparen']):
        return 'ZICHT'
    if any(x in combo for x in ['watptn', 'totale bedekking', 'total cover']):
        return 'BEDKG'
    return str(raw_value or parameter_hint or '').strip() or 'BEDKG'


def _observation_to_raw_contract(observation: dict) -> dict:
    flat = _flatten(observation)
    epsg, wkt = _geometry_to_epsg_wkt(flat)
    meting_datum = _pick(flat, [
        'collectiondate', 'collectionDate', 'collectionstartdate', 'collectionStartDate',
        'phenomenonTime', 'resultTime', 'date', 'timestamp', 'observationTime',
    ], default='')
    meetobject = _pick(flat, [
        'locationcode', 'location.code', 'location.id', 'location.name',
        'sitecode', 'site.code', 'site.id', 'gebied', 'area',
    ], default='')
    project = _pick(flat, [
        'project', 'projectcode', 'project.name', 'programme', 'program',
        'measurementpurpose', 'campaign', 'nivo', 'compartment',
    ], default='DD-ECO API')
    collectie_ref = _pick(flat, [
        'collectionreference', 'collectionReference', 'observationid', 'observationId',
        'id', '@id', 'eventid', 'eventId', 'sampleid', 'sampleId',
    ], default='')
    parameter = _pick(flat, [
        'taxonname', 'taxon.name', 'parametercode', 'parameter.code', 'parametername',
        'parameter.name', 'observedproperty.name', 'observedproperty.code',
        'property.name', 'property.code', 'label',
    ], default='')
    grootheid_raw = _pick(flat, [
        'parametertype', 'parameter.type', 'quantityname', 'quantity.name',
        'observedproperty.category', 'observedproperty.type', 'type', 'type_n', 'grootheid',
    ], default='')
    waarde = _pick(flat, [
        'measuredvalue', 'measuredValue', 'calculatedvalue', 'calculatedValue',
        'result', 'result.value', 'value',
    ], default='')
    eenheid = _pick(flat, [
        'calculatedunit', 'calculatedUnit', 'unit', 'unitofmeasure', 'unitOfMeasure', 'result.unit',
    ], default='')
    grootheid = _normalize_grootheid(grootheid_raw, parameter_hint=parameter)
    return {
        'MetingDatumTijd': meting_datum,
        'MeetObject': meetobject,
        'Projecten': project,
        'GeografieDatum': epsg,
        'GeografieVorm': wkt,
        'CollectieReferentie': collectie_ref,
        'Parameter': parameter,
        'Grootheid': grootheid,
        'WaardeGemeten': waarde,
        'EenheidGemeten': eenheid,
    }


def observations_to_raw_contract_frame(observations: List[dict]) -> pd.DataFrame:
    rows = [_observation_to_raw_contract(obs) for obs in (observations or [])]
    if not rows:
        return pd.DataFrame(columns=RAW_CONTRACT_COLUMNS)
    df = pd.DataFrame(rows)
    for col in RAW_CONTRACT_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    return df[RAW_CONTRACT_COLUMNS]


def fetch_raw_measurements_frame() -> pd.DataFrame:
    return observations_to_raw_contract_frame(fetch_observations())


def fetch_references_frame() -> pd.DataFrame:
    refs = fetch_references()
    if not refs:
        return pd.DataFrame()
    return pd.DataFrame([_flatten(r) for r in refs])


def ensure_dd_eco_measurements_parquet(out_path: Path | None = None, refs_path: Path | None = None) -> Path:
    out_path = out_path or DD_API_MEAS_PARQUET
    refs_path = refs_path or DD_API_REFERENCES_PARQUET
    df = fetch_raw_measurements_frame()
    if df.empty:
        raise RuntimeError('DD-ECO API leverde geen observation-records op voor de ruwe contractoutput.')
    _write_parquet(df, out_path)
    try:
        refs_df = fetch_references_frame()
        if not refs_df.empty:
            _write_parquet(refs_df, refs_path)
    except Exception:
        pass
    return out_path


__all__ = [
    'RAW_CONTRACT_COLUMNS',
    'fetch_raw_measurements_frame',
    'fetch_references_frame',
    'ensure_dd_eco_measurements_parquet',
    'observations_to_raw_contract_frame',
]
