from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

from waterplanten_app.config.settings import (
    DD_API_ACCEPT_CRS,
    DD_API_BASE_URL,
    DD_API_DEFAULT_EXPAND,
    DD_API_DEFAULT_FILTER,
    DD_API_DEFAULT_SELECT,
    DD_API_MAX_PAGES,
    DD_API_OBSERVATIONS_ENDPOINT,
    DD_API_PAGE_SIZE,
    DD_API_REFERENCES_ENDPOINT,
    DD_API_TIMEOUT_SECONDS,
)


def _base_headers() -> Dict[str, str]:
    headers = {'Accept': 'application/json'}
    if DD_API_ACCEPT_CRS:
        headers['Accept-Crs'] = str(DD_API_ACCEPT_CRS)
    return headers


def _build_url(endpoint: str) -> str:
    base = str(DD_API_BASE_URL).rstrip('/')
    ep = str(endpoint or '').strip()
    if not ep.startswith('/'):
        ep = '/' + ep
    return base + ep


def _paged_get(url: str, params: Optional[Dict[str, Any]] = None) -> List[dict]:
    items: List[dict] = []
    page = 0
    next_url: Optional[str] = url
    next_params: Optional[Dict[str, Any]] = dict(params or {})
    with requests.Session() as session:
        session.headers.update(_base_headers())
        while next_url and page < int(DD_API_MAX_PAGES):
            response = session.get(next_url, params=next_params, timeout=int(DD_API_TIMEOUT_SECONDS))
            response.raise_for_status()
            payload = response.json() if response.content else {}
            values = payload.get('value') if isinstance(payload, dict) else None
            if isinstance(values, list):
                items.extend(values)
            elif isinstance(payload, list):
                items.extend(payload)
            next_url = payload.get('@odata.nextLink') if isinstance(payload, dict) else None
            next_params = None
            page += 1
    return items


def fetch_observations(
    filter_expr: Optional[str] = None,
    select: Optional[str] = None,
    expand: Optional[str] = None,
    top: Optional[int] = None,
    orderby: Optional[str] = None,
) -> List[dict]:
    params: Dict[str, Any] = {'$top': top or int(DD_API_PAGE_SIZE)}
    if filter_expr or DD_API_DEFAULT_FILTER:
        params['$filter'] = filter_expr or DD_API_DEFAULT_FILTER
    if select or DD_API_DEFAULT_SELECT:
        params['$select'] = select or DD_API_DEFAULT_SELECT
    if expand or DD_API_DEFAULT_EXPAND:
        params['$expand'] = expand or DD_API_DEFAULT_EXPAND
    if orderby:
        params['$orderby'] = orderby
    return _paged_get(_build_url(DD_API_OBSERVATIONS_ENDPOINT), params=params)


def fetch_references(
    filter_expr: Optional[str] = None,
    select: Optional[str] = None,
    expand: Optional[str] = None,
    top: Optional[int] = None,
    orderby: Optional[str] = None,
) -> List[dict]:
    params: Dict[str, Any] = {'$top': top or int(DD_API_PAGE_SIZE)}
    if filter_expr:
        params['$filter'] = filter_expr
    if select:
        params['$select'] = select
    if expand:
        params['$expand'] = expand
    if orderby:
        params['$orderby'] = orderby
    return _paged_get(_build_url(DD_API_REFERENCES_ENDPOINT), params=params)


__all__ = ['fetch_observations', 'fetch_references']
