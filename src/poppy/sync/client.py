"""HTTP client for the Trags memories KV endpoints.

Wraps the 5 endpoints exposed by trags-apps:
  POST   /api/memories
  GET    /api/memories?updated_since=&limit=&cursor=
  GET    /api/memories/{id}
  PUT    /api/memories/{id}
  DELETE /api/memories/{id}

Auth: ``Authorization: Bearer <api_key>`` (Trags `usr_xxx` API key).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class TragsAuthError(Exception):
    """401 from Trags — bad or missing API key."""


class TragsConflictError(Exception):
    """409 from Trags — client `updated_at` older than server's copy."""


class TragsError(Exception):
    """Any other non-2xx response."""


@dataclass(frozen=True)
class Page:
    items: list[dict]
    next_cursor: str | None


class TragsClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TragsClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Endpoints -----------------------------------------------------------

    def upsert(self, memory: dict) -> tuple[dict, bool]:
        """POST /api/memories. Returns (row, created) — True if HTTP 201."""
        resp = self._client.post(f"{self.base_url}/api/memories", json=memory)
        self._raise_for_status(resp)
        return resp.json(), resp.status_code == 201

    def get(self, memory_id: str) -> dict | None:
        """GET /api/memories/{id}. Returns None on 404."""
        resp = self._client.get(f"{self.base_url}/api/memories/{memory_id}")
        if resp.status_code == 404:
            return None
        self._raise_for_status(resp)
        return resp.json()

    def list_since(
        self,
        updated_since: str | None = None,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page:
        """GET /api/memories?updated_since=&limit=&cursor=."""
        params: dict[str, str | int] = {"limit": int(limit)}
        if updated_since:
            params["updated_since"] = updated_since
        if cursor:
            params["cursor"] = cursor
        resp = self._client.get(f"{self.base_url}/api/memories", params=params)
        self._raise_for_status(resp)
        body = resp.json()
        return Page(items=list(body.get("items") or []), next_cursor=body.get("next_cursor"))

    def iter_all_since(
        self,
        updated_since: str | None = None,
        *,
        page_size: int = 100,
    ):
        """Generator that walks the cursor pagination."""
        cursor: str | None = None
        while True:
            page = self.list_since(updated_since, limit=page_size, cursor=cursor)
            yield from page.items
            cursor = page.next_cursor
            if cursor is None:
                return

    def replace(self, memory_id: str, memory: dict) -> dict:
        """PUT /api/memories/{id}. Raises TragsConflictError on 409."""
        resp = self._client.put(
            f"{self.base_url}/api/memories/{memory_id}",
            json=memory,
        )
        if resp.status_code == 409:
            raise TragsConflictError(resp.text)
        self._raise_for_status(resp)
        return resp.json()

    def delete(self, memory_id: str) -> bool:
        """DELETE /api/memories/{id}. Returns False on 404."""
        resp = self._client.delete(f"{self.base_url}/api/memories/{memory_id}")
        if resp.status_code == 404:
            return False
        self._raise_for_status(resp)
        return True

    # -- Internals -----------------------------------------------------------

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if 200 <= resp.status_code < 300:
            return
        if resp.status_code == 401:
            raise TragsAuthError(resp.text or "401 Unauthorized")
        raise TragsError(f"{resp.status_code} {resp.text[:500]}")
