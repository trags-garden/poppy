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


class TragsError(Exception):
    """Base for any non-2xx response from Trags.

    Auth and conflict errors subclass this so a single ``except TragsError``
    catches every server-side failure (the sync loops rely on that), while
    callers that must react differently — abort on a revoked key, skip a
    conflicting row — still catch the specific subclass first.
    """


class TragsAuthError(TragsError):
    """401 from Trags — bad, missing, or revoked API key."""


class TragsQuotaError(TragsError):
    """402 from Trags — the account's memory quota is exhausted (free-plan cap).

    Carries the server's human-facing ``detail`` message verbatim so the CLI and
    the auto-sync worker can show it (e.g. "Free plan memory limit reached.
    Upgrade to Trags Pro to remove the cap."). Distinct from a generic
    ``TragsError`` so push stops on the first 402 and surfaces the upgrade prompt
    instead of silently swallowing it as a per-item soft failure.
    """


class TragsConflictError(TragsError):
    """409 from Trags — client `updated_at` older than server's copy."""


def auth_error_message(exc: BaseException) -> str:
    """The one wording for a rejected key, shared by the CLI and the auto worker.

    Names the status itself. ``TragsAuthError`` carries only the response BODY,
    and Trags answers a revoked key with ``{"detail":"Unauthorized"}`` — no
    number anywhere — so a message built from ``str(exc)`` alone left
    `poppy sync status` showing a failure that never said 401.
    """
    return f"auth failed (401, check the Trags API key): {exc}"


class TragsTransportError(TragsError):
    """The request got no usable answer from Trags: DNS, connect, TLS, timeout.

    A ``TragsError`` so the sync loops' single ``except TragsError`` renders one
    offline line instead of letting httpx's own exception escape the CLI as a
    raw traceback. Its own class because the distinction still
    matters to ``push``: a server response means the host is alive and the next
    row is worth attempting, while a dead host is not, so only this one trips
    the consecutive-failure circuit breaker.

    Carries what the run had already done when the host went away, because the
    line the user reads has to describe THAT run. ``outcome_unknown`` marks the
    failures that cannot prove the last request was not applied, so nothing
    downstream promises a write did not land when it may well have.
    """

    def __init__(self, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.outcome_unknown = outcome_unknown
        # Filled in by the sync layer as it unwinds (see ``_note_transport_progress``).
        self.applied_pulled = 0
        self.sent_pushed = 0
        # The counts above describe a ``--dry-run``, so they are what WOULD have
        # happened, not work anything did. A simulated count reported as a
        # completed one is the same lie in the other direction.
        self.simulated = False


# Transport faults that failed BEFORE any request byte could reach the server:
# the connection never came up, so the write provably did not land. Every other
# ``RequestError`` - a read timeout, a socket dropped mid-flight, a server that
# hung up, a redirect loop - leaves the last request's fate unknowable, and the
# offline message has to say so instead of guessing.
_NEVER_REACHED_SERVER = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)


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
        resp = self._send("POST", f"{self.base_url}/api/memories", json=memory)
        self._raise_for_status(resp)
        return resp.json(), resp.status_code == 201

    def get(self, memory_id: str) -> dict | None:
        """GET /api/memories/{id}. Returns None on 404."""
        resp = self._send("GET", f"{self.base_url}/api/memories/{memory_id}")
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
        resp = self._send("GET", f"{self.base_url}/api/memories", params=params)
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

    def ping(self) -> None:
        """One cheap authenticated round trip, purely to prove the key still works.

        Used by a push that has nothing above its watermark to send: without it
        such a push contacts the server not at all and reports success, so a
        revoked key or a deleted account reads as a clean sync.

        It asks the list endpoint for a single row because that is the endpoint
        the `usr_` API key authenticates against. Trags' `/api/me` is for the
        browser session (cookie auth), so it would answer 401 for every CLI key
        and turn every idle push into a false alarm.
        """
        self.list_since(limit=1)

    def replace(self, memory_id: str, memory: dict) -> dict:
        """PUT /api/memories/{id}. Raises TragsConflictError on 409."""
        resp = self._send(
            "PUT",
            f"{self.base_url}/api/memories/{memory_id}",
            json=memory,
        )
        if resp.status_code == 409:
            raise TragsConflictError(resp.text)
        self._raise_for_status(resp)
        return resp.json()

    def delete(self, memory_id: str) -> bool:
        """DELETE /api/memories/{id}. Returns False on 404."""
        resp = self._send("DELETE", f"{self.base_url}/api/memories/{memory_id}")
        if resp.status_code == 404:
            return False
        self._raise_for_status(resp)
        return True

    # -- Internals -----------------------------------------------------------

    def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """One request, with transport faults wrapped as ``TragsTransportError``.

        ``_raise_for_status`` only sees requests that got an answer; a host that
        is down, renamed or unroutable fails in httpx before that, and those
        exceptions are not ``TragsError``, so every sync handler missed them.

        ``RequestError`` rather than its ``TransportError`` subset: a redirect
        loop or an undecodable body from a misconfigured ``trags-api-url`` is
        the same thing to a caller - no usable response - and would otherwise
        still escape as the raw traceback this exists to remove.
        """
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            raise TragsTransportError(
                f"Cannot reach Trags at {self.base_url}: {exc}",
                outcome_unknown=not isinstance(exc, _NEVER_REACHED_SERVER),
            ) from exc

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if 200 <= resp.status_code < 300:
            return
        if resp.status_code == 401:
            raise TragsAuthError(resp.text or "401 Unauthorized")
        if resp.status_code == 402:
            message = _quota_message(resp)
            if message is not None:
                raise TragsQuotaError(message)
        raise TragsError(f"{resp.status_code} {resp.text[:500]}")


def _quota_message(resp: httpx.Response) -> str | None:
    """Return the server's detail for a 402 ``quota_exceeded`` body, else None.

    The Trags server sends ``{"detail": "...", "code": "quota_exceeded"}`` when a
    free-plan account hits its memory cap. Any other 402 (or an unparseable body)
    falls through to a generic ``TragsError``.
    """
    try:
        body = resp.json()
    except ValueError:
        return None
    if isinstance(body, dict) and body.get("code") == "quota_exceeded":
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        return "Trags memory quota reached. Upgrade to Trags Pro to remove the cap."
    return None
