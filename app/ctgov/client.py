"""The least-privilege HTTP client (ARCHITECTURE_SPEC §A(a)/(d)) — Phase 0 stub.

Structurally encodes the security posture even before any real HTTP call exists:
base-URL-pinned to the registry host, HTTPS-forced, and GET-only by construction
(no other HTTP verb method is defined anywhere on this class — the read/write
boundary is absolute, not enforced by convention or prompt).
"""

from __future__ import annotations

import logging
import random
import time
from urllib.parse import urlparse

import httpx

from app import config
from app.ctgov.enums import BASE_URL

logger = logging.getLogger(__name__)

# The one host this whole system is allowed to call (SSRF guard, ARCHITECTURE_SPEC
# §A(d)). Matched as an EXACT ``hostname`` after parsing — never via
# ``str.startswith``/``str.endswith``, which a userinfo trick
# (``https://clinicaltrials.gov@evil.com/...`` — real host is ``evil.com``) or a
# suffix trick (``https://clinicaltrials.gov.evil.com/...``) can both defeat.
_PINNED_HOST = "clinicaltrials.gov"

# --- Retry / throttle policy (LESSON L4) — sourced from app.config so the DoS
# knobs are genuinely operator-tunable (defaults match the historical literals). -
_MAX_RETRIES = config.MAX_RETRIES  # up to N+1 attempts total; GETs are idempotent so this is safe.
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0
_MAX_PAGE_SIZE = config.PAGE_SIZE  # the API's hard cap ≤1000 (SPEC_INTERROGATION §C).

# Every transient network class worth a retry (LESSON L4/K4). ``httpx.TransportError``
# is the base of the whole transient family — timeouts (Connect/Read/Write/Pool),
# network errors (ConnectError AND ReadError/WriteError/CloseError — a mid-response
# reset is a ReadError, distinct from ReadTimeout), ProtocolError (RemoteProtocolError),
# and ProxyError — so catching the base retries them ALL instead of an
# enumeration that silently omitted ReadError/WriteError/CloseError/ProxyError.
# A non-429 4xx is a STATUS (handled below), never an exception here, so it is not
# retried. Any OTHER httpx error (a non-transport HTTPError) is redacted, not retried.
_RETRYABLE_EXCEPTIONS = (httpx.TransportError,)


class UpstreamError(RuntimeError):
    """A redacted egress failure (LESSON B4).

    The message is a fixed, generic string and ``code`` is a machine-readable
    tag for callers. The real httpx error — which embeds the full URL and query
    params (a leak vector) — is logged server-side only and never rides on this
    exception's message.
    """

    def __init__(self, code: str, message: str = "upstream request failed") -> None:
        self.code = code
        super().__init__(message)


class CTGovClient:
    """The sole egress point to clinicaltrials.gov (ARCHITECTURE_SPEC §A(a)).

    Every tool (§3.5) routes its HTTP traffic through an instance of this class.
    It is read-only by construction: ``get`` is the only network method defined —
    there is no ``post``/``put``/``delete``/``patch`` anywhere in the system.
    """

    # Monotonic timestamp of the last issued request, powering the ~``self.rps``
    # min-interval throttle. Class-level default so ``__init__`` stays verbatim;
    # the first ``_throttle`` call shadows it with a per-instance value.
    _last_request_at: float = 0.0

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = config.PER_CALL_TIMEOUT_SECONDS,  # per-call DoS bound (SEC-29)
        rps: float = config.RATE_LIMIT_RPS,  # shared politeness limiter (SEC-30)
    ) -> None:
        """Base-pin + https-force the client at construction time.

        ``base_url`` must be the real registry host — no user-supplied host,
        port, or path is ever accepted (SSRF guard, ARCHITECTURE_SPEC §A(d)).
        Parsed with ``urllib.parse.urlparse`` and checked structurally (scheme,
        exact ``hostname``, no userinfo, no unexpected port) rather than with a
        raw ``str.startswith``/``endswith``, which a userinfo trick
        (``https://clinicaltrials.gov@evil.com/...``) or a suffix trick
        (``https://clinicaltrials.gov.evil.com/...``) both defeat.
        ``timeout`` bounds a single call; ``rps`` is the shared politeness rate
        limit (~3 req/s, ``[UNVERIFIED]`` per the API brief but the safe default).
        """
        parsed = urlparse(base_url)

        if parsed.scheme != "https":
            raise ValueError(
                f"CTGovClient requires https; got scheme={parsed.scheme!r} in base_url={base_url!r}"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "CTGovClient rejects userinfo (user:pass@host) in base_url — the "
                f"real host may not be clinicaltrials.gov; got base_url={base_url!r}"
            )
        if parsed.hostname != _PINNED_HOST:
            raise ValueError(
                f"CTGovClient is base-pinned to host {_PINNED_HOST!r} (exact match, "
                f"not a prefix/suffix); got hostname={parsed.hostname!r} in base_url={base_url!r}"
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"CTGovClient rejects a malformed port in base_url={base_url!r}") from exc
        if port is not None and port != 443:
            raise ValueError(
                f"CTGovClient rejects a non-standard port; got port={port!r} in base_url={base_url!r}"
            )

        self.base_url = base_url
        self.timeout = timeout
        self.rps = rps

    def get(self, path: str, params: dict) -> dict:
        """Issue one throttled, retrying GET against ``{base_url}{path}`` with ``params``.

        - **GET-only.** The only HTTP method this client ever issues.
        - **No redirects.** ``follow_redirects=False``: a 3xx from the pinned
          host is REFUSED as an error, never followed — strictly stronger than
          "same-host redirects only" (SEC redirect rule).
        - **Throttled** to ~``self.rps`` and bound by ``self.timeout``.
        - **Retries** (max 3, exponential backoff + full jitter, honoring
          ``Retry-After`` on 429) on 429 / 5xx and on transient transport
          errors; NEVER on a non-429 4xx. GETs are idempotent.
        - **Redacted errors** (LESSON B4): on exhaustion / an unexpected status
          it raises :class:`UpstreamError` (generic message + machine ``code``);
          the real httpx error is logged server-side only.

        Returns the parsed JSON body as a ``dict``.
        """
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            for attempt in range(_MAX_RETRIES + 1):
                self._throttle()
                try:
                    response = client.get(url, params=params)
                except _RETRYABLE_EXCEPTIONS as exc:
                    if attempt >= _MAX_RETRIES:
                        logger.warning("ctgov transport failure (retries exhausted): %r", exc)
                        raise UpstreamError("upstream_unreachable") from exc
                    self._sleep_backoff(attempt)
                    continue
                except httpx.HTTPError as exc:
                    # A non-transport httpx error (never retryable, must not escape
                    # raw past this boundary — LESSON B4). Redacted, logged server-side.
                    logger.warning("ctgov unexpected httpx error: %r", exc)
                    raise UpstreamError("upstream_unreachable") from exc

                status = response.status_code

                if 200 <= status < 300:
                    try:
                        body = response.json()
                    except (ValueError, httpx.DecodingError) as exc:
                        # Non-JSON / corrupt-encoding body (DecodingError is raised
                        # by .json() on a bad gzip/brotli stream — not caught by
                        # ValueError alone).
                        logger.warning("ctgov undecodable body from %s: %r", url, exc)
                        raise UpstreamError("upstream_bad_response") from exc
                    if not isinstance(body, dict):
                        # A 200 that is a JSON array/scalar, not the documented
                        # ``{totalCount, studies, nextPageToken}`` object.
                        logger.warning("ctgov non-object JSON body from %s", url)
                        raise UpstreamError("upstream_bad_response")
                    return body

                if 300 <= status < 400:
                    # A redirect from the pinned host is refused, never followed.
                    logger.warning(
                        "ctgov refused %s redirect to %r",
                        status,
                        response.headers.get("Location"),
                    )
                    raise UpstreamError("upstream_redirect_refused")

                if status == 429 or status >= 500:
                    if attempt >= _MAX_RETRIES:
                        logger.warning(
                            "ctgov upstream %s (retries exhausted) for %s", status, url
                        )
                        raise UpstreamError(f"upstream_status_{status}")
                    retry_after = self._retry_after_seconds(response) if status == 429 else None
                    self._sleep_backoff(attempt, retry_after=retry_after)
                    continue

                # Any other 4xx is a permanent client error — do NOT retry.
                logger.warning("ctgov upstream client error %s for %s", status, url)
                raise UpstreamError(f"upstream_status_{status}")

        raise UpstreamError("upstream_request_failed")  # pragma: no cover — loop always returns/raises

    def count(self, search_params: dict) -> int:
        """Return the exact ``totalCount`` for ``search_params`` — the cheap oracle.

        One ``countTotal=true&pageSize=1`` call; ``fields=NCTId`` keeps the
        (discarded) single record tiny. Every aggregation reconciles against this.
        """
        params = {**search_params, "countTotal": "true", "pageSize": 1, "fields": "NCTId"}
        response = self.get("/studies", params)
        total = response.get("totalCount")
        if not isinstance(total, int) or isinstance(total, bool):
            # A 200 dict without a valid integer totalCount is a malformed oracle;
            # never return None/str (it would TypeError at the budget gate) — the
            # count primitive stays total (LESSON K5).
            logger.warning("ctgov missing/invalid totalCount")
            raise UpstreamError("upstream_bad_response")
        return total

    def iter_studies(
        self,
        search_params: dict,
        *,
        fields: str,
        page_size: int = config.PAGE_SIZE,
        max_pages: int = config.PAGE_BUDGET_PAGES,  # the 20-page budget (§B.4), operator-tunable
    ) -> tuple[list[dict], bool]:
        """Cursor-page ``/studies`` under a page budget.

        ``page_size`` is clamped to the API's hard cap (``<=1000``). Follows
        ``nextPageToken`` until it is absent (complete) or ``max_pages`` is
        reached (truncated). The SAME ``search_params`` must be used here and in
        :meth:`count` or reconciliation breaks (one population, G-23).

        Returns ``(records, truncated)`` where ``truncated`` is ``True`` iff the
        walk stopped on ``max_pages`` with a ``nextPageToken`` still pending.
        """
        page_size = max(1, min(page_size, _MAX_PAGE_SIZE))  # floor at 1 (no pageSize=0/-n)
        records: list[dict] = []
        page_token: str | None = None
        pages_read = 0
        while pages_read < max_pages:
            params = {**search_params, "pageSize": page_size, "fields": fields}
            if page_token:
                params["pageToken"] = page_token
            response = self.get("/studies", params)
            studies = response.get("studies")
            # Validate BOTH the container and each element: ``studies`` must be a list
            # AND every record a dict — a JSON ``null``/scalar element (seen in the
            # wild) would otherwise reach a ``key_fn`` and crash the aggregation (K1/K5).
            if isinstance(studies, list):
                records.extend(record for record in studies if isinstance(record, dict))
            pages_read += 1
            page_token = response.get("nextPageToken")
            if not page_token:  # absent OR empty-string cursor → walk complete (K5)
                break
        # Broke on a falsy token → complete; stopped on the budget with a real
        # token still pending → truncated.
        truncated = bool(page_token)
        return records, truncated

    # --- internals ----------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep just enough to hold issue-rate at ~``self.rps`` (min-interval gate)."""
        if self.rps <= 0:
            return
        min_interval = 1.0 / self.rps
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _sleep_backoff(self, attempt: int, *, retry_after: float | None = None) -> None:
        """Back off before a retry: honor ``Retry-After``, else full-jitter exp backoff."""
        if retry_after is not None:
            time.sleep(min(retry_after, _BACKOFF_CAP_S))
            return
        ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
        time.sleep(random.uniform(0.0, ceiling))  # full jitter over [0, ceiling]

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """Parse a numeric ``Retry-After`` (seconds). HTTP-date form → ``None`` (use backoff)."""
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None
