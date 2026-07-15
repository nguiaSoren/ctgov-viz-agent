"""NCT-id format guard against path injection (G-8 / R-20 / SEC-25 / C-45).

An NCT id (e.g. ``"NCT01234567"``) is the ONE token that would ever be
interpolated into a URL *path* (``GET /studies/{nctId}``) rather than a query
*value* slot. Every other user- or record-derived string reaches CT.gov as a
query-parameter value, where httpx percent-encodes it; a path segment is not
encoded the same way, so an unvalidated id is a path-traversal / path-injection
vector (``NCT../../something``) the moment it is interpolated.

**No user-controlled nctId reaches a path TODAY** — nctIds are read out of
CT.gov *records* (the citation / drill-down flow), not accepted from request
input, and ``get_trial`` is still a documented ``NotImplementedError`` stub in
``app.ctgov.tools``. This module is therefore a real, tested, FORWARD-COMPATIBLE
guard, not dead defensive code: :func:`validate_nct_id` is the mandatory gate
that ``get_trial`` MUST call before it interpolates an id into
``/studies/{nctId}``. Wiring ``get_trial`` is out of scope here — this only
provides the gate it will call.

Everything here is pure and total: no id is ever echoed verbatim into an error
(a rejected value is shown only as a short, truncated repr), and no input shape
raises — malformed ids fail as a clean ``bool``/``ValueError``, never a crash.
"""

from __future__ import annotations

import re

# Exact wire format: the literal "NCT" followed by exactly 8 ASCII digits. Nothing
# else is a valid CT.gov id. Anchored, and matched with ``fullmatch`` so the
# ``$``-before-trailing-newline gotcha ("NCT01234567\n") cannot slip through.
# ``[0-9]`` (NOT ``\d``) on purpose: ``\d`` matches the whole Unicode decimal-digit
# category, so "NCT0123456٩" (7 ASCII + one Arabic-Indic 9) or a fullwidth digit
# would satisfy ``\d{8}`` yet NFKC-fold to a DIFFERENT real id — a validate-vs-
# normalize divergence a downstream canon/proxy could exploit. ASCII-only closes it.
NCT_ID_RE = re.compile(r"^NCT[0-9]{8}$")

# A rejected value is never echoed in full — an attacker-supplied id could be a
# multi-kilobyte traversal payload. We surface only a short, truncated repr so
# logs/messages stay bounded and safe to render.
_MAX_REPR = 32


def _redacted(value: object) -> str:
    """Bounded, safe repr of a rejected value — never an unbounded echo.

    Returns ``repr(value)`` truncated to :data:`_MAX_REPR` characters so a large
    injection payload cannot ride into an error message or log line.
    """
    text = repr(value)
    if len(text) > _MAX_REPR:
        return text[:_MAX_REPR] + "...<truncated>"
    return text


def is_valid_nct_id(value: object) -> bool:
    """Return ``True`` iff ``value`` is a ``str`` matching :data:`NCT_ID_RE`.

    TOTAL and non-raising for any input. ``False`` for ``None``, a non-``str``
    (e.g. an ``int``), the wrong digit count, a lowercase ``"nct"`` prefix, a
    non-digit tail character, surrounding whitespace, or the empty string.
    """
    return isinstance(value, str) and NCT_ID_RE.fullmatch(value) is not None


def validate_nct_id(value: object) -> str:
    """Return the validated NCT id, or raise ``ValueError`` if it is malformed.

    This is the guard a future ``get_trial`` MUST call before interpolating an
    id into the ``/studies/{nctId}`` path. On rejection the ``ValueError``
    message describes the expected format and includes only a short, truncated
    repr of the offending value (see :func:`_redacted`) — never the raw payload
    verbatim.
    """
    if is_valid_nct_id(value):
        # is_valid_nct_id has confirmed value is a str matching NCT_ID_RE.
        return value  # type: ignore[return-value]
    raise ValueError(
        "validate_nct_id: not a well-formed NCT id "
        f"(expected 'NCT' followed by exactly 8 digits), got {_redacted(value)}"
    )
