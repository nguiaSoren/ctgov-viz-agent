"""Essie query-language injection defense for ``query.<area>`` values (SEC-24 / G-31).

``build_search_params`` used to place user entity values (drug/condition/sponsor/
country/term) UN-encoded into ``query.<area>``, relying only on httpx URL-encoding.
But CT.gov FULLY PARSES the Essie query language on those values after url-decode,
so a raw value could smuggle an operator (``cancer OR HIV`` → a union, live 131313
vs the literal phrase's 2) or a cross-field selector (``AREA[Phase]PHASE1`` → 65069,
a whole different population). ``neutralize_query_value`` closes that hole: a clean
value passes through for full recall; a dirty one is wrapped as an inert Essie
``StringLiteral``.

Three layers, matching the project's live-test discipline (test_aggregate_live.py):

* Unit (offline) — the neutralization rule itself: clean unchanged, dirty quoted,
  substrings (``ANDROGEN``/``Organon``) NOT triggered, internal quote escaped, TOTAL.
* Integration (offline) — the rule is actually applied at the ``build_search_params``
  boundary (an operator value is neutralized, a clean value is untouched).
* Live (skip-on-transport-error) — the neutralized ``"cancer OR HIV"`` yields a
  DIFFERENT (smaller/literal) totalCount than the raw operator union, i.e. the
  neutralization genuinely changed the query semantics. Skips (never fails) on any
  transport-level ``UpstreamError`` so an offline / rate-limited run stays green.
"""

from __future__ import annotations

import pytest

from app.ctgov.client import CTGovClient, UpstreamError
from app.ctgov.params import build_search_params, neutralize_query_value

# --- unit: the neutralization rule (offline) --------------------------------


def test_clean_value_passes_through_unchanged() -> None:
    # The 99% case: a plain entity phrase has no Essie metacharacter / operator, so
    # it is returned verbatim — full recall, no quoting, no recall cost.
    for clean in [
        "pancreatic cancer",
        "pembrolizumab",
        "Crohn's disease",  # an apostrophe is not an Essie metacharacter
        "Non-small cell lung cancer",  # a hyphen is not an Essie metacharacter
        "Keytruda",
    ]:
        assert neutralize_query_value(clean) == clean


def test_lowercase_operator_word_is_a_literal_term_not_an_operator() -> None:
    # Essie is CASE-SENSITIVE: only UPPERCASE ``OR`` is the union operator; the
    # lowercase/titlecase forms are literal terms, so they stay unchanged.
    assert neutralize_query_value("headache or migraine") == "headache or migraine"
    assert neutralize_query_value("headache Or migraine") == "headache Or migraine"


def test_uppercase_operator_keywords_are_quoted() -> None:
    # A standalone UPPERCASE operator keyword makes the value dirty → wrapped as a
    # StringLiteral (starts and ends with a quote), so the parser can't act on it.
    for dirty in ["cancer OR HIV", "cancer AND HIV", "cancer NOT HIV"]:
        out = neutralize_query_value(dirty)
        assert out.startswith('"') and out.endswith('"')
        assert out != dirty


def test_metacharacters_and_cross_field_selector_are_quoted() -> None:
    # The real injection: brackets/parens/quotes + a cross-field AREA[...] selector.
    for dirty in ["AREA[Phase]PHASE1", "a (b)", "cancer]", 'say "hi"', "[x]"]:
        out = neutralize_query_value(dirty)
        assert out.startswith('"') and out.endswith('"'), f"{dirty!r} → {out!r} not wrapped"


def test_substring_keywords_do_not_trigger() -> None:
    # Word-boundary + case-sensitive detection: a keyword that is only a SUBSTRING of
    # a real term (not a standalone token) must NOT be treated as an operator.
    for clean in ["ANDROGEN", "Organon", "MINOXIDIL", "MAXILLA", "NOTCH", "COVERAGES"]:
        assert neutralize_query_value(clean) == clean


def test_internal_quote_is_escaped_not_stripped() -> None:
    # Live-verified: the ``\"`` StringLiteral escape form is API-accepted, so an
    # internal quote is ESCAPED (backslash-quote) inside the wrap, never dropped.
    out = neutralize_query_value('a"b')
    assert out == '"a\\"b"'
    # a backslash is escaped first so the added quote-escapes can't be doubled up
    assert neutralize_query_value('a\\"b') == '"a\\\\\\"b"'


def test_total_never_raises_on_odd_input() -> None:
    # TOTAL by contract: non-str coerced (None → ""), empty → "" — never raises.
    assert neutralize_query_value("") == ""
    assert neutralize_query_value(None) == ""  # type: ignore[arg-type]
    assert neutralize_query_value(12345) == "12345"  # type: ignore[arg-type]


# --- integration: the rule is applied at the build boundary (offline) -------


def test_build_search_params_neutralizes_operator_value() -> None:
    params = build_search_params({"cond": "cancer OR HIV"}, {})
    value = params["query.cond"]
    assert value != "cancer OR HIV"  # NOT the raw operator union
    assert value == '"cancer OR HIV"'  # the neutralized StringLiteral form


def test_build_search_params_leaves_clean_value_untouched() -> None:
    params = build_search_params({"cond": "pancreatic cancer"}, {})
    assert params["query.cond"] == "pancreatic cancer"


def test_build_search_params_neutralizes_every_area() -> None:
    # Every user-free-text area routes through neutralization, not just ``cond``.
    params = build_search_params(
        {"term": "AREA[Phase]PHASE1", "intr": "a OR b", "spons": "Merck", "locn": "France"}, {}
    )
    assert params["query.term"] == '"AREA[Phase]PHASE1"'
    assert params["query.intr"] == '"a OR b"'
    assert params["query.spons"] == "Merck"  # clean → unchanged
    assert params["query.locn"] == "France"  # clean → unchanged


# --- live: neutralization actually changes the query semantics --------------


def _skip_if_transient(exc: Exception) -> None:
    """Skip (never fail) on any transport-level ``UpstreamError`` (offline, retry-
    exhausted rate-limit, redirect refusal, bad JSON). Mirrors test_aggregate_live's
    skip: the count call is a pure transport op here, so an ``UpstreamError`` is
    always a network condition, never a logic bug. Anything else re-raises."""
    if isinstance(exc, UpstreamError):
        pytest.skip(f"clinicaltrials.gov transport error ({exc.code}) -- live test skipped")
    raise exc


def test_live_neutralization_changes_the_union_semantics() -> None:
    """The neutralized ``"cancer OR HIV"`` (a literal phrase) matches FAR fewer trials
    than the raw ``cancer OR HIV`` operator union — proof the fix defused the operator,
    not merely re-encoded the string. Live-observed 2026-07-16: 131313 vs 2."""
    client = CTGovClient()
    raw_value = "cancer OR HIV"
    neutralized = neutralize_query_value(raw_value)
    assert neutralized == '"cancer OR HIV"'  # sanity: it IS the quoted form
    try:
        raw_count = client.count({"query.cond": raw_value})
        neutralized_count = client.count({"query.cond": neutralized})
    except Exception as exc:  # noqa: BLE001 -- narrowed to transient-skip below
        _skip_if_transient(exc)
        raise
    assert neutralized_count < raw_count, (
        f"neutralized literal count ({neutralized_count}) is not smaller than the raw "
        f"operator-union count ({raw_count}) -- neutralization did not change semantics"
    )
