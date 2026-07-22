"""The provenance projection must come from ONE authority — regression guard.

Why this file exists
--------------------
``meta.query_provenance.params.fields`` tells a reader which ``fields=`` projection the run
actually requested. For that to be an audit trail rather than decoration, the value stamped into
provenance and the value the tool really pages with must come from the *same* place.

They once did not. ``_execute_single`` carried a second, hand-written projection switch whose
``else`` branch returned the literal ``"NCTId|Phase"``, so every non-timeseries / non-geographic /
non-duration aggregation stamped ``NCTId|Phase`` regardless of the field it actually paged. A
status distribution requested ``NCTId|OverallStatus|BriefTitle`` (``FIELD_SPEC["overallStatus"]``)
and reported ``NCTId|Phase``. No computed number was affected — the bug was confined to the audit
metadata — which is exactly why nothing caught it: ``scripts/verify_examples.py`` checks citation
provenance, reconciliation and count coherence, but never ``meta.query_provenance``.

The fix routed provenance through :func:`app.graph.nodes._projection`, which reads the authority
per path instead of re-deriving it. These tests pin that: they compare ``_projection`` against the
authorities themselves, so a re-introduced duplicate switch fails here even though the frozen
``examples/*.json`` (a recorded pre-fix run, deliberately left byte-for-byte as submitted — see
EXAMPLE_RUNS.md) still carry the old values.
"""

from __future__ import annotations

import typing

import pytest

from app.ctgov.fields import FIELD_SPEC
from app.ctgov.tools import _DURATION_FIELDS as DURATION_FIELDS
from app.ctgov.tools import DATE_PROJECTION
from app.graph.nodes import _projection
from app.plan.models import Plan, QueryClass


def _plan(**kw) -> Plan:
    """A minimal legal Plan; callers override only what the assertion is about."""
    base = {"query_class": "distribution", "field": "phase", "chart_type": "bar", "filters": {}}
    base.update(kw)
    return Plan(**base)


@pytest.mark.parametrize("alias", sorted(FIELD_SPEC))
def test_aggregation_projection_is_field_spec(alias: str) -> None:
    """Every FIELD_SPEC alias projects exactly what ``aggregate_by`` will page with.

    This is the assertion the original bug violated: it is not enough that the projection *looks*
    plausible, it must be the identical string the aggregation module owns.
    """
    plan = _plan(query_class="distribution", field=alias)
    assert _projection(plan) == FIELD_SPEC[alias].fields_projection


def test_status_distribution_is_not_the_old_hardcoded_phase() -> None:
    """The exact historical defect, pinned by name so it cannot return quietly."""
    projection = _projection(_plan(field="overallStatus"))
    assert projection != "NCTId|Phase"
    assert projection == FIELD_SPEC["overallStatus"].fields_projection
    # Case matters: the old switch also emitted a lowercased `overallStatus` wire token.
    assert "OverallStatus" in projection


def test_geographic_ignores_plan_field_and_uses_country() -> None:
    """``geographic`` aggregates on ``country`` regardless of ``plan.field`` (see
    ``_aggregation_field``), so its provenance must follow the field it really groups by."""
    plan = _plan(query_class="geographic", field="country")
    assert _projection(plan) == FIELD_SPEC["country"].fields_projection


def test_study_duration_uses_the_tools_authority() -> None:
    """The histogram pages a date pair, not a FIELD_SPEC row — its authority is ``tools``."""
    assert _projection(_plan(field="study_duration")) == DURATION_FIELDS


@pytest.mark.parametrize("date_field", sorted(DATE_PROJECTION))
def test_timeseries_projection_tracks_date_projection(date_field: str) -> None:
    """``tools.timeseries`` builds NCTId + the date field's wire token + BriefTitle."""
    plan = _plan(query_class="timeseries", field=None, date_field=date_field)
    assert _projection(plan) == f"NCTId|{DATE_PROJECTION[date_field]}|BriefTitle"


def test_unknown_field_degrades_to_bare_nctid_not_a_guess() -> None:
    """Unreachable in practice (the checker validates ``plan.field`` against the same alias
    table), but the degradation must be the honest minimum rather than a plausible-looking guess —
    a wrong-but-plausible projection is what made the original bug invisible."""
    assert _projection(_plan(field="not_a_real_field")) == "NCTId"


def test_every_query_class_is_covered_by_a_projection_path() -> None:
    """Coverage guard: if a seventh query class is added, this fails until its projection
    authority is decided, rather than silently inheriting a default."""
    handled = {"distribution", "timeseries", "compare", "geographic", "network", "single_value"}
    assert set(typing.get_args(QueryClass)) == handled
