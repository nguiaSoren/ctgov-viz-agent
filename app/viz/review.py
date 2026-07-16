"""Output-Reviewer deterministic pre-checks (ARCHITECTURE_SPEC §3.8) -- Wave 3.

The *code* half of the Output Reviewer: a total, never-raising checker that runs
BEFORE the LLM half in ``app.graph.nodes.review_output``. It proves four things
about an already-built ``ok`` visualization spec, using only computed data (it
introduces no number, authors no excerpt):

1. **excerpt provenance** -- every ``citation.matched_value`` is an element-precise quote
   of its own ``citation.value`` (the field value the core string-extracted from
   the record); fabricated / broken excerpts are a hard fail.
2. **reconciliation** (G-26) -- the **distinct-nctId** count (``distinct_trials``,
   the CC-16 anchor for BOTH modes) reconciles to the API's exact ``countTotal``;
   an ok spec with no oracle is a hard fail, a small live drift is disclosed, a
   large one is a hard fail (G-41a).
3. **partial iff truncated** -- ``meta.partial`` is present exactly when the
   aggregation actually truncated.
4. **cited-or-derived** (G-35) -- every datum carries >=1 citation, is a derived
   value, or is a legitimate zero-count bucket.

A hard fail becomes a redacted ``status:"error"`` envelope upstream; a within-
tolerance reconciliation drift becomes a precise ``meta.notes`` disclosure. The
checker is **total**: it returns a ``PrecheckResult``, it never raises (LESSON
B2 -- a checker that can crash is not a checker).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.api.schemas import Citation, VisualizeResponse
from app.ctgov.citations import is_substring_at

_DIGIT_RUN = re.compile(r"\d+")


def computed_numbers(spec: VisualizeResponse) -> set[str]:
    """Every maximal digit-run present in the spec EXCEPT ``meta.notes`` — the set a caveat's
    digits must be drawn from (ARCHITECTURE_SPEC §1: ``meta.notes`` is code-generated OR run
    through a deterministic post-check that rejects any digit not present in the computed data).

    Computed from a copy with ``notes`` cleared, so a previously-appended note can't launder a
    number, and from the whole envelope (data cells, ``count_basis``, ``partial``, the scalar
    ``answer``, ``source_ids``) — i.e. every number the deterministic engine actually produced."""
    probe = spec.model_copy(update={"meta": spec.meta.model_copy(update={"notes": []})})
    return set(_DIGIT_RUN.findall(probe.model_dump_json()))


def note_number_safe(note: str, allowed: set[str]) -> bool:
    """True iff every digit-run in ``note`` appears among ``allowed`` (the computed numbers).

    The deterministic §1 post-check on an LLM-authored ``meta.notes`` entry (the Output Reviewer's
    ``flag`` reason): a fabricated count — a digit-run absent from the computed data — makes the
    note unsafe, so the caller drops it for a fixed code-owned caveat. A digit-free note is always
    safe. Digit-run granularity (not per-character) is the meaningful reading: it rejects a
    fabricated number like ``99999`` while a legitimate reference to a real datum value passes."""
    return all(run in allowed for run in _DIGIT_RUN.findall(note))


@dataclass(frozen=True)
class PrecheckResult:
    """The deterministic pre-check verdict.

    ``reason`` carries the machine error code on a hard fail (``citation_invalid``
    / ``reconciliation_failed`` / ``partial_inconsistent`` / ``uncited_datum``) so
    the caller can stamp it straight onto an ``ErrorObj.code``. ``disclosure`` is a
    precise, human-readable note for ``meta.notes`` when a within-tolerance
    reconciliation drift is accepted (never set alongside a hard fail).
    """

    ok: bool
    hard_fail: bool
    reason: str | None = None
    disclosure: str | None = None


def _excerpt_in_value(excerpt: str, value: object) -> bool:
    """Is ``excerpt`` a legitimate quote of ``value`` (the record's own field value
    at ``citation.field_path``, which the citation core string-extracted — never
    LLM-authored)?

    Element-precise, matching ``citations.is_substring_at`` (LESSON K2 — the old
    loose ``excerpt in str(list)`` accepted false positives):

    * An **empty** excerpt is an *absence* citation (e.g. the MISSING phase bucket:
      no field value to quote) — valid ONLY when ``value`` is itself absent/empty;
      an empty excerpt against a PRESENT value proves nothing (would be a universal
      pass) → False.
    * A **list** value (token array, e.g. ``["PHASE1","PHASE2"]``): the excerpt must
      EQUAL one element — ``"PHASE1"`` is not a quote of a ``["PHASE10"]`` trial,
      and the stringified-list repr punctuation (``"', '"``, ``"["``) is not data.
    * A **scalar** value: substring (a genuine free-text excerpt of a longer
      string).
    """
    if excerpt == "":
        return value is None or value == [] or value == ""
    if isinstance(value, list):
        return any(excerpt == str(element) for element in value)
    return excerpt in str(value)


def _citation_valid(citation: Citation) -> bool:
    """A citation passes iff its ``excerpt`` AND every ``matched_tokens`` member are
    element-precise quotes of its own ``value`` (the record's real field value the
    core string-extracted). ``matched_tokens`` (CC-15 composite buckets, e.g.
    ``PHASE1|PHASE2``) carries the additional member literals; each must be as
    verbatim as the primary excerpt, so a fabricated composite token can't ride in
    unverified (the Citation invariant with teeth)."""
    if not _excerpt_in_value(citation.matched_value, citation.value):
        return False
    for token in citation.matched_tokens or []:
        if not _excerpt_in_value(token, citation.value):
            return False
    return True


def _spec_citations(spec: VisualizeResponse):
    """Yield every ``Citation`` a spec carries — row data + both endpoints of every
    network edge — so the record-grounded re-verify covers the whole provenance surface."""
    if spec.visualization is None:
        return
    data = spec.visualization.data
    if isinstance(data, list):
        for datum in data:
            yield from datum.citations
    else:  # NetworkData
        for edge in getattr(data, "edges", None) or []:
            yield from edge.citations


def record_grounded_reverify(
    spec: VisualizeResponse, fetched_records: dict[str, dict] | None
) -> PrecheckResult:
    """Phase-4 hardening (§3.8, SEC-19): a **best-effort, bounded-sample** re-verify of each
    citation excerpt against the ACTUAL fetched record, not the citation's own stored ``value``.

    The PRIMARY provenance guarantee stays the build-time one: every citation is code-built from a
    real record, and ``deterministic_precheck`` checks ``excerpt`` against ``citation.value`` (the
    record's real field). This adds a second, independent check for defense-in-depth once the LLM
    is in the loop: for each citation whose ``nct_id`` is present in the bounded ``fetched_records``
    index (the records ``execute`` paged), assert the excerpt — and every ``matched_tokens``
    member — is a real substring **at its ``field_path`` in that independent record** via
    :func:`is_substring_at`. A cited record that's PRESENT but whose excerpt doesn't appear in it
    HARD-FAILS ``citation_invalid`` **even when ``excerpt == value``** (defeating the tautology a
    fabricated citation could otherwise pass). This is ``is_substring_at``'s load-bearing RUNTIME
    caller (LESSON M3), not just a test one.

    **Coverage is bounded, and honestly so.** The index caps at ``_RECORD_INDEX_CAP`` records; a
    combine/explode board's cited set (≤ K×#buckets) fits, so those are FULLY covered, but a dense
    network cites nctIds across the whole paged set and may exceed the cap. A citation whose
    ``nct_id`` is NOT in the index is therefore SKIPPED (not failed) — the build-time value-check
    already grounded it, and a bounded sample cannot fail-closed without rejecting honest
    citations outside the sample. So this is a strong spot-check + a runtime caller for the
    primitive, NOT a complete gate over every citation. TOTAL: returns a ``PrecheckResult``, never
    raises. A ``None``/empty index (the path didn't page — too_large / offline sentinels) is a
    clean pass."""
    if not fetched_records:
        return PrecheckResult(ok=True, hard_fail=False)
    for citation in _spec_citations(spec):
        record = fetched_records.get(citation.nct_id)
        if record is None:
            continue  # not in the bounded sample — build-time value-check already covered it
        if not is_substring_at(record, citation.field_path, citation.matched_value):
            return PrecheckResult(ok=False, hard_fail=True, reason="citation_invalid")
        for token in citation.matched_tokens or []:
            if not is_substring_at(record, citation.field_path, token):
                return PrecheckResult(ok=False, hard_fail=True, reason="citation_invalid")
    return PrecheckResult(ok=True, hard_fail=False)


def deterministic_precheck(
    spec: VisualizeResponse,
    *,
    count_total: int | None,
    mode: str | None,
    distinct_trials: int | None,
    truncated: bool,
    reconcile: bool = True,
    drift_pct: float = 0.005,
    drift_abs: int = 20,
) -> PrecheckResult:
    """Run the deterministic Output-Reviewer checks over ``spec``.

    Runs ONLY when ``spec.status == "ok"`` and ``spec.visualization.data`` is a
    list (G-32; ``network`` / ``answer`` / ``too_large`` / ``empty`` are EXEMPT --
    they carry no row array to reconcile). Returns a ``PrecheckResult``; NEVER
    raises.

    ``reconcile`` gates ONLY the Σ==``countTotal`` step (2). A ``compare`` spec is
    a row list but spans TWO populations (two countTotals) -- there is no single
    oracle to reconcile the union against, so the caller passes ``reconcile=False``
    and each arm is instead self-reconciled inside the ``compare`` tool. The
    excerpt-substring (1), partial-iff-truncated (3), and cited-or-derived (4)
    checks STILL run -- provenance/tamper-evidence is never waived, only the
    single-oracle count check that doesn't apply to a multi-population union.
    """
    # --- exemption gate (G-32) ------------------------------------------------
    if spec.status != "ok" or spec.visualization is None:
        return PrecheckResult(ok=True, hard_fail=False)
    data = spec.visualization.data
    if not isinstance(data, list):
        # Network (NetworkData): reconciliation-exempt (no row array, no single
        # oracle) -- but the excerpt tamper-evidence STILL runs on every edge's two
        # citations. The coarse list-only exemption used to waive check (1) as well,
        # so a fabricated edge excerpt shipped unverified (F2).
        for edge in getattr(data, "edges", None) or []:
            for citation in edge.citations:
                if not _citation_valid(citation):
                    return PrecheckResult(ok=False, hard_fail=True, reason="citation_invalid")
        return PrecheckResult(ok=True, hard_fail=False)

    # --- (1) excerpt substring ------------------------------------------------
    for datum in data:
        for citation in datum.citations:
            if not _citation_valid(citation):
                return PrecheckResult(
                    ok=False, hard_fail=True, reason="citation_invalid"
                )

    # --- (2) reconciliation (mode-aware, G-26; skipped for multi-population) ---
    disclosure: str | None = None
    if reconcile:
        # An ok list-spec with no oracle CANNOT be certified as reconciled — shipping
        # it as "ok" would be a false provenance claim (in Phase 1 execute always
        # stamps count_total for ok, so None here means a real upstream failure).
        if count_total is None:
            return PrecheckResult(ok=False, hard_fail=True, reason="reconciliation_unavailable")

        # Anchor on the DISTINCT-nctId count for BOTH modes — that IS the CC-16 claim
        # ("distinct-nctId == countTotal"), not a raw bar-sum. With the core's
        # per-bucket nctId dedup (K3), Σ count_trials == distinct_trials for combine,
        # so the displayed bars and this anchor agree. Fall back to the bar-sum only if
        # the core reported no distinct total.
        observed = (
            distinct_trials
            if distinct_trials is not None
            else sum(datum.count_trials for datum in data)
        )
        drift = abs(observed - count_total)
        if drift == 0:
            pass  # exact -- nothing to disclose
        elif drift <= drift_pct * count_total and drift <= drift_abs:
            disclosure = (
                f"Reconciliation: observed distinct-trial total {observed:,} differs "
                f"from the API countTotal {count_total:,} by {drift} "
                f"(within tolerance {drift_pct:.1%} and {drift_abs}); likely live data drift."
            )
        else:
            return PrecheckResult(ok=False, hard_fail=True, reason="reconciliation_failed")

    # --- (2b) combine bar-sum consistency (R1-F1/F2) --------------------------
    # For a combine field every trial is in EXACTLY one bucket, so Σ displayed bars
    # MUST equal the distinct-trial anchor. The reconciliation above checks the
    # SCALAR anchor == countTotal; this checks the DISPLAYED bars == that anchor, so a
    # deflated/inflated bar (or a cross-key double-count) that left the scalar correct
    # can no longer ship silently. Explode is exempt (Σ bars = memberships ≥ distinct,
    # by design — a multi-value trial counts in each value's bar).
    if reconcile and mode == "combine" and distinct_trials is not None:
        bar_sum = sum(datum.count_trials for datum in data)
        if bar_sum != distinct_trials:
            return PrecheckResult(ok=False, hard_fail=True, reason="bar_sum_mismatch")

    # --- (3) partial iff truncated --------------------------------------------
    if (spec.meta.partial is not None) != bool(truncated):
        return PrecheckResult(ok=False, hard_fail=True, reason="partial_inconsistent")

    # --- (4) cited-or-derived (G-35) ------------------------------------------
    for datum in data:
        if not (datum.citations or datum.derived or datum.count_trials == 0):
            return PrecheckResult(ok=False, hard_fail=True, reason="uncited_datum")

    return PrecheckResult(ok=True, hard_fail=False, disclosure=disclosure)
