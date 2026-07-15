"""Offline, $0, no-network correctness harness over the shipped example runs.

For every ``examples/run_*.json`` (filename-agnostic glob) this asserts the five
invariants that make the system's core guarantee — *"the LLM decides WHAT to
compute; deterministic tools compute it — the model never emits a number"* —
mechanically checkable on a real output envelope:

* **(I) SCHEMA** — ``VisualizeResponse.model_validate`` round-trips the file.
* **(II) PROVENANCE TEETH** — every deep citation (row data + both endpoints of
  every network edge) is an element-precise quote of its own field ``value``; a
  fabricated excerpt fails.
* **(III) COUNT COHERENCE** — the dual-count / sample bookkeeping on each row
  datum is internally consistent.
* **(IV) RECONCILIATION** — Σ ``count_trials`` reconciles to
  ``meta.count_basis.trials`` (exactly for combine fields, ≥ for explode).
* **(V) NO LLM-AUTHORED NUMBER** — the computed-number set is note-independent,
  so a caveat can never launder a fabricated count onto the wire.

The load-bearing checks **reuse the runtime's own verifier primitives** from
``app.viz.review`` / ``app.ctgov.citations`` (``_citation_valid`` /
``_excerpt_in_value`` / ``_spec_citations`` / ``computed_numbers`` /
``note_number_safe``), so this offline gate proves the same thing the live
Output Reviewer (``app.graph.nodes.review_output``) does, with the same code.

Run:  ``./.venv/bin/python scripts/verify_examples.py``
Exits 0 iff every example passes every applicable invariant, else nonzero. A
malformed file yields a clear FAIL line, never a traceback (the harness is
total).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.api.schemas import VisualizeResponse
from app.viz.review import (
    _citation_valid,  # element-precise excerpt+tokens check (the runtime's teeth)
    _spec_citations,  # yields row-data + edge citations (the runtime provenance surface)
    computed_numbers,  # every computed digit-run in the envelope EXCEPT meta.notes
    note_number_safe,  # True iff a note's digit-runs are all drawn from the computed set
)

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _ROOT / "examples"
_GLOB = "run_*.json"

# Phrases the aggregation tools emit onto meta.notes for an EXPLODE (multi-value)
# field, per the CC-3 disclosure contract (e.g. "sum to MORE", "counted once per").
_EXPLODE_NOTE_PHRASES = ("sum to more", "counted once per")

# The five invariant labels, in report order.
_I_SCHEMA = "I  SCHEMA"
_I_PROV = "II PROVENANCE"
_I_COUNT = "III COUNT-COHERENCE"
_I_RECON = "IV RECONCILIATION"
_I_NONUM = "V  NO-LLM-NUMBER"


# --- report value objects -------------------------------------------------


@dataclass
class Check:
    """One invariant's verdict for one file."""

    label: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str = ""


@dataclass
class FileReport:
    """Every invariant verdict for one example file (+ any load error)."""

    name: str
    checks: list[Check] = field(default_factory=list)
    load_error: str | None = None

    @property
    def ok(self) -> bool:
        """True iff the file loaded and no invariant hard-failed (SKIP is fine)."""
        return self.load_error is None and all(c.status != "FAIL" for c in self.checks)

    def failure_lines(self) -> list[str]:
        """Human-readable lines for every FAIL (+ a load error), for a test's assert msg."""
        lines: list[str] = []
        if self.load_error is not None:
            lines.append(f"{self.name}: LOAD ERROR: {self.load_error}")
        lines += [f"{self.name}: {c.label}: FAIL: {c.detail}" for c in self.checks if c.status == "FAIL"]
        return lines


# --- per-invariant checks -------------------------------------------------


def _check_schema(obj: dict[str, Any]) -> tuple[Check, VisualizeResponse | None]:
    """(I) The whole envelope validates against the frozen wire contract."""
    try:
        spec = VisualizeResponse.model_validate(obj)
    except Exception as exc:  # noqa: BLE001 — total harness: any validation error is a clean FAIL
        return Check(_I_SCHEMA, "FAIL", f"model_validate raised: {type(exc).__name__}: {exc}"), None
    return Check(_I_SCHEMA, "PASS", f"status={spec.status} kind={spec.kind}"), spec


def _check_provenance(spec: VisualizeResponse) -> Check:
    """(II) Every deep citation is an element-precise quote of its own ``value``.

    Reuses the runtime's ``_spec_citations`` (row data + both endpoints of every
    edge) and ``_citation_valid`` (excerpt AND every ``excerpt_tokens`` member
    must be a verbatim element/substring of ``citation.value``; an empty excerpt
    is valid only against a genuinely-absent value). A fabricated excerpt fails.
    """
    n = 0
    for cit in _spec_citations(spec):
        n += 1
        if not _citation_valid(cit):
            return Check(
                _I_PROV,
                "FAIL",
                f"fabricated/mismatched excerpt: nct={cit.nct_id} field={cit.field_path} "
                f"excerpt={cit.excerpt!r} tokens={cit.excerpt_tokens!r} value={cit.value!r}",
            )
    if n == 0:
        return Check(_I_PROV, "PASS", "no row/edge citations (kind carries no chart data)")
    return Check(_I_PROV, "PASS", f"{n} citations element-precise")


def _check_count_coherence(spec: VisualizeResponse) -> Check:
    """(III) Per-row-datum dual-count / sample bookkeeping is internally consistent.

    * ``contributing_count >= len(citations)`` (a capped sample never claims more
      cited than contributing) — checked only when ``contributing_count`` is set.
    * ``citations_truncated == (contributing_count > len(citations))`` — the flag
      means exactly "the sample dropped a contributing record" — same guard.
    * every ``source_id`` resolves to a citation: ``set(source_ids) ⊆
      {c.nct_id for inline citations} ∪ top-level citations{} keys`` (the schema's
      stated contract; ``source_ids`` is a ≤K SAMPLE that may be empty, so a strict
      ``==`` would false-fail e.g. a single_value datum with an empty sample).
    * ``count_trials >= 0``.

    Structurally N/A (SKIP) for network / answer / clarification / too_large — no
    row array to reconcile.
    """
    if spec.visualization is None or not isinstance(spec.visualization.data, list):
        return Check(_I_COUNT, "SKIP", "no row-list data")
    top_keys = set(spec.citations.keys())
    for datum in spec.visualization.data:
        cits = datum.citations
        inline = {c.nct_id for c in cits}
        cc = datum.contributing_count
        if cc is not None:
            if cc < len(cits):
                return Check(_I_COUNT, "FAIL", f"{datum.value!r}: contributing_count {cc} < len(citations) {len(cits)}")
            if bool(datum.citations_truncated) != (cc > len(cits)):
                return Check(
                    _I_COUNT,
                    "FAIL",
                    f"{datum.value!r}: citations_truncated={datum.citations_truncated} "
                    f"but contributing_count {cc} vs len(citations) {len(cits)}",
                )
        unresolved = set(datum.source_ids) - (inline | top_keys)
        if unresolved:
            return Check(
                _I_COUNT,
                "FAIL",
                f"{datum.value!r}: source_ids do not resolve to a citation: {sorted(unresolved)}",
            )
        if datum.count_trials < 0:
            return Check(_I_COUNT, "FAIL", f"{datum.value!r}: count_trials {datum.count_trials} < 0")
    return Check(_I_COUNT, "PASS", f"{len(spec.visualization.data)} row data coherent")


def _is_explode(spec: VisualizeResponse) -> tuple[bool, str]:
    """Is this an EXPLODE (multi-value) distribution? Returns (explode, why).

    Two independent signals, EITHER sufficient:
    * the CC-3 disclosure phrase on ``meta.notes`` ("sum to MORE" / "counted once
      per") — what the tools emit, and
    * ``count_basis.mentions != trials`` — the structural fingerprint of a
      multi-value field (a trial counted in >1 bucket lifts mentions above the
      distinct-trial total). Note-scan alone MISSES a genuine explode whose tool
      did not emit the phrase (the degenerate-network drug-frequency fallback),
      so the count-basis signal is load-bearing, not redundant.
    """
    blob = " ".join(spec.meta.notes).lower()
    if any(p in blob for p in _EXPLODE_NOTE_PHRASES):
        return True, "meta.notes explode phrase"
    cb = spec.meta.count_basis
    if cb is not None and cb.mentions is not None and cb.mentions != cb.trials:
        return True, f"count_basis mentions {cb.mentions} != trials {cb.trials}"
    return False, ""


def _check_reconciliation(spec: VisualizeResponse) -> Check:
    """(IV) Σ ``count_trials`` reconciles to ``meta.count_basis.trials`` (T).

    Runs ONLY for ``status=="ok"`` + ``kind=="visualization"`` + a non-empty ROW
    LIST (network / answer / clarification / too_large / empty are EXEMPT — no
    single row array with a single count oracle). Combine → ``Σ == T``; explode →
    ``Σ >= T`` (a multi-value trial is counted once per value).
    """
    if spec.status != "ok" or spec.kind != "visualization":
        return Check(_I_RECON, "SKIP", f"exempt (status={spec.status} kind={spec.kind})")
    if spec.visualization is None or not isinstance(spec.visualization.data, list):
        return Check(_I_RECON, "SKIP", "exempt (network / non-row data)")
    data = spec.visualization.data
    if not data:
        return Check(_I_RECON, "SKIP", "exempt (0-bucket / empty chart)")
    if spec.meta.count_basis is None:
        return Check(_I_RECON, "FAIL", "ok row-list chart carries no count_basis to reconcile against")
    total = spec.meta.count_basis.trials
    sigma = sum(d.count_trials for d in data)
    explode, why = _is_explode(spec)
    if explode:
        if sigma < total:
            return Check(_I_RECON, "FAIL", f"explode: Σcount_trials {sigma} < count_basis.trials {total} ({why})")
        return Check(_I_RECON, "PASS", f"explode Σ {sigma} >= T {total} ({why})")
    if sigma != total:
        return Check(_I_RECON, "FAIL", f"combine: Σcount_trials {sigma} != count_basis.trials {total}")
    return Check(_I_RECON, "PASS", f"combine Σ == T == {total}")


def _check_no_llm_number(spec: VisualizeResponse) -> Check:
    """(V) The computed-number set is note-independent — a caveat cannot launder a
    fabricated count onto the wire.

    Reuses the runtime's ``computed_numbers`` (every digit-run in the envelope
    EXCEPT ``meta.notes``) and ``note_number_safe``. The teeth: a fabricated
    data-magnitude number placed in a note is rejected, proving ``meta.notes``
    contributes nothing to the allowed set (the §1 anti-laundering guarantee).

    NB (a genuine scoping finding — see the module report): the runtime applies
    ``note_number_safe`` as an ADMISSION gate on untrusted candidate notes only
    (planner notes in ``build_spec``; the LLM flag reason in ``review_output``),
    NOT over the whole ``meta.notes`` array. Code-templated ``build_envelope``
    notes legitimately carry non-data digit-runs — spec-reference codes ("CC-3",
    "G-40"), config caps ("top 50/60"), comma-grouped totals ("121,770"), and
    compare captions ("N=2903; N=2011") absent from the computed set — and bypass
    the gate by design. So "every note is note_number_safe" is NOT a system
    invariant and is reported, not asserted.
    """
    allowed = computed_numbers(spec)
    # A digit-run guaranteed absent from the computed set (extend 9s until unique).
    sentinel = "9"
    while sentinel in allowed:
        sentinel += "9"
    if note_number_safe(f"fabricated count {sentinel}", allowed):
        return Check(_I_NONUM, "FAIL", f"anti-laundering breached: note number {sentinel} accepted as computed")
    # Informational: how many shipped notes carry only computed digits vs non-data numerics.
    notes = spec.meta.notes
    gated = sum(1 for n in notes if note_number_safe(str(n), allowed))
    return Check(
        _I_NONUM,
        "PASS",
        f"notes cannot launder numbers (sentinel {sentinel} rejected); "
        f"{gated}/{len(notes)} notes gate-clean, rest = code-templated non-data numerics",
    )


# --- driver ---------------------------------------------------------------


def verify_obj(obj: dict[str, Any], name: str) -> FileReport:
    """Run all five invariants over an already-parsed envelope dict.

    Split from :func:`verify_file` so a test can feed a tampered in-memory
    envelope (e.g. a fabricated excerpt) and confirm the harness has teeth.
    """
    report = FileReport(name=name)
    schema_check, spec = _check_schema(obj)
    report.checks.append(schema_check)
    if spec is None:
        # Can't build the typed model — everything downstream is unverifiable.
        for label in (_I_PROV, _I_COUNT, _I_RECON, _I_NONUM):
            report.checks.append(Check(label, "SKIP", "schema invalid — cannot verify"))
        return report
    for fn in (_check_provenance, _check_count_coherence, _check_reconciliation, _check_no_llm_number):
        try:
            report.checks.append(fn(spec))
        except Exception as exc:  # noqa: BLE001 — total harness: never a traceback
            label = {
                _check_provenance: _I_PROV,
                _check_count_coherence: _I_COUNT,
                _check_reconciliation: _I_RECON,
                _check_no_llm_number: _I_NONUM,
            }[fn]
            report.checks.append(Check(label, "FAIL", f"checker raised: {type(exc).__name__}: {exc}"))
    return report


def verify_file(path: Path) -> FileReport:
    """Load one example file and verify it. A malformed file → a clean FAIL, never a traceback."""
    name = path.name
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — total harness: bad JSON is a reported FAIL
        report = FileReport(name=name, load_error=f"{type(exc).__name__}: {exc}")
        report.checks.append(Check(_I_SCHEMA, "FAIL", f"could not parse JSON: {exc}"))
        return report
    if not isinstance(obj, dict):
        report = FileReport(name=name, load_error="top-level JSON is not an object")
        report.checks.append(Check(_I_SCHEMA, "FAIL", "top-level JSON is not an object"))
        return report
    return verify_obj(obj, name)


def iter_example_paths() -> list[Path]:
    """Every ``examples/run_*.json`` (filename-agnostic), sorted for stable output."""
    return sorted(_EXAMPLES_DIR.glob(_GLOB))


def main(argv: list[str] | None = None) -> int:
    """Verify every shipped example; print per-file/per-invariant verdicts + summary.

    Returns 0 iff every example passed every applicable invariant, else 1.
    """
    paths = iter_example_paths()
    if not paths:
        print(f"FAIL: no examples matched {_EXAMPLES_DIR}/{_GLOB}", file=sys.stderr)
        return 1

    reports = [verify_file(p) for p in paths]
    for report in reports:
        flag = "PASS" if report.ok else "FAIL"
        print(f"\n{'=' * 78}\n[{flag}] {report.name}")
        for c in report.checks:
            print(f"    [{c.status:4s}] {c.label:20s} {c.detail}")

    passed = sum(1 for r in reports if r.ok)
    total = len(reports)
    print(f"\n{'=' * 78}")
    if passed != total:
        print("FAILURES:")
        for r in reports:
            for line in r.failure_lines():
                print(f"  - {line}")
    print(f"\n{passed}/{total} examples passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
