"""String-extraction citation primitives (ARCHITECTURE_SPEC §3.6 / §A(c), CC-9).

The core anti-hallucination guarantee for provenance: an excerpt is **never**
authored by the LLM — it is walked out of the fetched JSON record at a known
``field_path`` and is round-trip verifiable (the Output Reviewer asserts each
excerpt is a real substring of the source at that path). ``extract_excerpt``
and ``is_substring_at`` are the two pure, real primitives that guarantee;
everything else in the citation pipeline (paging, bucketing) is Phase 1.
"""

from __future__ import annotations

from typing import Any

from app.api.schemas import Citation


def _resolve_path(record: dict, field_path: str) -> Any:
    """Walk a dotted JSON path, descending into the first element of any
    ``name[]`` list segment. Returns ``None`` if any segment is absent.

    Handles the two path shapes used throughout this codebase: a plain
    ``a.b.c`` dict walk, and ``a[].b`` where ``a`` is a list and we take its
    first element before continuing the walk (the "first match" contract).
    """
    current: Any = record
    for part in field_path.split("."):
        if current is None:
            return None
        if part.endswith("[]"):
            key = part[:-2]
            current = current.get(key) if isinstance(current, dict) else None
            if isinstance(current, list):
                current = current[0] if current else None
            else:
                current = None
        else:
            current = current.get(part) if isinstance(current, dict) else None
    return current


def extract_excerpt(record: dict, field_path: str) -> str:
    """Walk ``field_path`` in ``record`` and return the literal value as a string.

    This is the string-extraction primitive (CC-9): excerpts are never
    LLM-authored, only ever pulled out of the fetched record. When the
    resolved value is a list (e.g. ``phases: ["PHASE1","PHASE2"]``), the
    excerpt is its first element (or ``""`` for an empty/absent list) — the
    literal token that decided membership in that bucket. This is the
    "first/whole value" primitive; per-bucket extraction that targets a
    *specific* list element (e.g. the second intervention, not the first)
    is a Phase-1 citation-core feature, not this function's job.
    """
    value = _resolve_path(record, field_path)
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _resolve_all(record: Any, field_path: str) -> list[Any]:
    """Like :func:`_resolve_path`, but when a segment is a ``name[]`` list,
    continues the walk over **every** element instead of only the first.

    ``_resolve_path`` (and therefore ``extract_excerpt``) intentionally keeps
    the "first element" convention — that's the literal token that decided a
    bucket. But round-trip *verification* must not share that blind spot: a
    genuinely-present value at a non-first list index (e.g. the second of two
    interventions, or the second of two countries) has to verify True, not
    silently fail because only element 0 was ever inspected. Returns the list
    of every leaf value reachable by expanding each ``[]`` segment across all
    of its list's elements (``[None]`` when a segment can't be walked).
    """
    parts = field_path.split(".")

    def walk(current: Any, idx: int) -> list[Any]:
        if idx == len(parts):
            return [current]
        if current is None:
            return [None]
        part = parts[idx]
        if part.endswith("[]"):
            key = part[:-2]
            nxt = current.get(key) if isinstance(current, dict) else None
            if not isinstance(nxt, list) or not nxt:
                return [None]
            results: list[Any] = []
            for item in nxt:
                results.extend(walk(item, idx + 1))
            return results
        nxt = current.get(part) if isinstance(current, dict) else None
        return walk(nxt, idx + 1)

    return walk(record, 0)


def is_substring_at(record: dict, field_path: str, excerpt: str) -> bool:
    """Round-trip verify: is ``excerpt`` really present at ``field_path`` in ``record``?

    The record-grounded provenance primitive (§3.8). Scans **every** value
    reachable at ``field_path`` (every element of a ``name[]`` list, not just the
    first — CC-3/CC-13). Matching is **element-precise, not loose substring**
    (LESSON K2):

    * A **token-array** value (e.g. ``phases: ["PHASE1","PHASE2"]``) round-trips
      only when the excerpt EQUALS one element — ``"PHASE1"`` is NOT present in a
      ``["PHASE10"]`` trial, and the stringified-list repr punctuation (``"', '"``,
      ``"["``) is not data (both were false positives under the old
      ``excerpt in str(list)`` fallback).
    * A **scalar** value round-trips on substring (a genuine free-text excerpt of a
      longer string; Phase-2 token scalars should tighten this to equality).
    * An **empty** excerpt is a legitimate *absence* citation (e.g. the MISSING
      phase bucket: no field value to quote) ONLY when the path resolves to no
      value; an empty excerpt against a PRESENT value proves nothing → False.
    """
    values = _resolve_all(record, field_path)
    present = [value for value in values if value is not None and value != []]
    if not present:
        return excerpt == ""
    if excerpt == "":
        return False

    for value in present:
        if isinstance(value, list):
            if any(excerpt == str(item) for item in value):
                return True
        elif excerpt in str(value):
            return True
    return False


def build_citation(record: dict, field_path: str) -> Citation:
    """Build a ``Citation`` for ``record`` at ``field_path`` (CC-9).

    ``nct_id`` is read from the fixed identification path (real on every
    fetched record); ``value`` is the literal resolved value at ``field_path``
    (may be an array); ``excerpt`` is string-extracted via
    :func:`extract_excerpt`, never authored.
    """
    nct_id = _resolve_path(record, "protocolSection.identificationModule.nctId")
    value = _resolve_path(record, field_path)
    excerpt = extract_excerpt(record, field_path)
    return Citation(nct_id=nct_id or "", field_path=field_path, value=value, excerpt=excerpt)


def build_bucket_citations(
    records: list[dict],
    field_path: str,
    *,
    k: int = 20,
    member_tokens: list[str] | None = None,
) -> tuple[list[Citation], int, bool]:
    """Build the per-bucket citation sample for ONE bucket's contributing records.

    Returns ``(citations, contributing_count, truncated)`` — the CC-9 split of an
    **exact** contributing total from a **capped** provenance sample:

    * ``contributing_count`` is the true size of the contributing set
      (``len(records)``), computed BEFORE any capping — it is the number the
      count reconciles against, never the length of the (possibly-capped) sample.
    * The sample is deterministic: records are sorted by their nctId
      (``protocolSection.identificationModule.nctId``) and the first ``k`` are
      taken. A record missing that path sorts as ``""`` (kept deterministic; it
      never raises). ``build_citation`` (the existing primitive) turns each
      sampled record into a ``Citation`` whose excerpt is string-extracted from
      the record, never authored.
    * ``truncated`` is ``True`` iff ``contributing_count > k`` — i.e. the sample
      dropped at least one contributing record.

    ``member_tokens`` — the composite-bucket contract (CC-15). When a bucket is a
    combined multi-value bucket (e.g. a ``PHASE1|PHASE2`` phase bucket formed from
    ``["PHASE1","PHASE2"]``), a single excerpt (the first token) UNDER-identifies
    the composite in a drill-down. Pass the member tokens to identify EVERY member:

    * ``None`` or length ≤ 1 (single-value bucket): behavior UNCHANGED — each
      citation's ``excerpt`` is string-extracted via ``build_citation`` and
      ``excerpt_tokens`` stays ``None``.
    * length ≥ 2 (composite bucket): each citation keeps ``value`` = the record's
      real resolved value and sets ``excerpt`` = the first member token (display),
      and ``excerpt_tokens`` = the subset of ``member_tokens`` actually present in
      THAT record at ``field_path``, each verified verbatim via
      :func:`is_substring_at`, in token order. Only verified tokens are included,
      so ``excerpt_tokens`` stays honest even on a malformed record. No excerpt
      text is ever synthesized (Citation invariant).

    Pure: no I/O, and the input ``records`` list and its dicts are never mutated
    (``sorted`` returns a new list; the walk is read-only). TOTAL: never raises on
    a malformed record.
    """
    contributing_count = len(records)

    def _nct_key(record: dict) -> str:
        nct_id = _resolve_path(record, "protocolSection.identificationModule.nctId")
        return nct_id if isinstance(nct_id, str) else ""

    sample = sorted(records, key=_nct_key)[:k]

    tokens = member_tokens if member_tokens is not None and len(member_tokens) >= 2 else None
    citations: list[Citation] = []
    for record in sample:
        citation = build_citation(record, field_path)
        if tokens is not None:  # composite bucket: identify EVERY verified member
            verified = [t for t in tokens if is_substring_at(record, field_path, t)]
            citation = citation.model_copy(
                update={"excerpt": tokens[0], "excerpt_tokens": verified}
            )
        citations.append(citation)

    truncated = contributing_count > k
    return citations, contributing_count, truncated
