"""String-extraction citation primitives (ARCHITECTURE_SPEC §3.6 / §A(c), CC-9).

The core anti-hallucination guarantee for provenance: **no citation text is ever
authored by the LLM** — every field of a ``Citation`` is walked out of a fetched
JSON record here.

A ``Citation`` carries TWO extracted strings, and they come from different paths
(this is the shape after the ``excerpt``→``matched_value`` rename; the two names
are not interchangeable):

* ``matched_value`` — the literal value at the datum's own ``field_path`` (the
  token that decided bucket membership, e.g. ``"PHASE1"`` / ``"France"`` /
  ``"2015-01-28"``), produced by :func:`extract_excerpt` or by the element-targeted
  builders in ``app.ctgov.tools`` / ``app.ctgov.network``. This is the anchor the
  Output Reviewer round-trip verifies with :func:`is_substring_at`
  (``app.viz.review``), together with ``matched_tokens`` for a composite bucket.
* ``excerpt`` — the trial's human-readable **brief title**, walked out of the FIXED
  identification path :data:`_BRIEF_TITLE_PATH` (see :func:`brief_title`), NOT out
  of ``field_path``. It is assignment §5's readable supporting excerpt. It is
  code-extracted and never authored, but it is **not** re-verified by the Output
  Reviewer — the reviewer checks ``matched_value``/``matched_tokens`` only, since
  the brief title provably does not live at ``field_path``.

``extract_excerpt`` and ``is_substring_at`` are the two pure primitives that carry
the guarantee; everything else here (sampling, bucketing) is bookkeeping.
"""

from __future__ import annotations

from typing import Any

from app import config
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

    This is the string-extraction primitive (CC-9) behind ``Citation.matched_value``
    (NOT behind ``Citation.excerpt``, which is the brief title — see the module
    docstring): the value is never LLM-authored, only ever pulled out of the
    fetched record. When the
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

    The ``excerpt`` parameter keeps its pre-rename name, but every caller passes a
    **``Citation.matched_value`` / ``matched_tokens`` member** — the value extracted
    AT ``field_path``. ``Citation.excerpt`` (the brief title) is deliberately never
    passed here: it comes from a different path and would always verify False.

    The record-grounded provenance primitive (§3.8). Scans **every** value
    reachable at ``field_path`` (every element of a ``name[]`` list, not just the
    first — CC-3/CC-13). Matching is **element-precise, not loose substring**
    (LESSON K2):

    * A **token-array** value (e.g. ``phases: ["PHASE1","PHASE2"]``) round-trips
      only when the excerpt EQUALS one element — ``"PHASE1"`` is NOT present in a
      ``["PHASE10"]`` trial, and the stringified-list repr punctuation (``"', '"``,
      ``"["``) is not data (both were false positives under the old
      ``excerpt in str(list)`` fallback).
    * A **scalar** value round-trips on plain **substring** — the one deliberately
      loose spot in the provenance chain, worth naming rather than glossing. It is
      loose in exactly one direction: a claimed value SHORTER than the record's real
      one passes (``"COMPLET"`` verifies against ``"COMPLETED"``). No shipped path
      can produce such a value — every scalar ``matched_value`` is either the whole
      resolved value from :func:`extract_excerpt` or a validated enum token, never a
      trimmed one — so this is a latent looseness, not a live hole. Tightening
      scalars to equality would break the genuinely free-text scalars (sponsor name,
      dates), where quoting part of a longer string is legitimate. Known watch-item.
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


_BRIEF_TITLE_PATH = "protocolSection.identificationModule.briefTitle"


def brief_title(record: dict) -> str | None:
    """The trial's human-readable **brief title** — the descriptive text excerpt that
    supports the datum (assignment §5), string-extracted from the record at the fixed
    identification path (never LLM-authored). Returns ``None`` when the record did not
    project ``BriefTitle`` (a fetch that didn't request it), so an un-enriched citation
    stays honest rather than inventing context. TOTAL: never raises."""
    value = _resolve_path(record, _BRIEF_TITLE_PATH)
    return value if isinstance(value, str) and value else None


def build_citation(record: dict, field_path: str) -> Citation:
    """Build a ``Citation`` for ``record`` at ``field_path`` (CC-9).

    Field by field (all four are extracted from ``record``, none authored):

    * ``nct_id`` — the fixed identification path (real on every fetched record).
    * ``value`` — the literal resolved value at ``field_path`` (may be an array).
    * ``matched_value`` — the string-extracted value at ``field_path``
      (:func:`extract_excerpt`); the anchor the Output Reviewer verifies.
    * ``excerpt`` — the record's brief title (:func:`brief_title`), the readable
      supporting excerpt of assignment §5. It falls back to ``matched_value`` when
      the record carries no ``BriefTitle`` (an un-projected fetch), so a citation
      always ships some human-readable text.

    There is no ``title`` field on ``Citation`` — the brief title IS ``excerpt``.
    """
    nct_id = _resolve_path(record, "protocolSection.identificationModule.nctId")
    value = _resolve_path(record, field_path)
    excerpt = extract_excerpt(record, field_path)
    return Citation(
        nct_id=nct_id or "", field_path=field_path, value=value,
        matched_value=excerpt, excerpt=brief_title(record) or excerpt,
    )


def build_bucket_citations(
    records: list[dict],
    field_path: str,
    *,
    k: int = config.CITATION_SAMPLE_K,
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
      taken (``k`` defaults to ``config.CITATION_SAMPLE_K``, operator-tunable). A
      record missing that path sorts as ``""`` (kept deterministic; it never
      raises). ``build_citation`` (the existing primitive) turns each sampled
      record into a ``Citation`` whose strings are extracted from the record,
      never authored.
    * ``truncated`` is ``True`` iff ``contributing_count > k`` — i.e. the sample
      dropped at least one contributing record.

    ``member_tokens`` — the composite-bucket contract (CC-15). When a bucket is a
    combined multi-value bucket (e.g. a ``PHASE1|PHASE2`` phase bucket formed from
    ``["PHASE1","PHASE2"]``), a single ``matched_value`` (the first token)
    UNDER-identifies the composite in a drill-down. Pass the member tokens to
    identify EVERY member:

    * ``None`` or length ≤ 1 (single-value bucket): behavior UNCHANGED — each
      citation's ``matched_value`` is string-extracted via ``build_citation`` and
      ``matched_tokens`` stays ``None``.
    * length ≥ 2 (composite bucket): each citation keeps ``value`` = the record's
      real resolved value, sets ``matched_value`` = the first member token
      (display), and sets ``matched_tokens`` = the subset of ``member_tokens``
      actually present in THAT record at ``field_path``, each verified verbatim via
      :func:`is_substring_at`, in token order. Only verified tokens are included,
      so ``matched_tokens`` stays honest even on a malformed record. ``excerpt``
      (the brief title) is untouched by the composite branch. No citation text is
      ever synthesized (Citation invariant).

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
                update={"matched_value": tokens[0], "matched_tokens": verified}
            )
        citations.append(citation)

    truncated = contributing_count > k
    return citations, contributing_count, truncated
