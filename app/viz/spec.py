"""The viz-spec builder (ARCHITECTURE_SPEC §3.7).

Assembles a schema-valid ``VisualizeResponse`` envelope from a validated
``Plan`` plus the bucket / graph / scalar results the executor produced. This is
the file that upholds G-30 for prose fields: ``title`` is **code-templated** from
the validated ``Plan`` only -- assembled here, never authored by the adapter --
so no digit or interpretation the model produced can leak into a user-facing
string. The scalar ``answer`` for ``too_large`` and for the ``single_value``
yes/no path is likewise code-templated from the computed total.

Envelope shapes, keyed by the ``status`` × ``kind`` pair a frontend switches on
(``status`` alone does NOT determine the shape -- ``ok`` and ``empty`` each carry
two kinds):

* ``ok`` / ``visualization`` -- a populated ``Visualization``: a row chart, a
  ``network_graph``, or a ``single_value`` stat card. ``vega_lite`` is set for the
  standard marks only (``None`` for network / single_value / table, see
  ``.vega``).
* ``ok`` / ``answer`` -- the ``single_value`` yes/no path (CC-7): no
  visualization, no vega_lite, a code-templated scalar ``answer``.
* ``empty`` / ``visualization`` -- the row-chart shape with an empty ``data``
  list and a "No trials matched this query." note (an empty chart, not an
  answer).
* ``empty`` / ``clarification`` -- :func:`build_clarification_envelope` (E-13):
  a code-owned ``question``, no visualization / answer / vega_lite (nothing was
  queried).
* ``too_large`` / ``answer`` -- no visualization/vega_lite, and ``meta.partial``
  stays null (refusing to chart is not truncating, G-39).
* ``error`` / ``answer`` -- no visualization, a populated ``ErrorObj`` (never a
  half-built viz, API-22).
"""

from __future__ import annotations

from app.api.schemas import (
    ChartType,
    Citation,
    CountBasis,
    Datum,
    Edge,
    EncodingChannel,
    ErrorObj,
    Meta,
    NetworkData,
    Node,
    Partial,
    Visualization,
    VisualizeResponse,
)
from app.plan.models import Plan

from .vega import to_vega_lite

# --- title templating (G-30: code-only, never adapter-authored) -----------

# Human display name for a recipe's aggregation field. Falls back to the raw
# field token for anything not listed here (still code, never LLM prose).
_FIELD_LABELS: dict[str, str] = {
    "phase": "Phase",
    "studyType": "Study type",
    "overallStatus": "Status",
    "sponsorClass": "Sponsor class",
    "sponsorName": "Sponsor",
    "interventionType": "Intervention type",
    "interventionName": "Intervention",
    "country": "Country",
}


def _field_label(field: str | None) -> str:
    if field is None:
        return "Value"
    return _FIELD_LABELS.get(field, field)


def _entity_phrase(plan: Plan) -> str:
    """The plan's headline entity, for the title -- condition preferred (the
    common case), falling back through the other dimensions."""
    for key in ("condition", "drug", "sponsor", "country", "term"):
        value = plan.entities.get(key)
        if value:
            return value
    return "matching"


# Human date-intent phrase for a time-series title (CC-4). The chosen date field
# is always ALSO disclosed in meta.date_field_used; this is only display prose.
_DATE_INTENT: dict[str, str] = {
    "startDate": "started",
    "primaryCompletionDate": "with primary completion",
    "completionDate": "completed",
    "studyFirstPostDate": "first registered",
    "lastUpdatePostDate": "last updated",
}


def _title(plan: Plan) -> str:
    """Code-templated title (G-30), dispatched by query_class -- assembled purely
    from the validated Plan's own fields. Any digit that later accompanies it
    (e.g. a trial count in a caller's note) still traces to computed data, never
    to this string, which the model never sees or authors."""
    entity = _entity_phrase(plan)
    if plan.query_class == "single_value":
        qualifier = "interventional " if plan.interventional_only else ""
        return f"Number of {qualifier}{entity} trials"
    if plan.query_class == "timeseries":
        intent = _DATE_INTENT.get(plan.date_field or "", "recorded")
        grain = plan.grain or "year"
        return f"{entity.capitalize()} trials {intent} per {grain}"
    if plan.query_class == "geographic":
        return f"{entity.capitalize()} trials by country"
    if plan.query_class == "compare":
        labels = [s.label for s in (plan.series or []) if s.label]
        arms = " vs ".join(labels) if labels else entity
        return f"{arms} trials by {_field_label(plan.field).lower()}"
    if plan.query_class == "network":
        kind = plan.network.kind if plan.network else "drug_drug"
        if kind == "sponsor_drug":
            return f"Sponsor-drug network for {entity} trials"
        return f"Drugs studied together in {entity} trials"
    qualifier = "interventional " if plan.interventional_only else ""
    if plan.chart_type is ChartType.HISTOGRAM or plan.field == "study_duration":
        return f"Study-duration distribution of {qualifier}{entity} trials"
    return f"{_field_label(plan.field)} distribution of {qualifier}{entity} trials"


# Per-class channel keys a bucket dict may carry beyond the core measures. These
# are forwarded onto the Datum verbatim so a timeseries ``period``, a compare
# ``series``/``percent``, a histogram ``bin_start``/``bin_end``, or a future-year
# ``planned`` flag survive into the wire spec (``period``/``series``/``bin_*`` are
# named Datum fields; ``planned``/``percent`` ride on Datum's ``extra="allow"``).
_CHANNEL_KEYS: tuple[str, ...] = (
    "period", "series", "bin_start", "bin_end", "planned", "partial_year", "percent",
)


def _bucket_to_datum(bucket: dict) -> Datum:
    citations = [
        c if isinstance(c, Citation) else Citation(**c) for c in bucket.get("citations", [])
    ]
    channels = {key: bucket[key] for key in _CHANNEL_KEYS if key in bucket}
    return Datum(
        value=bucket["value"],
        label=bucket.get("label", bucket["value"]),
        count_trials=bucket.get("count_trials", 0),
        count_mentions=bucket.get("count_mentions"),
        source_ids=bucket.get("source_ids", []),
        citations=citations,
        citations_truncated=bucket.get("citations_truncated", False),
        contributing_count=bucket.get("contributing_count"),
        derived=bucket.get("derived", False),
        members=bucket.get("members"),
        **channels,
    )


def _encoding_for(plan: Plan) -> dict[str, EncodingChannel]:
    """The x/y(/color) channel spec for a row chart, by chart type (CC-10).

    * ``time_series``  -> x = ``period`` (ascending), y = distinct-trial count.
    * ``grouped_bar``  -> x = category ``value``, y = ``percent`` (% within series,
      the CC-14 headline), color = ``series`` (the per-arm split); raw
      ``count_trials`` stays on each datum for the tooltip.
    * ``histogram``    -> x = the duration ``value`` (bin label), y = count.
    * ``bar`` (distribution / geographic) -> x = categorical ``value``, y = count.
    """
    y_trials = EncodingChannel(field="count_trials", label="Trials", unit="trials", scale="linear")
    if plan.chart_type is ChartType.SINGLE_VALUE:
        # A stat card has one measure and no axes: the single "value" channel reads the
        # code-computed distinct-trial count (CC-7 scalar path).
        return {"value": EncodingChannel(field="count_trials", label="Trials", unit="trials")}
    if plan.chart_type is ChartType.TIME_SERIES:
        return {
            "x": EncodingChannel(field="period", label="Period", sort="ascending"),
            "y": y_trials.model_copy(update={"label": "Trials"}),
        }
    if plan.chart_type is ChartType.GROUPED_BAR:
        return {
            "x": EncodingChannel(field="value", label=_field_label(plan.field)),
            "y": EncodingChannel(field="percent", label="% within series", unit="%", scale="linear"),
            "color": EncodingChannel(field="series", label="Series"),
        }
    if plan.chart_type is ChartType.HISTOGRAM:
        return {
            "x": EncodingChannel(field="value", label="Study duration"),
            "y": y_trials,
        }
    return {
        "x": EncodingChannel(field="value", label=_field_label(plan.field)),
        "y": y_trials,
    }


def build_visualization(plan: Plan, buckets: list[dict]) -> Visualization:
    """Assemble a row-chart ``Visualization`` from computed buckets (§3.7).

    ``title`` is CODE-TEMPLATED (G-30), never adapter-authored; ``encoding`` is
    the per-chart-type channel spec (``_encoding_for``). Used for every non-network
    chart (bar / grouped_bar / time_series / histogram); networks go through
    :func:`build_network_visualization`.
    """
    data = [_bucket_to_datum(b) for b in buckets]
    return Visualization(
        type=plan.chart_type, title=_title(plan), encoding=_encoding_for(plan), data=data
    )


def build_network_visualization(plan: Plan, graph: dict) -> Visualization:
    """Assemble a ``network_graph`` ``Visualization`` from a ``{nodes, edges}``
    graph dict (``app.ctgov.network.build_graph`` output).

    ``data`` is a ``NetworkData`` (not a row list) -- the type<->shape invariant
    the schema's model_validator enforces. Edges carry their derived weight + the
    two-endpoint citations built by the network layer; nothing here authors a
    number or an excerpt. ``encoding`` names the node-id and edge-weight channels
    a force-directed renderer reads.
    """
    nodes = [n if isinstance(n, Node) else Node(**n) for n in graph.get("nodes", [])]
    edges = [_edge_from(e) for e in graph.get("edges", [])]
    encoding = {
        "nodes": EncodingChannel(field="id", label="Entity"),
        "edges": EncodingChannel(field="weight", label="Co-occurring trials", unit="trials"),
    }
    return Visualization(
        type=ChartType.NETWORK_GRAPH,
        title=_title(plan),
        encoding=encoding,
        data=NetworkData(nodes=nodes, edges=edges),
    )


def _edge_from(edge: dict | Edge) -> Edge:
    """Coerce an edge dict to an ``Edge``, converting its inline citations."""
    if isinstance(edge, Edge):
        return edge
    citations = [
        c if isinstance(c, Citation) else Citation(**c) for c in edge.get("citations", [])
    ]
    return Edge(
        source=edge["source"],
        target=edge["target"],
        weight=edge["weight"],
        source_ids=edge.get("source_ids", []),
        citations=citations,
    )


# --- single_value / no-viz path (CC-7) --------------------------------------


def build_single_value_visualization(
    plan: Plan, total_count: int, citations: list[dict] | None = None
) -> Visualization:
    """Assemble a ``single_value`` stat-card ``Visualization`` (CC-7).

    ``total_count`` is the exact ``countTotal`` from the ``count_trials`` tool,
    inserted by CODE here (G-30/CC-16) — never authored by the LLM. ``title`` is
    code-templated from the validated Plan; the one ``Datum`` carries the number in
    its code-computed ``count_trials`` field. ``citations`` (a small sample of
    contributing nctIds, each a ``{nct_id, field_path, value, matched_value,
    excerpt}`` dict — ``matched_value`` is the nctId itself at the identification
    path, ``excerpt`` the trial's brief title) prove membership in the counted set.
    ``vega_lite`` is None for a stat card (no natural mark — the caller sets it).
    """
    cites = [c if isinstance(c, Citation) else Citation(**c) for c in (citations or [])]
    datum = Datum(
        value=str(total_count),
        label=f"{total_count:,} trials",
        count_trials=total_count,
        contributing_count=total_count,
        # The sample is honest about being a sample: a stat card cites ≤K of the N
        # counted trials, so flag truncation when it is smaller than the total.
        citations_truncated=len(cites) < total_count,
        citations=cites,
    )
    return Visualization(
        type=ChartType.SINGLE_VALUE,
        title=_title(plan),
        encoding=_encoding_for(plan),
        data=[datum],
    )


def _single_value_answer(total_count: int) -> str:
    """Code-templated yes/no answer for the ``kind:"answer"`` scalar path (G-30).

    The number is inserted by code from the ``count_trials`` result; nothing here is
    LLM-authored. A positive count is a "Yes — N trials match."; zero is "No trials
    match." (the honest yes/no reading of a scalar count)."""
    if total_count <= 0:
        return "No trials match this query."
    return f"Yes — {total_count:,} trial{'s' if total_count != 1 else ''} match this query."


# --- clarification (no-data) envelope (E-13 / P5-INPUT) ---------------------


def build_clarification_envelope(
    *,
    question: str,
    plan: Plan | None = None,
    retrieved_at: str | None = None,
    query_provenance: dict | None = None,
) -> VisualizeResponse:
    """Assemble a ``kind:"clarification"`` envelope (E-13): the request was
    well-formed but named an unresolvable NL referent, so we ASK rather than
    guess. ``question`` is a fixed CODE-owned string (never LLM-authored, carries
    no number — the §1 invariant holds). No visualization / answer / vega, and
    ``status:"empty"`` (nothing was queried); the ``clarification`` kind is what a
    frontend switches on to prompt for the missing detail."""
    filters = dict(plan.filters) if plan is not None else {}
    return VisualizeResponse(
        status="empty",
        kind="clarification",
        visualization=None,
        vega_lite=None,
        answer=None,
        question=question,
        error=None,
        citations={},
        meta=Meta(
            count_basis=None,
            filters=filters,
            query_provenance=query_provenance or {},
            retrieved_at=retrieved_at,
            partial=None,
            notes=[
                "The query referred to an entity it did not name; asking for "
                "clarification rather than guessing (E-13)."
            ],
        ),
    )


# --- envelope assembly ------------------------------------------------------


def _extract_graph(tool_results: list[dict]) -> dict | None:
    """Pull the most recent network graph (``{nodes, edges, ...}``) out of
    ``tool_results`` -- the ``build_network`` tool result carries a ``graph`` key
    (or an inline ``nodes``/``edges`` pair). Returns ``None`` when no graph is
    present (a non-network run)."""
    for result in reversed(tool_results or []):
        if isinstance(result, dict):
            if isinstance(result.get("graph"), dict):
                return result["graph"]
            if "nodes" in result and "edges" in result:
                return result
    return None


def _extract_total(tool_results: list[dict]) -> int:
    """The exact matching total for a ``too_large`` refusal -- either an
    explicit ``total_count`` (a scalar ``count_trials``-style tool result) or,
    failing that, the sum of whatever buckets are present."""
    for result in reversed(tool_results or []):
        if "total_count" in result:
            return result["total_count"]
        if "buckets" in result:
            return sum(b.get("count_trials", 0) for b in result["buckets"])
    return 0


def _extract_single_value(tool_results: list[dict]) -> dict | None:
    """The ``count_trials`` single_value tool-result (CC-7), or ``None``.

    The trunk's single_value executor appends
    ``{"tool":"count_trials","total_count":N,"kind":"visualization"|"answer",
    "citations":[...]}``. The distinguishing marker vs the ``too_large`` count result
    (``{"tool":"count_trials","total_count":N}``, no ``kind``) is the ``kind`` key —
    so this reads the latest result carrying BOTH ``total_count`` and ``kind`` and no
    ``buckets``/``graph`` (a scalar count, not an aggregation)."""
    for result in reversed(tool_results or []):
        if not isinstance(result, dict):
            continue
        if "buckets" in result or "graph" in result:
            continue
        if "total_count" in result and "kind" in result:
            return result
    return None


def _count_basis(buckets: list[dict], distinct_total: int | None = None) -> CountBasis:
    """The dual-count basis (CC-3). ``trials`` is the DISTINCT-trial total
    (``distinct_total`` = the reconciliation anchor / countTotal) when the caller
    knows it -- for an explode field (country / interventionType) Σ bucket
    count_trials is the MEMBERSHIP total (a multi-country trial counts in each
    country's bar), which OVERSTATES the trial count; the distinct total is the
    honest headline. Falls back to Σ buckets only when no distinct total is known
    (e.g. compare, which has no single population)."""
    if distinct_total is not None:
        trials_total = distinct_total
    else:
        trials_total = sum(b.get("count_trials", 0) for b in buckets)
    if any(b.get("count_mentions") is not None for b in buckets):
        mentions_total = sum(b.get("count_mentions", 0) or 0 for b in buckets)
    else:
        mentions_total = None
    return CountBasis(trials=trials_total, mentions=mentions_total)


def _extract_result(tool_results: list[dict]) -> dict | None:
    """The most recent tool-result dict carrying a ``buckets`` list (the row-chart
    payload + its ``distinct_trials`` anchor / tool ``notes``)."""
    for result in reversed(tool_results or []):
        if isinstance(result, dict) and "buckets" in result:
            return result
    return None


def _extract_tool_notes(tool_results: list[dict]) -> list[str]:
    """The disclosures a tool attaches (timeseries gap-fill/planned, geographic
    Other-fold, network synonym-merge/placebo/cap) -- surfaced onto ``meta.notes``
    so a reviewer sees how the data was shaped. Reads the latest result's own
    ``notes`` (or its ``graph.notes`` for a network)."""
    for result in reversed(tool_results or []):
        if not isinstance(result, dict):
            continue
        graph = result.get("graph")
        if isinstance(graph, dict) and graph.get("notes"):
            return [str(note) for note in graph["notes"]]
        if result.get("notes"):
            return [str(note) for note in result["notes"]]
    return []


def _citations_index(buckets: list[dict]) -> dict[str, Citation]:
    """The top-level dedup citation index keyed by nctId (G-4) -- collected
    from each bucket's inline (authoritative) citations list."""
    index: dict[str, Citation] = {}
    for bucket in buckets:
        for citation in bucket.get("citations", []):
            citation_obj = citation if isinstance(citation, Citation) else Citation(**citation)
            index.setdefault(citation_obj.nct_id, citation_obj)
    return index


def _network_citations_index(edges: list) -> dict[str, Citation]:
    """The top-level dedup citation index for a network (G-4) -- collected from
    every edge's two inline endpoint citations (first-seen wins per nctId)."""
    index: dict[str, Citation] = {}
    for edge in edges:
        raw = edge.citations if isinstance(edge, Edge) else edge.get("citations", [])
        for citation in raw:
            citation_obj = citation if isinstance(citation, Citation) else Citation(**citation)
            index.setdefault(citation_obj.nct_id, citation_obj)
    return index


def _build_single_value_envelope(
    *,
    plan: Plan,
    sv: dict,
    status: str,
    kind: str | None,
    filters: dict,
    query_provenance: dict | None,
    retrieved_at: str | None,
    notes: list[str],
    answer: str | None,
) -> VisualizeResponse:
    """Assemble the ``single_value`` envelope (CC-7) from the ``count_trials``
    scalar tool-result ``sv``. The number is ``sv['total_count']``, inserted by
    CODE (G-30/CC-16). ``kind`` is decided by the tool-result's ``kind`` marker
    (an explicit ``kind`` arg wins, then the marker, then a visualization stat
    card): ``"visualization"`` → a stat card; ``"answer"`` → a code-templated
    yes/no. ``vega_lite`` is always None for this path (no natural mark, G-41d)."""
    total = int(sv["total_count"])
    sv_citations = sv.get("citations", [])
    resolved_kind = kind or sv.get("kind") or "visualization"
    citations_index = _citations_index([{"citations": sv_citations}])
    meta = Meta(
        count_basis=CountBasis(trials=total, mentions=None),
        filters=filters,
        query_provenance=query_provenance or {},
        retrieved_at=retrieved_at,
        partial=None,
        notes=notes,
    )
    if resolved_kind == "answer":
        return VisualizeResponse(
            status=status,
            kind="answer",
            visualization=None,
            vega_lite=None,
            answer=answer or _single_value_answer(total),
            error=None,
            citations=citations_index,
            meta=meta,
        )
    return VisualizeResponse(
        status=status,
        kind="visualization",
        visualization=build_single_value_visualization(plan, total, sv_citations),
        vega_lite=None,
        answer=None,
        error=None,
        citations=citations_index,
        meta=meta,
    )


def build_envelope(
    *,
    plan: Plan | None,
    tool_results: list[dict],
    status: str = "ok",
    kind: str | None = None,
    question: str | None = None,
    retrieved_at: str | None = None,
    query_provenance: dict | None = None,
    notes: list[str] | None = None,
    partial: dict | None = None,
    error: dict | None = None,
    answer: str | None = None,
) -> VisualizeResponse:
    """Assemble the response envelope for ``status`` (ARCHITECTURE_SPEC §3.7/§6).

    ``question`` is the user's raw NL question. Callers pass it (``build_spec``
    threads ``state["question"]`` through) but this builder **deliberately discards
    it**: every prose string it emits -- ``title``, the ``too_large`` refusal, the
    ``single_value`` yes/no -- is templated from the validated ``Plan`` and the
    computed numbers, so untrusted user text never reaches a user-facing field
    (G-30). It stays in the signature as the hook a future phrasing pass would use.
    """
    del question
    notes = list(notes or [])
    # Effective, validated filters actually applied (Meta.filters, ARCHITECTURE_SPEC §6).
    # The interventional toggle lives on its own Plan field, not in plan.filters, but it IS
    # a filter applied server-side (filter.advanced=...INTERVENTIONAL on both count and pages)
    # — surface it here so meta.filters honestly reflects the population, not just what was in
    # plan.filters (the raw filter.advanced string stays in meta.query_provenance).
    filters = dict(plan.filters) if plan is not None else {}
    if plan is not None and plan.interventional_only:
        filters.setdefault("interventional_only", True)
    resolved_kind = kind or ("answer" if status in ("too_large", "error") else "visualization")

    if status == "error":
        err = error or {"code": "internal", "message": "unspecified error"}
        return VisualizeResponse(
            status="error",
            kind=resolved_kind,
            visualization=None,
            vega_lite=None,
            answer=None,
            error=ErrorObj(**err),
            citations={},
            meta=Meta(
                count_basis=None,
                filters=filters,
                query_provenance=query_provenance or {},
                retrieved_at=retrieved_at,
                partial=None,
                notes=notes,
            ),
        )

    if status == "too_large":
        total = _extract_total(tool_results)
        templated_answer = answer or (
            f"{total:,} trials match this query -- too large to chart faithfully "
            "within the paging budget. Narrow the query (e.g. add a phase, status, "
            "or year range) to render a distribution."
        )
        return VisualizeResponse(
            status="too_large",
            kind=resolved_kind,
            visualization=None,
            vega_lite=None,
            answer=templated_answer,
            error=None,
            citations={},
            meta=Meta(
                count_basis=CountBasis(trials=total, mentions=None),
                filters=filters,
                query_provenance=query_provenance or {},
                retrieved_at=retrieved_at,
                partial=None,  # refusing to chart is not truncating (G-39)
                notes=notes,
            ),
        )

    # "ok" and "empty" share the same visualization-envelope shape.
    assert plan is not None, "build_envelope: plan is required for status in {'ok', 'empty'}"

    # single_value (CC-7): a scalar count → a stat card (kind:visualization) or a
    # yes/no (kind:answer). The NUMBER is the exact count_trials countTotal, inserted
    # by CODE (G-30/CC-16), never authored. Routed before the row/network logic.
    sv = _extract_single_value(tool_results)
    if plan.query_class == "single_value" and sv is not None:
        return _build_single_value_envelope(
            plan=plan,
            sv=sv,
            status=status,
            kind=kind,
            filters=filters,
            query_provenance=query_provenance,
            retrieved_at=retrieved_at,
            notes=notes,
            answer=answer,
        )

    # Network is a different data shape ({nodes, edges}, not a row list) and a
    # different citation surface (per-edge, two field_paths) -- route it to its own
    # builder. A DEGENERATE network (≤1 node / no co-occurrence, G-41e) instead
    # carries a `network_fallback` row-bucket result: render it as a BAR of individual
    # drug frequencies, NOT a network. Everything else is a row chart.
    result = _extract_result(tool_results)
    is_network_fallback = bool(result and result.get("network_fallback"))
    graph = (
        _extract_graph(tool_results)
        if plan.chart_type is ChartType.NETWORK_GRAPH and not is_network_fallback
        else None
    )
    if graph is not None:
        viz = build_network_visualization(plan, graph)
        citations = _network_citations_index(graph.get("edges", []))
        count_basis = CountBasis(trials=int(graph.get("distinct_trials", 0)), mentions=None)
    else:
        buckets = result.get("buckets", []) if result else []
        distinct_total = result.get("distinct_trials") if result else None
        # A degenerate network falls back to a BAR: override chart_type (BAR is the
        # network recipe's degeneracy_fallback) so the viz type<->data invariant holds
        # (a NETWORK_GRAPH type with row data would fail the schema validator), and
        # give it a bar-appropriate title + an explicit "too sparse to graph" note.
        eff_plan = plan.model_copy(update={"chart_type": ChartType.BAR}) if is_network_fallback else plan
        viz = build_visualization(eff_plan, buckets)
        if is_network_fallback:
            viz = viz.model_copy(update={"title": f"Most-studied drugs in {_entity_phrase(plan)} trials"})
            notes = [
                *notes,
                "Network too sparse to graph (≤1 entity or no repeated co-occurrence); "
                "showing individual drug frequencies instead (G-41e).",
            ]
        citations = _citations_index(buckets)
        count_basis = _count_basis(buckets, distinct_total=distinct_total)
        if status == "empty":
            notes = [*notes, "No trials matched this query."]

    # Surface the tool's own disclosures (gap-fill / planned / Other-fold / network
    # synonym-merge / placebo-drop / cap) onto meta.notes.
    notes = [*notes, *_extract_tool_notes(tool_results)]

    vega = to_vega_lite(viz)

    # Time-series discloses which of the 5 date fields it binned (CC-4) and the grain.
    date_field_used = plan.date_field if plan.query_class == "timeseries" else None
    time_granularity = plan.grain if plan.query_class == "timeseries" else None

    return VisualizeResponse(
        status=status,
        kind=kind or "visualization",
        visualization=viz,
        vega_lite=vega,
        answer=None,
        error=None,
        citations=citations,
        meta=Meta(
            count_basis=count_basis,
            date_field_used=date_field_used,
            time_granularity=time_granularity,
            filters=filters,
            query_provenance=query_provenance or {},
            retrieved_at=retrieved_at,
            partial=Partial(**partial) if partial else None,
            notes=notes,
        ),
    )
