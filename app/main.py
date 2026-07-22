"""FastAPI surface for the ClinicalTrials.gov query-to-visualization agent.

Two endpoints (ARCHITECTURE_SPEC §6):
  - POST /visualize         — sync; runs the LangGraph pipeline, returns the response envelope.
  - POST /visualize/stream  — SSE; emits a fixed enum of high-level status events, then the
                              terminal event carries the full envelope (never raw model reasoning).
Plus GET /healthz for liveness (DB-free — this service holds no persistent state), and a
static /demo mount that appears only when the `demo/` directory ships alongside the app — a
local dev affordance for the bundled citation viewer, not part of the service contract.

The pipeline is fully live: the LLM planner + reviewers run through the provider-agnostic adapter
(``LLM_PROVIDER``; unset ⇒ the zero-network StubAdapter), ``execute`` dispatches the six live
deterministic classes against ClinicalTrials.gov, and both endpoints return the same
schema-valid, cited, reconciled envelope. The SSE stream emits the fixed 8-member status enum
(no private model reasoning) and the terminal event always carries the full envelope.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app import config
from app.api.schemas import VisualizeRequest, VisualizeResponse
from app.graph.build import build_graph, initial_state, run_sync
from app.logging_setup import configure_logging, log_event

logger = logging.getLogger(__name__)

# Attach the redaction filter + structured-event logging at import (idempotent, §A(i)):
# the provider key + raw user query can never reach the logs at info level (the
# RedactionFilter backstop + log_event's field allowlist enforce this by construction).
configure_logging()

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    version="0.1.0",
    summary="A deterministic visualization engine orchestrated by a ReAct planner routing to validated recipes.",
)

# ---- Local demo surface (a dev affordance; not part of the service contract) -------------------
# The bundled citation viewer (`demo/viewer.html`) can drive this API live. Serving it FROM the app
# makes the viewer same-origin with `/visualize`, so the browser needs no CORS at all — that is the
# intended path (`http://localhost:8000/demo/viewer.html`). The narrow rule below covers only the
# other way an operator opens it: straight off the filesystem, whose `Origin` is the literal
# string "null". Both are scoped to a local operator — the regex admits loopback and `null` and
# nothing else, and the service holds no cookies, no auth and no persistent state, so credentials
# stay off. Gating on the directory's presence means a slim packaged deploy (no `demo/`) gets
# neither the mount nor the CORS rule: fail-closed in production, convenient locally.
# File-serving only: `demo/` holds exactly one file (`viewer.html`) and no `index.html`, so
# StaticFiles' `html=True` directory-index mode would have nothing to serve — `/demo` and
# `/demo/` 404 either way, and only `/demo/viewer.html` resolves. The flag is left off rather
# than carried as decoration.
_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if _DEMO_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=_DEMO_DIR), name="demo")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
        allow_methods=["GET", "POST"],  # GET is the viewer's /healthz capability probe
        allow_headers=["content-type", "accept"],
        allow_credentials=False,
    )

# The fixed, high-level SSE status enum (ARCHITECTURE_SPEC §3.9) — the 8-member CONTRACT a
# client may see. Never token-level reasoning / private chain-of-thought. `_NODE_TO_STATUS`
# below is the runtime mapping that actually emits these; this tuple is the published
# contract, and the two are tied together by an import-time equality check (see under the
# mapping) so neither can drift silently. `demo/viewer.html` keeps its own JS copy of these
# labels for display — that third copy is a client, not a source of truth, and is not
# machine-checked against this one.
#
# ORDERING NOTE (surfaced deviation): §3.9's prose lists `plan_approved` before `validating`,
# but the pipeline order is `check` (validating) → `review_intent` (plan_approved), so we emit
# in true EXECUTION order — planning → validating → plan_approved → fetching → aggregating →
# building_spec → verifying → done — which is what a client actually observes.
SSE_STATUS_ENUM = (
    "planning",
    "validating",
    "plan_approved",
    "fetching",
    "aggregating",
    "building_spec",
    "verifying",
    "done",
)

# Graph node → the status event(s) surfaced to the client (a node may map to more than one).
# `execute` is the single deterministic runner that both pages and aggregates (§3.6 — one
# serial step), so it surfaces the fused pair `fetching` then `aggregating`. Nodes not mapped
# emit nothing.
#
# `plan_approved` marks the intent-review STAGE completing (§3.9's fixed enum), not that the
# Intent Reviewer necessarily approved: on the best-effort path (revise ∧ budget exhausted, §B.5)
# the stage still fires. The authoritative outcome is always the terminal envelope — which
# carries the "shipped best-effort" `meta.notes` caveat in exactly that case.
_NODE_TO_STATUS: dict[str, tuple[str, ...]] = {
    "plan": ("planning",),
    "check": ("validating",),
    "review_intent": ("plan_approved",),
    "execute": ("fetching", "aggregating"),
    "build_spec": ("building_spec",),
    "review_output": ("verifying",),
    "respond": ("done",),
    "error": ("done",),
}

# Contract check, run once at import: the set of statuses the mapping can emit must be exactly
# the published enum. Catches both drift directions — a node emitting an event no client was
# told about, and an enum member no node can ever produce. Written as a plain `if`/raise rather
# than an `assert` so it survives `python -O`; a mismatch is a startup failure, not a warning,
# because the SSE enum is a client-facing contract.
_MAPPED_STATUSES = {status for statuses in _NODE_TO_STATUS.values() for status in statuses}
if _MAPPED_STATUSES != set(SSE_STATUS_ENUM):
    raise RuntimeError(
        "SSE status contract drift: _NODE_TO_STATUS emits "
        f"{sorted(_MAPPED_STATUSES)} but SSE_STATUS_ENUM declares {sorted(SSE_STATUS_ENUM)}"
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness. Deliberately DB-free and dependency-free so it can't be gated by a slow backend."""
    return {"status": "ok"}


@app.post("/visualize", response_model=VisualizeResponse)
def visualize(request: VisualizeRequest) -> VisualizeResponse:
    """Synchronous: run the full pipeline and return the response envelope.

    Pydantic validates the request at this boundary (a bad request is a 422 before the graph runs);
    the graph models its own failures as an `error` envelope, never a raised exception.
    """
    response = run_sync(request)
    # Structured event only — the DECIDED shape, never the raw query (§A(i)/SEC-47). Both keys
    # are closed enums and both are on `log_event`'s allowlist, so the emitted line is exactly
    # {"event":"visualize_complete","status":...,"kind":...}.
    log_event(logger, "visualize_complete", status=response.status, kind=response.kind)
    return response


@app.post("/visualize/stream")
async def visualize_stream(request: VisualizeRequest) -> EventSourceResponse:
    """SSE: stream high-level status events, then a terminal event carrying the full envelope.

    Runs the graph exactly ONCE, via multi-mode streaming (``stream_mode=["updates", "values"]``):
    the "updates" chunks drive the status events, and the last "values" chunk's ``spec`` is the
    same terminal envelope the status events describe -- there is no second, independent
    ``run_sync``/``graph.invoke`` call, so the streamed status and the terminal result can never
    diverge (they come from one execution).

    The terminal event always carries the envelope — so empty/too_large/error/answer outcomes
    stream too, not only a viz spec (G-16). A mid-stream failure still ends in a terminal event,
    with a fixed, generic message -- the real exception is logged server-side only and never
    echoed to the client (it may embed upstream URLs/params, §A(e)/§3.11).
    """

    async def event_generator():
        seen: set[str] = set()
        final_spec: VisualizeResponse | None = None
        try:
            graph = build_graph()
            # SSE gets the longer 90s wall-clock deadline (§B.4) — status events keep the
            # client informed, so a longer serial-paging window is acceptable.
            stream_state = initial_state(request, deadline_seconds=config.WALL_CLOCK_SSE_SECONDS)
            for mode, chunk in graph.stream(stream_state, stream_mode=["updates", "values"]):
                if mode == "updates":
                    for node_name in chunk:
                        for status in _NODE_TO_STATUS.get(node_name, ()):
                            if status not in seen:
                                seen.add(status)
                                yield {"event": "status", "data": status}
                elif mode == "values":
                    spec = chunk.get("spec")
                    if spec is not None:
                        final_spec = spec  # last "values" chunk with a spec = the terminal envelope

            if final_spec is None:
                # Defensive only: every graph path (respond/error) sets `spec` before END.
                final_spec = VisualizeResponse(
                    status="error",
                    kind="answer",
                    error={"code": "stream_error", "message": "internal error while streaming"},
                    meta={"notes": ["stream failed"]},
                )
            yield {"event": "result", "data": final_spec.model_dump_json()}
        except Exception:  # never hang — surface a terminal error event, never leak `exc` details
            logger.exception("SSE stream failed")
            err = VisualizeResponse(
                status="error",
                kind="answer",
                error={"code": "stream_error", "message": "internal error while streaming"},
                meta={"notes": ["stream failed"]},
            )
            yield {"event": "result", "data": err.model_dump_json()}

    return EventSourceResponse(event_generator())


# A tiny module-level self-check hook the doctor script imports (keeps import side effects out).
def _selfcheck_payload() -> dict:
    """Return a canned request payload the doctor can POST through the app."""
    return json.loads(
        VisualizeRequest(
            query="Phase distribution of interventional pancreatic cancer trials",
            condition="pancreatic cancer",
            interventional_only=True,
        ).model_dump_json()
    )
