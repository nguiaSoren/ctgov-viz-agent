# Architecture Spec — ClinicalTrials.gov Query-to-Visualization Agent

This is the implementation spec: how the agent reasons, what it is built on, and what
guarantees it makes. `ARCHITECTURE.md` is the short narrative version; this is the long
one, and §5 turns a conflict-and-ambiguity lens back on the architecture's own choices.

**Notation.** `CC-N`, `G-N`, `E-N`, `A-N`, `SEC-N` and `API-N` are stable decision IDs
from the project's working design record — respectively: behavioural rulings, runtime
guards, edge cases, requirements, security controls and API-contract items. Those
working documents are not part of the published repository, so an ID here identifies a
decision without linking to it. The decision itself is always stated inline where it
matters, and the code is authoritative in every case; blocks marked **SUPERSEDED BY
CODE** record where the shipped implementation diverged from this document's original
intent, kept rather than silently edited so the drift stays visible.

---

## 1. The invariant (the spine of the whole design)

> **The LLM decides *what* to compute; deterministic tools compute it. The model never emits a number.**

Every component below exists to hold this line. It is the architectural answer to avoiding hallucination-prone steps: the language model is structurally incapable of producing a quantitative result — it can only *request* a deterministic operation, and every number shown to the user is computed by code. That is a guarantee you can state in one sentence and defend line by line, which is worth more than any amount of "we prompt it not to."

**The invariant covers text fields too, not just data cells (G-30).** The user-facing text fields the LLM could otherwise author — `title`, the scalar `answer`, `meta.notes` — are the leak the invariant must also seal: **`title` and the scalar `answer` are code-templated** from the validated Plan + computed values (any digit in them is inserted by code from the reconciled `data`, never typed by the model), and **`meta.notes` is either code-generated or run through a deterministic post-check that rejects any digit not present in the computed `data`.** So "the model never emits a number" is mechanical everywhere a number could reach the user, not only in the `data` array.

---

## 2. Architecture at a glance

```
User request  (query + optional structured fields)
      │
      ▼
[ Planner · ReAct loop ]   reason → act → observe over high-level tools (bounded; usually 1 turn / 1 tool)
      │   the LLM loop lives HERE and emits a typed Plan {query_class, entities, filters, date_field, chart_type, alternates}
      ▼
[ Plan Checker · CODE ]  valid tokens / real field paths / sane ranges / supported type / whitelisted areas
      │   runs first (free); reject → escalate to re-plan with a precise machine reason
      ▼
[ Intent Reviewer · LLM ]  intent check on a mechanically-valid Plan — right metric / dimension / date-intent / apt chart?
      │   approve  │  revise → bounded escalation (n ≤ 1)
      ▼
[ Execute · deterministic runner ]  page → count → dedupe → bucket → cite   (NO LLM — the loop already closed)
      │   the LLM never touches raw records or tallies anything
      ▼
[ Viz-spec builder ]  custom canonical schema  (+ Vega-Lite projection for standard charts)
      │
      ▼
[ Output Reviewer ]  CODE: excerpt = real substring @ field_path · Σbuckets == countTotal   +   LLM: does it answer the question?
      │   approve  │  flag → annotate meta.notes (never re-runs aggregation)
      ▼
Response  (terminal SSE event, or sync JSON)

── throughout: SSE emits a fixed enum of high-level STATUS events (never raw reasoning) ──
── orchestrated as a LangGraph graph · LLM behind a provider-agnostic adapter · stateless + response cache ──
```

**The components, named (in execution order).** Every box in the diagram is one of these; nothing else touches the pipeline. The two gate families never blur: a **Checker** is deterministic *code* (mechanical legality); a **Reviewer** applies *LLM* judgment (semantic). There is exactly **one Checker and two Reviewers.**

| # | Component | Code / LLM | What it does | Runs |
|---|---|---|---|---|
| 1 | **Planner** (§3.2) | LLM (ReAct loop) | interprets the question, classifies it to a query-class recipe, fills the recipe's slots → a typed **Plan** | first |
| 2 | **Plan Checker** (§3.3) | **code** | mechanically checks the Plan's arguments are *legal* (real tokens / fields / ranges / chart type) — the anti-hallucination gate. "Are the arguments legal?" | after Planner |
| 3 | **Intent Reviewer** (§3.4) | LLM | judges whether the Plan *captures the user's intent* (right metric / dimension / date / apt chart). "Did the planner understand the question?" | after Plan Checker |
| 4 | **Executor** (§3.6) | **code** | runs the validated Plan's tools: page → count → dedupe → bucket → cite. No LLM here | after Intent Reviewer |
| 5 | **Viz-spec builder** (§3.7) | **code** | assembles the custom canonical spec (+ Vega-Lite projection) | after Executor |
| 6 | **Output Reviewer** (§3.8) | **code + LLM** | *code:* every citation excerpt is a real substring + counts reconcile; *LLM:* does the result faithfully answer? | last |

**Chosen stack (decided this session):** ReAct planner · LangGraph orchestration · two LLM reviewers (plan-time + output-time) · provider-agnostic LLM adapter · high-level deterministic task tools · custom canonical viz schema + Vega-Lite projection · SSE streaming of structured status events · stateless core + response cache. Stretch: MCP server, conversational memory.

**Where this lands (control × autonomy).** Topology is **cyclic** — a DAG spine plus two back-edges (the ReAct self-loop in `plan`, and the bounded escalation edge), not a DAG. Control is **Orchestrated**: a LangGraph graph routes from a known menu of nodes and tools; no agent-to-agent debate, no dynamic agent creation → not *emergent*. Autonomy is **Adaptive** — and the evidence is **runtime tool-choice + retry + early-stop** (§8's exact definition of Adaptive), which qualifies *on its own*. The bounded re-plan is **not** re-planning in the Self-directed sense; it is an **escalation**: n ≤ 1, externally gate-triggered (a checker/reviewer reject or zero-results, never planner-initiated), and structure-preserving (same plan-space, same 7-tool menu — it invents no new subtask, tool, or agent). That hard gate is what keeps us left of *self-directed*. In one line: **a controlled orchestration — cyclic and adaptive, not an emergent society.**

> **ANNOTATION (2026-07-22) — two things to settle before this is said aloud.** (a) *Topology:* "a DAG spine plus **two** back-edges (the ReAct self-loop in `plan`, and the bounded escalation edge)" is one back-edge too many — there is no self-loop in `plan` (see the §3.2 / §3.12 annotations). The shipped topology is **a DAG spine plus ONE back-edge** (the shared, ≤1 escalation into `plan`). Everything else in this paragraph survives that correction: it is still cyclic, still Orchestrated, and the Adaptive claim rests on runtime retry/early-stop/escalation, which all shipped. `app/graph/build.py:19-22` carries the same inherited "two cycles" framing. (b) *Provenance of the lens:* the control × autonomy grid is cited here as "§8", which is `AGENT_STUDY_NOTES.md` §8 — **the only substantive section of that file with no `*Refs:*` line**. Decide before being asked "whose taxonomy is that?"; presenting it as your own synthesis is the stronger and, on the evidence, the accurate answer.

**Headline (DECIDED — README).** Lead with the determinism, not the agent: *"A deterministic visualization engine orchestrated by a ReAct planner. The planner interprets the natural-language request and selects a validated query recipe; deterministic tools perform retrieval, aggregation, and spec generation, while a schema checker and two reviewers ensure correctness before and after execution. The LLM never computes a statistic."* This frames the LLM as intelligence-for-orchestration and code as correctness — a stronger engineering story than "we built an AI agent," and the accurate description of what was built.

---

## 3. Components (each a mini-spec)

### 3.1 LLM adapter — provider-agnostic
An interface (`propose`, `verify`, structured-output/tool-calling) that hides provider differences (tool-call vs tool_use shapes, `tool_choice` modes, result-message envelopes). Default implementation: Claude; swappable to OpenAI/open models without touching the graph. Each node picks its model through the adapter, so the planner can run a strong model and the reviewers a cheaper one later, with zero rewiring.
**Rationale:** the tool-calling contract is conceptually identical across providers but not portable in the wire shape; the adapter is where that non-portability is absorbed. Also the honest realization of "model-agnosticism."

> **SUPERSEDED BY CODE (2026-07-22):** there is no compiled-in "default: Claude". `get_adapter` is env-driven (`LLM_PROVIDER`) and falls back to `StubAdapter` when no provider/key is configured — see `app/llm/adapter.py:791-800`. Two real adapters shipped (`OpenAIAdapter` at `app/llm/adapter.py:374`, `AnthropicAdapter` at `:574`); the shipped example ladder was run OpenAI-primary with one Anthropic twin. The Anthropic model constants are `claude-opus-4-8` / `claude-haiku-4-5` (`app/llm/adapter.py:587-588`), not Sonnet.
> Original intent kept above for the record.

### 3.2 Planner — ReAct agent (the LLM loop lives here)
Input: the NL `query` plus optional structured fields, **merged by the CC-1 precedence rule** (fields authoritative on the dimension they name; query supplies intent and fills gaps; conflict → field wins, echoed in `meta.notes`). The planner runs the `reason → act → observe` loop — **this is the only LLM loop in the system**; `execute` (§3.6) is a deterministic runner, not a loop. The loop is **bounded** (§B) and usually terminates in **one turn / one tool**. The planner's job shrinks to two moves: **classify → `query_class`**, then **fill that recipe's slots** (entities, filters, `date_field` per intent CC-4, `chart_type` from the recipe's allowed alternates CC-8) — see the **recipe/skill registry** (§B.6). It composes **high-level tools** (§3.5) and receives back **Observation objects** — computed, bounded summaries (`{tool, ok, error_code?, total_count?, buckets_preview?, truncated?, date_field_used?}`), **never raw records, never stack traces** — which is what holds the "LLM never counts" invariant *through* the loop. It emits a typed **Plan** and a `finalize` stop signal.

> **SUPERSEDED BY CODE (2026-07-22):** the shipped planner is **single-shot classify→fill**, not a tool-calling ReAct loop. `plan_request` makes exactly ONE `adapter.propose(...)` call with `tools=None` and a closed `PlannerOutput` schema — see `app/llm/planner.py:441-453` (the `tools=None` literal is at `:446`). Consequences: (a) there is **no `finalize` field** on `PlannerOutput` (`app/llm/planner.py:129-154`); (b) the `Observation` model was written (`app/plan/models.py:96`) but is **instantiated nowhere** in `app/`, `tests/` or `scripts/` — re-plan feedback is a plain string threaded into the prompt (`app/graph/nodes.py:319`); (c) the model never sees an action space — dispatch is a hand-written `if/elif` on `plan.query_class` (`app/graph/nodes.py:557-568`). The reason→act→observe *cycle* is still real at the GRAPH level (execute makes live calls; a zero-result observation re-enters `plan` for one bounded escalation), but the loop is **not internal to the planner node** and the tool menu is not in the model's context.
> Original intent kept above for the record.

### 3.3 Plan Checker — mechanical validation (CODE, runs first)
Code, not a prompt. Runs **before any LLM review and before any API call**. Rejects/normalizes the Plan against known constraints: phase strings → valid tokens (`PHASE1`…`PHASE4`/`EARLY_PHASE1`/`NA`, combined → array), fields exist, `start_year ≤ end_year`, chart_type ∈ supported enum, query areas whitelisted. This is the real anti-hallucination gate — the planner *cannot* invent a capability the tool set doesn't expose, and cannot pass an invalid argument through. On reject → escalate to one bounded re-plan with a precise machine reason. It answers **"are the arguments legal?"** — a mechanical question.

> **SUPERSEDED BY CODE (2026-07-22):** the shipped checker **rejects only — it normalizes nothing, and it does NOT check `start_year ≤ end_year`.**
> - *No normalization.* `check_plan` returns the identical object it was handed: `CheckResult(ok=True, normalized_plan=plan)` — `app/plan/checker.py:216`. Verified by running it: `check_plan(p).normalized_plan is p` → `True`. Phase-string normalization really lives **upstream in the planner** (`_normalize_field` at `app/llm/planner.py:71`, `normalize_trial_phase` inside `_apply_field_precedence` at `:350`); a human phase string that reaches the checker is *rejected*, not fixed (verified: `filters={"phase": "phase 1"}` → `ok=False, invalid_filter_token`).
> - *No year-ordering check.* `_check_filter_tokens` hits `token_set is None → continue` for `start_year`/`end_year` (`app/plan/checker.py:118-123`) and never compares them. Verified: `filters={"start_year": 2020, "end_year": 2010}` → `ok=True`. Ordering is enforced one layer out, on `VisualizeRequest` (`app/api/schemas.py:402-422`), and the `[1900, 2100]` range only at `build_search_params` (`_validate_year`, `app/ctgov/params.py:47-58`), which **raises** and becomes a redacted `upstream_error` rather than a clean re-plan. An inverted range arriving from the PLANNER (not the request) is therefore unchecked.
> Original intent kept above for the record.

### 3.4 Intent Reviewer — plan-time intent review (LLM)
Runs **after the Plan Checker** (§3.3), on a mechanically-valid Plan only — so it never spends an LLM call on a plan with invented tokens. Reads the original question and the Plan, and judges whether the plan *captures the user's intent*: right metric / right dimension mapping / right date-intent / chart *apt* (not merely legal)? — a bounded checklist (§B.3). It answers **"did the planner understand the question?"** — a semantic judgment, which is why it is an LLM and not code. Output `approve` or `revise + reason` (one bounded escalation). Cheap: no API calls, on a cheap model, one forced-structured call — and **skippable** on cache-hits and fully-structured-field plans (no NL parse happened → no misread to catch).

> **SUPERSEDED BY CODE (2026-07-22):** only the **fully-structured-field** skip shipped (`should_skip_intent_review`, `app/llm/reviewers.py:167`). The **cache-hit skip does not exist**: the response cache is consulted *inside* `execute` (`app/graph/nodes.py:484`), which runs AFTER `review_intent`, so a cache hit saves the aggregation calls but never the Intent-Reviewer LLM call. Also note the skip's docstring under-states its reach — it stops re-guarding **filters** too (Intent-Reviewer checklist item 5, `app/llm/reviewers.py:76-77`), and the Plan Checker only proves a filter token is *legal*, never that it was *asked for*.
> Original intent kept above for the record.

### 3.5 High-level task tools — the scoped surface
The tools the ReAct planner composes. **Each performs its full deterministic job internally** (paging, counting, dedupe, sort, Unknown-bucketing, dual counts, citations) and returns *computed* results. The LLM chooses *which* to call with *what* arguments; it never pages or tallies.

**Why this surface — and not a raw API/shell tool.** Scoped, purpose-built tools are safer to sandbox, easier to validate, and faster for the model to call correctly than a raw HTTP/shell escape hatch. The visible surface is deliberately **small** — the ~7 tools below, only what the five query classes need — which keeps the planner's context lean and cuts the wrong-tool-call rate (an exposed tool costs prompt tokens and a distraction whether it's used or not). Every argument is typed and validated **at the tool boundary**, so the security/validation layer lives in code, never in a prompt instruction. And the whole surface is **read-only**: no write, delete, or destructive operation exists anywhere, so the read/write privilege boundary is absolute by construction (see §4).

| Tool | Does (deterministically, inside) | Returns |
|---|---|---|
| `count_trials(query, filters)` | one `countTotal=true` call | exact total (the `trial_count` primitive + correctness oracle) |
| `aggregate_by(query, filters, field)` | page under budget → bucket by `field` → Unknown bucket (CC-5) → dual counts (CC-3) → combined-phase own bucket (CC-15) → citations (CC-9) | buckets `[{value,label,count_trials,count_mentions,citations…}]` |
| `timeseries(query, filters, date_field, grain)` | page → bin by chosen date (CC-4) → genuine-future dates → flagged "planned" bucket (not clamped, G-40) / normalize precision → fill gap periods | ordered points + `date_field` used |
| `compare(query_a, query_b, field)` | two aggregations → union categories + fill 0 → % within series (CC-14) | two labeled series |
| `build_network(query, kind, entity_a, entity_b)` | page → extract entities → edges = trial-pairs (CC-3) → synonym-merge → drop placebo → top-N + min-edge-weight (CC-12) → per-edge nctIds | `{nodes, edges}` |
| `get_trial(nctId)` | one record fetch | the raw record (for citation excerpt / drill-down) |
| `resolve_entity(name, kind)` | light alias canonicalization; API resolves search-recall synonyms | canonical label(s) |

> **SUPERSEDED BY CODE (2026-07-22):** the shipped surface is **8 registered, 6 live, 2 documented stubs** — not "~7". `TOOL_REGISTRY` (`app/ctgov/tools.py:674-683`) holds `count_trials · aggregate_by · timeseries · compare · build_network · study_duration_histogram · get_trial · resolve_entity`; `study_duration_histogram` is the G-20/G-28 addition this table predates. `get_trial` (`app/ctgov/tools.py:640`) and `resolve_entity` (`:661`) still `raise NotImplementedError`, so §A(c)'s "Planner via `get_trial`" touch point is currently unreachable. A 9th module-level callable, `aggregate_by_counts` (`app/ctgov/tools.py:365`, the exact-at-scale count path), is not in the registry at all. **Say "eight registered, six live, two documented stubs" — never "7".** Separately: `TOOL_REGISTRY` has **zero consumers in `app/`** (tests only) — the live dispatch is the `if/elif` at `app/graph/nodes.py:557-568`, so the registry is a contract + test fixture, not the routing table.
> Original intent kept above for the record.

### 3.6 Aggregation + citation core (deterministic)
The engine beneath the tools. Owns: paging under a **page budget** (CC-6, `countTotal` first), client-side aggregation, **dual counts** (distinct-trial + trial×value, CC-3), explicit **Unknown/NA buckets** (CC-5), **combined-phase own bucket** (CC-15), **country dedupe per trial** (CC-13), **`countTotal` reconciliation** (CC-16), and **citations** (CC-9: exact `contributing_count` + deterministic capped sample + `field_path` for round-trip; excerpts string-extracted, never LLM-authored; derived values labeled and citing members). This is where correctness lives and where every number is born.

**Concurrency (DECIDED: serial in-build; seam documented).** Within a query, paging is serial (cursor pagination forces it). Across independent operations — the two sides of a `compare`, several facets — the implementation runs **serial in this build** (deterministic, simplest, politeness-safe), and the tool interface is deliberately shaped so those independent branches **can** be scheduled concurrently under the same global rate limiter in a later iteration. That seam is documented, not built — the right 24h trade: concurrency is an optimization that doesn't reinforce the correctness story. (The adapter still normalizes any parallel `tool_use` blocks the planner emits, but correctness never depends on provider parallelism.)

### 3.7 Viz-spec builder
Produces the **custom canonical schema** (CC-10): discriminated-union `data` on `type` (`rows[]` for charts, `{nodes,edges}` for network), `{value,label}` pairs, `status` envelope, dual counts, rich `encoding` channels, per-datum citations. **Plus a Vega-Lite projection** for the standard chart types (bar / grouped bar / line / histogram / scatter) as a frontend convenience. The custom schema is the source of truth (it carries citations, dual counts, and the network graph, which Vega-Lite cannot); the Vega-Lite block is a derived view for standard charts only. **The builder — not the LLM — writes the user-facing text fields (G-30):** `title` and the scalar `answer` are **code-templated** from the validated Plan + the reconciled `data` (any digit in them is inserted by code from a computed value, never authored by the model), so the invariant ("the model never emits a number") holds for prose fields too, not only for the `data` array.

### 3.8 Output Reviewer — output-time faithfulness + provenance (CODE + LLM)
Runs *after* the spec is built, on already-computed data — and is **split**, because the provenance guarantee must not rest on an LLM: Output Reviewer reads registry free-text (the indirect-prompt-injection vector, §A). **Deterministic pre-checks (code, always, load-bearing):** every citation excerpt is a **real substring at its `field_path`** — walking each datum's **inline `citations[]` list** (rows AND the two entries per edge, G-34), not a top-level map; **reconciliation is gated** — it runs ONLY when `status=="ok"` AND `data.type=="rows"` (network / answer / too_large / empty are reconciliation-**exempt**, G-32) — and its selector is the recipe's per-field **combine/explode mode** (G-26): `combine` → Σbuckets == `countTotal`, `explode` → distinct-nctId == `countTotal` (so phase, though multi-valued, reconciles as Σ because it's `combine`); `meta.partial` present iff truncated; every datum has ≥1 citation or a `derived`+members label — **except a legitimately-empty `count==0` datum** (fill-0 bucket / gap year), which is exempt (G-35). An instruction injected into a `briefSummary` cannot make a fabricated excerpt pass a substring check. **LLM check (one bounded call, secondary, non-generative):** does the spec faithfully answer the question, and is the encoding an apt rendering of the computed data? Output `approve | flag{reason}`; it inspects computed output only, so it cannot introduce a number. On `flag`: annotate `meta.notes` and ship — it **never** re-runs aggregation (the numbers are correct by construction; only interpretation is in question). That split makes provenance a first-class, *tamper-evident* gate rather than a claim.

> **SUPERSEDED BY CODE (2026-07-22) — the code is STRONGER than this spec on two axes, and weaker on one wording:**
> - *Reconciliation is not mode-aware.* The shipped precheck anchors on **`distinct_trials` for BOTH modes** (`app/viz/review.py:229-238`) — that IS the CC-16 claim — and then adds a **separate combine-only bar-sum check** (`app/viz/review.py:251-261`, reason `bar_sum_mismatch`). So combine proves *two* things (anchor==countTotal AND Σbars==anchor) where §3.8 asked for one. Motivation is LESSON K3/L1 (`tasks/LESSONS.md:166,216`).
> - *There is a drift tolerance the spec never mentions.* A gap ≤0.5% AND ≤20 trials is disclosed in `meta.notes` rather than hard-failing (`app/viz/review.py:241-248`, decision P1-DRIFT).
> - *"every citation excerpt is a real substring at its `field_path`" is now the wrong field name.* The reviewer verifies **`citation.matched_value`** (`app/viz/review.py:109`) and `matched_tokens` (`:111`); the field literally named `excerpt` (the trial's `briefTitle`) is **never** verified by the reviewer. See the §6 citation-shape annotation below for the rename.
> - *The precheck proves FIVE things, not four* (the module docstring's stale "four" is a shipped-code defect, tracked separately): `citation_invalid · reconciliation_failed · reconciliation_unavailable · partial_inconsistent · uncited_datum · bar_sum_mismatch` are all reachable reason codes.
> Original intent kept above for the record.

### 3.9 Transport — SSE streaming
`POST /visualize/stream` emits a **fixed enum of high-level status events** — `planning → plan_approved → validating → fetching → aggregating → building_spec → verifying → done` — surfacing the planner/reviewer separation (legibility → the anti-hallucination signal) **without streaming the model's private reasoning**. The **terminal event carries the full viz spec**, so the stream is frontend-consumable end-to-end. A plain `POST /visualize` sync endpoint returns the same final envelope for non-streaming clients.

> **SUPERSEDED BY CODE (2026-07-22) — deliberate, and flagged in the code itself.** The emitted order is `planning → validating → plan_approved → …` (`app/main.py:73-82`), because the pipeline runs `check` (→ `validating`) before `review_intent` (→ `plan_approved`). The code documents the deviation in place at `app/main.py:68-72`, so **the spec's ordering is stale, not the code**. Two further notes: `plan_approved` marks the *stage completing*, not that the reviewer approved; and `SSE_STATUS_ENUM` itself has **zero readers** — the runtime truth is `_NODE_TO_STATUS` (`app/main.py:93-102`) and `demo/viewer.html` keeps a third JS copy, with no test tying the three together.
> Original intent kept above for the record.

### 3.10 State + cache
Stateless per request. A short-TTL response cache keys on the normalized plan (helps repeat queries, the demo, and rate-limit politeness ~3 req/s per the API brief). LangGraph state is per-request only. **Conversational memory is a stretch** (§8).

### 3.11 MCP server (stretch)
A thin, **read-only** MCP server exposing the query tools (`count_trials` / `aggregate_by` / `get_trial`) so an MCP client (Claude Desktop / Cursor) can explore the registry directly. Bounded, no write path, built only if the core lands with time to spare.

> **NOT BUILT (2026-07-22):** `app/mcp/__init__.py` is a **0-byte placeholder** and `app/mcp/server.py` does not exist. The stretch was never started; it is disclosed in the shipped README's Limitations. **Never say "we expose an MCP server."** The same applies to §3.10's conversational memory (the checkpointer is OFF by design and no memory layer was written) and to §B.8's function-call-accuracy eval (also not built — see the §B.8 annotation).
> Original intent kept above for the record.

### 3.12 LangGraph orchestration
The pipeline is a graph. **Nodes:** `plan` → `check` → `review_intent` → `execute` → `build_spec` → `review_output` → `respond` (+ a terminal `error` node). The **ReAct loop lives in `plan`** (§3.2); **`execute` is the deterministic tool-execution runner** over the validated plan — *not* an LLM loop. **Cycles:** the ReAct self-loop inside `plan`, plus one bounded **escalation** back-edge into `plan` (checker-reject / Intent-Reviewer-reject / zero-results, n ≤ 1); `review_output` **never** loops. Errors are modeled as **edges** (route to `error`), not raised exceptions. Checkpointing is **OFF** (`thread_id = request_id`, nothing persists → stateless + horizontally scalable; the conversational-memory stretch is exactly "flip a checkpointer on"). SSE events stream from node transitions, not checkpoint reads. The full node list, the conditional-edge routing table, and the state reader/writer matrix are in **§B.5**.

> **SUPERSEDED BY CODE (2026-07-22):** **there is no ReAct self-loop inside `plan`.** `plan` calls `plan_request` exactly once (`app/graph/nodes.py:254`), which calls `adapter.propose` exactly once with `tools=None` (`app/llm/planner.py:441-453`). The only genuine sub-loop anywhere in the LLM layer is the adapter's bounded **schema-repair re-ask** (`app/llm/adapter.py:504`, `:683`). So the graph has exactly ONE cycle — the shared, ≤1 escalation back-edge into `plan` — not two. `app/graph/build.py:9-10,23-24` inherited the "self-loop internal to `plan`" wording from this section and is stale for the same reason.
> Original intent kept above for the record.

---

## 4. Failure-mode guards (the three failure modes → concrete limits)

| Failure mode | Guard |
|---|---|
| **Infinite loop** | This is the **runtime harness** — the loop that turns the model's requests into actions and, critically, knows when to stop. Bounded: max ReAct iterations (e.g. 8) + max tool calls + **page budget** (CC-6); **progress tracked** (each iteration must advance the Plan, else it's terminated as stalled); transient API errors **retried with backoff**, hard errors surfaced rather than looped on. Two distinct terminations, never a hang: **(a)** required data gathered ∧ Output Reviewer approves → ship; **(b)** a ReAct-iteration / tool-call cap fires mid-gather → a **best-effort genuine partial** (`meta.partial.truncated:true`). The **page-budget over-run is NOT this path** — it returns `status:"too_large"` (a refuse, `meta.partial:null`, §B.7), never a sorted-prefix partial. |
| **Hallucinated planning** | Deterministic tool schemas + the Plan Checker (§3.3) + Intent Reviewer (§3.4); the "LLM never counts" invariant; the LLM's capabilities *are* the tool set, so it cannot promise an action no tool performs. |
| **Unsafe tool use** | **Least privilege:** each tool does exactly one job; none exposes raw HTTP or shell, so there is no over-privileged escape hatch. **Read/write distinction:** the entire surface is **read-only** (query a public registry) — no write/delete/destructive tool exists, so the boundary is absolute, not enforced by prompt. **Approval workflow:** none is needed because no action is destructive or irreversible; the standing rule is that any future write capability would sit behind an explicit human-approval gate. Params are whitelisted + validated at the tool boundary; the API client carries timeouts + a shared rate limiter. |

> **SUPERSEDED BY CODE (2026-07-22) — corrections to the "Infinite loop" row (the other two rows hold as written, except "shared rate limiter", see §A(f)):** (a) **termination (b) is wrong** — a cap firing does NOT produce a best-effort partial; `plan` aborts to a redacted guard error routed to `error` (`app/graph/nodes.py:238-241`). The only `partial` writer in the system is `_execute_single` on a genuine page-budget truncation (`app/graph/nodes.py:635`). (b) **"progress tracked (each iteration must advance the Plan, else terminated as stalled)"** — the stall detector is gated to `iter_count >= 2` and therefore cannot fire under v1; see the §B.7 annotation. (c) The read-only claim survives intact and is the strongest security line in the document — all 8 registered tools are GET-only against one pinned host, and 2 of them don't even execute.

---

## 5. Conflict-and-ambiguity analysis of *this* architecture

The same interrogation the requirements got, turned on the design itself. Each row: the tension → the ruling.

**Conflicts**
- **ReAct (LLM drives tools) vs CC-2 (never counts).** The natural reading of "the agent calls tools and reasons" flirts with letting the model tally. → **Ruling:** all counting/aggregation lives *inside* high-level deterministic tools (§3.5); the LLM composes tools and receives computed results. The invariant holds *by construction*, not by instruction.
- **Two LLM reviewers vs the anti-hallucination goal** (more LLM calls = more risk). → **Ruling:** both reviewers are **gates, not generators**. Intent Reviewer emits only `approve/revise` on a Plan; Output Reviewer inspects already-computed output and checks substrings + reconciliation. Neither can introduce a number.
- **LangGraph (a stateful-graph framework) vs a stateless service.** → **Ruling:** graph state is **per-request only**; nothing persists across requests. Cross-request memory is an explicit stretch (§8), not a hidden default.
- **SSE streaming vs "don't expose hidden reasoning."** → **Ruling:** stream a **fixed enum of high-level status events**, never token-level chain-of-thought; the terminal event is the spec.
- **Provider-agnostic adapter vs real provider-specific tool-calling shapes.** → **Ruling:** the adapter normalizes tool-call/tool-result shapes and `tool_choice`; every provider quirk lives *below* the adapter, nothing above it knows the provider.
- **Custom schema vs the Vega-Lite standard.** → **Ruling:** custom is **canonical/source-of-truth** (carries citations, dual counts, networks); Vega-Lite is a **convenience projection** for standard charts only; the network graph is never Vega-Lite.

**Ambiguities**
- **"Done" in the ReAct loop.** → Explicit predicate: the Plan's required data is gathered **and** Output Reviewer approves; else a cap fires and a labeled partial returns.
- **What each reviewer "approves."** → Concrete bounded checklists (§3.4, §3.8), not open-ended judgment.
- **Which model runs each node.** → Adapter default (Claude); planner strong, reviewers cheaper; configurable per node.
- **Re-plan trigger.** → Reviewer-#1 reject **or** zero-results → one bounded re-plan; then an empty-state spec + message (the CC zero-results ruling), never a loop.
- **MCP stretch scope.** → Read-only tools; no write path; clearly bounded.

**Boundary/degeneracy** (inherited from `SPEC_INTERROGATION.md`, enforced here): zero-results → valid empty spec + message; huge result set → page budget → `status:"too_large"` + exact `countTotal` (the §B.7 refuse contract, not a partial); one-node network → degrade to stat/bar; a plan the tools can't serve → checker rejects with a reason, streamed as an event.

---

## 6. The API contract

**Endpoints**
- `POST /visualize` — sync; returns the response envelope.
- `POST /visualize/stream` — SSE; status events + terminal spec event.
- *(stretch)* MCP server exposing the read tools.

**Request** — `query` (required, non-empty, ≤ ~500 chars) + optional structured fields (`drug_name`, `condition`, `trial_phase`, `sponsor`, `country`, `start_year`, `end_year`, …), merged by CC-1 precedence, per-field validated.

**Response envelope** —
```
{
  status: "ok" | "empty" | "too_large" | "error",
  kind: "visualization" | "answer",
  visualization: { type, title, encoding, data } | null,   // custom canonical (CC-10); null on kind:"answer"
  vega_lite: { … } | null,                                 // projection for standard charts
  answer: <text> | null,                                   // for kind:"answer" (yes/no + scalar); CC-7, E-34
  error: { code, message } | null,                         // for status:"error"; never a half-viz; E-35
  citations: { <nctId>: { field_path, value, excerpt } },  // CC-9; each data row carries source_ids[] → these keys (A-47)
  // ↑ SUPERSEDED — see the annotation below the block.
  meta: {
    count_basis: { trials, mentions },                     // CC-3 (both) — an OBJECT, supersedes the earlier string form; for status:"too_large", trials = the exact matching countTotal (G-39)
    date_field_used, time_granularity?,                    // CC-4 / E-31
    filters, query_provenance, retrieved_at,               // CC-18
    source: "clinicaltrials.gov",                          // A-33
    partial: { truncated, of_total } | null,               // ONLY a genuine defensible truncation (truncated:true); NULL for too_large — refusing to chart ≠ truncating (G-39)
    notes                                                  // interpretation / overrides (CC-1)
  }
}
```

> **SUPERSEDED BY CODE (2026-07-22) — three envelope divergences, all verifiable in `app/api/schemas.py`:**
>
> **(1) `kind` has THREE members, not two.** `Kind = Literal["visualization", "answer", "clarification"]` — `app/api/schemas.py:48`. `clarification` (decision P5-INPUT / E-13) is the first-class outcome for a well-formed request whose NL names an unresolvable referent ("this drug", no `drug_name`): the envelope carries a code-templated `question` (`app/api/schemas.py:316`) and no data, and the graph short-circuits `plan → build_spec` (`app/graph/build.py:104`). Live proof: `examples/run_11_*.json`. §6 and API-10 predate it.
>
> **(2) The Citation shape is `{nct_id, excerpt, field_path, value, matched_value, matched_tokens}`.** `app/api/schemas.py:96-110`. G-25 (`REQUIREMENTS.md:548`) already superseded API-13's singular `{field_path, value, excerpt}` by making citations a **per-datum list**; this is the further, later rename.
>
> **(3) ⚠️ THE ROLE REVERSAL — this is the single most load-bearing doc-vs-code divergence in the project.** CC-9 (`SPEC_INTERROGATION.md:29`) and `REQUIREMENTS.md:143` define `excerpt` as *the literal bucketing field value, augmented with `briefTitle` when present*. **The shipped code inverted that:**
> - `excerpt` = the trial's **`briefTitle`** (the citation contract's readable "text excerpt that supports the datum") — extracted at `identificationModule.briefTitle`, a DIFFERENT path from `field_path` (`app/ctgov/citations.py:133,160`). It falls back to `matched_value` only when a record has no briefTitle.
> - `matched_value` = the **literal value at `field_path` that decided membership** (`"PHASE1"`, `"2015-01-28"`, `"France"`) — the anti-fabrication anchor the Output Reviewer actually checks (`app/viz/review.py:109`).
> - `matched_tokens` = the composite-bucket member list. **The field named `excerpt_tokens` in P3-CITE (`BUILD_PLAN.md:217,225`) NEVER SHIPPED under that name** and does not exist anywhere in the codebase.
> - There is **no `title` field** on `Citation`, despite `BUILD_PLAN.md:377` describing the fix as "keep `excerpt` = the field value and ADD `title` = briefTitle". The rename (commit `12c5568`) landed the opposite way round and updated only the code lines, which is the root cause of the stale-docstring cluster in `app/ctgov/citations.py`, `app/api/schemas.py` and `app/viz/review.py`.
>
> **Consequence:** anyone reading CC-9, REQUIREMENTS.md:143 or BUILD_PLAN.md:377 will describe the citation schema **backwards**. The shipped `README.md:276-283` and `ARCHITECTURE.md:109-114` describe the CURRENT (reversed) behaviour correctly — quote those. And be ready for the honest follow-up: **the `briefTitle` excerpt is code-extracted but is NOT itself re-verified by the Output Reviewer** — only `matched_value`/`matched_tokens` are.
> Original intent kept above for the record.

---

## 7. Traceability — architecture → decisions

Which behavioural rulings each component exists to satisfy. Read it in either direction:
a component with no ruling behind it is speculative, and a ruling with no component
behind it is unimplemented.

| Component | Implements | Property it buys |
|---|---|---|
| Invariant + high-level tools (§3.1, 3.5, 3.6) | CC-2, CC-3, CC-5, CC-6, CC-15 | no hallucination-prone step anywhere a number reaches the user |
| Planner + Intent Reviewer + Checker (§3.2–3.4) | CC-1, CC-4, CC-8 | a plan is validated against a closed vocabulary before anything executes |
| Aggregation·Citation core + Output Reviewer (§3.6, 3.8) | CC-9, CC-16, CC-18 | every datum traces to a real record, and the totals reconcile |
| Viz-spec builder (§3.7) | CC-7, CC-10, CC-13, CC-14 | the encoding fits the data, including deciding not to chart |
| Network tool (§3.5) | CC-11, CC-12 | relationship queries without leaving the deterministic path |
| Transport + guards (§3.9, §4) | — | bounded work, and a refusal that states its own numbers |

---

## 8. Stretch / deferred (sequenced last, clearly labeled)

MCP server (§3.11) · conversational memory (§3.10) · deeper Vega-Lite coverage · a deployed endpoint / short demo GIF. Built only after the core ships end-to-end (the vertical-slice-first sequencing from `STEPS_TO_BUILD.md`).

---

## 9. The invariant, restated

> The LLM decides *what* to compute; deterministic tools compute it — so the model is structurally incapable of emitting a number. A ReAct planner turns the question into a Plan; a **Plan Checker** (code) confirms the arguments are real; an **Intent Reviewer** (LLM) confirms the planner understood the question; the tools do all the paging, counting, and citing; an **Output Reviewer** (code + LLM) confirms every excerpt is a real substring of the source and the result faithfully answers. Every value the user sees was computed by code and traces back to the trials that produced it.

---

## §A — Tool access & security model

The tool layer is where security lives. Every tool is minimally privileged: its only capability is an HTTPS **GET** to `clinicaltrials.gov`. No filesystem, no shell, no secrets, no write anywhere.

**(a) Per-tool least-privilege matrix**

| Tool | Egress (host · endpoint · method) | FS | Shell | Secret | Write |
|---|---|---|---|---|---|
| `count_trials` | clinicaltrials.gov · `GET /api/v2/studies?countTotal=true&pageSize=1` | — | — | — | — |
| `aggregate_by` | clinicaltrials.gov · `GET /api/v2/studies` (paged, `fields=`) | — | — | — | — |
| `timeseries` | clinicaltrials.gov · `GET /api/v2/studies` (paged, date fields) | — | — | — | — |
| `compare` | clinicaltrials.gov · 2× `GET /api/v2/studies` | — | — | — | — |
| `build_network` | clinicaltrials.gov · `GET /api/v2/studies` (paged) | — | — | — | — |
| `get_trial` | clinicaltrials.gov · `GET /api/v2/studies/{nctId}` | — | — | — | — |
| `resolve_entity` | **none** (local alias table); else read-only GET, same host | — | — | — | — |

The HTTP client is **GET-only** (rejects other methods), **https-forced**, **base-URL-pinned** to `https://clinicaltrials.gov` (no user host/port/path), with **same-host-only redirects**. Exactly **two egress destinations, partitioned by component:** tools/core → registry only; the LLM **adapter** → provider host only. Neither can reach the other's host.

> **SUPERSEDED BY CODE (2026-07-22) — the code is STRONGER than this spec, deliberately, and says so in place.** (a) Redirects: the client refuses **ALL** 3xx (`follow_redirects=False`; a 3xx raises `upstream_redirect_refused`) — `app/ctgov/client.py:140,175-182` — rather than allowing same-host ones. (b) Method: the client **defines no verb other than GET** (`app/ctgov/client.py:126-129`), so there is no "reject other methods" branch to point at; the absence IS the guarantee. (c) Base-URL pinning is done by `urlparse` with exact-hostname + no-userinfo + port ∈ {None,443} (`app/ctgov/client.py:23-27,104-108`), never `startswith` — see LESSON B1 (`tasks/LESSONS.md:233`); ⚠️ `app/ctgov/enums.py:15` still *says* "asserts every base_url starts with this", which describes the exact vulnerability the code fixes. Do not read that line aloud.
> Original intent kept above for the record.

**(b) Read/write distinction — absolute by construction.** All 7 tools are read-only; there is no write/delete axis to gate. State is per-request; the response cache is a derived politeness layer, never authoritative (CC-18). **Approval workflow** is a *standing policy*, not a runtime feature: any future write/mutating capability (saved reports, an MCP write tool, an external post) sits behind a human-in-the-loop gate + a separate privilege tier + an audit log.

**(c) Indirect prompt injection — the primary threat.** Registry free-text (`briefTitle`/`briefSummary`/`officialTitle`, names, countries) is attacker-authored (a sponsor writes it) and can carry "ignore previous instructions…". Where it touches an LLM, and how each point is defanged:

| Touch point | Untrusted content | Defense |
|---|---|---|
| Planner observations | bucket/node **labels** | labels are string keys only; no field value changes a tool's action; the planner's action space is the fixed 7 validated tools → injection cannot add capability |
| Planner via `get_trial` | raw record | raw JSON goes to the **code** citation core; only a **bounded, fenced excerpt** reaches the LLM, typed as untrusted `excerpt` data |
| **Output Reviewer** (biggest exposure) | citation **excerpts** | the provenance guarantee is a **deterministic substring/`field_path` check in code** (§3.8); the LLM opinion is secondary + non-generative |
| Intent Reviewer | Plan (structured) | structured tokens, not prose; lowest exposure |

Standing principle: **retrieved registry text is DATA, never instructions.** The planner routes/aggregates structurally and never executes content from a field; excerpts are string-extracted, never LLM-authored (CC-9), so an injected string can neither be *written* into a passing citation nor *expand* the tool catalog. Downstream renderers/MCP clients must treat excerpts + labels as untrusted display text.

**(d) SSRF / Essie-injection.** User text occupies **only the value slot of a whitelisted query-area param** (`query.term/cond/intr/spons/locn`), URL-encoded. Host, path, param **names**, and every `filter.advanced=AREA[…]RANGE[…]` Essie expression are **code-generated from validated tokens** (phase enum, integer years, status enum) — user text never becomes a URL, an operator, a filter expression, or a param name.

**(e) Secrets.** Exactly one secret in the system — the LLM provider key — held in env/config and read **only by the adapter** (§3.1). Never passed to tools, never in output/SSE, never logged; redacted on error. ClinicalTrials.gov contributes zero secrets (no auth).

**(f) Resource / DoS guards.** Per-call timeout · shared global rate limiter (~3 req/s, tunable — the number is `[UNVERIFIED]`; the 1000-page cap is `[LIVE]`) · page budget (CC-6, refined by §B.7) · `pageSize ≤ 1000` (hard) · response-size cap · max ReAct iterations · query length ≤ ~500 chars · per-request wall-clock deadline. Without these, "all cancer" (142,411 trials) pages ~143× serially → the budget converts it to `status:"too_large"` + the exact `countTotal` (the §B.7 decision — faithfulness over a biased partial), instead of a hang.

> **SUPERSEDED BY CODE (2026-07-22) — the rate limiter is PER-INSTANCE, not shared/global (same for SEC-30, `REQUIREMENTS.md:411`).** `CTGovClient._last_request_at` is declared as a class attribute (`app/ctgov/client.py:73`) but `_throttle` writes it back as an **instance** attribute (`app/ctgov/client.py:269`), which shadows the class value; and `app/ctgov/tools.py` constructs a **fresh `CTGovClient()` per tool call** (`:65, :88, :380, :475, :583, :623`). PROVEN by running it: 5 throttles on 5 fresh instances took **0.0001 s** (no throttling at all); 5 throttles on ONE instance took **1.35 s** (correct ~3 rps). **So throttling holds inside a single tool's paging loop, and never across tool calls.** Under a single-user demo this is harmless (the paging loop is where the burst lives), but the "shared global limiter" wording is not what shipped. Related caps that also drifted: `_MAX_PAGE_SIZE` is `config.PAGE_SIZE` (`app/ctgov/client.py:35`), so the "`pageSize ≤ 1000` (hard)" claim is **self-referential** — `PAGE_SIZE=5000` makes the clamp 5000 (`app/config.py:85`, `_env_int` has no maximum); and `MAX_QUERY_CHARS` / `MAX_STRUCTURED_FIELD_CHARS` (`app/config.py:105-106`) have **zero consumers** — the real caps are hardcoded `max_length=500` / `max_length=200` literals at `app/api/schemas.py:364-368`.
> Original intent kept above for the record.

**(g) Per-agent tool visibility** (tight surface = cheap context *and* small attack surface)

| Component | Tools visible | Why |
|---|---|---|
| Planner (ReAct) | the ~7 | only agent with tools; ~7 scoped schemas ≪ the ≈55k-tokens-for-80-tools figure |
| Intent Reviewer / Output Reviewer | **zero** | gates; read/approve only, no tool schemas in context |
| Checker, aggregation/citation core, viz builder | **zero (code, not agents)** | deterministic |

A tight catalog means injection can only ever request one of 7 read-only validated operations — there is no shell/HTTP/file/email tool to hijack (directly closes the "hallucinated planning promises an action no tool performs" hole).

> **SUPERSEDED BY CODE (2026-07-22) — the "7" is off by one in three places in §A** (`(a)`'s least-privilege matrix, `(b)`'s "all 7 tools are read-only", and `(g)`'s blast-radius line; same for SEC-38/SEC-41 in `REQUIREMENTS.md:419,422`). The registry holds **8**; `study_duration_histogram` is missing from every table. **The security substance is unchanged and arguably stronger** — all 8 are read-only GETs against one pinned host, and 2 of them (`get_trial`, `resolve_entity`) raise `NotImplementedError`, so the live blast radius is **six** read-only operations. Also note the planner sees **none** of them: `tools=None` (`app/llm/planner.py:446`), so the injection-can-only-request-a-validated-op argument holds *a fortiori* — the model has no action space to hijack, only a closed plan schema. Finally, the "≈55k-tokens-for-80-tools" figure in `(g)` is sourced to a talk title only (`AGENT_STUDY_NOTES.md:98,113`) — **do not assert it as measured.**
> Original intent kept above for the record.

**(h) MCP stretch security.** Read-only subset (`count_trials`/`aggregate_by`/`get_trial`), no write path, **reuses the same validated + rate-limited + allowlisted client**; bind local/stdio or require an auth token + host allowlist (so it isn't an open proxy); **no LLM/provider key in the MCP path** (the client's model reasons; we serve deterministic data).

**(i) Logging / PII hygiene.** Registry data is public (no PII). **User query text may be sensitive** (reveals what someone researches) → log structured events (SSE status enum, tool name, validated arg tokens, counts, timings, `retrieved_at`); never the provider key, never model reasoning, and don't persist raw query text at info level (redact / TTL). The cache keys on the normalized plan and holds only public computed results.

---

## §B — Engineering reference (the fine-grained contracts)

### B.1 LLM adapter capability matrix
The adapter exposes a **capability descriptor** nodes query — `{supports_forced_tool_choice, supports_parallel_tool_calls, supports_native_structured_output, system_prompt_style, json_schema_dialect, max_context}` — and normalizes each axis:

| Axis | Normalizes to | Does NOT hide |
|---|---|---|
| tool-call / result shape | canonical `{name, args, call_id}` | — |
| `tool_choice` | `{auto, none, required, forced-named}`; **emulates** absent modes (single-tool list, or structured-output + parse) | emulation may cost a round-trip |
| parallel `tool_use` | one-or-many blocks → `ToolCall[]`; **serializes** when a provider lacks it | it does not *make* the model parallelize |
| structured output | schema-validated typed object; native mode else forced-tool-schema; **always Pydantic-validate + bounded re-ask** | provider "strict" claims are not trusted |
| streaming | irrelevant upward — our SSE is our own status enum | provider token deltas stay below the adapter |

**Honest boundary:** the adapter normalizes wire shape + capability semantics, **not model behavior** (accuracy, latency, cost, context all still vary). Capability-driven degradations are **logged, never silent**. Nothing above the adapter names a provider or *assumes* a capability — it queries the descriptor.

> **SUPERSEDED BY CODE (2026-07-22) — this is the largest spec-vs-code gap in the document. The shipped adapter normalizes STRUCTURED OUTPUT only.** Row by row:
> - *tool-call / result shape → canonical `{name, args, call_id}`:* **not implemented.** No canonical `ToolCall` type exists anywhere in the codebase. OpenAI forwards `tools` raw and pins `tool_choice="auto"` (`app/llm/adapter.py:541-544`); Anthropic's `propose` **deletes** `tools` outright (`del tools, canned`, `app/llm/adapter.py:640`).
> - *`tool_choice` emulation:* **not implemented** — no emulation path exists.
> - *parallel `tool_use` → serialized:* **not implemented.** `supports_parallel_tool_calls` exists only as an unread flag (`app/llm/adapter.py:55`).
> - *structured output → schema-validated typed object + bounded re-ask:* **implemented and real** (OpenAI `response_format:json_schema` vs Anthropic `output_config.format` + a forced-`emit`-tool fallback; both always Pydantic-validate + re-ask). This row alone is what "provider-agnostic" rests on — and it is genuinely proven by two real providers (LESSON AA1).
> - *"nothing above the adapter … queries the descriptor":* **`CapabilityDescriptor` (`app/llm/adapter.py:47`) has ZERO production consumers** — its only caller anywhere is `tests/test_llm.py:72`. Both real adapters return identical hardcoded descriptors, including `max_context=200_000` for every model (`app/llm/adapter.py:414-421, 608-615`). And `AnthropicAdapter`'s own class docstring calls its native→fallback switch "capability-driven" (`:580-582`) when the implementation is **exception-driven** (`except BadRequestError/TypeError` at `:718`).
> **Honest framing:** the adapter is a *structured-output* + *model-selection* abstraction proven across two real wire formats — not the full capability-matrix normalizer §B.1 describes. Since the shipped planner passes `tools=None`, none of the unimplemented rows are on any live path; they are unbuilt design, not dead code that runs.
> Original intent kept above for the record.

### B.2 Observation object + three-tier error taxonomy
Planner observation = `{tool, ok, error_code?, total_count?, buckets_preview?, truncated?, date_field_used?, note?}` (computed, bounded; never raw records/stack traces). Errors are tiered:
- **Transient** (429 / 5xx / timeout) — absorbed **below** the tool with backoff; the planner never sees them.
- **Actionable-semantic** (zero-results, unresolved/ambiguous entity, 1-node graph) — surfaced as a typed observation the planner may **escalate on once**.
- **Hard** (persistent 5xx after retries, un-fixable request) — short-circuit to `status:"error"`, never looped.

### B.3 Checker + reviewer checklists
**Deterministic checker (code, un-skippable):** phase tokens ∈ `{EARLY_PHASE1,PHASE1..4,NA}` (combined→array; reject `"PHASE1/PHASE2"`, `"Phase 5"`) · `date_field` ∈ the 5 real date fields · JSON paths ∈ known-field allowlist (reject invented) · `start_year ≤ end_year` ∈ `[1900, currentYear+N]` · `chart_type` ∈ `{bar, grouped_bar, time_series, histogram, scatter, network_graph, single_value, table}` · query areas ∈ `{term,cond,intr,spons,locn}` · status/studyType/sponsor-class/intervention-type ∈ their real token sets · **tool exists + args typed-match schema** · **data-shape ↔ chart** compat (no date → reject time_series; no two continuous fields → reject scatter) · query length ≤ ~500. *(Entity resolvability is NOT checked — a typo'd drug is a valid request with 0 results.)*
**Intent Reviewer (LLM, semantic):** metric matches ask · entity → right dimension · `date_field` matches date-intent · chart *apt* not merely legal · filters faithful (none invented). Output `approve | revise{field,reason}`.
**Output Reviewer:** deterministic pre-checks (substring @ `field_path` over each datum's inline `citations[]`; reconciliation *gated* to `status=="ok"`∧`data.type=="rows"` and *mode-aware* — combine → Σ==`countTotal`, explode → distinct-nctId==`countTotal`; `partial` iff truncated; every datum cited-or-derived except a legit `count==0` bucket) **+** one bounded LLM `approve | flag{reason}` (answers question? apt encoding?). `flag` annotates, never rebuilds.

> **SUPERSEDED BY CODE (2026-07-22) — four checker items in this list do not live where §B.3 puts them:**
> - **"tool exists + args typed-match schema" — no analogue exists.** `check_plan` never touches tools; `Recipe.allowed_tools` (`app/plan/recipes.py:34`) is read by **no runtime code at all**. The constraint is enforced at BUILD time only, by `tests/test_ctgov_plan.py:45` (`allowed_tools ⊆ TOOL_NAMES`). Two sibling `Recipe` fields are equally inert: `count_basis_rule` (`:37`) and `date_field_disclosed` (`:38`) — disclosure is driven off `query_class` directly (`app/viz/spec.py:646`) and count basis is computed from buckets (`app/viz/spec.py:385,635`). Only `chart_type` / `alternates` / `degeneracy_fallback` are load-bearing, and only at `app/plan/checker.py:195-197`.
> - **`start_year ≤ end_year ∈ [1900, currentYear+N]`** — not in `check_plan`; see the §3.3 annotation.
> - **query length ≤ ~500** — a Pydantic `max_length` on `VisualizeRequest` (`app/api/schemas.py:364-368`), not a checker item.
> - **"no two continuous fields → reject scatter"** — moot: `scatter` is in the closed `ChartType` enum but **no recipe emits it** (deferred by G-20, `REQUIREMENTS.md:538`), so the branch is unreachable by construction.
> - *What §B.3 MISSES:* the shipped `check_plan` runs an **8th** check the numbered list never mentions — `_check_class_shape` (`app/plan/checker.py:212-214`), the G-33 per-class structural check that stops a shapeless compare/network plan from reaching execute. Arguably the most important check is the one the docstring drops.
> - *Two checks are honest defence-in-depth but cannot fire today:* `network_graph_requires_network_query_class` (`:206-207`) is pre-empted by the recipe check at `:198`, and `unknown_query_class` (`:193`) is reachable only via attribute mutation. Present them as belt-and-braces, not as active guards.
> - *Totality caveat:* the "a checker must never raise" contract holds for **Pydantic-constructed** plans only. `Plan` sets no `validate_assignment` (`app/plan/models.py:66`), so post-construction mutation bypasses the Literal/Enum guards — verified: `plan.chart_type = "network_graph"` → `AttributeError` at `app/plan/checker.py:202`. Not reachable from the LLM path (and `app/graph/nodes.py:498` would redact it), but the claim needs the qualifier.
> - *Filter-shape check is narrower than its docstring:* `_check_filter_tokens` shape-checks only keys that HAVE a token set — the `token_set is None → continue` at `app/plan/checker.py:119-123` runs BEFORE `_is_malformed_token_shape` at `:125`. Verified: `filters={"country": {"a": 1}}` → `ok=True`.
> Original intent kept above for the record.

### B.4 Guard-values table

| Guard | Value | Rationale |
|---|---|---|
| Max ReAct iterations | **8** | normal plan = 2–4 tool calls; 8 = 2× headroom incl. one escalation + a `compare`. >8 ⇒ spin, not depth → hard abort-to-error |
| Max tool calls | **12** | bounds total fan-out independent of iterations |
| Max wall-clock | **60s sync / 90s SSE** | dominated by serial paging; SSE longer since status events keep the client informed |
| Page budget | **20 pages = 20,000 trials** | `countTotal` first; ≤20k → aggregate fully (exact); **>20k → refuse the chart, return the exact `countTotal` + a "too large to chart faithfully" note** (DECIDED — a sorted prefix is a *biased* sample, not just incomplete; faithfulness over completeness). Covers virtually all *scoped* queries fully; only bare broad conditions (cancer=142k) hit the refusal, and even they get an exact number |
| Response-size cap | top-N cats **50** + "Other" · network top-N **60** nodes, min-edge-weight ≥ k · citations **K=20**/datum (+ exact `contributing_count`, `citations_truncated`) | keeps JSON + SSE terminal event renderable |

Scalars via `countTotal` are **never partial** (one exact call regardless of size); only paging aggregations truncate. The cap is deploy-time config, never agent-tunable.

> **SUPERSEDED BY CODE (2026-07-22) — three notes on this table:**
> - **The 8-iteration and 12-tool-call caps CANNOT FIRE under the shipped v1 planner** (single-shot + shared escalation ≤ 1 means `plan` runs at most twice). They are real, active, tested graph-runner backstops (`app/graph/guards.py`, `tests/test_guards.py`) — decision P5-GUARDS, "the headroom is stated, not hidden" — but do **not** claim "the ReAct loop is bounded by an 8-iteration cap" as an exercised guarantee. Note also that §B.4's stated rationale ("normal plan = 2–4 tool calls") has no measurement anywhere in the repo; live is 1–2.
> - **The categorical top-N is 50, not 15** (decision P5-TOPN, `app/config.py:89`) — but `app/ctgov/fields.py:429` still says "Σ(top-15 + Other + Unknown)". Consequence worth knowing: with 50+1 buckets × K=20 citations, a combine/explode board can cite up to ~1020 records against a 500-record re-verify index (`RECORD_INDEX_CAP`, `app/config.py:97`), so some cited records are genuinely absent from the index and are skipped (`app/viz/review.py:161-162`). Measured on `examples/run_06_geographic_ranked_bar.json`: 51 buckets, 836 citation objects, 566 unique cited nctIds. Behaviour is safe (LESSON Z1's bounded-defence framing); the "fully covered" comment at `app/ctgov/tools.py:170-172` overclaims.
> - **"deploy-time config" is only partially true.** `CITATION_SAMPLE_K` (`app/config.py:96`) is honoured at `app/ctgov/tools.py:267,483` but hardcoded to 20 at `app/ctgov/histogram.py:150,175` and defaulted to 20 at `app/ctgov/network.py:613` and `app/ctgov/citations.py:168`; `aggregate.py:95` hardcodes `budget_pages=20` and `aggregate.py:29` hardcodes `_PAGE_SIZE=1000`, so `PAGE_BUDGET_PAGES` is silently ignored on the `aggregate_by`/`timeseries` paths (PROVEN: `PAGE_BUDGET_PAGES=3` changed `iter_studies` but not `page_and_group`). Honest framing: **"partially operator-tunable"** — this is LESSON AF1 recurring inside the very file that fixed it.
> Original intent kept above for the record.

### B.5 LangGraph — nodes, conditional edges, state
**Nodes:** `merge_inputs` (**raw structured-field normalization/validation only** — the CC-1 *dimension-precedence* is the Planner's job, since deciding which dimensions the free-text query names requires the NL parse) · `plan` (ReAct loop, LLM, **owns CC-1 precedence**) · `check` (code) · `review_intent` (LLM) · `execute` (deterministic runner) · `build_spec` · `review_output` (code+LLM) · `respond` · `error`.

**Conditional edges (predicate → target):**

| From | Predicate | To |
|---|---|---|
| `merge_inputs` | valid / invalid | `plan` / `respond`(error 422) |
| `plan` | **internal ReAct loop** (reason→act→observe over tools; bounded by iter<8 ∧ progressing ∧ ¬stalled; then finalize — or cap → best-effort finalize with `partial`) | `check` |
| `check` | ok / reject∧esc<1 / reject∧esc≥1 | `review_intent` / **`plan`** / `respond`(error) |
| `review_intent` | approve / revise∧esc<1 / revise∧esc≥1 | `execute` / **`plan`** / `execute`(best-effort) |
| `execute` (runs the finalized plan **once** — no LLM, no self-loop) | done / zero-results∧esc<1 / zero-results∧esc≥1 / **over-budget** (totalCount > budget) / hard-error | `build_spec` / **`plan`** / `build_spec`(empty) / **`build_spec`(too_large)** / `error` |
| `build_spec` | — | `review_output` |
| `review_output` | approve / flag | `respond` / `respond`(+caveat, no rebuild) |

Escalation budget is **shared, ≤ 1** across the three re-plan triggers. Errors route to `error` as edges; no Python exception escapes a node. **The ReAct iteration and its iter/stall/progress guards are internal to the `plan` node — they are not a graph edge; `execute` runs the finalized plan's aggregation exactly once** (consistent with §3.2/§3.12: the loop lives in `plan`, `execute` is a deterministic runner).

**State (`TypedDict`), writer→reader:** `question/raw_fields` (merge→plan,verify,vout) · `merged_inputs` (merge→plan,validate) · `plan` (plan→validate,verify,execute,build) · `escalation_count` (plan++→validate,verify,execute) · `validation` (validate→verify) · `tool_results` append-only (plan/execute→build,vout) · `iter_count/scratch` (**plan RW** — the ReAct loop is internal to `plan`) · `partial` (plan/execute→build,respond) · `spec` (build→vout,respond) · `verifications` (verify/vout→respond) · `error` (any→respond/error) · `status` (build/respond) · `retrieved_at/query_provenance` (execute→build,respond) · `events` append-only (all; a derived stream via node transitions). Checkpointing OFF.

> **SUPERSEDED BY CODE (2026-07-22) — the `plan` routing row and the writer→reader matrix both drifted.** `app/graph/state.py`'s per-field comments were inherited verbatim from this matrix and are stale in the same places, so fixing one without the other re-introduces the drift.
> - **`plan` guard-trip does NOT "best-effort finalize with `partial`".** It aborts to a **redacted guard error** routed to the `error` node — `return {"error": guards.guard_error(tripped), "status": "error", …}`, `app/graph/nodes.py:238-241`. Genuine code-vs-spec conflict, not wording. (And per the §3.12 annotation, there is no internal ReAct loop to bound in the first place, so `iter_count`'s "plan RW — the ReAct loop is internal to `plan`" gloss is describing a loop that was never built.)
> - **`partial` is written by `execute` ONLY** — the sole writer is `_execute_single` at `app/graph/nodes.py:635`. `plan` never writes it.
> - **`spec` has three writers, not one.** `build_spec`, plus `execute` on a **cache hit** (`app/graph/nodes.py:488` — the mechanism the whole cache short-circuit depends on), plus `review_output` twice (`:962, :1004`).
> - **`status` has six writers, not three.** `execute`/`build_spec`/`respond` **plus** `plan` (guard trip `:241`, clarification `:298`), `review_output` (`:963`) and `error` (`:1051`).
> - **`deadline_at` is read at exactly two nodes, not "each expensive node"** — `plan` (via `check_pre_plan_guards`) and `execute` (`app/graph/nodes.py:455`). `review_intent` and `review_output` each make a real LLM call with **no deadline check**, and there is no explicit client timeout in `app/llm/` at all.
> - *Three routing edges exist that §B.5's table never lists:* `plan → build_spec` (clarification) and `plan → error` (guard) at `app/graph/build.py:104`, and `execute → respond` (cache hit) at `app/graph/build.py:121`. Conversely `app/graph/build.py:111-115` declares `error` as a target of `route_after_intent`, but `route_after_intent` (`app/graph/nodes.py:1094-1108`) can only return `execute` or `plan` — a dead branch left over from the Phase-0 hard-stop routing that P4-ROUTING replaced.
> - *Two undocumented behaviours worth knowing before the demo:* a **cached `empty` skips the zero-results re-plan** (`route_after_execute` checks `cache_hit` at `app/graph/nodes.py:1119` before the empty-escalation branch at `:1126`); and the tool-budget check runs BEFORE the cache lookup (`:464` vs `:483`) while the hit's return dict omits `tool_call_count`, so a hit is budget-checked without recording a spend.
> Original intent kept above for the record.

### B.6 Recipe / skill registry

**What it is, plainly.** There are ~5 natural query *classes* (time-trends, distributions, comparisons, geographic, networks). Rather than let the LLM improvise *how* to handle each one (flexible, but non-deterministic and prone to one-off hacks), we predefine a **recipe** per class: a fixed, deterministic procedure. The **registry** is just the collection of those recipes, keyed by `query_class` — it is **data (a config table), not code branches.**

The LLM's job then shrinks to **two moves**: **(1) classify** the question → which class? and **(2) fill the recipe's blanks** → which drug? which filter? which date field? It does **not** invent the procedure — the recipe already encodes it. The LLM decides *what*; the recipe decides *how*. This is the operationalization of the "Skills" idea (a reusable, deterministic procedure selected when needed).

**Worked example.** User asks: *"How are diabetes trials distributed across phases?"*
1. Planner **classifies** → `query_class = "distribution"`.
2. It loads the *distribution* recipe, which already specifies: tool = `aggregate_by(field=phase)` · chart = `bar` · count distinct-trials + mentions (CC-3) · include an `NA/Unknown` bucket (CC-5) · combined phase → its own bucket (CC-15).
3. Planner **fills the slots**: `condition = "diabetes"`, `field = "phase"`.
4. Done — the LLM never decided *how* to count or *which* chart; the recipe did. It only decided *what* (diabetes, by phase).

**The registry row (schema).** Each entry: `{allowed_tools[], chart_type default + alternates (CC-8), required date_field disclosure (CC-4), count-basis/denominator rule (CC-3), degeneracy fallback (e.g. network→bar, CC-12)}`. After the planner fills a recipe, the **Plan Checker** (§3.3) confirms the filled Plan satisfies that recipe's constraints (e.g. `chart_type ∈ recipe.allowed`).

**Why it's worth it.** (a) It makes *"single coherent approach, no one-off hacks"* **literal** — adding a query class = **adding a registry row, not a code branch** (the extensibility story). (b) It raises reliability — the LLM picks from a known menu, it doesn't improvise. (c) The ReAct loop still exists; it runs *inside* a recipe (retries, the one escalation, and multi-step composition like a comparison = two aggregations). Under the hood, all tools delegate to one `AggregationCore.page_and_group(key_fn, projection)`; they differ only in grouping key + output projection (`compare` = 2× + union; `timeseries` = key is a date-bin; `build_network` = key is an entity-*pair*).

> **SUPERSEDED BY CODE (2026-07-22):** the registry ships **SIX** rows, not "~5" — `distribution · timeseries · compare · geographic · network · single_value` (`app/plan/recipes.py`, verified by `list(RECIPES)`). The 6th, **`single_value`, is Soren's own addition**, not a spec item: it implements the "identify **if** a visualization is needed" decision, CC-7) so a yes/no or scalar question returns a stat card or `kind:"answer"` instead of being forced into a bar. Present it as a deliberate scope decision. The precise phrasing that survives scrutiny is **"five chart classes + `single_value`"**. Two further corrections to this section: (a) "the ReAct loop still exists; it runs *inside* a recipe" — it does not, see the §3.2/§3.12 annotations; (b) `Recipe.allowed_tools` / `count_basis_rule` / `date_field_disclosed` from the registry-row schema are **declared but never read at runtime** — see the §B.3 annotation. The *extensibility* claim itself survives: `_dispatch_execute` really does route every class over the one `AggregationCore`, and `Plan.alternates` — which CC-8 wanted returned — is never validated by the checker and **never reaches the wire** (`app/viz/spec.py` never reads it; `app/graph/cache.py:68` deliberately excludes it).
> Original intent kept above for the record.

### B.7 Retry policy · stall detector · partial contract
**Retry:** 429 / 5xx / connection-timeout with exponential backoff + full jitter (base ~0.5s, ×2, cap ~8s), **max 3**; honor `Retry-After`; **never retry 4xx**; GETs idempotent; retries count against wall-clock only (not page budget / iterations); exhausted → surfaced, never looped.
**Stall detector:** signature = `tool_name` + canonical-JSON(normalized args). Abort if a signature repeats, or 2 consecutive no-progress iterations (no new resolved Plan field **and** no new signature), or two escalations yield the same normalized Plan. On a duplicate signature: return the **memoized** result + a "no new data" observation (no API hit).

> **SUPERSEDED BY CODE (2026-07-22):** the shipped signature is over the **normalized Plan**, not `tool_name + args` (the planner emits no tool calls), and — importantly — **the stall detector CANNOT fire under v1 either.** `app/graph/nodes.py:270` gates the check on `iter_count >= 2` (a 3rd+ plan entry), while the shared escalation budget of ≤1 caps `plan` entries at 2. This is *deliberate*: LESSON AH2 (`tasks/LESSONS.md:74`) records that an ungated detector first aborted the legitimate zero-results escalation, turning a clean `empty` into an `error`. ⚠️ **`app/graph/guards.py:13-15` asserts the opposite** ("The stall detector CAN fire under v1") and contradicts the inline comment at `app/graph/nodes.py:262-268` in the same repo; `tests/test_guards.py:132` asserts the *code's* behaviour (no fire on the sanctioned single re-plan). `tasks/LESSONS.md` AH1/AH2 are correct; the `guards.py` module docstring is the wrong one. **Correct line to say aloud: "all four runtime guards are active, tested backstops that cannot fire under the v1 single-shot planner — the headroom is stated, not hidden."**
> Original intent kept above for the record.
**Over-budget contract (DECIDED — faithfulness over completeness, refines CC-6):** an aggregation whose match set exceeds the page budget does **not** ship a biased sorted-prefix chart. It returns `status:"too_large"` with the **exact** `countTotal` and a note ("N trials match — too large to chart faithfully within the paging budget; here is the exact total"). Only the exact scalar is offered above budget — it's one cheap `countTotal` call, always exact. This extends the invariant: *if the system cannot produce a trustworthy visualization, it says so rather than producing a misleading one.* (`meta.partial` remains only for the rare case a partial is genuinely defensible; it never claims completeness.)

### B.8 Function-call-accuracy eval
An **offline** fixture suite `(NL query [+fields]) → golden {tool sequence, tool, args}` across the 5 classes + edge cases (zero-result, dangling ref, huge set, combined-phase). Metrics: tool-selection accuracy · arg-validity rate (passes the Plan Checker) · arg-exactness (normalized == golden). Run against **recorded/mocked API fixtures** (VCR-style) → deterministic, $0, CI-able. Adversarial assertion: **no** plan ever clears the Plan Checker with an invented token/path (the anti-hallucination gate gets a test — the CC-9 substring-test discipline applied to planning). Production proxies: checker-reject rate + Intent-Reviewer-revise rate.

> **NOT BUILT (2026-07-22):** the function-call-accuracy eval is **the largest unbuilt piece of this spec**, and the one that would have measured the planner. There is no golden `(NL → tool/args)` fixture suite and no VCR-style recorded fixtures anywhere (`respx` is declared in `pyproject.toml:33` and imported by nothing; `asyncio_mode="auto"` at `pyproject.toml:48` is likewise inert — the suite has zero `async def test_`). What DID ship measures a different guarantee: `scripts/verify_examples.py` + `tests/test_examples_offline.py` verify **output** invariants (reconciliation, citation validity, no LLM-authored number) on the shipped examples; `tests/test_planner.py` only round-trips the stub. Say plainly: *"the planner's accuracy is evidenced by a 15-rung live ladder that all reconciles, not by a measured tool-selection score — the FCA harness is designed and unbuilt."*
> Original intent kept above for the record.
