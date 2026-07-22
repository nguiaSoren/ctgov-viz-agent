# Architecture

A companion to the README's design section: the *why* behind the structure, in one place. The
service turns a natural-language clinical-trials question into a structured visualization
specification backed by the ClinicalTrials.gov Data API v2, with per-datum citations.

## The one idea everything follows from

> **The LLM decides *what* to compute; deterministic code computes it. The model never emits a number.**

A registry aggregate ("how many Phase 2 trials?") has one correct answer, and it is not one an LLM
should be trusted to produce token-by-token. So the language model is confined to *planning* — it
reads the question and chooses a query recipe, a field, a set of filters, a chart — and every number
the user sees is computed by code that pages the real API and reconciles its result against the API's
own `countTotal`. This is not a convention that can drift: the planner's typed output declares **no
count, total, or value field** — its only numbers are the two year bounds of a filter, which select a
population rather than describe one — so a fabricated total is unrepresentable. The same constraint is
what makes the deep citations claimable — each excerpt is *string-extracted* from a fetched record,
never authored by the model — and what makes the whole system testable: correctness reduces to an
arithmetic identity a reviewer can re-check offline.

## The pipeline

A request flows through nine nodes on a LangGraph state graph:

```
merge_inputs → plan → check → review_intent → execute → build_spec → review_output → respond
                                                                                    (+ error → respond)
```

There is exactly **one Checker and two Reviewers**, and the two kinds never blur:

| Stage | Kind | Responsibility |
|---|---|---|
| `plan` | LLM | classify the question into one of six query classes, then fill that recipe's slots — **one** structured-output call, `tools=None` |
| `check` | **code** | mechanical legality — is every token, field, and range real? (the anti-hallucination gate) |
| `review_intent` | LLM | semantic — did the plan capture what was actually asked? (right metric/dimension/date/chart) |
| `execute` | **code** | run the validated plan's tools: page → count → dedupe → bucket → cite |
| `build_spec` | **code** | assemble the canonical visualization spec + a Vega-Lite projection |
| `review_output` | **code + LLM** | *code:* every citation's `matched_value` is a real element/substring of its own field value, counts reconcile; *LLM:* does it faithfully answer? |

*ReAct, precisely.* The *reason → act → observe* cycle is the **graph's**, not the model's: `plan`
makes one structured-output call with `tools=None`, and it is the *recipe* — selected by the
classified query class — that decides which tool `execute` runs. The gates and `execute` are the
observe step; the bounded back-edge threads the rejection reason back into the next prompt as
feedback. A planner that never sees an action space cannot invent one, which is exactly why the
collapse from tool-calling ReAct to classify→fill + bounded re-plan is a strengthening, not a
shortcut.

A **Checker** is deterministic code that returns a verdict on structure; a **Reviewer** applies LLM
judgment to meaning. Both reviewers are **gates, not generators** — they emit `approve` / `revise` /
`flag` on already-typed or already-computed data, so neither can introduce a number. The provenance
guarantee deliberately rests on the *code* half of the Output Reviewer, never on LLM vigilance: an
instruction injected into a trial's free-text summary cannot make a fabricated citation pass a
substring check.

## Where this lands on control × autonomy

The system is **Orchestrated × Adaptive**, and the distinction is deliberate:

- **Orchestrated, not emergent.** A graph routes among a fixed menu of nodes and a fixed catalog of
  tools. There is no agent-to-agent debate and no dynamic creation of new agents or tools — the LLM's
  capabilities *are* the tool set, so it cannot promise an action no tool performs.
- **Adaptive, not self-directed.** Autonomy shows up as runtime tool-choice, retry with backoff, and a
  bounded early-stop re-plan — but the re-plan is an **escalation**, not open-ended planning: it fires
  at most once, only when a gate rejects or a query returns zero results (never planner-initiated), and
  it stays within the same recipe menu. That hard bound is what keeps the system left of "self-directed."

In one line: a controlled orchestration — cyclic and adaptive, not an agent society.

## Why a cyclic graph, not a DAG

The spine above is acyclic, but one edge points *backward*: `check` / `review_intent` / `execute`
can escalate to `plan` for a single bounded re-plan. A backward edge means cycles, so the graph is
**cyclic, not a DAG** — but bounded, because the escalation budget is ≤ 1 and gate-triggered, so every
execution trace is finite. That combination (a cyclic control flow with runtime adaptation) is exactly
what a plain DAG runner cannot express, and the reason the pipeline is a LangGraph graph rather than a
straight function pipeline. Checkpointing is off — the graph is stateless per request, so it scales
horizontally, and conversational memory would be "turn a checkpointer on," not a redesign.

## The layers

- **Wire schema** — the request model and the response envelope. The lowest layer; it imports nothing
  from the rest of the app, so every other module depends on the contract, never the reverse.
- **Planner** — emits a *closed* typed object whose filter vocabulary is a fixed set of real tokens, so
  a hallucinated filter key is unrepresentable. Code lowers it to an internal plan and re-asserts input
  precedence before anything runs.
- **Plan Checker** — an allowlist gate over the whole surface: unknown fields, unknown filter keys,
  unknown tokens, and out-of-range values are all rejected. Total on its real input — for any
  Pydantic-constructed `Plan` it returns a verdict and never raises. (The totality is a property of
  the typed input, not magic: mutating a validated `Plan`'s attributes in place bypasses the enum
  guards and can raise, which is unreachable from the LLM path and would land on the redacted-error
  route anyway.)
- **The tool layer** — a small set of high-level, read-only, GET-only tools. Eight are registered
  (count, aggregate, timeseries, compare, network, duration histogram, single-record fetch, entity
  resolution); **six are live** — `get_trial` and `resolve_entity` are
  deliberate `NotImplementedError` stubs, documented as such and out of scope for v1 (no
  user-controlled id reaches a URL path today, though the `^NCT[0-9]{8}$` guard for it is wired and
  tested). Each live tool does its full deterministic job internally and returns *computed* results;
  the plan's query class selects which one runs.
- **The aggregation core** — one `page → group → dual-count → cite` engine that every query class
  composes. Breadth comes from composing this core, not from per-class code, which is what lets very
  different queries run off one engine without one-off handlers.
- **Viz builder** — the canonical spec is the source of truth (it carries citations, dual counts, and
  the graph shape Vega-Lite cannot express); a Vega-Lite projection is emitted alongside for standard
  charts so a frontend gets a render for free.

## The correctness model

The registry has no external "right answer" to an aggregate query, so correctness here is **internal
consistency against the one server number you can check** — the API's `countTotal`:

- The executor issues one `countTotal=true` call → the exact matching total `T`, then pages and
  aggregates client-side. The count call and every page route through the *same* parameter builder, so
  a filter is applied to both or neither — the two populations can never desync.
- **Reconciliation anchors on `distinct-nctId == T` for *both* counting modes.** That is the claim the
  citations back, and it is the only anchor that survives a multi-value field (a trial spanning several
  countries or intervention types), whose bars sum to more than `T` by design — a convention that is
  disclosed rather than hidden. For a single-value field (one bucket per trial, e.g. phase, status) the
  reviewer adds a **second** assertion: `Σ displayed bars == distinct-nctId`, so a deflated bar or a
  cross-bucket double-count cannot ship behind a correct scalar. A drift of ≤ 0.5% *and* ≤ 20 trials is
  taken as live registry drift and disclosed in `meta.notes`; anything larger is a hard fail.
- Reconciliation runs only where it is meaningful, and the exemptions are explicit: networks, scalar
  answers, the over-budget refuse, a two-population `compare` (each arm self-reconciles in-tool; there
  is no single oracle for the union), and the degenerate-network fallback bar. The citation checks
  still run on every one of them — the exemption is scoped to the count, never to provenance.

## Citations

Each datum carries the exact `contributing_count` (always the true bucket size) plus a bounded,
deterministic sample of up to twenty citations (the first twenty contributing nctIds, sorted — stable
across runs) with a `citations_truncated` flag when the true set exceeds the cap. On the exact-count
path (one count call per token, used when paging would be refused) the sample is ten per bucket,
pulled inside that same call so provenance costs no extra request. A citation is
**two-part**: an `excerpt` — the trial's human-readable brief title (§5's descriptive "text excerpt
that supports the datum"), string-extracted from the record — and a `matched_value`, the exact field
value at `field_path` that decided membership, verified element-precise against the record (a
fabricated value fails at build time). The excerpt reads like a source; the matched_value is the
rigorous "why this trial is in this bucket." The verification is deliberately asymmetric and worth
stating plainly: the Output Reviewer re-checks `matched_value` (and every `matched_tokens` member on a
composite bucket), **not** the `excerpt` — the brief title comes from a different, fixed path, so it is
not a quote of the bucketing value and checking it there would reject every honest citation. Both are
string-extracted; only one is the anti-fabrication anchor.
A *derived* value (a network edge weight) cites its **members** — the contributing trials — since it
has no single source field to quote. When a trial is in a bucket *because* a field is missing, the
citation is an honest absence: `matched_value: ""` against a `null` value, accepted only when the
field really is absent at `field_path`.

## Failure modes and the control for each

| Failure mode | Control |
|---|---|
| Hallucinated planning | closed typed planner output + the Plan Checker + the Intent Reviewer; the LLM can only pick real tokens and a real query class — the tool follows from the class, so it cannot name a tool that does not exist |
| Fabricated numbers / citations | the "LLM never counts" invariant + the deterministic substring/reconciliation checks in code |
| Unbounded loop | a bounded runtime harness — max iterations, max tool calls, a max node-visit count, a page budget, a stall detector, and a wall-clock deadline; every trip aborts to a redacted error, never a hang (see the note below) |
| A query too broad to chart faithfully | refuse with the exact total (`too_large`) rather than ship a biased sorted prefix — except for bounded-token fields, which chart exactly via one count per token at any scale |
| Indirect prompt injection | retrieved registry text is **data, never instructions**; the planner routes structurally and never executes field content, and excerpts are string-extracted |
| A malformed live record | every descent is type-guarded and one bad record is skipped, never allowed to sink the batch |

*On the loop harness, stated rather than hidden:* under v1's single-shot planner and its ≤ 1
escalation budget, the iteration / tool-call / node-visit caps **cannot fire in normal operation** —
`plan` is entered at most twice and `execute` runs once. The stall detector is gated the same way (it
arms on a third plan entry, `iter_count >= 2`), because the one sanctioned re-plan may legitimately
reproduce the same plan and must be allowed to settle into a clean `empty` rather than abort. All
four are real, active, unit-tested code — defense-in-depth against a routing defect or a future
multi-tool planner, with the headroom declared instead of implied. The two guards that *do* bind on
every request are the wall-clock deadline and the page budget.

## The security model

- **Least-privilege egress.** A base-URL-pinned, GET-only HTTP client (the host is parsed, not
  prefix-matched; userinfo, non-standard ports, and redirects are refused). Tools reach only the
  registry; the LLM adapter reaches only its provider. The provider key is read solely in the adapter,
  passed to no tool, and redacted from all logs and output.
- **Query-injection neutralization.** The API parses the Essie query language on `query.*` values after
  URL-decoding, so a user-supplied field value could smuggle operators or a cross-field selector. The
  single parameter builder neutralizes this: a clean value passes through for full recall; a value
  carrying an Essie metacharacter or a standalone uppercase operator is wrapped as an inert string
  literal. Everything else in a request URL is code-generated from validated tokens.
- **Bounded inputs.** Query and structured-field lengths are capped; the compare/entity fan-out is
  capped; a single id that would ever reach a URL path is format-locked to `^NCT[0-9]{8}$`.

## Key design decisions

- **Field precedence with disclosure.** A structured field is authoritative for its dimension; the
  query supplies intent and gap-fills. On conflict the field wins *and* the override is echoed — never
  a silent pick.
- **Show both counts.** Multi-value fields double-count by nature; every bucket emits the distinct-trial
  count and the mention count, and the headline anchors on distinct trials.
- **Expose the date field.** "Over time" is ambiguous (started vs registered vs completed); the planner
  picks per intent and always discloses which date field it used, and genuine future dates go into a
  flagged "planned" bucket rather than being clamped.
- **Ask when intent is incomplete.** A syntactically valid request whose language names an unresolvable
  referent ("this drug", with no drug field) is neither a 422 nor a guess — it is a first-class
  clarification.

## Deliberately out of scope (v1)

Scatter (trials lack an honest two-continuous-axis pair; a study-duration histogram ships instead);
entity *display* canonicalization beyond heuristics (the API resolves search-recall synonyms already);
network node types beyond sponsor↔drug and drug↔drug; intra-request concurrency (serial paging, bounded
by the page budget); a read-only MCP surface and conversational memory (both designed-for, neither
built). Each is a documented boundary, not an omission.
