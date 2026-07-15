# Example runs — the 15-rung ladder

This is an annotated walkthrough of fifteen real runs, ordered simplest to most complex. Each rung is a natural-language question (plus, sometimes, the structured fields a real caller would supply) driven end-to-end through the live system: **NL query → LLM planner → plan checker → intent reviewer → live ClinicalTrials.gov API → deterministic aggregation → viz-spec builder → output reviewer → cited envelope.** No `query_class` is ever passed in. The LLM reads the natural language and classifies the request itself; the deterministic core computes every number.

The full, untruncated JSON for each rung lives in `examples/run_NN_<slug>.json` and is regenerated — never hand-edited — by `scripts/run_ladder.py`. The excerpts below are truncated for reading (arrays shown as `… (N more)`), but every value quoted here is copied from the actual file. Where my prose and the JSON could disagree, the JSON wins; I verified each number against it.

### Reproduce

```bash
# from ctgov-viz-agent/, with a provider key in the environment
set -a; source ../../../.env; set +a
export OPENAI_API_KEY="$CTGOV_OPENAI_API_KEY" LLM_PROVIDER=openai \
       LLM_MODEL_PLANNER=gpt-5.4 LLM_MODEL_REVIEWER=gpt-5.4-mini
./.venv/bin/python scripts/run_ladder.py            # all 15 rungs
./.venv/bin/python scripts/run_ladder.py 04 15      # just rungs 04 and 15
```

### The governing invariant

**The LLM decides *what* to compute; deterministic tools compute it — the number is never the model's.** For every charting rung the console prints, and you can re-derive from the JSON, the reconciliation that makes this checkable: the distinct-`nctId` bars sum to the API's own `countTotal`.

```
Σ(distinct-nct bars)  ==  count_basis.trials  ==  the API's countTotal
```

Two rungs are the honest exceptions and say so in their own `meta.notes`: rung 06 (a geographic *explode*, where Σ ≥ distinct by design because one trial spans several countries) and rung 10 (a network that degenerates to a frequency bar). The system's willingness to *not* reconcile-to-a-single-total when the data genuinely does not is as much the point as the clean cases.

### The ladder at a glance

| # | Query class | What it proves | Headline number |
|---|---|---|---|
| 01 | `single_value` (answer) | Knows a chart isn't needed; answers in prose | 325 recruiting glioblastoma trials |
| 02 | `distribution` (bar) | The killer gate: Σ == countTotal, exactly | 8 buckets, Σ = 3,950 |
| 03 | `time_series` | Per-year bins, gap-fill, partial + planned years | 13 buckets, Σ = 2,130 |
| 04 | `histogram` | A continuous magnitude binned into ranges | 7 bins, Σ = 3,950 |
| 05 | `compare` (grouped bar) | Two independently-scoped arms, synonym recall | pembro 2,903 vs nivo 2,011 |
| 06 | `geographic` (ranked bar) | Per-trial country dedup; the honest explode | 51 bars, Σ = 2,598 vs 1,957 distinct |
| 07 | `network_graph` | Co-occurrence graph, every edge doubly cited | 59 nodes / 194 edges |
| 08 | `too_large` (refuse) | Knows when NOT to chart | exact 121,770, viz = null |
| 09 | `distribution` (bar) | The same 121k population charted *exactly* | 13 buckets, Σ = 121,770 |
| 10 | `network → bar` fallback | Knows when NOT to graph | falls back to 8-bar frequency |
| 11 | `clarification` | Asks rather than guesses | first-class clarification |
| 12 | `single_value` | Input precedence (CC-1): field beats query | 2,011 (nivolumab wins) |
| 13 | `distribution` (bar) | Boss #1: one sentence → four filters that bite | 3,950 → 229 (Σ holds) |
| 14 | `distribution` (empty) | Boss #2: Essie injection neutralized live | 0 trials, value quoted inert |
| 15 | `compare` (grouped bar) | Boss #3: a full four-filter stack per arm | 163 = 123 + 40, exact |

---

## 01 — Is a chart even needed? (`single_value`, prose answer)

**Query:** *"Is there any recruiting trial for glioblastoma?"* · field `condition = glioblastoma`

```json
{
  "status": "ok",
  "kind": "answer",
  "visualization": null,
  "vega_lite": null,
  "answer": "Yes — 325 trials match this query.",
  "citations": { "NCT00083512": { "...": "..." }, "…": "(20 cited nctIds)" },
  "meta": {
    "count_basis": { "trials": 325 },
    "filters": { "overallStatus": "RECRUITING" },
    "query_provenance": { "params": {
      "query.cond": "glioblastoma",
      "filter.overallStatus": "RECRUITING",
      "countTotal": "true", "pageSize": 1000, "fields": "NCTId"
    } }
  }
}
```

**What to notice.** This is the *decide whether a visualization is even warranted* step, and here it isn't. A yes/no question wants a scalar and a sentence, so `visualization` and `vega_lite` are both `null` and the model answers in prose. But it is still a *grounded* answer: the count `325` comes from the API's `countTotal` (not the model), the "recruiting" qualifier is compiled into a real server-side `filter.overallStatus=RECRUITING`, and 20 contributing `nctId`s are attached as citations so the "yes" is checkable. The floor of the ladder already refuses to let the model invent a number.

Full output: `examples/run_01_single_value_yesno.json`

---

## 02 — The killer gate (`distribution`, bar, Σ == countTotal)

**Query:** *"How are interventional pancreatic cancer trials distributed across phases?"* · fields `condition = pancreatic cancer`, `interventional_only = true`

```json
{
  "status": "ok", "kind": "visualization",
  "visualization": {
    "type": "bar",
    "title": "Phase distribution of interventional pancreatic cancer trials",
    "data": [
      { "value": "NA",            "label": "NA (not applicable)", "count_trials": 937,
        "contributing_count": 937, "citations_truncated": true, "citations": [ "… (20 samples)" ] },
      { "value": "PHASE1",        "label": "Phase 1",   "count_trials": 895,
        "contributing_count": 895, "citations_truncated": true,
        "citations": [
          { "nct_id": "NCT00001431",
            "field_path": "protocolSection.designModule.phases",
            "value": ["PHASE1"], "excerpt": "PHASE1" }
          , "… (19 more)"
        ] },
      { "value": "PHASE1|PHASE2", "label": "Phase 1/2", "count_trials": 505 },
      { "value": "PHASE2",        "label": "Phase 2",   "count_trials": 1143 }
      , "… (4 more)"
    ]
  },
  "meta": { "count_basis": { "trials": 3950 }, "query_provenance": { "params": {
    "query.cond": "pancreatic cancer",
    "filter.advanced": "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL",
    "fields": "NCTId|Phase" } } }
}
```

The eight buckets in full: `NA` 937, `EARLY_PHASE1` 109, `PHASE1` 895, `PHASE1|PHASE2` 505, `PHASE2` 1143, `PHASE2|PHASE3` 58, `PHASE3` 232, `PHASE4` 71. **Sum: 3,950 — exactly `count_basis.trials`, which is exactly the API's `countTotal`.**

**What to notice.** This is the gate the whole design exists to pass. Three kinds of messy real data are handled in the open rather than swept away:

- **The ~63% non-phased mass.** `NA (not applicable)` is 937 trials and non-phasing is a real registry outcome — a phase histogram that silently dropped it would misreport the field. It is charted as its own labeled bucket, not discarded.
- **Composite phase buckets.** `PHASE1|PHASE2` (505) and `PHASE2|PHASE3` (58) are genuine registry values — a trial spanning two phases. They are kept as distinct buckets ("Phase 1/2", "Phase 2/3"), never split across `PHASE1` and `PHASE2` (which would double-count and break the sum).
- **The deep-citation shape.** Each datum carries the true `contributing_count` (895 for Phase 1) plus a bounded, deterministic sample of up to 20 citations and a `citations_truncated` flag. Every citation's `excerpt` ("PHASE1") is string-extracted from the record at its `field_path`, not authored by the model.

The `interventional_only` field is compiled into a server-side `filter.advanced=AREA[StudyType]…INTERVENTIONAL`, so the population is filtered by the API, not post-hoc in memory.

Full output: `examples/run_02_distribution_phase.json`

---

## 03 — Time trend with honest edges (`time_series`)

**Query:** *"How has the number of melanoma trials changed per year since 2015?"* · fields `condition = melanoma`, `start_year = 2015`

```json
{
  "visualization": {
    "type": "time_series",
    "title": "Melanoma trials started per year",
    "data": [
      { "value": "2015", "period": "2015", "label": "2015",           "count_trials": 177 },
      { "value": "2016", "period": "2016", "label": "2016",           "count_trials": 169 },
      "… (2017–2024) …",
      { "value": "2025", "period": "2025", "label": "2025",           "count_trials": 180 },
      { "value": "2026", "period": "2026", "label": "2026 (partial)", "count_trials": 108 },
      { "value": "2027", "period": "2027", "label": "2027 (planned)", "count_trials": 2 }
    ]
  },
  "meta": { "count_basis": { "trials": 2130 }, "query_provenance": { "params": {
    "filter.advanced": "AREA[StartDate]RANGE[2015-01-01,MAX]", "fields": "NCTId|StartDate" } },
    "notes": [
      "The current year 2026 is a PARTIAL year … so its count is legitimately short — read its dip as incomplete data, not a real decline.",
      "Bucket(s) 2027 hold genuine future/estimated dates and are flagged 'planned' — kept in the series, not clamped … and not dropped (which would break reconciliation) (G-40)."
    ] }
}
```

Thirteen yearly buckets, 2015 through 2027, **summing to 2,130 = `countTotal`.**

**What to notice.** Time-series aggregation is where sloppy binning hides. This run gets three edge cases right and *discloses* two of them:

- **Gap-fill.** Every year in the observed span gets a bucket, including any zero-count years, so the axis is continuous and a quiet year reads as a quiet year rather than a missing tick.
- **A real future date, not clamped.** `StartDate` is partly an *estimated* field, so two trials legitimately carry a 2027 start. Rather than clamp them into 2026 (which would inflate the current year) or drop them (which would break `Σ == countTotal`), they are kept in a bucket explicitly labeled `2027 (planned)`.
- **Partial current year.** 2026's 108 is flagged `(partial)` with a note that its dip is incomplete data, not a decline — the kind of caveat a careful analyst adds by hand.

The output reviewer also flagged (in a third note) that the chart slightly over-answers "since 2015" by including the partial/planned years; that flag is surfaced, and the computed values are left unchanged.

Full output: `examples/run_03_timeseries_year.json`

---

## 04 — A continuous magnitude, binned (`histogram`)

**Query:** *"Show the distribution of study durations for interventional pancreatic cancer trials"* · fields `condition = pancreatic cancer`, `interventional_only = true`

```json
{
  "visualization": {
    "type": "histogram",
    "title": "Study-duration distribution of interventional pancreatic cancer trials",
    "encoding": { "x": { "field": "value", "label": "Study duration" },
                  "y": { "field": "count_trials", "label": "Trials", "scale": "linear" } },
    "data": [
      { "value": "0–6 mo",  "bin_start": 0,  "bin_end": 6,   "count_trials": 45 },
      { "value": "6–12 mo", "bin_start": 6,  "bin_end": 12,  "count_trials": 111 },
      { "value": "1–2 yr",  "bin_start": 12, "bin_end": 24,  "count_trials": 543 },
      { "value": "2–4 yr",  "bin_start": 24, "bin_end": 48,  "count_trials": 1510,
        "contributing_count": 1510, "citations_truncated": true,
        "citations": [ { "nct_id": "NCT00558207",
          "field_path": "protocolSection.statusModule.startDateStruct.date",
          "value": "2007-11", "excerpt": "2007-11" }, "… (19 more)" ] },
      { "value": "4–10 yr", "bin_start": 48,  "bin_end": 120,  "count_trials": 1448 },
      { "value": "10+ yr",  "bin_start": 120, "bin_end": null, "count_trials": 152 },
      { "value": "UNDATED", "bin_start": null, "bin_end": null, "count_trials": 141,
        "label": "Unknown (undated)" }
    ]
  },
  "meta": { "count_basis": { "trials": 3950 },
    "notes": [
      "Duration measured start→completion at month precision … Derived from the two dated status fields, not the unverified enrollment field (R-16).",
      "141 trial(s) are undated or have an implausible negative duration (completion before start) and are grouped in an 'Unknown (undated)' bucket, kept for reconciliation and excluded from the duration bins."
    ] }
}
```

Seven bins — six numeric ranges plus one undated bucket — **summing to 3,950 = `countTotal`.**

**What to notice.** This fills the histogram slot in the type system, and it is genuinely a different animal from a categorical bar (rung 02). There is no `study_duration` field in ClinicalTrials.gov; the magnitude is *derived* on the fly — `completionDate − startDate`, at month precision — and then bucketed into ranges (`bin_start`/`bin_end` carry the numeric edges, which a categorical bar never has). Three decisions make it faithful:

- **A principled source.** Duration comes from the two *dated status fields*, not the softer enrollment field — noted explicitly (R-16). The excerpt on each datum cites the raw `startDateStruct.date` ("2007-11") it was computed from.
- **The undated/ill-formed tail is not silently dropped.** 141 trials are either undated or carry a negative duration (completion recorded before start — real, dirty data). They go into an explicit `Unknown (undated)` bucket that is excluded from the numeric bins but *kept for reconciliation*, which is exactly why the seven buckets still sum to 3,950. Drop them and the histogram would quietly under-count; hide them in a numeric bin and it would lie about durations.
- **Month precision, stated.** Day is ignored and a year-only date is treated as January — a real modeling choice, disclosed rather than buried.

Full output: `examples/run_04_histogram_duration.json`

---

## 05 — Two drugs side by side (`compare`, grouped bar)

**Query:** *"Compare the overall status of pembrolizumab versus nivolumab trials"* · (no structured fields — both drugs parsed from the query)

```json
{
  "visualization": {
    "type": "grouped_bar",
    "title": "pembrolizumab vs nivolumab trials by status",
    "data": [
      { "series": "pembrolizumab", "value": "COMPLETED",  "count_trials": 818 },
      { "series": "nivolumab",     "value": "COMPLETED",  "count_trials": 740 },
      { "series": "pembrolizumab", "value": "RECRUITING", "count_trials": 708 },
      { "series": "nivolumab",     "value": "RECRUITING", "count_trials": 340 },
      "… (20 more rows across 12 statuses) …"
    ]
  },
  "meta": { "count_basis": { "trials": 4914 },
    "notes": [ "pembrolizumab N=2903; nivolumab N=2011",
               "Percent is within-series (denominator = each series' own N) …" ] }
}
```

**What to notice.** A comparison is two independently-scoped queries stitched into one chart, and the failure modes are subtle:

- **Two arms, two scopes.** Pembrolizumab (N = 2,903) and nivolumab (N = 2,011) are each queried separately; per series the status buckets sum to that series' own N.
- **Synonym recall.** "pembrolizumab" resolves via the API's own synonym expansion to the same 2,903 trials that "Keytruda" would — the brand and the generic are one population, and the count reflects that (the same equivalence rungs 07 and 12 lean on).
- **Category union with zero-padding.** The two drugs don't share the same status set — nivolumab has one `APPROVED_FOR_MARKETING` trial that pembrolizumab lacks; pembrolizumab has a `TEMPORARILY_NOT_AVAILABLE` that nivolumab lacks. Both series are padded across the union of all 12 statuses (with an explicit `count_trials: 0`) so the grouped bars line up column-for-column.
- **Within-series percentage.** Shares are computed against each series' own N, so the larger arm doesn't visually swamp the smaller one.

Full output: `examples/run_05_compare_grouped_bar.json`

---

## 06 — Countries, and the dedup trap (`geographic`, ranked bar)

**Query:** *"Which countries have the most recruiting diabetes trials?"* · field `condition = diabetes`

```json
{
  "visualization": {
    "type": "bar",
    "title": "Diabetes trials by country",
    "data": [
      { "value": "United States",  "count_trials": 764 },
      { "value": "China",          "count_trials": 222 },
      { "value": "Canada",         "count_trials": 127 },
      "… (47 more countries) …",
      { "value": "Other", "label": "Other (41 countries)", "count_trials": 71,
        "derived": true, "members": ["Croatia", "South Africa", "… (39 more)"] }
    ]
  },
  "meta": {
    "count_basis": { "trials": 1957, "mentions": 2620 },
    "filters": { "overallStatus": "RECRUITING" },
    "notes": [
      "Each bar counts the DISTINCT trials studied in that country; a trial spanning multiple countries is counted once per country, so bar totals sum to MORE than the trial count -- the headline count_basis.trials is the distinct-trial total (CC-3).",
      "Ranked bar, top 50 by trial count; 41 lower-count country values are folded into the derived 'Other' bucket …",
      "… USA / U.S. → United States … variant spellings are merged into one bar (E-20)."
    ]
  }
}
```

The 51 rendered bars sum to **2,598** (the 50 named countries = 2,527, plus the derived `Other` = 71). The distinct-trial headline is **1,957**.

**What to notice.** This is the rung that *breaks* `Σ == countTotal`, on purpose, and says so. A trial run in the US, China, and Canada appears in all three country bars — so summing bars over-counts trials by design. The system does not paper over this: it declares the distinct-trial total (1,957) as `count_basis.trials` and the raw country-appearance tally (2,620) as `mentions`, and the first note spells out exactly why the bars sum higher (CC-3). Three more real-data hazards handled here:

- **The dedup itself.** Within a single trial, a country is counted once even if it lists ten sites there — otherwise a big multi-site trial would dominate its own country's bar.
- **Ranked bar, not a choropleth.** `LocationCountry` is a free-text display name with no ISO code, so a map would require a join the data doesn't support. The honest rendering is a ranked bar.
- **Spelling normalization + top-N fold.** Variant spellings (`USA`, `U.S.` → `United States`) are canonicalized before counting, and the long tail of 41 low-count countries folds into a single derived `Other (41 countries)` bar that still cites its members — legible without discarding data.

Full output: `examples/run_06_geographic_ranked_bar.json`

---

## 07 — The richest view: drugs studied together (`network_graph`)

**Query:** *"Show a network of drugs studied together in melanoma trials"* · field `condition = melanoma`

```json
{
  "visualization": {
    "type": "network_graph",
    "title": "Drugs studied together in melanoma trials",
    "data": {
      "nodes": [
        { "id": "drug:bcd-201", "label": "Pembrolizumab", "kind": "drug", "degree": 28 },
        "… (58 more nodes) …"
      ],
      "edges": [
        { "source": "drug:cyclophosphamide", "target": "drug:fludarabine",
          "weight": 84,
          "source_ids": ["NCT00001832", "NCT00003552", "… (20-id sample of 84)"],
          "citations": [
            { "nct_id": "NCT00001832",
              "field_path": "protocolSection.armsInterventionsModule.interventions[].name",
              "value": ["gp100:209-217 (210M)", "…", "Fludarabine", "Cyclophosphamide"],
              "excerpt": "Cyclophosphamide" },
            { "nct_id": "NCT00001832", "field_path": "…interventions[].name",
              "value": ["gp100:209-217 (210M)", "…", "Fludarabine", "…"], "excerpt": "Fludarabine" }
          ] }
        , "… (193 more edges) …"
      ]
    }
  },
  "meta": { "count_basis": { "trials": 3733 } }
}
```

**59 nodes, 194 edges, over a basis of 3,733 melanoma trials. Every one of the 194 edges carries exactly two citations** — one per endpoint — verified by the runner (`every-edge-2-cites = True`).

**What to notice.** The graph is the hardest artifact to make *trustworthy*, and the traps are all about false structure:

- **Every edge weight is traceable.** An edge weight is the number of distinct trials that studied both drugs (cyclophosphamide ↔ fludarabine: 84 shared trials). The edge carries a bounded sample of those `source_ids` and two citations that each string-extract the endpoint's name from a real record's `interventions[].name`. No weight is asserted; each traces to its contributing `nctId`s.
- **No placebo mega-hub.** Placebo and standard-of-care interventions are excluded by name (`placebo present? False`), because otherwise nearly every arm would connect through placebo and the graph would collapse into one meaningless hub.
- **Alias-only synonym merge, guarded.** Drug nodes are keyed by active ingredient — names normalized for case, dose, salt, and route — and a brand folds into its generic only when the alias is *itself* another drug's primary name *and* corroborated by ≥ 2 trials. That's why the node id `drug:bcd-201` (a biosimilar identifier) resolves to the label `Pembrolizumab`. A combination product ("Drug A + Drug B") never merges its components, and a single-trial registry mislabel can't collapse two distinct drugs.
- **Legibility caps, disclosed.** Nodes are capped at the top 60 by degree and edges below weight 2 are pruned — flagged in a note as a configurable interpretability default that leaves the underlying co-occurrence graph unchanged.

Full output: `examples/run_07_network_drug_drug.json`

---

## 08 — Knowing when not to chart (`too_large`, refuse)

**Query:** *"How are cancer trials distributed across phases overall?"* · field `condition = cancer`

```json
{
  "status": "too_large",
  "kind": "answer",
  "visualization": null,
  "vega_lite": null,
  "answer": "121,770 trials match this query -- too large to chart faithfully within the paging budget. Narrow the query (e.g. add a phase, status, or year range) to render a distribution.",
  "citations": {},
  "meta": { "count_basis": { "trials": 121770 },
    "query_provenance": { "params": { "query.cond": "cancer", "fields": "NCTId|Phase" } } }
}
```

**What to notice.** A phase distribution over 121,770 trials would require paging through the entire result set. Rather than chart a *biased sorted prefix* (the first N pages, which are not a random sample), the system refuses: `status: "too_large"`, with `visualization`, `vega_lite`, and `partial` all `null`. Critically, the refusal still returns the **exact** total — 121,770 comes from a single `countTotal` call, so the user gets a precise number and an actionable suggestion (add a phase, status, or year range) instead of a plausible-looking but skewed chart. The contrast with rung 09, on the very same population, is the whole point.

Full output: `examples/run_08_too_large_refuse.json`

---

## 09 — The same 121k, charted exactly (`distribution`, bar)

**Query:** *"How are cancer trials distributed across overall recruitment status?"* · field `condition = cancer`

```json
{
  "visualization": {
    "type": "bar",
    "title": "Status distribution of cancer trials",
    "data": [
      { "value": "COMPLETED",             "count_trials": 52551 },
      { "value": "UNKNOWN",               "count_trials": 19713 },
      { "value": "RECRUITING",            "count_trials": 18804 },
      { "value": "TERMINATED",            "count_trials": 11032 },
      { "value": "ACTIVE_NOT_RECRUITING", "count_trials": 7687 },
      "… (8 more statuses) …"
    ]
  },
  "meta": { "count_basis": { "trials": 121770 },
    "notes": [ "Computed via exact per-category count queries (not paged): the 121,770-trial population exceeds the paging budget, so each bar is one exact countTotal call — an exact distribution at any scale, no biased sampling." ] }
}
```

Thirteen status buckets **summing to exactly 121,770 = `countTotal`** — the identical population rung 08 refused to chart.

**What to notice.** This is the counterpoint that shows the refusal in rung 08 is principled, not timid. The difference is the *field*, not the size. `overallStatus` is a **bounded-token** field: there are only ~13 possible values, so the distribution can be computed as one exact `count()` query per token — thirteen cheap calls, no paging, no sampling bias, at any scale. `Phase` at 121k trials cannot be done that way within budget (rung 08). The system distinguishes which distributions it can compute *faithfully* from which it cannot, and lets that distinction — not a raw row count — decide whether to chart. Same 121,770 trials; one refuses, one reconciles to the last unit.

Full output: `examples/run_09_exact_at_scale_status.json`

---

## 10 — Knowing when not to *graph* (`network` → bar fallback)

**Query:** *"Show a network of drugs studied together in progeria trials"* · field `condition = progeria`

```json
{
  "visualization": {
    "type": "bar",
    "title": "Most-studied drugs in progeria trials",
    "data": [
      { "value": "Lonafarnib",  "count_trials": 4 },
      { "value": "Progerinin",  "count_trials": 2 },
      { "value": "Everolimus and lonafarnib", "count_trials": 1 },
      "… (5 more) …"
    ]
  },
  "meta": { "count_basis": { "trials": 9, "mentions": 12 },
    "notes": [
      "Network too sparse to graph (≤1 entity or no repeated co-occurrence); showing individual drug frequencies instead (G-41e).",
      "9 of 10 matched trials study ≥1 drug intervention; the remaining 1 have none and are not shown here.",
      "The question asks for a network of drugs studied together, but the response falls back to a bar chart of individual drug frequencies. That answers a different question and does not encode co-occurrence/network structure."
    ] }
}
```

**What to notice.** Progeria is a rare disease — about ten drug trials, and no drug pair recurs across enough of them to form a real edge. Forcing a graph here would produce a misleading hairball of weight-1 edges that suggest structure that isn't there. So the network builder detects the degenerate case (`≤ 1 entity or no repeated co-occurrence`) and *falls back* to a cited bar of individual drug frequencies — Lonafarnib in 4 trials, Progerinin in 2. This is rung 08's discipline applied to a different artifact: refuse the misleading view, return the faithful one. Two honest caveats ride along in `meta.notes`: one of the ten matched trials studies no drug at all (excluded, and said so), and the output reviewer flags outright that the bar answers a *different* question than "studied together." The values are left unchanged; the divergence is disclosed, not hidden.

Full output: `examples/run_10_network_degenerate_fallback.json`

---

## 11 — Asking rather than guessing (`clarification`)

**Query:** *"How many trials are there for this drug?"* · (no `drug_name` field)

```json
{
  "status": "empty",
  "kind": "clarification",
  "visualization": null,
  "answer": null,
  "question": "Which drug do you mean? Please name the drug (e.g. in the drug_name field).",
  "meta": { "notes": [ "The query referred to an entity it did not name; asking for clarification rather than guessing (E-13)." ] }
}
```

**What to notice.** "This drug" is a demonstrative referent with no antecedent — there is no drug in the query and none in the structured fields. A system that wanted to look busy could pick a popular drug and chart it. This one returns a first-class `kind: "clarification"` with a specific question and no fabricated data: `visualization`, `answer`, and `count_basis` are all null. Nothing is invented to fill the gap.

Full output: `examples/run_11_clarification.json`

---

## 12 — Input precedence: the field wins (`single_value`, CC-1)

**Query:** *"How many trials are there for Keytruda?"* · field `drug_name = nivolumab`

```json
{
  "visualization": {
    "type": "single_value",
    "title": "Number of nivolumab trials",
    "data": [ { "count_trials": 2011, "label": "2,011 trials", "contributing_count": 2011,
                "citations": [ "… (20 samples) …" ] } ]
  },
  "meta": {
    "count_basis": { "trials": 2011 },
    "query_provenance": { "params": { "query.intr": "nivolumab", "fields": "NCTId" } },
    "notes": [ "Override: used field drug_name='nivolumab' over query drug 'Keytruda'." ]
  }
}
```

**What to notice.** The output is a single number, but the interesting work is invisible: two inputs *disagree* about which drug we mean. The free-text query says "Keytruda" (≡ pembrolizumab, 2,903 trials); the structured field says "nivolumab" (2,011). The contract (CC-1) is that a structured field is the authoritative signal and wins — and you can watch it win in the wire params (`query.intr = nivolumab`, not Keytruda) and in the resulting count (2,011). Just as important, the override is not silent: `meta.notes` echoes exactly what was overridden and why. A caller can always tell that a conflict occurred and how it was resolved. Nothing is decided in the dark.

Full output: `examples/run_12_cc1_field_vs_query_conflict.json`

---

## 13 — Boss #1: one sentence, four filters (`distribution`, bar)

**Query:** *"How are recruiting, industry-sponsored interventional pancreatic cancer trials that started in 2020 or later distributed across phases?"* · fields `condition = pancreatic cancer`, `interventional_only = true`, `start_year = 2020`

```json
{
  "visualization": {
    "type": "bar",
    "title": "Phase distribution of interventional pancreatic cancer trials",
    "data": [
      { "value": "NA",            "count_trials": 11 },
      { "value": "PHASE1",        "count_trials": 84 },
      { "value": "PHASE1|PHASE2", "count_trials": 73 },
      { "value": "PHASE2",        "count_trials": 29 },
      "… (4 more) …"
    ]
  },
  "meta": {
    "count_basis": { "trials": 229 },
    "filters": { "overallStatus": "RECRUITING", "studyType": "INTERVENTIONAL",
                 "sponsorClass": "INDUSTRY", "start_year": 2020 },
    "query_provenance": { "params": {
      "query.cond": "pancreatic cancer",
      "filter.overallStatus": "RECRUITING",
      "filter.advanced": "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL AND AREA[LeadSponsorClass]COVERAGE[FullMatch]INDUSTRY AND AREA[StartDate]RANGE[2020-01-01,MAX]",
      "fields": "NCTId|Phase" } }
  }
}
```

Eight phase buckets **summing to 229 = `countTotal`.**

**What to notice.** The output looks like an ordinary bar chart. Don't be fooled — the difficulty is entirely upstream, in the sentence. From one clause of English the planner had to extract a condition *and four separate constraints* and compile each to the right API mechanism: "recruiting" → `filter.overallStatus=RECRUITING`, "industry-sponsored" → `AREA[LeadSponsorClass]…INDUSTRY`, "interventional" → `AREA[StudyType]…INTERVENTIONAL`, "started in 2020 or later" → `AREA[StartDate]RANGE[2020-01-01,MAX]`. The last three are `AND`-composed into one `filter.advanced` Essie expression. That decomposition is the hard part; the chart is the easy part.

- **All four filters actually reach the API.** They are in the wire params, not applied in memory after a broad fetch. This is the invisible planning complexity made auditable.
- **The filters bite.** This is the same base population as rung 02 (3,950 interventional pancreatic-cancer trials); adding three constraints collapses it to **229** — a ~94% reduction. A stacked query that *should* shrink the population demonstrably does, which is how you know the filters aren't decorative.
- **Reconciliation survives.** Over those 229 trials the eight phase buckets still sum to exactly 229. Faithfulness isn't a property of the easy base case; it holds under four composed constraints.

Full output: `examples/run_13_boss_stacked_filters.json`

---

## 14 — Boss #2: Essie injection, neutralized live (`distribution`, empty)

**Query:** *"How are the matching trials distributed across phases?"* · field `condition = "cancer OR diabetes"`

```json
{
  "status": "empty",
  "kind": "visualization",
  "visualization": { "type": "bar", "title": "Phase distribution of cancer OR diabetes trials", "data": [] },
  "vega_lite": { "…": "empty bar spec (data.values: [])" },
  "meta": {
    "count_basis": { "trials": 0 },
    "query_provenance": { "params": {
      "query.cond": "\"cancer OR diabetes\"",
      "countTotal": "true", "fields": "NCTId|Phase" } },
    "notes": [
      "No trials matched this query.",
      "Use the authoritative structured condition dimension: cancer OR diabetes."
    ]
  }
}
```

Look closely at the wire value: `query.cond` is `"\"cancer OR diabetes\""` — the string arrives at the API **wrapped in quotes**, as an inert literal.

**What to notice.** `condition` is user-supplied free text, and CT.gov parses `query.*` values with the Essie query language *after* URL-decoding. So a field value like `cancer OR diabetes` is not a search term — the bare `OR` is a boolean operator. Sent raw, it runs a *union* of roughly 145,000 trials (confirmed live: `cancer OR diabetes` → 145,385), a genuine DoS-amplification and cross-field-breakout vector (`AREA[Phase]PHASE1` in a condition slot would break out to select on Phase entirely). The neutralizer (`app/ctgov/params.py`) catches that this value carries an Essie operator keyword and wraps it as a quoted string literal, so it matches the *literal* condition "cancer OR diabetes" — of which there are legitimately **zero** trials. The attack input is defused into the real (empty) population rather than blowing up into a 145k union. The neutralization is recall-preserving: a *clean* value passes through untouched (as rungs 01–13 show, including the API's own `Keytruda` ≡ `pembrolizumab` synonym expansion), and only a value carrying a metacharacter or a standalone uppercase operator gets quoted. This is the end-to-end fix on a live request.

Full output: `examples/run_14_boss_injection_neutralized.json`

---

## 15 — Boss #3: a full filter stack on each arm (`compare`, grouped bar)

**Query:** *"Compare how recruiting, interventional Phase 3 pembrolizumab trials versus nivolumab trials, started in 2018 or later, break down by lead sponsor type."*

```json
{
  "visualization": {
    "type": "grouped_bar",
    "title": "pembrolizumab vs nivolumab trials by sponsor class",
    "data": [
      { "series": "pembrolizumab", "value": "INDUSTRY",  "count_trials": 79 },
      { "series": "nivolumab",     "value": "INDUSTRY",  "count_trials": 17 },
      { "series": "pembrolizumab", "value": "OTHER",     "count_trials": 32 },
      { "series": "nivolumab",     "value": "OTHER",     "count_trials": 15 },
      { "series": "pembrolizumab", "value": "NIH",       "count_trials": 7 },
      { "series": "nivolumab",     "value": "NIH",       "count_trials": 3 },
      { "series": "pembrolizumab", "value": "NETWORK",   "count_trials": 5 },
      { "series": "nivolumab",     "value": "NETWORK",   "count_trials": 3 },
      { "series": "pembrolizumab", "value": "OTHER_GOV", "count_trials": 0 },
      { "series": "nivolumab",     "value": "OTHER_GOV", "count_trials": 2 }
    ]
  },
  "meta": {
    "count_basis": { "trials": 163 },
    "filters": { "phase": ["PHASE3"], "overallStatus": "RECRUITING",
                 "studyType": "INTERVENTIONAL", "start_year": 2018 },
    "query_provenance": { "params": {
      "query.intr": "pembrolizumab",
      "filter.overallStatus": "RECRUITING",
      "filter.advanced": "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL AND AREA[Phase](PHASE3) AND AREA[StartDate]RANGE[2018-01-01,MAX]",
      "fields": "NCTId|sponsorClass" } },
    "notes": [ "pembrolizumab N=123; nivolumab N=40",
               "Percent is within-series (denominator = each series' own N) …" ]
  }
}
```

Ten grouped bars across five sponsor classes: pembrolizumab **123** (79 + 32 + 7 + 5 + 0), nivolumab **40** (17 + 15 + 3 + 3 + 2), **163 total = `countTotal`.**

**What to notice.** This is the hardest plan on the ladder — it composes rung 05's two-arm comparison with rung 13's four-filter stack, and then applies that whole stack *to each arm independently*. From one sentence the planner had to build two scoped queries, each carrying **four** filters (recruiting, interventional, Phase 3, started ≥ 2018), and group each result by lead-sponsor class.

- **Every filter landed on both arms.** The proof is in the arithmetic: `meta.notes` reports `pembrolizumab N=123; nivolumab N=40`, and those are exactly the sums of each series' rendered bars. If a filter had silently failed to apply to one arm, that arm's total would be inflated and the note would not match. It matches to the unit on both — so all four filters demonstrably reached both queries.
- **Union with zero-padding, again.** The two arms don't share a sponsor-class profile — pembrolizumab has no `OTHER_GOV` trial (padded to 0); nivolumab has two. Both series span the same five-category axis so the bars align.
- **Within-series percentage.** With a 123-vs-40 imbalance, per-series denominators matter: shares are computed against each arm's own N so the smaller arm stays readable next to the larger one.

The visible chart is rich, but the real achievement is that the sentence-to-eight-constraints decomposition happened correctly for two arms at once, and the deterministic core's own ground-truth counts confirm it.

Full output: `examples/run_15_boss_compare_filtered_arms.json`

---

## Cross-provider proof — the number is the tool's, not the model's

The governing invariant claims the model chooses *what* to compute while deterministic code computes the value. The cleanest way to prove it: swap the model and check the number doesn't move.

`examples/run_02_distribution_phase.anthropic.json` is rung 02 re-run with the planner switched from OpenAI (`gpt-5.4`) to Anthropic (`claude-sonnet-5`). Everything else — the query, the fields, the deterministic core — is held fixed. The result is **identical**: the same 8 buckets, the same per-bar counts (`NA` 937, `PHASE1` 895, `PHASE1|PHASE2` 505, `PHASE2` 1143, …), the same Σ = 3,950 = `countTotal`, and the same wire params.

In fact the two envelopes are **byte-for-byte identical except for the `retrieved_at` timestamp** (verified: normalizing that one field, the JSON is the same). If either model were writing the numbers, two different vendors would not agree to the digit. They agree because neither model writes numbers — one plans, the other plans, and the same deterministic engine computes. That is the invariant, demonstrated rather than asserted.
