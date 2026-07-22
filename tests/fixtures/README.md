# `tests/fixtures/golden_*.json` — schema fixtures, NOT engine output

**Read this before quoting a golden as "what the API returns". It is not.**

These six files are **hand-written Phase-0 material**, authored before the engine
existed, one per envelope shape:

| file | shape |
|---|---|
| `golden_distribution.json` | `status:"ok"` / `kind:"visualization"`, bar rows |
| `golden_timeseries.json` | `status:"ok"` / `kind:"visualization"`, time-series rows |
| `golden_network.json` | `status:"ok"` / `kind:"visualization"`, `NetworkData` |
| `golden_answer.json` | `status:"ok"` / `kind:"answer"` |
| `golden_error.json` | `status:"error"` |
| `golden_too_large.json` | `status:"too_large"` / `kind:"answer"` |

## What they are used for (the whole list)

* `tests/test_schemas.py` — each one validates as a `VisualizeResponse` and
  stamps `meta.source == "clinicaltrials.gov"`; all six shapes exist on disk.
* `tests/test_hardening_schemas.py` — every `source_id` on every datum/edge
  resolves to a real `Citation` (inline or in the top-level dedup index, G-4).
* `tests/test_reviewers.py` — `golden_distribution.json` is fed to the LLM
  Output Reviewer through `StubAdapter` as a *well-formed input*; the assertions
  are about the reviewer's verdict, not about the fixture's contents.
* `app/doctor.py` check C3 — same validation as above, run by `ct-doctor`.

Every one of those is a **schema-validity** use. No test asserts that a golden
equals what the engine produces, and none should: they are **not behaviour
oracles**.

## Why they can't be oracles — concrete divergences from real output

Compare `golden_distribution.json` with the real shipped run of the same query,
`examples/run_02_distribution_phase.json`:

| | golden (hand-written) | engine (real) |
|---|---|---|
| chart title | `... trials (1,000 trials)` | `... trials` — `app/viz/spec.py::_title` never appends a count |
| composite phase datum `value` | `"PHASE1/2"` | `"PHASE1|PHASE2"` — the pipe is the wire token; the slash is only the *label* |
| `encoding.x.sort` | an explicit token-order string | `null` — the engine sets `sort` only on time series, and only to `"ascending"` |
| citation fields | `nct_id`, `field_path`, `value`, `matched_value` | plus `excerpt` (the trial's brief title) and `matched_tokens` |

## The rule

* Demoing or explaining **what the API returns** → use `examples/run_*.json`.
  Those are real captured runs and are gated by `scripts/verify_examples.py`
  (and `tests/test_examples_offline.py`).
* Testing **that the envelope contract holds for a shape** → a golden is fine,
  and that is the only thing it proves.
