"""Regenerate the Phase-1 killer-gate example run (X-2) from a LIVE API call.

This is NOT hand-edited output (A-57/V-9): it runs the real pipeline through the
FastAPI route and writes the ACTUAL response envelope, pretty-printed, verbatim.

Run:  ./.venv/bin/python scripts/run_gate.py
Writes: examples/run_x2_distribution_interventional_pancreatic.json
Prints: the reconciliation proof (Σ distinct-trial counts == the API's countTotal).

X-2 = "Phase distribution of interventional pancreatic cancer trials" — the
sub-20k population that charts (not too_large), with the interventional filter
applied SERVER-SIDE (filter.advanced=AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL)
on BOTH the countTotal call and every page, so one population is seen and the
distinct-nctId aggregate reconciles to the exact countTotal (CC-16, the gate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "examples" / "run_x2_distribution_interventional_pancreatic.json"

_REQUEST = {
    "query": "Phase distribution of interventional pancreatic cancer trials",
    "condition": "pancreatic cancer",
    "interventional_only": True,
}


def main() -> int:
    client = TestClient(app)
    resp = client.post("/visualize", json=_REQUEST)
    if resp.status_code != 200:
        print(f"FAIL: HTTP {resp.status_code}: {resp.text[:400]}")
        return 1

    body = resp.json()
    if body["status"] != "ok" or not body.get("visualization"):
        # Distinguish a transient upstream rate-limit (the public API 429s under a
        # burst — e.g. running this right after the full test suite) from a real
        # failure. The example JSON is NOT overwritten in either case.
        err = body.get("error") or {}
        if body["status"] == "error" and err.get("code") == "upstream_error":
            print("TRANSIENT: clinicaltrials.gov rate-limited/failed this call "
                  "(retry in ~30s; the public API is ~3 req/s). Example JSON left untouched.")
            return 2
        print(f"FAIL: expected an ok visualization, got status={body['status']}")
        print(json.dumps(body, indent=2)[:800])
        return 1

    sigma = sum(d["count_trials"] for d in body["visualization"]["data"])
    total = body["meta"]["count_basis"]["trials"]
    reconciles = sigma == total

    _OUT.parent.mkdir(exist_ok=True)
    _OUT.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {_OUT.relative_to(_ROOT)}")
    print(f"status={body['status']} kind={body['kind']} chart={body['visualization']['type']}")
    print(f"buckets: {[(d['value'], d['count_trials']) for d in body['visualization']['data']]}")
    print(f"Σ distinct-trial counts = {sigma}  |  API countTotal = {total}  |  RECONCILES: {reconciles}")
    print(f"citations index: {len(body['citations'])} nctIds  |  vega_lite: {body['vega_lite'] is not None}")
    if body["meta"]["notes"]:
        print(f"notes: {body['meta']['notes']}")
    return 0 if reconciles else 1


if __name__ == "__main__":
    sys.exit(main())
