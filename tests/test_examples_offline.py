"""Offline gate: every shipped ``examples/run_*.json`` passes the correctness harness.

This puts ``scripts/verify_examples.py`` in the test suite — the same five
invariants (schema, provenance teeth, count coherence, reconciliation, no
LLM-authored number) the harness prints per-file are asserted here, parametrized
per file so a regression names the offending example. Pure + offline: it only
reads the checked-in example JSONs and reuses the runtime verifier primitives; no
network, no graph, no LLM.

The harness lives under ``scripts/`` (not an installed package), so it is loaded
by file path via ``importlib`` — robust regardless of how ``sys.path`` is set up.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_HARNESS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_examples.py"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_examples", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass definitions can resolve
    # ``cls.__module__`` in sys.modules (dataclasses needs it for field typing).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_examples = _load_harness()

_PATHS = verify_examples.iter_example_paths()


def test_examples_are_present() -> None:
    """Guard against a silent vacuous-green: the ladder ships 13 rungs + 1 twin."""
    assert len(_PATHS) >= 13, f"expected the shipped example ladder, found {len(_PATHS)} at {_HARNESS_PATH.parent.parent / 'examples'}"


@pytest.mark.parametrize("path", _PATHS, ids=[p.name for p in _PATHS])
def test_example_passes_all_invariants(path: Path) -> None:
    """Every applicable invariant holds on this shipped example (SKIP is allowed)."""
    report = verify_examples.verify_file(path)
    assert report.ok, "\n".join(report.failure_lines())


def test_harness_exit_code_is_zero_over_all_examples() -> None:
    """The CLI entrypoint exits 0 when every example passes (the gate's contract)."""
    assert verify_examples.main() == 0


# --- negative controls: the checks have teeth --------------------------------
#
# A harness that only ever passes proves nothing. These construct a tampered
# in-memory envelope from a real example and confirm the corresponding invariant
# HARD-FAILS — so a genuine regression cannot slip through green.


def _first_row_example() -> dict:
    """A real example whose visualization carries a row list with citations."""
    for path in _PATHS:
        obj = json.loads(path.read_text(encoding="utf-8"))
        viz = obj.get("visualization")
        if viz and isinstance(viz.get("data"), list) and viz["data"] and viz["data"][0].get("citations"):
            return obj
    pytest.skip("no row-list example with citations found")


def _check_by_label(report, label: str):
    """The single Check with an exact label (harness labels are unique per file)."""
    return next(c for c in report.checks if c.label == label)


def test_fabricated_excerpt_is_caught() -> None:
    """(II) A fabricated citation excerpt (not present in its value) hard-fails."""
    obj = copy.deepcopy(_first_row_example())
    obj["visualization"]["data"][0]["citations"][0]["matched_value"] = "FABRICATED-NOT-IN-VALUE"
    report = verify_examples.verify_obj(obj, "tampered-excerpt")
    assert not report.ok
    assert _check_by_label(report, verify_examples._I_PROV).status == "FAIL"


def test_inflated_bar_breaks_reconciliation() -> None:
    """(IV) Inflating a combine bar so Σ != count_basis.trials hard-fails."""
    obj = copy.deepcopy(_first_row_example())
    if obj["status"] != "ok" or obj["kind"] != "visualization":
        pytest.skip("first row example is not an ok visualization")
    obj["visualization"]["data"][0]["count_trials"] += 1  # Σ now overshoots T
    report = verify_examples.verify_obj(obj, "inflated-bar")
    # An inflated combine bar breaks Σ == T; an explode chart would still satisfy
    # Σ >= T, so only assert the failure for the combine case we picked.
    recon = _check_by_label(report, verify_examples._I_RECON)
    if recon.status != "SKIP":
        assert recon.status == "FAIL", recon.detail


def test_broken_schema_is_caught() -> None:
    """(I) A structurally invalid envelope (bad chart_type↔data shape) hard-fails."""
    obj = copy.deepcopy(_first_row_example())
    obj["visualization"]["type"] = "network_graph"  # a row list under a network mark
    report = verify_examples.verify_obj(obj, "type-shape-mismatch")
    assert not report.ok
    assert _check_by_label(report, verify_examples._I_SCHEMA).status == "FAIL"
