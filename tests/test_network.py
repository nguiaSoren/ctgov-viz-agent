"""Offline unit tests for ``app.ctgov.network.build_graph`` (§W1b).

Pure — no network. Records are synthetic ClinicalTrials.gov-shaped dicts. Covers:
sponsor_drug + drug_drug weights, DEVICE-only trials contributing no drug node,
placebo drop-by-name, Keytruda/pembrolizumab synonym merge, basket-trial skip +
note, degeneracy, the two-citation-per-edge contract with round-trip excerpts,
TOTAL behavior on malformed records, and schema validation of the output shape.

Phase-3 (P3) coverage: the alias-only over-merge regression (distinct drugs
sharing a non-primary code stay separate), sponsor name-variant canonicalization
(merge same-org variants, never fold parent/subsidiary), the per-kind
``min_edge_weight`` default (drug_drug k=2), and the degeneracy ``fallback`` bar.
"""

from __future__ import annotations

import pytest

from app.api.schemas import Edge, NetworkData, Node
from app.ctgov.citations import is_substring_at
from app.ctgov.network import build_graph

_SPONSOR_PATH = "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
_DRUG_PATH = "protocolSection.armsInterventionsModule.interventions[].name"


# --- Synthetic record builders ----------------------------------------------


def _drug(name: str, other: list[str] | None = None, type_: str = "DRUG") -> dict:
    return {"type": type_, "name": name, "otherNames": other}


def _rec(nct: str, sponsor: str | None = None, interventions: list[dict] | None = None) -> dict:
    ps: dict = {"identificationModule": {"nctId": nct}}
    if sponsor is not None:
        ps["sponsorCollaboratorsModule"] = {"leadSponsor": {"name": sponsor}}
    if interventions is not None:
        ps["armsInterventionsModule"] = {"interventions": interventions}
    return {"protocolSection": ps}


def _edge(graph: dict, source: str, target: str) -> dict | None:
    for e in graph["edges"]:
        if {e["source"], e["target"]} == {source, target}:
            return e
    return None


# --- sponsor_drug ------------------------------------------------------------


def test_sponsor_drug_weights() -> None:
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000003", "Merck", [_drug("Lenvatinib")]),
        _rec("NCT00000004", "Bristol-Myers Squibb", [_drug("Nivolumab")]),
    ]
    graph = build_graph(records, kind="sponsor_drug")

    assert graph["degenerate"] is False
    assert graph["distinct_trials"] == 4

    edge = _edge(graph, "sponsor:merck", "drug:pembrolizumab")
    assert edge is not None
    assert edge["weight"] == 2  # trials 1 + 2
    assert sorted(edge["source_ids"]) == ["NCT00000001", "NCT00000002"]
    assert edge["contributing_count"] == 2

    assert _edge(graph, "sponsor:merck", "drug:lenvatinib")["weight"] == 1
    assert _edge(graph, "sponsor:bristol-myers-squibb", "drug:nivolumab")["weight"] == 1

    merck = next(n for n in graph["nodes"] if n["id"] == "sponsor:merck")
    assert merck["kind"] == "sponsor"
    assert merck["degree"] == 2  # pembrolizumab + lenvatinib


def test_min_edge_weight_prunes() -> None:
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000003", "Merck", [_drug("Lenvatinib")]),  # weight-1, pruned
    ]
    graph = build_graph(records, kind="sponsor_drug", min_edge_weight=2)
    assert _edge(graph, "sponsor:merck", "drug:pembrolizumab") is not None
    assert _edge(graph, "sponsor:merck", "drug:lenvatinib") is None  # below min weight


# --- drug_drug ---------------------------------------------------------------


def test_drug_drug_weights() -> None:
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000003", "Merck", [_drug("Pembrolizumab"), _drug("Lenvatinib")]),
    ]
    # This test pins the weight MECHANICS, so pin k=1 explicitly — the drug_drug
    # DEFAULT is now k=2 (P3-WEIGHT), which would prune the weight-1 edge below.
    graph = build_graph(records, kind="drug_drug", min_edge_weight=1)

    assert graph["degenerate"] is False
    assert _edge(graph, "drug:pembrolizumab", "drug:nivolumab")["weight"] == 2
    assert _edge(graph, "drug:pembrolizumab", "drug:lenvatinib")["weight"] == 1
    # every node is a drug in a drug_drug graph
    assert all(n["kind"] == "drug" for n in graph["nodes"])


# --- filtering: type, placebo -----------------------------------------------


def test_device_only_contributes_no_drug_node() -> None:
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000002", "Merck", [_drug("Coronary Stent", type_="DEVICE")]),
    ]
    graph = build_graph(records, kind="sponsor_drug")
    labels = {n["label"] for n in graph["nodes"]}
    assert "Coronary Stent" not in labels
    drug_nodes = [n for n in graph["nodes"] if n["kind"] == "drug"]
    assert [n["label"] for n in drug_nodes] == ["Pembrolizumab"]


def test_placebo_dropped_by_name() -> None:
    # Placebo has type DRUG, so only a name filter removes it (G-36).
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Placebo")]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab"), _drug("Matching Placebo")]),
    ]
    graph = build_graph(records, kind="sponsor_drug")
    labels = {n["label"].lower() for n in graph["nodes"]}
    assert not any("placebo" in label for label in labels)
    assert _edge(graph, "sponsor:merck", "drug:pembrolizumab")["weight"] == 2


# --- synonym merge (CC-12) ---------------------------------------------------


def test_synonym_merge_via_other_names() -> None:
    # Corroborated (residual-1 rule): the alias is attested by ≥2 trials — here
    # BIDIRECTIONALLY (Keytruda→pembrolizumab and Pembrolizumab→Keytruda) — so it merges;
    # a one-off single-trial alias would not (see test_single_trial_mislabel_not_merged).
    records = [
        _rec("NCT00000001", "Merck", [_drug("Keytruda", other=["pembrolizumab"])]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab", other=["Keytruda"])]),
    ]
    graph = build_graph(records, kind="sponsor_drug")
    drug_nodes = [n for n in graph["nodes"] if n["kind"] == "drug"]
    assert len(drug_nodes) == 1  # Keytruda ≡ pembrolizumab → one node
    assert drug_nodes[0]["label"] in {"Keytruda", "Pembrolizumab"}
    # both trials pair the (single) sponsor with the (single) merged drug
    edge = graph["edges"][0]
    assert edge["weight"] == 2
    assert sorted(edge["source_ids"]) == ["NCT00000001", "NCT00000002"]


# --- basket-trial skip (G-41c) ----------------------------------------------


def test_basket_trial_skipped_and_noted() -> None:
    basket = _rec(
        "NCT00000009",
        "Merck",
        [_drug(f"Drug{i}") for i in range(6)],  # 6 distinct drugs
    )
    normal = _rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")])
    # k=1 pinned: this test is about the basket SKIP, not the P3-WEIGHT default k=2.
    graph = build_graph(
        [basket, normal], kind="drug_drug", max_drugs_per_trial=5, min_edge_weight=1
    )

    # Only the normal trial's single pair survives; the basket's C(6,2) is skipped.
    assert len(graph["edges"]) == 1
    assert _edge(graph, "drug:pembrolizumab", "drug:nivolumab") is not None
    assert any("basket" in note.lower() or "skipped" in note.lower() for note in graph["notes"])
    # none of the basket drugs formed a node
    assert not any(n["label"].startswith("Drug") for n in graph["nodes"])


# --- degeneracy (G-41e) ------------------------------------------------------


def test_degenerate_single_entity() -> None:
    # A sponsor with only a DEVICE intervention → no drug edge → nothing survives.
    records = [_rec("NCT00000001", "Merck", [_drug("Stent", type_="DEVICE")])]
    graph = build_graph(records, kind="sponsor_drug")
    assert graph["degenerate"] is True
    assert len(graph["nodes"]) <= 1
    assert graph["edges"] == []


def test_degenerate_all_isolated_no_cooccurrence() -> None:
    # Each trial has a single drug → no co-occurring pair → no edges.
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000002", "Merck", [_drug("Nivolumab")]),
        _rec("NCT00000003", "Merck", [_drug("Lenvatinib")]),
    ]
    graph = build_graph(records, kind="drug_drug")
    assert graph["degenerate"] is True
    assert graph["edges"] == []


# --- per-edge citations (G-25) ----------------------------------------------


def test_sponsor_drug_edge_has_two_endpoint_citations() -> None:
    records = [_rec("NCT00000001", "Merck Sharp & Dohme LLC", [_drug("Pembrolizumab")])]
    graph = build_graph(records, kind="sponsor_drug")
    edge = graph["edges"][0]

    assert len(edge["citations"]) == 2
    field_paths = {c.field_path for c in edge["citations"]}
    assert field_paths == {_SPONSOR_PATH, _DRUG_PATH}

    # Each excerpt round-trips against the contributing record (never authored).
    for citation in edge["citations"]:
        assert is_substring_at(records[0], citation.field_path, citation.excerpt)


def test_drug_drug_edge_has_two_drug_citations() -> None:
    records = [_rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")])]
    # k=1 pinned: this test is about the two-citation contract, not the P3-WEIGHT
    # default k=2 (which would prune this lone weight-1 edge).
    graph = build_graph(records, kind="drug_drug", min_edge_weight=1)
    edge = graph["edges"][0]

    assert len(edge["citations"]) == 2
    # both endpoints are drugs → both cite the interventions[].name path
    assert {c.field_path for c in edge["citations"]} == {_DRUG_PATH}
    excerpts = {c.excerpt for c in edge["citations"]}
    assert excerpts == {"Pembrolizumab", "Nivolumab"}
    for citation in edge["citations"]:
        assert is_substring_at(records[0], citation.field_path, citation.excerpt)


# --- TOTAL over malformed records (LESSON K1) -------------------------------


def test_malformed_records_do_not_raise() -> None:
    records = [
        {"protocolSection": None},  # null section
        {"protocolSection": {"armsInterventionsModule": {"interventions": "not-a-list"}}},
        _rec("NCT00000002", "Merck", ["not-a-dict-intervention"]),
        _rec("NCT00000003", "Merck", [{"type": "DRUG"}]),  # missing name
        _rec("NCT00000004", "Merck", [{"type": "DRUG", "name": "Aspirin", "otherNames": None}]),
        _rec("NCT00000005", None, [_drug("Pembrolizumab")]),  # no sponsor
        "not-a-record",  # a bare string in the list
        {"protocolSection": {"sponsorCollaboratorsModule": {"leadSponsor": None}}},
        # one valid co-occurrence so the graph still produces an edge:
        _rec("NCT00000006", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000007", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
    ]
    graph = build_graph(records, kind="drug_drug")  # must not raise
    assert set(graph) == {
        "nodes", "edges", "degenerate", "notes", "distinct_trials", "fallback"
    }
    # NCT6 + NCT7 give a weight-2 pembrolizumab↔nivolumab edge, so it survives the
    # P3-WEIGHT default (drug_drug k=2) without pinning.
    assert _edge(graph, "drug:pembrolizumab", "drug:nivolumab")["weight"] == 2


def test_unknown_kind_raises() -> None:
    # kind is a caller contract (not a record) → fail-fast is correct here.
    with pytest.raises(ValueError):
        build_graph([], kind="sponsor_sponsor")


# --- schema validation of the emitted shape ---------------------------------


def test_output_validates_as_networkdata() -> None:
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000003", "Bristol-Myers Squibb", [_drug("Nivolumab"), _drug("Ipilimumab")]),
    ]
    graph = build_graph(records, kind="drug_drug")

    network = NetworkData(
        nodes=[Node(**n) for n in graph["nodes"]],
        edges=[Edge(**e) for e in graph["edges"]],
    )
    assert network.nodes
    assert network.edges
    # the raw edge dict carries contributing_count (Edge ignores the extra key)
    assert all("contributing_count" in e for e in graph["edges"])
    assert all(len(e.citations) == 2 for e in network.edges)


# --- P3-MERGE: alias-only merge (over-merge regression) ----------------------


def test_over_merge_regression_shared_noncanonical_code() -> None:
    # THE CORE FIX. Two DISTINCT drugs whose otherNames share a common NON-primary
    # code token (a protocol code) must NOT merge. Under the old
    # union-any-shared-token logic, "XYZ-123" bridged them into one mega-node
    # (this is the pembrolizumab↔trametinib defect). Alias-only merge keeps them
    # distinct because "XYZ-123" is not any drug's primary .name.
    records = [
        _rec(
            "NCT00000001",
            "Merck",
            [
                _drug("Pembrolizumab", other=["XYZ-123"]),
                _drug("Vemurafenib", other=["XYZ-123"]),
            ],
        ),
    ]
    graph = build_graph(records, kind="drug_drug", min_edge_weight=1)

    drug_labels = sorted(n["label"] for n in graph["nodes"] if n["kind"] == "drug")
    assert drug_labels == ["Pembrolizumab", "Vemurafenib"]  # TWO nodes, not merged
    # the co-occurrence edge connects the two genuinely-distinct drugs
    edge = _edge(graph, "drug:pembrolizumab", "drug:vemurafenib")
    assert edge is not None
    assert edge["weight"] == 1
    # both endpoint excerpts round-trip against the contributing record
    excerpts = {c.excerpt for c in edge["citations"]}
    assert excerpts == {"Pembrolizumab", "Vemurafenib"}
    for citation in edge["citations"]:
        assert is_substring_at(records[0], citation.field_path, citation.excerpt)


def test_brand_generic_still_merges() -> None:
    # The legitimate brand↔generic merge MUST survive the alias-only rule: the
    # otherName "pembrolizumab" IS another intervention's primary .name, and it is
    # corroborated by ≥2 trials (bidirectional, residual-1 rule).
    records = [
        _rec("NCT00000001", "Merck", [_drug("Keytruda", other=["pembrolizumab"])]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab", other=["Keytruda"])]),
        _rec("NCT00000003", "Merck", [_drug("Nivolumab")]),
    ]
    graph = build_graph(records, kind="sponsor_drug", min_edge_weight=1)
    drug_labels = {n["label"].lower() for n in graph["nodes"] if n["kind"] == "drug"}
    # Keytruda ≡ pembrolizumab folded to ONE node (label is one of the two).
    assert "nivolumab" in drug_labels
    merged = drug_labels - {"nivolumab"}
    assert len(merged) == 1
    assert merged.issubset({"keytruda", "pembrolizumab"})


# --- P3-SPONSOR: sponsor name-variant canonicalization -----------------------


def test_sponsor_canonicalization_merges_variants() -> None:
    # Legal-suffix / punctuation variants of the SAME org collapse to one node.
    records = [
        _rec("NCT00000001", "Merck Sharp & Dohme LLC", [_drug("Pembrolizumab")]),
        _rec("NCT00000002", "Merck Sharp & Dohme, Inc.", [_drug("Pembrolizumab")]),
    ]
    graph = build_graph(records, kind="sponsor_drug", min_edge_weight=1)
    sponsor_nodes = [n for n in graph["nodes"] if n["kind"] == "sponsor"]
    assert len(sponsor_nodes) == 1  # both raw variants → ONE sponsor node
    # the single sponsor↔pembrolizumab edge carries both trials
    edge = graph["edges"][0]
    assert edge["weight"] == 2
    assert sorted(edge["source_ids"]) == ["NCT00000001", "NCT00000002"]


def test_sponsor_canon_no_parent_subsidiary_fold() -> None:
    # Mechanical suffix strip merges Novartis Pharmaceuticals ≡ Novartis, but
    # Genentech and Roche (parent/subsidiary) MUST stay two distinct nodes.
    records = [
        _rec("NCT00000001", "Novartis Pharmaceuticals", [_drug("Drug A")]),
        _rec("NCT00000002", "Novartis", [_drug("Drug A")]),
        _rec("NCT00000003", "Genentech", [_drug("Drug B")]),
        _rec("NCT00000004", "Roche", [_drug("Drug B")]),
    ]
    graph = build_graph(records, kind="sponsor_drug", min_edge_weight=1)
    sponsor_labels = {n["label"] for n in graph["nodes"] if n["kind"] == "sponsor"}

    novartis_nodes = [
        n
        for n in graph["nodes"]
        if n["kind"] == "sponsor" and "novartis" in n["label"].lower()
    ]
    assert len(novartis_nodes) == 1  # Novartis Pharmaceuticals ≡ Novartis → one node
    # NO corporate-parent fold: Genentech and Roche remain separate legal entities.
    assert "Genentech" in sponsor_labels
    assert "Roche" in sponsor_labels


def test_canonicalize_sponsor_alias_and_suffix() -> None:
    # Direct unit coverage of the two P3-SPONSOR concepts.
    from app.ctgov.network import canonicalize_sponsor

    # (1) mechanical: case, punctuation, trailing-suffix folding
    assert canonicalize_sponsor("Merck Sharp & Dohme LLC") == "merck sharp dohme"
    assert canonicalize_sponsor("Merck Sharp & Dohme, Inc.") == "merck sharp dohme"
    assert canonicalize_sponsor("Novartis Pharmaceuticals") == "novartis"
    # (2) disclosed same-org alias applied AFTER mechanical norm
    assert canonicalize_sponsor("MSD") == "merck sharp dohme"
    # never stripped to empty even when the whole name is a suffix token
    assert canonicalize_sponsor("Pharma") == "pharma"
    # NO parent/subsidiary fold — these stay distinct keys
    assert canonicalize_sponsor("Genentech") != canonicalize_sponsor("Roche")


# --- P3-WEIGHT: per-kind min_edge_weight default -----------------------------


def test_default_k_degeneracy_drug_drug() -> None:
    # A drug_drug population whose only pairs are weight-1: under the DEFAULT
    # (drug_drug k=2) every edge prunes → edges==0 → degenerate.
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000002", "Merck", [_drug("Lenvatinib"), _drug("Ipilimumab")]),
    ]
    graph = build_graph(records, kind="drug_drug")  # DEFAULT → k=2
    assert graph["degenerate"] is True
    assert graph["edges"] == []

    # Explicit k=1 recovers the two weight-1 edges — the underlying graph is
    # unchanged, the threshold is just a configurable default.
    graph1 = build_graph(records, kind="drug_drug", min_edge_weight=1)
    assert graph1["degenerate"] is False
    assert len(graph1["edges"]) == 2


def test_sponsor_drug_default_k_is_one() -> None:
    # The sponsor_drug default stays k=1 (a weight-1 sponsor→drug edge survives).
    records = [_rec("NCT00000001", "Merck", [_drug("Pembrolizumab")])]
    graph = build_graph(records, kind="sponsor_drug")  # DEFAULT → k=1
    assert _edge(graph, "sponsor:merck", "drug:pembrolizumab") is not None


# --- P3 degeneracy fallback bar ----------------------------------------------


def test_fallback_payload_buckets_cited() -> None:
    records = [
        _rec("NCT00000001", "Merck", [_drug("Pembrolizumab"), _drug("Nivolumab")]),
        _rec("NCT00000002", "Merck", [_drug("Pembrolizumab")]),
        _rec("NCT00000003", "Merck", [_drug("Lenvatinib")]),
    ]
    graph = build_graph(records, kind="drug_drug")
    fallback = graph["fallback"]

    assert fallback["mode"] == "explode"
    assert fallback["distinct_trials"] == 3  # 3 trials each have >=1 drug node
    buckets = fallback["buckets"]
    assert buckets  # non-empty for a population with drugs

    # Pembrolizumab is studied in two DISTINCT trials (NCT1 + NCT2).
    pem = next(b for b in buckets if b["label"] == "Pembrolizumab")
    assert pem["count_trials"] == 2  # distinct-trial count
    assert pem["count_mentions"] == 2
    assert pem["contributing_count"] == 2
    assert sorted(pem["source_ids"]) == ["NCT00000001", "NCT00000002"]

    # Every bucket citation round-trips against its own contributing record.
    by_nct = {r["protocolSection"]["identificationModule"]["nctId"]: r for r in records}
    for bucket in buckets:
        for citation in bucket["citations"]:
            record = by_nct[citation.nct_id]
            assert is_substring_at(record, citation.field_path, citation.excerpt)


# --- Phase-3 hardening regressions (adversarial pass) ------------------------


def test_combination_intervention_does_not_merge_components() -> None:
    # L1-F1: a combination product "A + B" lists its components (each a primary
    # elsewhere) as otherNames — it must NOT merge A and B into one node (the
    # over-merge vector that survived the first alias-only fix).
    from app.ctgov.network import _build_drug_index, _canonical_key, _drug_interventions

    records = [
        _rec("NCT00000001", "S", [_drug("Sunitinib Malate + Valproic Acid",
                                        other=["Sunitinib", "Valproic Acid"])]),
        _rec("NCT00000002", "S", [_drug("Sunitinib")]),
        _rec("NCT00000003", "S", [_drug("Valproic Acid")]),
    ]
    uf, _labels = _build_drug_index(records)
    keys: dict[str, str | None] = {}
    for record in records:
        for iv in _drug_interventions(record):
            keys[iv["name"]] = _canonical_key(iv, uf)
    assert keys["Sunitinib"] != keys["Valproic Acid"]  # distinct drugs stay distinct


def test_keytruda_still_merges_after_combination_guard() -> None:
    # The combination guard must NOT break the legit brand↔generic merge: Keytruda
    # [other: pembrolizumab] still folds into the pembrolizumab canonical node.
    from app.ctgov.network import _build_drug_index, _canonical_key, _drug_interventions

    records = [
        _rec("NCT00000001", "S", [_drug("Keytruda", other=["pembrolizumab"])]),
        _rec("NCT00000002", "S", [_drug("Pembrolizumab", other=["Keytruda"]), _drug("Nivolumab")]),
    ]
    uf, _labels = _build_drug_index(records)
    keys: dict[str, str | None] = {}
    for record in records:
        for iv in _drug_interventions(record):
            keys[iv["name"]] = _canonical_key(iv, uf)
    assert keys["Keytruda"] == keys["Pembrolizumab"]  # brand↔generic still merged (corroborated)
    assert keys["Keytruda"] != keys["Nivolumab"]  # a distinct drug stays separate


def test_edge_citation_value_is_real_list_not_copy() -> None:
    # L2-F1: the citation value must be the record's REAL interventions[].name list,
    # not a scalar copy of the excerpt (which made the precheck a tautology).
    records = [_rec("NCT00000001", "S", [_drug("Pembrolizumab"), _drug("Nivolumab")])]
    graph = build_graph(records, kind="drug_drug", min_edge_weight=1)
    edge = graph["edges"][0]
    for citation in edge["citations"]:
        assert isinstance(citation.value, list)  # real list, teeth
        assert citation.excerpt in citation.value  # excerpt equals a genuine element


def test_fallback_notes_exclude_graph_notes_and_disclose_truncation() -> None:
    # L3-1 + L1-F6: the degeneracy fallback bar carries ONLY bar-appropriate notes
    # (no edge/cap/prune notes), discloses the top-N drug truncation, and its
    # citations carry a real list value (teeth).
    records = [_rec(f"NCT{i:08d}", "S", [_drug(f"Drug{i}")]) for i in range(30)]
    graph = build_graph(records, kind="drug_drug")  # default k=2 → all isolated → degenerate
    assert graph["degenerate"] is True
    fb = graph["fallback"]
    joined = " ".join(fb["notes"])
    assert "Edges are derived" not in joined
    assert "capped" not in joined and "pruned" not in joined
    assert fb["buckets_truncated"] is True  # 30 drugs > top_n=25
    assert any("lower-frequency" in note for note in fb["notes"])
    for citation in fb["buckets"][0]["citations"]:
        assert isinstance(citation.value, list)


def test_punctuation_variant_names_merge_no_empty_citation() -> None:
    # "Nab paclitaxel" and "Nab-paclitaxel" are the SAME drug (differ only by a hyphen).
    # They must merge to ONE canonical node — NOT collide two distinct canons onto one
    # node id, which gave the other endpoint an empty, unverifiable citation (a
    # pre-existing latent bug the value-is-real-list teeth-fix exposed).
    records = [
        _rec("NCT00000001", "S", [_drug("Nab paclitaxel"), _drug("Carboplatin")]),
        _rec("NCT00000002", "S", [_drug("Nab-paclitaxel"), _drug("Carboplatin")]),
    ]
    graph = build_graph(records, kind="drug_drug", min_edge_weight=1)
    drug_nodes = [n for n in graph["nodes"] if n["kind"] == "drug"]
    assert len(drug_nodes) == 2  # nab-paclitaxel variants collapse to ONE node + carboplatin
    assert graph["edges"][0]["weight"] == 2  # both trials merged into the one edge
    for edge in graph["edges"]:
        for citation in edge["citations"]:
            assert citation.excerpt != ""  # never an empty endpoint citation
            assert citation.excerpt in citation.value  # element-precise (teeth)


def test_single_trial_mislabel_not_merged() -> None:
    # Residual 1: a ONE-OFF registry mislabel — a single trial's "Pembrolizumab"
    # intervention wrongly listing "Lenvatinib" (a different, established drug) as an
    # otherName — must NOT merge the two drugs. Lenvatinib is attested standalone
    # elsewhere; the alias {pembrolizumab, lenvatinib} is attested by only ONE trial,
    # so corroboration (≥2 trials) blocks it.
    from app.ctgov.network import _build_drug_index, _canonical_key, _drug_interventions

    records = [
        _rec("NCT00000001", "S", [_drug("Pembrolizumab", other=["Lenvatinib"])]),  # the mislabel
        _rec("NCT00000002", "S", [_drug("Lenvatinib")]),
        _rec("NCT00000003", "S", [_drug("Pembrolizumab")]),
        _rec("NCT00000004", "S", [_drug("Lenvatinib")]),
    ]
    uf, _labels = _build_drug_index(records)
    keys = {iv["name"]: _canonical_key(iv, uf) for r in records for iv in _drug_interventions(r)}
    # Pembrolizumab and Lenvatinib stay DISTINCT (the mislabel did not collapse them).
    pembro = {v for k, v in keys.items() if k == "Pembrolizumab"}
    lenva = {v for k, v in keys.items() if k == "Lenvatinib"}
    assert pembro.isdisjoint(lenva)


def test_dose_and_salt_variants_merge_to_ingredient() -> None:
    # Residual 2: dose/strength and salt/formulation variants of the SAME drug fold to
    # one active-ingredient node (no alias needed — they normalize to one key).
    records = [
        _rec("NCT00000001", "S", [_drug("Selumetinib, 100mg"), _drug("Carboplatin")]),
        _rec("NCT00000002", "S", [_drug("Selumetinib, 225mg"), _drug("Carboplatin")]),
        _rec("NCT00000003", "S", [_drug("Selumetinib"), _drug("Carboplatin")]),
    ]
    graph = build_graph(records, kind="drug_drug", min_edge_weight=1)
    drug_nodes = [n for n in graph["nodes"] if n["kind"] == "drug"]
    assert len(drug_nodes) == 2  # selumetinib (all doses) + carboplatin
    # the selumetinib↔carboplatin edge aggregates all three trials
    assert graph["edges"][0]["weight"] == 3
    for edge in graph["edges"]:
        for citation in edge["citations"]:
            assert citation.excerpt != ""
            assert citation.excerpt in citation.value
