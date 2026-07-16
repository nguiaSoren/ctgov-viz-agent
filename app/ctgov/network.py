"""Pure sponsor↔drug / drug↔drug graph assembly over already-paged records.

``build_graph`` turns a list of already-fetched ClinicalTrials.gov study dicts
into a node-link graph payload (``{nodes, edges, ...}``) for the
``network_graph`` chart class. It performs NO I/O — no client, no network — it is
a pure function over record dicts the tools layer has already paged.

Design invariants honored here (Interface Contract v2 §W1b):

* **TOTAL over live data (LESSON K1/B2)** — never raise on a malformed record.
  ``protocolSection: null``, a non-list ``interventions``, a non-dict
  intervention, ``otherNames: null``, a missing ``name`` — every descent is
  ``isinstance``-guarded and a bad record is silently skipped, never fatal.
* **Excerpts are string-extracted, never authored (CC-9)** — each edge carries
  two citations whose ``excerpt`` is walked out of a contributing record via the
  ``app.ctgov.citations`` primitives, round-trip verifiable by ``is_substring_at``.
* **Alias-only drug synonym merge (CC-12, P3-MERGE)** — drug nodes are merged by
  the **Alias invariant**: *an alias (``otherName``) belongs only to its own
  canonical drug and never bridges two distinct primary names; only an
  ``otherName`` that equals another intervention's PRIMARY ``.name`` merges them
  (the true brand↔generic case, e.g. Keytruda [other: pembrolizumab] folds into
  the primary "pembrolizumab").* A protocol code or regimen word that is NOT any
  drug's primary name is just a label and creates NO merge — this kills the
  transitive over-merge that collapsed distinct drugs through a shared
  ``otherName`` (the confirmed pembrolizumab↔trametinib defect).
* **Placebo exclusion by NAME (G-36)** — placebo / standard-of-care arms are
  dropped by name (their ``type`` is DRUG, so the type filter alone keeps them),
  preventing a false mega-hub.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from app import config
from app.api.schemas import Citation
from app.ctgov.citations import _resolve_path, brief_title, build_citation

# --- Fixed JSON paths (the endpoint field_paths cited per edge, G-25) --------

_NCT_PATH = "protocolSection.identificationModule.nctId"
_SPONSOR_PATH = "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
# The ``[]`` convention: round-trips via ``is_substring_at`` because the excerpt
# equals one element of the interventions list (the drug's own name).
_DRUG_PATH = "protocolSection.armsInterventionsModule.interventions[].name"

_VALID_KINDS = ("sponsor_drug", "drug_drug")

# Placebo / standard-of-care substrings to exclude BY NAME (case-insensitive,
# G-36). Substring match so "Matching Placebo" / "Placebo tablets" are caught.
_PLACEBO_TOKENS = (
    "placebo",
    "standard of care",
    "standard care",
    "best supportive care",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# A combiner token in an intervention ``.name`` marks it as a COMBINATION product
# (e.g. "Sunitinib Malate + Valproic Acid", "Elexacaftor/tezacaftor/ivacaftor"). A
# combination must NOT use its ``otherNames`` to union two distinct primary drug
# names — that reintroduced the over-merge via the combination vector (adversarial
# L1-F1): a component listed as an otherName is another drug's primary, so a naive
# alias-union folds two genuinely-distinct drugs into one mislabeled node. Broad by
# design: over-flagging only ever causes a SAFE under-merge (two nodes instead of a
# wrong single one), the chosen tradeoff (alias-only, brand↔generic merge).
_COMBINER_RE = re.compile(r"\+|/|\bplus\b|\band\b|\bwith\b", re.IGNORECASE)


# --- Small TOTAL descent helpers --------------------------------------------


def _get(obj: Any, key: str) -> Any:
    """``obj[key]`` only when ``obj`` is a dict, else ``None`` (never raises)."""
    return obj.get(key) if isinstance(obj, dict) else None


def _clean_str(value: Any) -> str | None:
    """Return a stripped non-empty string, else ``None`` (a bare non-str → None)."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _slug(text: str) -> str:
    """Deterministic ``a-z0-9-`` slug for a node id; hashed fallback if empty."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    if slug:
        return slug
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]  # noqa: S324 — id only, not security


# Dose/strength tokens — a number (+ optional decimal/comma) followed by a unit.
# Removing them collapses "Selumetinib, 100mg" / "Selumetinib, 225mg" onto the
# ingredient (residual 2). A bare number NOT followed by a unit (e.g. "5" in "5-FU",
# "2" in "Interleukin-2", "0" in "Drug0") is NEVER matched — the unit is required, so a
# number that is part of the drug identity is preserved.
_DOSE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|ug|g|kg|ml|l|iu|units?|u|%|mmol|nmol|mm|nm|meq|mg/ml|mg/kg|mg/m2)\b",
    re.IGNORECASE,
)

# Trailing salt / dose-form / route tokens stripped iteratively (whole trailing tokens
# only, never to empty), so a drug and its salt/formulation/route variants normalize to
# ONE active-ingredient key: "Sunitinib Malate" ≡ "Sunitinib", "Doxorubicin
# Hydrochloride" ≡ "Doxorubicin", "Pembrolizumab ... Intravenous Solution" ≡
# "Pembrolizumab". TRAILING-only (mirrors the sponsor legal-suffix strip) so a leading
# occurrence that is part of the identity ("Sodium Chloride") is left intact.
_INGREDIENT_TRAILING = frozenset(
    {
        # salts / esters
        "hydrochloride", "hcl", "dihydrochloride", "hydrobromide", "sulfate", "sulphate",
        "mesylate", "mesilate", "malate", "maleate", "tartrate", "citrate", "phosphate",
        "acetate", "succinate", "fumarate", "besylate", "tosylate", "sodium", "potassium",
        "calcium",
        # dose forms
        "solution", "injection", "injectable", "infusion", "tablet", "tablets", "capsule",
        "capsules", "suspension", "cream", "ointment", "gel", "powder", "spray", "patch",
        "syrup", "drops", "concentrate", "lyophilized", "lyophilisate", "film", "coated",
        # routes / release modifiers
        "oral", "intravenous", "iv", "intramuscular", "subcutaneous", "topical",
        "intrathecal", "inhaled", "nasal", "ophthalmic", "sublingual", "release",
        "extended", "immediate", "sustained", "modified", "delayed", "prolonged",
    }
)


def _norm(text: str) -> str:
    """Normalized ACTIVE-INGREDIENT key for a drug primary/alias name.

    Lowercase → drop dose/strength tokens → collapse every run of non-alphanumerics to a
    single space → strip trailing salt / dose-form / route tokens (never to empty). So
    punctuation, dose, salt, and formulation variants of the SAME drug normalize to ONE
    key — "Nab paclitaxel" ≡ "Nab-paclitaxel", "Selumetinib, 100mg" ≡ "Selumetinib",
    "Sunitinib Malate" ≡ "Sunitinib", "5-FU" ≡ "5 FU". This merges genuine variants AND
    keeps the canonical key stable so a node's identity can't split on a stray
    hyphen/dose/salt (that split collided distinct canons onto one node id → empty
    citations). TRAILING-only strip + required-unit dose match keep it from eating a
    number/word that is part of the identity."""
    base = _DOSE_RE.sub(" ", text.lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", base).split()
    while len(tokens) > 1 and tokens[-1] in _INGREDIENT_TRAILING:
        tokens.pop()
    return " ".join(tokens).strip()


def _nct_id(record: Any) -> str | None:
    """Read the trial's nctId (TOTAL)."""
    return _clean_str(_resolve_path(record, _NCT_PATH)) if isinstance(record, dict) else None


def _lead_sponsor(record: Any) -> str | None:
    """Read ``leadSponsor.name`` — lead sponsor ONLY, collaborators excluded (G-19)."""
    return _clean_str(_resolve_path(record, _SPONSOR_PATH)) if isinstance(record, dict) else None


def _other_names(intervention: dict) -> list[str]:
    """Normalize ``otherNames`` (often ``null``, sometimes a bare string) to a list."""
    raw = _get(intervention, "otherNames")
    if isinstance(raw, list):
        return [s for s in (_clean_str(item) for item in raw) if s]
    single = _clean_str(raw)
    return [single] if single else []


def _is_placebo(tokens: Iterable[str]) -> bool:
    """True if any name/otherName token matches a placebo/standard-of-care marker."""
    return any(marker in tok for tok in tokens for marker in _PLACEBO_TOKENS)


def _drug_interventions(record: Any) -> list[dict]:
    """Return the DRUG-type, non-placebo intervention dicts of a trial (TOTAL).

    Only ``type == "DRUG"`` interventions become drug nodes (G-17: DEVICE /
    PROCEDURE / GENETIC / BIOLOGICAL are skipped). Placebo / standard-of-care
    arms are dropped by name (G-36). A missing ``name`` → skipped (a node cannot
    be labelled without one).
    """
    module = _get(_get(record, "protocolSection"), "armsInterventionsModule")
    interventions = _get(module, "interventions")
    if not isinstance(interventions, list):
        return []
    out: list[dict] = []
    for item in interventions:
        if not isinstance(item, dict):
            continue
        if _get(item, "type") != "DRUG":
            continue
        name = _clean_str(_get(item, "name"))
        if name is None:
            continue
        low_tokens = [name.lower(), *(o.lower() for o in _other_names(item))]
        if _is_placebo(low_tokens):
            continue
        out.append(item)
    return out


# --- Drug synonym-merge union-find (CC-12, alias-only P3-MERGE) --------------


class _UnionFind:
    """Tiny union-find over normalized primary drug names; the root of a component
    is its lexicographically smallest name (deterministic canonical key)."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, token: str) -> str:
        self.parent.setdefault(token, token)
        root = token
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[token] != root:  # path compression
            self.parent[token], token = root, self.parent[token]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Smaller string becomes the root → canonical key is deterministic.
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


def _build_drug_index(records: list[dict]) -> tuple[_UnionFind, dict[str, str]]:
    """Alias-only synonym merge (P3-MERGE), then pick a display label per canon.

    The **Alias invariant** (see the module docstring): an ``otherName`` merges
    two interventions ONLY when that ``otherName`` is ITSELF some drug's PRIMARY
    ``.name`` — the true brand→generic case. An ``otherName`` that is not any
    drug's primary (a protocol code, a regimen word) is just a label and creates
    NO union. Two ``otherName`` tokens are NEVER unioned with each other. This is
    the exact deviation from the old union-any-shared-token logic that caused the
    transitive over-merge (distinct drugs collapsed through a shared code).

    Returns ``(uf, labels)`` where the union-find is over normalized primary
    names and ``labels[canonical_key]`` is the most-frequently-seen primary
    ``.name`` in the component, ties broken by first-seen order.
    """
    # Pass 1: the set of every normalized primary ``.name`` in the population.
    primaries: set[str] = set()
    for record in records:
        for intervention in _drug_interventions(record):
            name = _clean_str(_get(intervention, "name"))
            if name is not None:
                primaries.add(_norm(name))

    uf = _UnionFind()
    for primary in primaries:  # every primary is at least a singleton component
        uf.find(primary)

    # Pass 2: alias-only unions — merge p↔o ONLY when the otherName o is itself a
    # primary (brand↔generic). A non-primary otherName never bridges two drugs.
    # COMBINATION GUARD (L1-F1): a combination/multi-component intervention must NOT
    # use its otherNames to union two distinct primaries — a combination lists its
    # components as otherNames, each of which is another drug's primary, so a naive
    # union folds the components into one mislabeled node (the over-merge, via the
    # combination vector). Skip the unions when the name carries a combiner token OR
    # the intervention references ≥2 distinct OTHER primaries (a multi-drug regimen).
    #
    # CORROBORATION (residual 1): a single-trial registry MISLABEL (one intervention
    # wrongly listing a DIFFERENT drug's name as an otherName) is shape-identical to a
    # real brand↔generic and can't be told apart structurally. So only merge an
    # alias pair attested by ≥2 DISTINCT trials (a real synonym like Keytruda↔
    # pembrolizumab is attested by many trials, often bidirectionally; a one-off
    # mislabel is attested once) — the ontology-free defense. A legit single-attestation
    # pair under-merges (the SAFE direction), and it is disclosed.
    attest: dict[frozenset[str], set[str]] = {}
    for record in records:
        nct = _nct_id(record) or ""
        for intervention in _drug_interventions(record):
            name = _clean_str(_get(intervention, "name"))
            if name is None:
                continue
            primary = _norm(name)
            alias_primaries = {
                a
                for other in _other_names(intervention)
                if (a := _norm(other)) in primaries and a != primary
            }
            if _COMBINER_RE.search(name) or len(alias_primaries) >= 2:
                continue  # combination — do not merge its components together
            for alias in alias_primaries:
                attest.setdefault(frozenset((primary, alias)), set()).add(nct)

    for pair, ncts in attest.items():
        if len(ncts) >= 2:  # corroborated by ≥2 distinct trials → merge
            a, b = tuple(pair)
            uf.union(a, b)

    # Pass 3: unions settled → tally the display label per canonical component.
    name_stats: dict[str, dict[str, tuple[int, int]]] = {}
    seen = 0
    for record in records:
        for intervention in _drug_interventions(record):
            name = _clean_str(_get(intervention, "name"))
            if name is None:
                continue
            canon = uf.find(_norm(name))
            bucket = name_stats.setdefault(canon, {})
            count, order = bucket.get(name, (0, seen))
            bucket[name] = (count + 1, order)
            seen += 1

    labels: dict[str, str] = {}
    for canon, stats in name_stats.items():
        # max frequency, tie-break earliest first-seen order.
        best = min(stats.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
        labels[canon] = best[0]
    return uf, labels


def _canonical_key(intervention: dict, uf: _UnionFind) -> str | None:
    """Canonical union-find key of an intervention = ``uf.find(norm(name))``."""
    name = _clean_str(_get(intervention, "name"))
    return uf.find(_norm(name)) if name is not None else None


# --- Sponsor name-variant canonicalization (P3-SPONSOR) ---------------------

# Trailing legal / corporate suffix tokens stripped iteratively (whole trailing
# tokens only, never mid-name, never to empty). SAFE MECHANICAL canonicalization.
_SPONSOR_SUFFIXES = frozenset(
    {
        "inc", "llc", "ltd", "corp", "co", "company", "gmbh", "sa", "bv", "plc",
        "ag", "ab", "nv", "lp", "llp", "pharmaceuticals", "pharmaceutical",
        "pharma", "therapeutics", "laboratories", "labs", "group", "holdings",
        "limited", "incorporated", "corporation",
    }
)

# DISCLOSED same-org alias table — abbreviations / legal-name variants of the
# SAME org ONLY, applied AFTER mechanical norm. Deliberately tiny. NO
# parent/subsidiary or merger consolidation (folding "merck sharp dohme"→"merck"
# or "genentech"→"roche" would risk colliding distinct legal entities).
_SPONSOR_ALIASES = {
    "msd": "merck sharp dohme",
}


def canonicalize_sponsor(name: str) -> str:
    """Return a normalized canonical KEY for a lead-sponsor name (P3-SPONSOR).

    Two SEPARATE concepts: (1) safe mechanical canonicalization — lowercase,
    NFKC-normalize, non-alphanumeric → space, collapse whitespace, then strip
    trailing legal/corporate suffix tokens iteratively (never to empty); (2) a
    tiny DISCLOSED same-org alias table applied after mechanical norm. It does
    NOT do corporate parent/subsidiary/merger consolidation — different legal
    entities stay distinct so we can never collide two real companies.
    """
    mechanical = unicodedata.normalize("NFKC", name.lower())
    mechanical = re.sub(r"[^a-z0-9]+", " ", mechanical)
    tokens = mechanical.split()
    while len(tokens) > 1 and tokens[-1] in _SPONSOR_SUFFIXES:
        tokens.pop()
    key = " ".join(tokens)
    return _SPONSOR_ALIASES.get(key, key)


def _build_sponsor_index(records: list[dict]) -> dict[str, str]:
    """Map each canonical sponsor key → display label (most-frequent RAW variant).

    Mirrors the drug label logic: the label is the most-frequently-seen raw
    ``leadSponsor.name`` for that canonical key, ties broken by first-seen order.
    """
    stats: dict[str, dict[str, tuple[int, int]]] = {}
    seen = 0
    for record in records:
        raw = _lead_sponsor(record)
        if raw is None:
            continue
        canon = canonicalize_sponsor(raw)
        bucket = stats.setdefault(canon, {})
        count, order = bucket.get(raw, (0, seen))
        bucket[raw] = (count + 1, order)
        seen += 1

    labels: dict[str, str] = {}
    for canon, variants in stats.items():
        best = min(variants.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
        labels[canon] = best[0]
    return labels


# --- Citations (string-extracted from a contributing record, CC-9) ----------


def _drug_name_in_record(record: dict, canon: str, uf: _UnionFind) -> str | None:
    """The literal ``.name`` string for canonical ``canon`` as it appears here."""
    for intervention in _drug_interventions(record):
        if _canonical_key(intervention, uf) == canon:
            return _clean_str(_get(intervention, "name"))
    return None


def _drug_names_in_record(record: dict) -> list[str]:
    """The REAL list of DRUG intervention names in ``record`` — the citation ``value``
    for a drug endpoint. Storing the real list (not a scalar copy of the excerpt) is
    what gives the Output-Reviewer excerpt check TEETH (L2-F1, mirroring the explode
    path's F1 fix): ``_excerpt_in_value`` then requires the excerpt to EQUAL a genuine
    element, so a FABRICATED drug excerpt fails ``citation_invalid`` instead of passing
    against a copy of itself. Matches ``is_substring_at``'s element scan at ``.name``."""
    return [name for iv in _drug_interventions(record) if (name := _clean_str(_get(iv, "name")))]


def _endpoint_citation(record: dict, node_kind: str, node_key: str, uf: _UnionFind) -> Citation:
    """Build one endpoint citation, excerpt string-extracted from ``record``."""
    if node_kind == "sponsor":
        # build_citation resolves value + excerpt from the sponsor path directly.
        return build_citation(record, _SPONSOR_PATH)
    name = _drug_name_in_record(record, node_key, uf) or ""
    return Citation(
        nct_id=_nct_id(record) or "",
        field_path=_DRUG_PATH,
        value=_drug_names_in_record(record),  # REAL list → excerpt must equal an element (teeth)
        matched_value=name,  # the targeted drug's own literal .name → round-trips (is_substring_at)
        excerpt=brief_title(record) or name,
    )


# --- Edge / node assembly ----------------------------------------------------


class _NodeMeta:
    __slots__ = ("id", "label", "kind", "key")

    def __init__(self, node_id: str, label: str, kind: str, key: str) -> None:
        self.id = node_id
        self.label = label
        self.kind = kind
        self.key = key  # sponsor: raw name; drug: canonical union-find key


def _accumulate_edges(
    records: list[dict],
    *,
    kind: str,
    uf: _UnionFind,
    labels: dict[str, str],
    sponsor_labels: dict[str, str],
    max_drugs_per_trial: int,
) -> tuple[dict[tuple[str, str], dict[str, dict]], dict[str, _NodeMeta], int]:
    """Walk records → ``edge_key -> {nctId: record}`` plus a node registry.

    Returns ``(edges, node_meta, skipped_basket)``. ``edge_key`` is an ordered
    ``(source_id, target_id)`` tuple; the inner dict maps a contributing nctId to
    one contributing record (for citation extraction). ``skipped_basket`` counts
    drug_drug trials skipped for exceeding ``max_drugs_per_trial`` distinct drugs.
    """
    edges: dict[tuple[str, str], dict[str, dict]] = {}
    node_meta: dict[str, _NodeMeta] = {}
    skipped_basket = 0

    def drug_node(canon: str) -> _NodeMeta:
        # id from the UNIQUE canonical key (one per union-find component), NOT the
        # label slug — two distinct canons whose display labels slug to the same
        # string ("Nab paclitaxel"/"Nab-paclitaxel") must NOT collapse onto one node
        # (that gave the other endpoint an empty, unverifiable citation).
        node_id = f"drug:{_slug(canon)}"
        meta = node_meta.get(node_id)
        if meta is None:
            meta = _NodeMeta(node_id, labels.get(canon, canon), "drug", canon)
            node_meta[node_id] = meta
        return meta

    def sponsor_node(raw: str) -> _NodeMeta:
        # Node identity is the CANONICAL key (P3-SPONSOR): every raw variant of the
        # same org collapses to one stable id; label is the most-frequent raw name.
        canon = canonicalize_sponsor(raw)
        node_id = f"sponsor:{_slug(canon)}"
        meta = node_meta.get(node_id)
        if meta is None:
            meta = _NodeMeta(node_id, sponsor_labels.get(canon, raw), "sponsor", raw)
            node_meta[node_id] = meta
        return meta

    def add(a: _NodeMeta, b: _NodeMeta, nct: str | None, record: dict, *, ordered: bool) -> None:
        if nct is None:
            return
        # sponsor_drug is bipartite with a fixed sponsor→drug orientation (ordered);
        # drug_drug is symmetric → key by sorted ids for a deterministic single edge.
        if ordered:
            key = (a.id, b.id)
        else:
            key = (a.id, b.id) if a.id <= b.id else (b.id, a.id)
        edges.setdefault(key, {}).setdefault(nct, record)

    for record in records:
        if not isinstance(record, dict):
            continue
        nct = _nct_id(record)
        drug_canons = []
        for intervention in _drug_interventions(record):
            canon = _canonical_key(intervention, uf)
            if canon is not None:
                drug_canons.append(canon)
        distinct_canons = sorted(set(drug_canons))

        if kind == "sponsor_drug":
            sponsor_name = _lead_sponsor(record)
            if sponsor_name is None:
                continue
            s_node = sponsor_node(sponsor_name)
            for canon in distinct_canons:
                add(s_node, drug_node(canon), nct, record, ordered=True)
        else:  # drug_drug
            if len(distinct_canons) > max_drugs_per_trial:
                skipped_basket += 1
                continue
            for canon_a, canon_b in combinations(distinct_canons, 2):
                add(drug_node(canon_a), drug_node(canon_b), nct, record, ordered=False)

    return edges, node_meta, skipped_basket


def _prune_and_shape(
    edges: dict[tuple[str, str], dict[str, dict]],
    node_meta: dict[str, _NodeMeta],
    uf: _UnionFind,
    *,
    max_nodes: int,
    min_edge_weight: int,
) -> tuple[list[dict], list[dict]]:
    """Weight-prune, cap nodes by degree, drop isolated → node/edge dicts."""
    # 1. weight prune.
    kept: list[tuple[tuple[str, str], dict[str, dict]]] = [
        (key, contrib) for key, contrib in edges.items() if len(contrib) >= min_edge_weight
    ]

    # 2. degree over surviving edges.
    def degrees(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
        deg: dict[str, int] = {}
        for src, tgt in pairs:
            deg[src] = deg.get(src, 0) + 1
            deg[tgt] = deg.get(tgt, 0) + 1
        return deg

    deg = degrees(k for k, _ in kept)

    # 3. cap top-N nodes by degree (tie-break id), then refilter edges.
    if len(deg) > max_nodes:
        keep_ids = {
            nid for nid, _ in sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))[:max_nodes]
        }
        kept = [(k, c) for k, c in kept if k[0] in keep_ids and k[1] in keep_ids]
        deg = degrees(k for k, _ in kept)

    # 4. nodes = endpoints of surviving edges (isolated nodes thereby dropped).
    edge_dicts: list[dict] = []
    for (src, tgt), contrib in kept:
        ncts = sorted(contrib.keys())
        rep_record = contrib[ncts[0]]  # deterministic representative (min nctId)
        s_meta, t_meta = node_meta[src], node_meta[tgt]
        citations = [
            _endpoint_citation(rep_record, s_meta.kind, s_meta.key, uf),
            _endpoint_citation(rep_record, t_meta.kind, t_meta.key, uf),
        ]
        edge_dicts.append(
            {
                "source": src,
                "target": tgt,
                "weight": len(contrib),
                "source_ids": ncts[:20],
                "citations": citations,
                "contributing_count": len(contrib),
            }
        )

    node_dicts = [
        {"id": nid, "label": node_meta[nid].label, "kind": node_meta[nid].kind, "degree": d}
        for nid, d in deg.items()
        if d > 0
    ]
    node_dicts.sort(key=lambda n: (-n["degree"], n["id"]))
    edge_dicts.sort(key=lambda e: (-e["weight"], e["source"], e["target"]))
    return node_dicts, edge_dicts


# Node-formation disclosures shared by the full graph notes AND the degeneracy
# fallback bar (which forms the SAME drug nodes but shows no edges) — so the fallback
# carries only the disclosures that apply to it (drug-node formation), never the
# edge-derivation / node-cap / edge-weight-prune notes that describe a graph it does
# not render (L3-1).
_ALIAS_MERGE_NOTE = (
    "Drug nodes are keyed by active ingredient: names are normalized (case, punctuation, "
    "dose/strength, salt, and dose-form/route stripped — 'Sunitinib Malate'≡'Sunitinib', "
    "'Selumetinib, 100mg'≡'Selumetinib'), then a brand folds into its generic via an "
    "alias-only synonym merge — an otherName merges two drugs ONLY when it is itself "
    "another drug's primary name (Keytruda↔pembrolizumab) AND that alias is corroborated "
    "by ≥2 trials, so a single-trial registry mislabel can't collapse two drugs. A "
    "combination product ('Drug A + Drug B') never merges its components (CC-12). "
    "Conservative tradeoff: a brand/generic pair attested by only ONE trial stays split "
    "(the safe direction)."
)
_PLACEBO_NOTE = (
    "Placebo / standard-of-care interventions are excluded by name to avoid a "
    "false mega-hub (G-36)."
)


def _build_fallback(
    records: list[dict], uf: _UnionFind, labels: dict[str, str], *, top_n: int = 25, k: int = 20
) -> dict:
    """The degenerate-network fallback payload: a cited BAR of drug frequencies.

    Computed ALWAYS (cheap); consumed only when the network is degenerate (the
    caller then renders individual per-drug distinct-trial counts instead of an
    empty/1-node graph). Each bucket = one canonical drug with element-targeted
    citations (excerpt = the drug's OWN literal ``.name`` in each contributing
    record, round-trip verifiable by ``is_substring_at``).
    """
    # canonical drug key -> {nctId -> one contributing record} (dedup per trial).
    canon_trials: dict[str, dict[str, dict]] = {}
    for record in records:
        nct = _nct_id(record)
        if nct is None:
            continue
        canons = {
            canon
            for intervention in _drug_interventions(record)
            if (canon := _canonical_key(intervention, uf)) is not None
        }
        for canon in canons:
            canon_trials.setdefault(canon, {}).setdefault(nct, record)

    trials_with_drug: set[str] = set()
    for trials in canon_trials.values():
        trials_with_drug.update(trials.keys())

    # Top-N canonical drugs by distinct-trial count (desc), tie-break by label.
    ranked = sorted(
        canon_trials.items(), key=lambda kv: (-len(kv[1]), labels.get(kv[0], kv[0]))
    )[:top_n]

    buckets: list[dict] = []
    for canon, trials in ranked:
        label = labels.get(canon, canon)
        ncts = sorted(trials.keys())
        contributing_count = len(ncts)  # exact, pre-cap
        sample = ncts[:k]
        citations: list[Citation] = []
        for nct in sample:
            name = _drug_name_in_record(trials[nct], canon, uf) or ""
            citations.append(
                # value = the record's REAL drug-name list (not a copy of the excerpt),
                # so the excerpt check has TEETH — a fabricated drug name fails (L2-F1).
                Citation(
                    nct_id=nct,
                    field_path=_DRUG_PATH,
                    value=_drug_names_in_record(trials[nct]),
                    matched_value=name,
                    excerpt=brief_title(trials[nct]) or name,
                )
            )
        buckets.append(
            {
                "value": label,
                "label": label,
                # each trial counts once per drug → count_mentions == count_trials.
                "count_trials": contributing_count,
                "count_mentions": contributing_count,
                "source_ids": sample,
                "citations": citations,
                "citations_truncated": contributing_count > k,
                "contributing_count": contributing_count,
            }
        )

    total_drugs = len(canon_trials)
    buckets_truncated = total_drugs > top_n
    # Only the disclosures that apply to a drug-frequency BAR (node formation), NOT the
    # graph's edge/cap/prune notes (L3-1). The count-basis-gap note (some matched trials
    # have no drug) needs the executor's countTotal, so the caller adds it.
    notes = [_ALIAS_MERGE_NOTE, _PLACEBO_NOTE]
    if buckets_truncated:
        notes.append(
            f"Showing the top {top_n} drugs by distinct-trial count; "
            f"{total_drugs - top_n} lower-frequency drug(s) are not shown."
        )
    return {
        "mode": "explode",
        "distinct_trials": len(trials_with_drug),
        "total_drugs": total_drugs,
        "buckets_truncated": buckets_truncated,
        "buckets": buckets,
        "notes": notes,
        "note": (
            "Degenerate-network fallback: individual drug frequencies (distinct "
            "trials studying each canonical drug), each cited to the drug's own "
            "name in a contributing record."
        ),
    }


def _build_notes(
    kind: str, *, max_nodes: int, min_edge_weight: int, max_drugs_per_trial: int, skipped: int,
    n_edges: int = 0, n_weight1: int = 0,
) -> list[str]:
    """Standing disclosures always emitted with a network payload."""
    notes = [
        "Edges are derived; weight = number of distinct trials linking the two "
        "endpoints. Each edge carries two citations, one per endpoint field_path (G-25).",
        _ALIAS_MERGE_NOTE,
        _PLACEBO_NOTE,
    ]
    if kind == "sponsor_drug":
        notes.append("Only the lead sponsor is used; collaborators are excluded (G-19).")
        notes.append(
            "Sponsor names are canonicalized by approximate normalization "
            "(case / punctuation / trailing legal-suffix folding) plus a tiny "
            "same-org alias table; NO corporate parent/subsidiary consolidation — "
            "distinct legal entities stay distinct. Stripping a trailing descriptor "
            "(e.g. 'Pharmaceuticals' / 'Therapeutics') can, rarely, merge two "
            "same-stem names."
        )
    # Honest cap/prune note: state the ACTUAL applied threshold; a threshold of 1
    # prunes NOTHING, so don't claim it does.
    if min_edge_weight > 1:
        notes.append(
            f"Nodes capped at the top {max_nodes} by degree; edges below weight "
            f"{min_edge_weight} are pruned (only pairs co-occurring in ≥{min_edge_weight} "
            f"trials are shown). This is a configurable default optimizing "
            f"interpretability, not maximal edge inclusion — the underlying "
            f"co-occurrence graph is unchanged."
        )
    else:
        notes.append(
            f"Nodes capped at the top {max_nodes} by degree; every co-occurrence edge "
            f"(weight ≥ 1) is shown — no edge-weight threshold applied in this view."
        )
    # Density disclosure: a weight-1-heavy graph LOOKS rich but most edges are a single
    # shared trial — say so rather than let the reader over-read the structure (H5).
    if n_edges and n_weight1:
        pct = round(100 * n_weight1 / n_edges)
        notes.append(
            f"Dense baseline view: {n_weight1} of {n_edges} edges ({pct}%) rest on a single "
            f"shared trial (weight 1); raising the minimum edge weight sharpens it to the "
            f"strongest co-occurrences (a Phase-3 legibility knob)."
        )
    if skipped:
        notes.append(
            f"{skipped} trial(s) with more than {max_drugs_per_trial} distinct drugs were "
            "skipped for co-occurrence pairing (basket-trial pair-explosion guard, G-41c)."
        )
    return notes


def build_graph(
    records: list[dict],
    *,
    kind: str,
    max_nodes: int = config.NETWORK_MAX_NODES,
    min_edge_weight: int | None = None,
    max_drugs_per_trial: int = config.NETWORK_MAX_DRUGS_PER_TRIAL,
) -> dict:
    """Assemble a sponsor↔drug or drug↔drug graph from already-paged records.

    Parameters
    ----------
    records:
        Already-fetched study dicts (no I/O happens here). Malformed records are
        skipped, never fatal (TOTAL).
    kind:
        ``"sponsor_drug"`` (bipartite sponsor→drug) or ``"drug_drug"`` (drug
        co-occurrence). An unknown ``kind`` is a caller contract error → ValueError.
    min_edge_weight:
        Minimum edge weight to keep. ``None`` (default) resolves to a
        **per-kind default (P3-WEIGHT): 2 for ``drug_drug``, 1 for
        ``sponsor_drug``** — an explicit int always overrides. Rationale: the
        drug↔drug baseline is a weight-1 hairball (~84% of edges rest on a single
        shared trial), so a default of 2 shows only pairs co-occurring in ≥2
        trials. The underlying graph is unchanged; the knob is configurable;
        degeneracy detection guards over-pruning.
    max_nodes, max_drugs_per_trial:
        Pruning / guard knobs (see §W1b).

    Returns
    -------
    ``{"nodes": [...], "edges": [...], "degenerate": bool, "notes": [str],
    "distinct_trials": int, "fallback": {...}}``. ``degenerate`` is ``True`` iff
    ``len(nodes) <= 1`` or ``len(edges) == 0`` (G-41e) — the caller then falls
    back to the cited ``fallback`` bar of individual drug frequencies.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    # Resolve the per-kind default edge-weight threshold (P3-WEIGHT), from config.
    if min_edge_weight is None:
        min_edge_weight = (
            config.NETWORK_MIN_EDGE_WEIGHT_DRUG_DRUG
            if kind == "drug_drug"
            else config.NETWORK_MIN_EDGE_WEIGHT_SPONSOR_DRUG
        )

    safe_records = [r for r in records if isinstance(r, dict)]
    distinct_trials = len({nct for nct in (_nct_id(r) for r in safe_records) if nct})

    uf, labels = _build_drug_index(safe_records)
    sponsor_labels = _build_sponsor_index(safe_records)
    edges, node_meta, skipped = _accumulate_edges(
        safe_records,
        kind=kind,
        uf=uf,
        labels=labels,
        sponsor_labels=sponsor_labels,
        max_drugs_per_trial=max_drugs_per_trial,
    )
    node_dicts, edge_dicts = _prune_and_shape(
        edges,
        node_meta,
        uf,
        max_nodes=max_nodes,
        min_edge_weight=min_edge_weight,
    )

    degenerate = len(node_dicts) <= 1 or len(edge_dicts) == 0
    n_weight1 = sum(1 for edge in edge_dicts if edge.get("weight") == 1)
    notes = _build_notes(
        kind,
        max_nodes=max_nodes,
        min_edge_weight=min_edge_weight,
        max_drugs_per_trial=max_drugs_per_trial,
        skipped=skipped,
        n_edges=len(edge_dicts),
        n_weight1=n_weight1,
    )
    # Silence "no edges" degenerate networks with an explicit disclosure.
    if degenerate:
        notes.append(
            "Graph is degenerate (≤1 node or no co-occurring edges); the caller "
            "should fall back to a bar chart (G-41e)."
        )

    # The fallback bar is cheap and always computed; consumed only on degeneracy.
    fallback = _build_fallback(safe_records, uf, labels)

    return {
        "nodes": node_dicts,
        "edges": edge_dicts,
        "degenerate": degenerate,
        "notes": notes,
        "distinct_trials": distinct_trials,
        "fallback": fallback,
    }
