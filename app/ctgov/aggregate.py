"""The aggregation core (ARCHITECTURE_SPEC §3.6 / §B.6) -- Wave 2.

The engine every high-level tool (``aggregate_by``, ``timeseries``, ``compare``,
``build_network``) delegates to. Under the hood they all reduce to one
``page_and_group`` primitive and differ only in grouping key + counting mode
(§B.6): ``timeseries``'s key is a date bin, ``build_network``'s key is an entity
*pair*, ``compare`` runs this twice and unions the categories.

This is where correctness lives: paging under a page budget (CC-6), dual counts
(distinct-trial + trial×value mention, CC-3), explicit Missing/NA buckets via the
``key_fn`` (CC-5), combined-value own-bucket semantics (CC-15), and the material
that ``countTotal`` reconciliation (CC-16) is proven against. ``count_trials`` is
a **distinct-nctId** count per bucket (a duplicate page row / mid-walk repeat is
deduped, K3), so for ``combine`` ``Σ count_trials`` over buckets equals
``distinct_trials`` equals the API's exact ``totalCount`` -- the reconciliation is
against the distinct-trial count it claims to be, not a raw record tally.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from app.ctgov.client import CTGovClient

# Mirrors ``CTGovClient.iter_studies``'s default page size, used only to report
# an informational ``pages_read`` (nothing downstream depends on it).
_PAGE_SIZE = 1000

_NCT_PATH = ("protocolSection", "identificationModule", "nctId")


def _nct_id(record: dict) -> str | None:
    """Read the nctId from a record, or ``None`` if the path is absent/malformed."""
    current: object = record
    for part in _NCT_PATH:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, str) else None


@dataclass
class Bucket:
    """One grouped bucket: its identity, dual counts, and contributing records.

    ``records`` is the list of contributing record dicts (the citation surface).
    For both modes ``len(records) == count_trials`` -- one entry per distinct
    trial in the bucket -- so it is the exact contributing set the per-bucket
    citation sample is drawn from.
    """

    value: str
    label: str
    count_trials: int
    count_mentions: int
    records: list[dict] = field(default_factory=list)


@dataclass
class GroupResult:
    """The output of one ``page_and_group`` pass.

    ``distinct_trials`` is the count of distinct nctIds seen across ALL records
    (the reconciliation anchor for explode; equal to ``Σ count_trials`` for
    combine). ``truncated`` is True iff paging stopped on the budget with more
    pages pending.
    """

    buckets: list[Bucket]
    distinct_trials: int
    truncated: bool
    pages_read: int


class AggregationCore:
    """Pages a ClinicalTrials.gov search under a budget and buckets the results.

    Every high-level tool (§3.5) is a thin wrapper over this one primitive
    (§B.6) -- the "single general aggregation core covers all 5 query classes"
    decision (CC-11) that makes breadth cheap.
    """

    def __init__(self, client: CTGovClient) -> None:
        self.client = client

    def page_and_group(
        self,
        search_params: dict,
        *,
        fields: str,
        key_fn: Callable[[dict], list[tuple[str, str]]],
        mode: str,
        budget_pages: int = 20,
    ) -> GroupResult:
        """Page ``search_params`` under a budget and bucket records by ``key_fn``.

        Parameters
        ----------
        search_params:
            The search-selecting wire params (from ``build_search_params``) --
            the SAME params the exact-count call uses, or reconciliation breaks
            (one population, G-23).
        fields:
            The pipe-separated ``fields=`` projection (e.g. ``"NCTId|Phase"``).
        key_fn:
            Maps one record to its ``(value, label)`` bucket key(s).
        mode:
            ``"combine"`` -- ``key_fn`` returns exactly 1 key/record; the trial
            counts once for ``count_trials`` and once for ``count_mentions``
            (they are equal). ``"explode"`` -- ``key_fn`` returns >=1 distinct
            key/record; the trial counts once per DISTINCT value for
            ``count_trials`` and once per OCCURRENCE for ``count_mentions``.
        budget_pages:
            Page budget (default 20 = 20,000 trials at pageSize=1000). Above it,
            paging stops and ``truncated`` is True -- callers refuse the chart
            rather than ship a biased prefix (§B.7).

        Returns
        -------
        GroupResult
            Buckets in first-seen order (the tools layer re-sorts them), the
            distinct-trial total, the truncation flag, and an informational
            page count.
        """
        records, truncated = self.client.iter_studies(
            search_params, fields=fields, max_pages=budget_pages
        )

        # value -> accumulator, insertion-ordered so the label from the FIRST
        # time a value is seen is the one kept (dicts preserve insertion order).
        accums: dict[str, dict] = {}
        distinct_ids: set[str] = set()
        combine_seen: set[str] = set()  # walk-global dedup for combine (first-key-wins)

        for record in records:
            # A non-dict page element (a JSON ``null``/scalar inside ``studies[]``)
            # must NOT sink the whole aggregation (K1/B2): the client validates that
            # ``studies`` is a list but not that every ELEMENT is a dict, and a bare
            # ``key_fn(None)`` would raise. Skip it here (the record stream is the one
            # choke point every combine/explode key_fn flows through).
            if not isinstance(record, dict):
                continue
            nct = _nct_id(record)

            keys = key_fn(record)
            if not keys:  # a total key_fn never returns []; defend anyway (no IndexError)
                continue

            # Count a distinct trial once we know it will be bucketed (an id-ful record
            # whose key_fn returned [] must not inflate the distinct anchor above Σbars).
            if nct is not None:
                distinct_ids.add(nct)

            if mode == "combine":
                # Exactly one bucket per record; trials == mentions.
                value, label = keys[0]
                # count_trials is a DISTINCT-TRIAL count via a WALK-GLOBAL dedup
                # (first-key-wins): a same-key duplicate page row AND a CROSS-key
                # repeat -- the same nctId appearing under a different combine key
                # later in the walk because the registry mutated between cursor pages
                # (CT.gov paging is not snapshot-isolated) -- both count ONCE, so
                # Σ bars == distinct_trials always holds (K3 + the cross-key case). An
                # id-less record has no identity to reconcile and is not counted.
                if nct is None or nct in combine_seen:
                    continue
                combine_seen.add(nct)
                accum = self._accum_for(accums, value, label)
                accum["count_trials"] += 1
                accum["count_mentions"] += 1
                accum["records"].append(record)
            elif mode == "explode":
                for value, label in dict.fromkeys(keys):  # distinct (value,label) per record
                    accum = self._accum_for(accums, value, label)
                    accum["count_mentions"] += 1  # once per (trial, distinct value)
                    # count_trials = distinct nctIds carrying this value; dedup by
                    # nctId across records (same reconciliation discipline as combine).
                    if nct is not None and nct not in accum["seen_ids"]:
                        accum["seen_ids"].add(nct)
                        accum["count_trials"] += 1
                        accum["records"].append(record)
            else:
                raise ValueError(f"unknown aggregation mode {mode!r}; expected 'combine' or 'explode'")

        buckets = [
            Bucket(
                value=value,
                label=accum["label"],
                count_trials=accum["count_trials"],
                count_mentions=accum["count_mentions"],
                records=accum["records"],
            )
            for value, accum in accums.items()
        ]

        pages_read = math.ceil(len(records) / _PAGE_SIZE) if records else 0
        return GroupResult(
            buckets=buckets,
            distinct_trials=len(distinct_ids),
            truncated=truncated,
            pages_read=pages_read,
        )

    @staticmethod
    def _accum_for(accums: dict[str, dict], value: str, label: str) -> dict:
        """Get or create the accumulator for ``value``, keeping the first-seen label."""
        accum = accums.get(value)
        if accum is None:
            accum = {
                "label": label,
                "count_trials": 0,
                "count_mentions": 0,
                "records": [],
                "seen_ids": set(),  # nctIds already counted in this bucket (dedup, K3)
            }
            accums[value] = accum
        return accum
