"""Real ClinicalTrials.gov v2 token sets + field-path allowlist.

Every value here is grounded in the live-verified API brief
(``SPEC_INTERROGATION.md`` §C) — nothing is invented. This module is the token
vocabulary the Plan Checker (``app.plan.checker``) validates a planner's
tokens/fields against, so a hallucinated phase string can never clear validation.

Precisely which table does that work:

* The token frozensets (:data:`PHASE_TOKENS`, :data:`OVERALL_STATUS_TOKENS`, …) are
  checked directly by the checker AND re-checked at the wire boundary in
  ``app.ctgov.params.build_search_params``.
* :data:`FIELD_ALIASES` is what the checker validates ``plan.field`` against — an
  invented alias is rejected there.
* :data:`FIELD_PATHS` has NO runtime importer. Its only consumer is the
  self-consistency ``assert`` at the bottom of this module, which pins every
  alias VALUE to the allowlist. So "an invented JSON path can never reach the
  wire" holds *indirectly*: paths are never planner-supplied — they are looked up
  from ``FIELD_ALIASES``, whose values this allowlist constrains at import.
"""

from __future__ import annotations

# --- Base pin (ARCHITECTURE_SPEC §A(a)/(d)) --------------------------------

# The one host this whole system is allowed to call for registry data. Compiled in
# here, never read from the environment (``.env.example``'s CTGOV_BASE_URL is
# documentation only — no code reads it). The HTTP client
# (app.ctgov.client.CTGovClient) parses any base_url with ``urlparse`` and requires
# an EXACT hostname match against "clinicaltrials.gov" — deliberately NOT a
# ``startswith`` check, which a userinfo trick
# ("https://clinicaltrials.gov@evil.com/...") or a suffix trick
# ("https://clinicaltrials.gov.evil.com/...") would both defeat.
BASE_URL = "https://clinicaltrials.gov/api/v2"

# --- Enum token sets (SPEC_INTERROGATION §C, live-verified 2026-07-15) ----

# protocolSection.designModule.phases (array). 6 tokens, no slashes — a combined
# phase is ["PHASE1","PHASE2"], never the string "PHASE1/PHASE2" (CC-15).
PHASE_TOKENS = frozenset({"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"})

# protocolSection.statusModule.overallStatus (single, never missing). All 14 real values.
OVERALL_STATUS_TOKENS = frozenset(
    {
        "COMPLETED",
        "UNKNOWN",
        "RECRUITING",
        "TERMINATED",
        "NOT_YET_RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "WITHDRAWN",
        "ENROLLING_BY_INVITATION",
        "SUSPENDED",
        "WITHHELD",
        "NO_LONGER_AVAILABLE",
        "AVAILABLE",
        "APPROVED_FOR_MARKETING",
        "TEMPORARILY_NOT_AVAILABLE",
    }
)

# protocolSection.designModule.studyType.
STUDY_TYPE_TOKENS = frozenset({"INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"})

# protocolSection.sponsorCollaboratorsModule.leadSponsor.class (9 tokens).
SPONSOR_CLASS_TOKENS = frozenset(
    {"OTHER", "INDUSTRY", "OTHER_GOV", "NIH", "NETWORK", "FED", "INDIV", "UNKNOWN", "AMBIG"}
)

# protocolSection.armsInterventionsModule.interventions[].type (11 tokens).
INTERVENTION_TYPE_TOKENS = frozenset(
    {
        "DRUG",
        "OTHER",
        "DEVICE",
        "BEHAVIORAL",
        "PROCEDURE",
        "BIOLOGICAL",
        "DIAGNOSTIC_TEST",
        "DIETARY_SUPPLEMENT",
        "RADIATION",
        "COMBINATION_PRODUCT",
        "GENETIC",
    }
)

# The whitelisted ClinicalTrials.gov search-area params (query.<area>=...).
QUERY_AREAS = frozenset({"term", "cond", "intr", "spons", "locn"})

# The 5 real date fields the API exposes (mirrors app.plan.models.DateField's
# Literal values — kept as an independent frozenset here, not imported, so
# app.ctgov stays a foundational layer with no dependency on app.plan).
DATE_FIELDS = frozenset(
    {"startDate", "primaryCompletionDate", "completionDate", "studyFirstPostDate", "lastUpdatePostDate"}
)

# --- Field-path allowlist (the closed set FIELD_ALIASES' values are pinned to) ---
# Read by nothing at runtime except the ``assert`` at the bottom of this file; see
# the module docstring for how it constrains the paths that DO reach the wire.

FIELD_PATHS = frozenset(
    {
        "protocolSection.identificationModule.nctId",
        "protocolSection.designModule.phases",
        "protocolSection.designModule.studyType",
        "protocolSection.statusModule.overallStatus",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
        "protocolSection.armsInterventionsModule.interventions[].type",
        "protocolSection.armsInterventionsModule.interventions[].name",
        "protocolSection.contactsLocationsModule.locations[].country",
        "protocolSection.statusModule.startDateStruct.date",
        "protocolSection.statusModule.primaryCompletionDateStruct.date",
        "protocolSection.statusModule.completionDateStruct.date",
        "protocolSection.statusModule.studyFirstPostDateStruct.date",
        "protocolSection.statusModule.lastUpdatePostDateStruct.date",
    }
)

# Short alias -> real JSON path, so recipes/checker/tools can refer to a field by
# a human name (e.g. "phase") instead of repeating the full path everywhere.
FIELD_ALIASES: dict[str, str] = {
    "nctId": "protocolSection.identificationModule.nctId",
    "phase": "protocolSection.designModule.phases",
    "studyType": "protocolSection.designModule.studyType",
    "overallStatus": "protocolSection.statusModule.overallStatus",
    "sponsorClass": "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
    "sponsorName": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "interventionType": "protocolSection.armsInterventionsModule.interventions[].type",
    "interventionName": "protocolSection.armsInterventionsModule.interventions[].name",
    "country": "protocolSection.contactsLocationsModule.locations[].country",
    "startDate": "protocolSection.statusModule.startDateStruct.date",
    "primaryCompletionDate": "protocolSection.statusModule.primaryCompletionDateStruct.date",
    "completionDate": "protocolSection.statusModule.completionDateStruct.date",
    "studyFirstPostDate": "protocolSection.statusModule.studyFirstPostDateStruct.date",
    "lastUpdatePostDate": "protocolSection.statusModule.lastUpdatePostDateStruct.date",
}

# Self-consistency guard (not a user-input check): every alias must resolve to a
# path that's actually in the allowlist, so the two tables can never drift apart.
# It is an ``assert``, so it is a DEVELOPER guard, not a runtime one — ``python -O``
# strips it. That is acceptable precisely because no user input flows through it:
# it can only ever fail on a source edit, which the test suite runs unoptimized.
assert set(FIELD_ALIASES.values()) <= FIELD_PATHS
