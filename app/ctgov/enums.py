"""Real ClinicalTrials.gov v2 token sets + field-path allowlist (Phase 0).

Every value here is grounded in the live-verified API brief
(``SPEC_INTERROGATION.md`` §C) — nothing is invented. This module is the single
source of truth the Plan Checker (``app.plan.checker``) validates a planner's
tokens/fields against, so a hallucinated phase string or an invented JSON path
can never clear validation.
"""

from __future__ import annotations

# --- Base pin (ARCHITECTURE_SPEC §A(a)/(d)) --------------------------------

# The one host this whole system is allowed to call for registry data. The HTTP
# client (app.ctgov.client.CTGovClient) asserts every base_url starts with this.
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

# --- Field-path allowlist (the JSON paths the checker/citations validate against) ---

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
assert set(FIELD_ALIASES.values()) <= FIELD_PATHS
