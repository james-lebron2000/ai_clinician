# Guideline evidence ingestion protocol

Recommendation records never carry an independent locked/released status. Only a hash-bound `GuidelineRelease` can be locked or published after the required role-separated reviews; every v0.1 artifact remains research-only.

Guideline ingestion occurs only in the approved document-management environment. The public repository contains an empty metadata catalogue and strict schemas; it contains no licensed guideline text.

For each source, the steward records organization, jurisdiction, exact version, effective dates, document checksum when permitted, source URL, stable section/page/table locator, license status, and the permitted storage mode. A `review_required` entry cannot be used to build a recommendation. Restricted sources remain external: only allowed metadata, locator, checksum where permitted, and link are stored.

Each encoded recommendation must include a complete PICO, eligibility and exclusions, evidence certainty, recommendation strength, hard safety constraints, uncertainty, abstention conditions, and a human-authored GRADE Evidence-to-Decision record. A recommendation hash excludes only creation time; a release binds every recommendation ID to that hash. Baseline content requires authenticated expert-committee governance and three independent reviews before lock. AI-candidate content remains research-only and has no automatic promotion path.

The CPG-on-FHIR exporter creates `Library`, proposal-only `ActivityDefinition`, and `PlanDefinition` resources. It regenerates the Bundle from the stored typed recommendation before accepting it into the audit store. `MedicationRequest`, `ServiceRequest`, patient resources, source excerpts, executable orders, and unlicensed full text are prohibited.
