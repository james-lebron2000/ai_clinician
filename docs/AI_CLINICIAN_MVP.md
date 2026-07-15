# AI Clinician CRC Continuum MVP

## Intended use

The first release is a locked research system for reconstructing mCRC timelines from governed Chinese records, finding reproducible candidate state transitions, and estimating benefit/harm trade-offs for guideline-backed regimen classes. It is not a medical device release, a clinical-practice guideline, or a substitute for multidisciplinary review.

`baseline_approved` contains faithfully encoded, versioned recommendations that an authorized expert panel has reviewed. `ai_candidate` contains hypotheses generated from observational data. Candidate content cannot overwrite approved content and cannot be promoted without provenance completeness, validation evidence, and independent expert approvals.

## Data lifecycle

Raw workbooks are opened read-only. Four header rows are flattened into stable column names, workbook-level counts are calculated, and patient identifiers are used only in memory for cross-table aggregate overlap. Identifiers are HMAC-pseudonymized only when an approved patient-level timeline artifact is created outside Git. Missing or duplicated identifiers are represented as quarantine counts. No raw identifier or source text is written to the aggregate metadata store.

Timeline extraction creates structured events for diagnosis, metastatic disease, ECOG, pathology/TNM, molecular testing, surgery, systemic therapy, response/progression, toxicity, follow-up, and death. Every event records a source reference, text hash, offsets, confidence, time precision, and the time at which its source was clinically available. A decision-time view rejects evidence that became available later.

The production timeline Go/No-Go requires a complete, patient-keyed decision-time snapshot manifest. Missing snapshot times fail the gate; they are never treated as evidence of zero leakage. A local model adapter must bind every event to the current source document, byte-for-byte evidence hash, valid offsets, and the source record's actual availability time.

## Modeling boundary

Patients are divided chronologically by their first eligible mCRC decision time: 70% development, 15% tuning, and 15% locked temporal test. Feature fitting happens after the split. Treatment variables are excluded from state discovery features and supplied separately as actions or transition context.

The executable MVP state engine is an interpretable continuous-time mixture-state baseline with person-time transition intensities. It is intentionally simpler than the planned switching state-space/HSMM primary analysis; an HSMM may replace it only through the same locked split, leakage, stability, calibration, predictive-gain, and expert-interpretability contracts.

State candidates must be stable under resampling, calibrated, improve held-out prognosis over TNM/treatment-line baselines, and receive independent clinical interpretation. Target-trial analyses define eligibility, time zero, feasible strategies, outcomes, causal contrast, censoring, and analysis before estimation. Unsupported actions and low-overlap populations are excluded rather than extrapolated.

The v0.1 repository does not yet contain the locked-test calibration/Brier evaluator. Its `states evaluate` entry point therefore rejects user-supplied metrics, and no candidate state can be promoted from a hand-authored report. State fitting and patient-cluster bootstrap stability are available for development diagnostics only.

The initial decision points are first-line initiation, response/progression reassessment, second-line initiation, and third-line initiation. Outputs are Pareto treatment options over survival, progression, response, and severe toxicity. They are never rendered as a unique best treatment.

## Evidence and guideline lifecycle

Guideline documents remain in an approved document-management location. The repository stores only schema and synthetic examples. Each recommendation requires jurisdiction, document version, effective dates, PICO, applicability and exclusion logic, evidence certainty, recommendation strength, safety constraints, source locator, and redistribution status.

The aggregate API returns only enumerated status, counts, hashes, and non-narrative identifiers. It does not serialize guideline rationale, recommendation titles, context-of-use prose, review rationale, source excerpts, or a generic stored payload. Reviewers obtain the full licensed artifact from the approved document-management system and bind their decision to its content hash.

Computable exports use FHIR R4 guideline resources and CQL-compatible expression references. The exporter intentionally emits proposal activities only; it does not create a FHIR medication request, service request, or executable order.

## Validation

The locked timeline set is 20% of a 500-case, stratified, independently annotated reference set. Critical treatment-line, progression, and death events require precision and recall of at least 0.90; line classification macro-F1 at least 0.90; at least 90% of key dates within 14 days; Cohen's kappa at least 0.80; and zero future leakage.

Policy estimation starts only when a decision point has at least 1,000 eligible patients, each compared action has at least 200 observations, at least 80% of the target population lies in common support, and each weighted arm has effective sample size of at least 100. Post-weighting absolute standardized mean differences must remain below 0.10. Estimates include 95% confidence intervals and explicit abstention reasons.

The executable binary-action estimator currently provides IPTW marginal structural and cross-fitted AIPW diagnostics. Explicit OOD status, a negative control, sensitivity analysis, and directional agreement are mandatory. Manual causal-audit JSON ingestion and generic-store writes of causal audits or locked target-trial results are prohibited. Until a validated deterministic negative-control and unmeasured-confounding runner is implemented, `target-trial run` fails those gates, returns an abstention, and cannot create promotion evidence. Clone-censor-weight and sequential g-formula analyses remain registered extensions for longitudinal adherence questions; they must not be claimed as implemented by this MVP.

Candidate states require bootstrap adjusted Rand index of at least 0.75, locked-test calibration slope between 0.8 and 1.2, at least 5% relative integrated-Brier improvement over the registered baseline, and confirmation by at least two clinical experts.

## Clinical translation

After the research release, the required sequence is external-center retrospective validation, prospective silent operation, DECIDE-AI-style human factors evaluation, and then an appropriately designed comparative clinical study. Models are locked and updated in controlled batches; there is no online learning or automatic guideline promotion.

Reporting packages should follow [TARGET](https://www.bmj.com/content/390/bmj-2025-087179), [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), RECORD, and the relevant design-specific guidance. Evidence-to-decision judgments remain human decisions organized with [GRADE EtD](https://book.gradepro.org/guideline/introduction-to-the-evidence-to-decision-frameworks); computable exports follow [HL7 CPG-on-FHIR](https://hl7.org/fhir/uv/cpg/). External retrospective validation, prospective silent operation, and [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9) evaluation are prerequisites to any clinical-use discussion.
