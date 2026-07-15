# AI Clinician

AI Clinician is a research-only platform for reconstructing longitudinal metastatic colorectal-cancer (mCRC) timelines, representing current Chinese and US guidance as computable artifacts, discovering candidate disease states, and evaluating constrained dynamic treatment strategies with causal methods.

It does **not** diagnose, prescribe, place orders, or autonomously change an approved guideline. AI-generated outputs remain in the `ai_candidate` channel until independent expert review and later external/prospective validation. The legacy `virtualhuman_agents` package remains available as a separate, non-clinical disease-model R&D reference implementation.

## Safety boundary

- Raw and patient-level derived data stay outside the Git worktree. `AI_CLINICIAN_RAW_ROOT` is read-only; `AI_CLINICIAN_DERIVED_ROOT` is a separate approved location.
- Patient text is processed by deterministic rules and optional local model adapters. The package contains no cloud extractor.
- The API exposes aggregate research metadata and guideline artifacts only; patient text is handled by local batch commands.
- Every recommendation records jurisdiction, version, evidence certainty, source locator, license status, applicability, uncertainty, and abstention reasons.
- Low overlap, OOD input, missing critical variables, timeline conflicts, wide confidence intervals, or failed validation gates force abstention.

## Architecture

```text
governed XLSX / local records
        -> aggregate data profile + quarantine counts
        -> evidence-linked clinical timelines
        -> locked patient/time cohort split
        -> candidate continuous states
        -> target-trial emulation and benefit/harm estimates
        -> ai_candidate computable guideline
        -> expert review (never automatic promotion)
```

The computable guideline layer follows the concepts in the [HL7 Clinical Practice Guidelines implementation guide](https://hl7.org/fhir/uv/cpg/) and exports FHIR R4 `PlanDefinition`, `ActivityDefinition`, and `Library` resources. Evidence-to-recommendation decisions are represented explicitly rather than delegated to an LLM.

## Install and test

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/check_repository_safety.py
```

## Local data configuration

```bash
export AI_CLINICIAN_RAW_ROOT=/approved/read-only/clinical-data
export AI_CLINICIAN_DERIVED_ROOT=/approved/derived/ai-clinician
export AI_CLINICIAN_PSEUDONYM_SECRET='provision-with-your-secret-manager'
export AI_CLINICIAN_API_KEY='provision-with-your-secret-manager'
export AI_CLINICIAN_AUDIT_SIGNING_KEY='provision-at-least-32-random-bytes'
```

Both directories must resolve outside the repository, and must not be the same directory or contain one another. Configuration is validated before any patient-level command starts.

## CLI

```bash
ai-clinician data profile --workbook cohort-1.xlsx --id-column 住院号
ai-clinician data link --workbook cohort-1.xlsx --workbook cohort-2.xlsx --id-column 住院号 --output linked-patient-views.json
ai-clinician timeline build --input records.jsonl --output timelines.jsonl
ai-clinician timeline validate --predicted timelines.jsonl --gold gold.jsonl --decision-times decision-times.json --manifest-registration-id timeline_manifest_xxx
ai-clinician states fit --input state-features.json --output state-model.json
ai-clinician target-trial run --input target-trial.json
ai-clinician guideline build --input recommendation.json --db metadata.db
ai-clinician guideline lock --release-id release_xxx --db metadata.db
ai-clinician serve --db metadata.db
```

Raw inputs resolve inside `AI_CLINICIAN_RAW_ROOT`; normalized inputs, decision-time manifests, outputs, and databases resolve inside `AI_CLINICIAN_DERIVED_ROOT`. Patient-level JSONL is an interchange format inside that approved environment and is ignored by Git. CLI responses suppress raw patient identifiers and source text. State fitting performs the patient-level 70/15/15 temporal split itself and fits only the development partition.

Two promotion paths deliberately remain closed in v0.1. `states evaluate` rejects externally supplied metrics until the locked-test evaluator computes calibration and Brier improvement directly from a frozen model and partition. `target-trial run` can produce an abstained diagnostic from a signed frozen input, but manual negative-control or sensitivity-result upload is prohibited; without a validated deterministic audit runner it cannot register a locked `TargetTrialResult`. Therefore the current code cannot advance an AI candidate to `candidate_guideline` or `research_release` by self-reported evidence.

## API

Start the aggregate-only research API:

```bash
ai-clinician serve --host 127.0.0.1 --port 8000 --db metadata.db
```

The service refuses to start without a strong local API key and can bind only a loopback address. Every governed store write re-authenticates an independently provisioned role token and requires an HMAC-signed, actor-attributed hash-chain entry; caller-supplied reviewer identity or publisher role is never trusted. The service exposes health, non-narrative study/release/recommendation summaries, expert-review actions, and audit verification. Full recommendation prose remains in the approved guideline document/review system. The API deliberately has no patient-text or automated-order endpoint.

Individual recommendations can be draft or in review, but can never declare themselves locked or released. Publication status exists only on a hash-bound `GuidelineRelease`; the MVP forces both recommendations and releases to remain `research_only`.

## Validation gates

The MVP encodes the pre-specified Go/No-Go thresholds in `configs/clinical_mvp.example.json`. Timeline reconstruction must pass critical-event precision/recall, treatment-line macro-F1, date tolerance, inter-rater agreement, and zero future leakage. Strategy evaluation additionally requires adequate sample size, common support, effective sample size, covariate balance, confidence intervals, and sensitivity checks. Failure returns a research gap or abstention, not a treatment recommendation.

CPG-on-FHIR exports are non-ordering research artifacts. Until institution-specific FHIR search mappings are independently verified, the emitted CQL applicability expression is intentionally fail-closed and cannot mark a recommendation applicable.

## Repository contents

- `src/ai_clinician/`: clinical research schemas, privacy controls, timeline pipeline, computable guideline lifecycle, state and causal engines, CLI/API.
- `src/virtualhuman_agents/`: legacy non-clinical disease-model R&D control plane.
- `tests/`: synthetic-only unit, integration, red-team, and privacy tests.
- `docs/AI_CLINICIAN_MVP.md`: implementation and validation detail.
- `docs/CLINICAL_REGULATORY_ROUTE.md`: function-specific US/China research planning boundaries.
- `docs/GUIDELINE_INGESTION.md`: licensing, provenance, GRADE, and CPG export controls.
- `configs/clinical_mvp.example.json`: non-patient configuration template.

No clinical dataset, guideline full text, model weight, database, generated package, or patient-level manifest belongs in this public repository.

The legacy API is also local-only: it requires `VIRTUALHUMAN_API_KEY` and refuses non-loopback binding.
