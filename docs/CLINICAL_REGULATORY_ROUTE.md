# Clinical regulatory route (research planning only)

This document is a planning aid, not a legal classification or regulatory determination. The v0.1 software is restricted to retrospective research, has no patient-facing function, creates no order, and cannot reach `research_release` using self-reported evidence.

## United States

FDA's January 2026 final guidance explains the statutory criteria used to assess whether a health-care-professional clinical decision-support function may be excluded from the device definition. Classification is function-specific; calling a product “CDS” is not itself determinative. See the official [Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software) and [FDA CDS FAQ](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs).

The baseline-guideline viewer is designed so a clinician can inspect the source version, PICO, applicability criteria, safety constraints, rationale, and evidence location independently. That architecture supports a future function-by-function assessment against the non-device CDS criteria, but v0.1 makes no non-device determination.

AI-discovered disease states, patient-specific causal strategy estimates, opaque ranking, or functions whose basis cannot be independently reviewed are tracked on the device-software workstream. Before any clinical deployment, the sponsor should document each software function, intended user, inputs, outputs, role in the decision, explainability, foreseeable misuse, and whether an FDA Q-Submission is appropriate. No current artifact is labeled FDA-cleared, approved, or exempt.

## China

The project conservatively treats patient-specific AI-assisted diagnostic or treatment decision functions as medical-device candidates until the competent authority determines otherwise. NMPA has published [principles for classification of AI-based medical software](https://english.nmpa.gov.cn/2021-07/08/c_660267.htm), and its current [medical-device classification service](https://zwfw.nmpa.gov.cn/web/taskview/11100000MB0341032Y100207202300001) describes the process for products not clearly covered by the classification catalogue or whose class is uncertain.

Before clinical use, the sponsor should prepare a function list, intended purpose, algorithm role, clinical risk analysis, data provenance, cybersecurity and software-lifecycle evidence, human-factors evidence, and a proposed classification rationale, then seek classification consultation through the applicable NMPA/provincial process. “High risk” in project documents is a conservative engineering assumption, not an NMPA classification decision.

## Evidence sequence shared by both routes

1. Complete the retrospective locked research MVP and its timeline Go/No-Go.
2. Conduct external-center temporal validation with a frozen model and guideline version.
3. Run prospectively in silent mode with no influence on care.
4. Evaluate human–AI interaction, comprehension, automation bias, failure recovery, and workflow impact using a DECIDE-AI-aligned protocol.
5. Obtain function-specific regulatory advice before an interventional or decision-influencing study.
6. Consider a pragmatic randomized evaluation only after technical, clinical, human-factors, privacy, and regulatory gates pass.

Every release must preserve the distinction between `baseline_approved` and `ai_candidate`. There is no automatic promotion path, online learning path, or mechanism that turns a research association into a treatment-effect claim.
