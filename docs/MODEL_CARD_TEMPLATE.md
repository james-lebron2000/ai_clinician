# Model card template — AI Clinician

## Identification and intended use

Record model name/version, code hash, model hash, data snapshot IDs, preprocessing version, context of use, research-only status, forbidden uses, owner, and lock date. State explicitly that the model does not diagnose, prescribe, select a unique best treatment, or create an order.

## Development and evaluation

Describe the patient-level temporal split, state features, prohibited treatment/action features, baselines, hyperparameters, bootstrap stability, locked-test calibration, Brier improvement, expert interpretability reviews, and all subgroup analyses. For causal estimates, include the registered target trial, overlap, ESS, balance, 95% CI, negative control, sensitivity analysis, and agreement across estimators.

## Refusal behavior

List thresholds for missing critical variables, timeline conflict, OOD, positivity, common support, effective sample size, covariate balance, confidence-interval width, failed controls, and estimator disagreement. Show tests proving that each condition returns an abstention without a numerical treatment claim or order.

## Governance and monitoring

Bind validation reports and independent review IDs to the locked artifact hash. Describe batch-only update control, audit verification, external validation status, shadow-mode plan, and retirement criteria. Online learning and automatic guideline promotion are prohibited.
