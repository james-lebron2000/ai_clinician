# Timeline annotation and lock protocol

This protocol applies only inside the approved clinical-data environment. It is a prospective specification for creating the 500-case timeline reference set; it does not claim that annotation has already been completed.

Cases are sampled across diagnosis year, age group, sex, primary site, metastatic pattern, record completeness, treatment line, and outcome availability. At least three colorectal-cancer clinicians participate. Two clinicians independently annotate every selected case; a third clinician adjudicates disagreements without seeing model output. Patient identifiers and note text remain in the approved annotation system and never enter this repository.

The annotation schema covers diagnosis, metastasis, ECOG, pathology/TNM, molecular testing, surgery, treatment start and end, regimen class, treatment line, response, progression, grade 3 or higher toxicity, follow-up, and death. Each item records event time, time precision, source availability time, source location, negation, uncertainty, and conflict status. A decision-time snapshot manifest is created separately so feature leakage can be tested rather than inferred from an empty flag.

Twenty percent of the 500 cases are selected once, before extraction-model iteration, as the locked test subset. Patient keys—not rows—define every split. The lock manifest contains only pseudonymous keys, hashes, versions, and timestamps and is stored outside Git. Changes require a new version, documented rationale, data-steward approval, and a fresh lock; the old lock is retained.

The analysis-cohort split manifest is also stored outside Git and has four required members: ordered `development`, `tuning`, and `locked_test` patient-key lists plus an `index_dates` object containing each patient's first eligible mCRC decision time. `data lock-cohort` recomputes the chronological 70/15/15 split with stable patient-key tie-breaking, verifies disjointness and full coverage, and stores separate membership hashes and counts. State fitting recomputes the split and must reproduce all three locked hashes exactly.

Go/No-Go requires precision and recall of at least 0.90 for treatment-line, progression, and death events; treatment-line macro-F1 of at least 0.90; at least 90% of key dates within 14 days; Cohen's kappa of at least 0.80; and zero detected future information. A failed gate automatically limits the project to static stratification and computable baseline guidance.
