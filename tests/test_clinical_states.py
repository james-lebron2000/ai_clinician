from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ai_clinician import states as states_module
from ai_clinician.states import (
    AnalysisColumnManifest,
    ColumnRole,
    ContinuousTimeStateDiscovery,
    StateDiscoveryError,
    assert_no_future_information,
    chronological_patient_split,
    evaluate_state_validation_gate,
)
from ai_clinician.cli import main as cli_main


def _analysis_manifest(
    *,
    extra_columns: list[dict[str, str]] | None = None,
    treatment_name: str = "treatment",
) -> AnalysisColumnManifest:
    return AnalysisColumnManifest.model_validate(
        {
            "schema_version": "1.0",
            "locked": True,
            "columns": [
                {"name": "patient_id", "role": ColumnRole.PATIENT_ID},
                {"name": "decision_date", "role": ColumnRole.INDEX_DATE},
                {"name": "event_time", "role": ColumnRole.EVENT_TIME},
                {
                    "name": "feature_available_time",
                    "role": ColumnRole.FEATURE_AVAILABLE_TIME,
                },
                {"name": "ecog", "role": ColumnRole.STATE_FEATURE},
                {"name": "cea", "role": ColumnRole.STATE_FEATURE},
                *(extra_columns or []),
                {"name": treatment_name, "role": ColumnRole.TREATMENT_ACTION},
            ],
        }
    )


def _longitudinal_records() -> list[dict[str, object]]:
    rng = np.random.default_rng(10)
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    for patient_index in range(20):
        decision = origin + timedelta(days=patient_index * 10)
        for visit in range(3):
            severe = (patient_index + visit) % 2
            rows.append(
                {
                    "patient_id": f"patient-{patient_index:02d}",
                    "decision_date": decision,
                    "event_time": decision + timedelta(days=visit * (15 + patient_index % 3)),
                    "feature_available_time": decision
                    + timedelta(days=visit * (15 + patient_index % 3)),
                    "ecog": float(severe * 2 + rng.normal(0, 0.08)),
                    "cea": float(severe * 80 + rng.normal(0, 2)),
                    "treatment": "external-A" if patient_index % 2 else "external-B",
                }
            )
    return rows


def test_patient_temporal_split_is_70_15_15_and_patient_disjoint():
    partition = chronological_patient_split(_longitudinal_records())
    assert len(partition.development) == 14
    assert len(partition.tuning) == 3
    assert len(partition.locked_test) == 3
    assert len(set(partition.all_patient_ids)) == 20
    assert partition.development[-1] == "patient-13"
    assert partition.locked_test[-1] == "patient-19"


def test_state_discovery_is_deterministic_interpretable_and_continuous_time():
    rows = _longitudinal_records()
    manifest = _analysis_manifest()
    first = ContinuousTimeStateDiscovery(
        n_states=2, random_seed=7, bootstrap_replicates=5
    ).fit(
        rows,
        feature_names=["ecog", "cea"],
        treatment_columns=["treatment"],
        column_manifest=manifest,
        manifest_sha256=manifest.sha256,
    )
    second = ContinuousTimeStateDiscovery(
        n_states=2, random_seed=7, bootstrap_replicates=5
    ).fit(
        rows,
        feature_names=["ecog", "cea"],
        treatment_columns=["treatment"],
        column_manifest=manifest,
        manifest_sha256=manifest.sha256,
    )
    assert first.labels == second.labels
    assert first.posterior_probabilities == second.posterior_probabilities
    assert first.bootstrap_ari > 0.9
    assert first.analysis_manifest_sha256 == manifest.sha256
    assert first.model.treatment_columns == ("treatment",)
    assert len(first.model.candidate_states) == 2
    assert all(state.research_only for state in first.model.candidate_states)
    assert all(not state.eligible_for_guideline for state in first.model.candidate_states)
    assert sum(state.observation_count for state in first.model.candidate_states) == len(rows)
    assert all(
        state.patient_count == len({str(row["patient_id"]) for row in rows})
        for state in first.model.candidate_states
    )
    assert all(
        state.patient_count < state.observation_count
        for state in first.model.candidate_states
    )
    assert first.model.transition_rates.shape == (2, 2)
    assert np.all(np.isfinite(first.model.transition_rates))
    assert all(
        transition.treatment_conditioned is False
        for transition in first.model.state_transitions
    )
    continuous = first.model.continuous_state(rows[:4])
    assert continuous.shape == (4, 2)
    assert np.allclose(continuous.sum(axis=1), 1.0)


def test_bootstrap_resamples_whole_patient_clusters():
    patient_ids = ["patient-a", "patient-a", "patient-b", "patient-b", "patient-b", "patient-c"]
    indices = states_module._patient_cluster_bootstrap_indices(
        patient_ids, np.random.default_rng(23)
    )

    patient_rows = {
        "patient-a": [0, 1],
        "patient-b": [2, 3, 4],
        "patient-c": [5],
    }
    cluster_multiplicities: list[int] = []
    for rows in patient_rows.values():
        row_multiplicities = [int(np.sum(indices == row_index)) for row_index in rows]
        assert len(set(row_multiplicities)) == 1
        cluster_multiplicities.append(row_multiplicities[0])
    assert sum(cluster_multiplicities) == len(patient_rows)


def test_treatment_and_future_information_can_never_define_state():
    rows = _longitudinal_records()
    manifest = _analysis_manifest()
    with pytest.raises(StateDiscoveryError, match="state_feature"):
        ContinuousTimeStateDiscovery(n_states=2, bootstrap_replicates=3).fit(
            rows,
            feature_names=["ecog", "treatment"],
            treatment_columns=["treatment"],
            column_manifest=manifest,
            manifest_sha256=manifest.sha256,
        )
    disguised = [dict(row, FOLFOX=float(index % 2)) for index, row in enumerate(rows)]
    with pytest.raises(StateDiscoveryError, match="absent from the locked manifest"):
        ContinuousTimeStateDiscovery(n_states=2, bootstrap_replicates=3).fit(
            disguised,
            feature_names=["ecog", "FOLFOX"],
            treatment_columns=["treatment"],
            column_manifest=manifest,
            manifest_sha256=manifest.sha256,
        )
    leaked = [dict(rows[0])]
    leaked[0]["feature_available_time"] = leaked[0]["event_time"] + timedelta(days=1)
    with pytest.raises(StateDiscoveryError, match="future-information leakage"):
        assert_no_future_information(leaked)
    missing_availability = [dict(rows[0])]
    missing_availability[0].pop("feature_available_time")
    with pytest.raises(StateDiscoveryError, match="no feature availability time"):
        assert_no_future_information(missing_availability)


def test_manifest_blocks_opaque_x7_and_nonexistent_fake_treatment_bypass():
    rows = [dict(row, x7=float(index % 2)) for index, row in enumerate(_longitudinal_records())]
    bypass_manifest = _analysis_manifest(
        extra_columns=[{"name": "x7", "role": ColumnRole.STATE_FEATURE}],
        treatment_name="fake_treatment",
    )
    with pytest.raises(StateDiscoveryError, match="locked manifest"):
        ContinuousTimeStateDiscovery(n_states=2, bootstrap_replicates=3).fit(
            rows,
            feature_names=["ecog", "cea", "x7"],
            treatment_columns=["fake_treatment"],
            column_manifest=bypass_manifest,
            manifest_sha256=bypass_manifest.sha256,
        )

    reviewed_manifest = _analysis_manifest(
        extra_columns=[{"name": "x7", "role": ColumnRole.TREATMENT_ACTION}]
    )
    with pytest.raises(StateDiscoveryError, match="state_feature"):
        ContinuousTimeStateDiscovery(n_states=2, bootstrap_replicates=3).fit(
            rows,
            feature_names=["ecog", "cea", "x7"],
            treatment_columns=["x7", "treatment"],
            column_manifest=reviewed_manifest,
            manifest_sha256=reviewed_manifest.sha256,
        )


def test_manifest_requires_complete_roles_and_recomputes_canonical_sha():
    manifest = _analysis_manifest()
    assert manifest.sha256 == manifest.sha256.lower()
    with pytest.raises(StateDiscoveryError, match="does not match canonical"):
        ContinuousTimeStateDiscovery(n_states=2, bootstrap_replicates=3).fit(
            _longitudinal_records(),
            feature_names=["ecog", "cea"],
            treatment_columns=["treatment"],
            column_manifest=manifest,
            manifest_sha256="a" * 64,
        )

    unclassified = [dict(_longitudinal_records()[0], surprise_column=1.0)] * 10
    with pytest.raises(StateDiscoveryError, match="absent from the locked manifest"):
        ContinuousTimeStateDiscovery(n_states=2, bootstrap_replicates=3).fit(
            unclassified,
            feature_names=["ecog", "cea"],
            treatment_columns=["treatment"],
            column_manifest=manifest,
            manifest_sha256=manifest.sha256,
        )


def test_cli_recomputes_manifest_and_rejects_arbitrary_64_character_hash(
    tmp_path, monkeypatch, capsys
):
    raw = tmp_path / "raw"
    derived = tmp_path / "derived"
    git_root = tmp_path / "repo"
    raw.mkdir()
    derived.mkdir()
    git_root.mkdir()
    monkeypatch.setenv("AI_CLINICIAN_RAW_ROOT", str(raw))
    monkeypatch.setenv("AI_CLINICIAN_DERIVED_ROOT", str(derived))
    monkeypatch.setenv("AI_CLINICIAN_GIT_ROOT", str(git_root))
    manifest = _analysis_manifest()
    (derived / "states.json").write_text(
        json.dumps(
            {
                "analysis_column_manifest": manifest.canonical_payload(),
                "treatment_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert cli_main(
        ["states", "fit", "--input", "states.json", "--output", "fit.json"]
    ) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert not (derived / "fit.json").exists()


def test_state_promotion_gate_requires_every_pre_specified_threshold():
    passing = evaluate_state_validation_gate(
        bootstrap_ari=0.80,
        calibration_slope=1.0,
        model_brier=0.18,
        baseline_brier=0.20,
        expert_confirmations=2,
    )
    assert passing.passed is True
    assert passing.relative_brier_improvement == pytest.approx(0.10)

    failing = evaluate_state_validation_gate(
        bootstrap_ari=0.74,
        calibration_slope=1.3,
        model_brier=0.195,
        baseline_brier=0.20,
        expert_confirmations=1,
    )
    assert failing.passed is False
    assert set(failing.reasons) == {
        "STATE_STABILITY_BELOW_THRESHOLD",
        "CALIBRATION_SLOPE_OUT_OF_RANGE",
        "BRIER_IMPROVEMENT_BELOW_THRESHOLD",
        "INSUFFICIENT_EXPERT_INTERPRETABILITY_CONFIRMATION",
    }
