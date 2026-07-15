from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .models import ExperimentalCondition, FactorKind, ResearchGoal


@dataclass(frozen=True)
class DesignResult:
    conditions: list[ExperimentalCondition]
    expected_information_gain: float


class ConditionSpace:
    def __init__(self, goal: ResearchGoal, seed: int = 17) -> None:
        self.goal = goal
        self.rng = np.random.default_rng(seed)
        self.factor_names = sorted(goal.factor_space)

    def _candidate_factors(self, count: int) -> list[dict[str, str | float | int | bool]]:
        candidates: list[dict[str, str | float | int | bool]] = []
        continuous_names = [
            name
            for name in self.factor_names
            if self.goal.factor_space[name].kind in {FactorKind.CONTINUOUS, FactorKind.INTEGER}
        ]
        strata = {
            name: self.rng.permutation(count) for name in continuous_names
        }
        for index in range(count):
            factors: dict[str, str | float | int | bool] = {}
            for name in self.factor_names:
                spec = self.goal.factor_space[name]
                if spec.kind in {FactorKind.CONTINUOUS, FactorKind.INTEGER}:
                    fraction = (strata[name][index] + 0.5) / count
                    value = float(spec.minimum) + fraction * (
                        float(spec.maximum) - float(spec.minimum)
                    )
                    factors[name] = int(round(value)) if spec.kind is FactorKind.INTEGER else round(value, 8)
                else:
                    factors[name] = spec.choices[index % len(spec.choices)]
            candidates.append(factors)

        # Add interpretable boundary and midpoint conditions before de-duplication.
        level_sets: list[list[str | float | int | bool]] = []
        for name in self.factor_names:
            spec = self.goal.factor_space[name]
            if spec.kind in {FactorKind.CONTINUOUS, FactorKind.INTEGER}:
                midpoint = (float(spec.minimum) + float(spec.maximum)) / 2
                values: list[str | float | int | bool] = [spec.minimum, midpoint, spec.maximum]
                if spec.kind is FactorKind.INTEGER:
                    values = [int(round(float(value))) for value in values]
                level_sets.append(values)
            else:
                level_sets.append(list(spec.choices))
        maximum_boundaries = min(64, math.prod(len(levels) for levels in level_sets))
        for combination in itertools.islice(itertools.product(*level_sets), maximum_boundaries):
            candidates.append(dict(zip(self.factor_names, combination, strict=True)))

        unique: dict[str, dict[str, str | float | int | bool]] = {}
        for factors in candidates:
            condition = ExperimentalCondition(factors=factors)
            unique[condition.fingerprint()] = factors
        return list(unique.values())

    def encode(self, factors: dict[str, str | float | int | bool]) -> np.ndarray:
        encoded: list[float] = []
        for name in self.factor_names:
            spec = self.goal.factor_space[name]
            value = factors[name]
            if spec.kind in {FactorKind.CONTINUOUS, FactorKind.INTEGER}:
                span = float(spec.maximum) - float(spec.minimum)
                encoded.append((float(value) - float(spec.minimum)) / span)
            else:
                encoded.extend(float(value == choice) for choice in spec.choices)
        return np.asarray(encoded, dtype=float)

    def _cost_per_condition(self) -> float:
        total_slots = self.goal.budget.initial_condition_limit + (
            self.goal.budget.maximum_rounds - 1
        ) * self.goal.budget.subsequent_condition_limit
        return self.goal.budget.maximum_cost / max(total_slots, 1)

    def initial_design(self, count: int) -> DesignResult:
        candidates = self._candidate_factors(max(256, count * 12))
        matrix = np.asarray([self.encode(candidate) for candidate in candidates])
        matrix = np.column_stack([np.ones(len(matrix)), matrix])
        selected: list[int] = []
        information = np.eye(matrix.shape[1]) * 1e-6
        remaining = set(range(len(candidates)))
        for _ in range(min(count, len(candidates))):
            best_index = max(
                remaining,
                key=lambda index: np.linalg.slogdet(
                    information + np.outer(matrix[index], matrix[index])
                )[1],
            )
            selected.append(best_index)
            remaining.remove(best_index)
            information += np.outer(matrix[best_index], matrix[best_index])
        cost = self._cost_per_condition()
        conditions = [
            ExperimentalCondition(
                factors=candidates[index],
                estimated_cost=cost,
                rationale="D-optimal coverage of the approved design envelope",
            )
            for index in selected
        ]
        return DesignResult(conditions=conditions, expected_information_gain=1.0)

    def adaptive_design(
        self,
        count: int,
        history: list[tuple[ExperimentalCondition, float]],
    ) -> DesignResult:
        if len(history) < 4:
            return self.initial_design(count)
        used = {condition.fingerprint() for condition, _ in history}
        candidates = [
            factors
            for factors in self._candidate_factors(max(320, count * 16))
            if ExperimentalCondition(factors=factors).fingerprint() not in used
        ]
        if not candidates:
            return DesignResult([], 0.0)
        x_train = np.asarray([self.encode(condition.factors) for condition, _ in history])
        y_train = np.asarray([score for _, score in history], dtype=float)
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(x_train.shape[1]), nu=2.5
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=17,
            n_restarts_optimizer=1,
        )
        model.fit(x_train, y_train)
        x_candidates = np.asarray([self.encode(candidate) for candidate in candidates])
        mean, std = model.predict(x_candidates, return_std=True)
        mean_norm = (mean - mean.min()) / (np.ptp(mean) + 1e-12)
        std_norm = std / (std.max() + 1e-12)
        acquisition = 0.65 * std_norm + 0.35 * mean_norm

        selected: list[int] = []
        while len(selected) < min(count, len(candidates)):
            best_index = -1
            best_score = -np.inf
            for index, base_score in enumerate(acquisition):
                if index in selected:
                    continue
                diversity = 1.0
                if selected:
                    diversity = min(
                        float(np.linalg.norm(x_candidates[index] - x_candidates[chosen]))
                        for chosen in selected
                    )
                score = float(base_score) + 0.15 * diversity
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(best_index)

        cost = self._cost_per_condition()
        conditions = [
            ExperimentalCondition(
                factors=candidates[index],
                estimated_cost=cost,
                rationale="cost-aware uncertainty and response optimization",
            )
            for index in selected
        ]
        information_gain = float(np.mean(std_norm[selected])) if selected else 0.0
        return DesignResult(conditions=conditions, expected_information_gain=information_gain)
