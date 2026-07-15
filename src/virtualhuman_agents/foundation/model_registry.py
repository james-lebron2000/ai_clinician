from __future__ import annotations

from .schemas import (
    DatasetSnapshot,
    FoundationReadinessReport,
    InferenceTrace,
    ModelRelease,
)


class FoundationModelRegistry:
    """Small reference registry; production deployments should use a validated model registry."""

    def __init__(self) -> None:
        self._snapshots: dict[str, DatasetSnapshot] = {}
        self._releases: dict[str, ModelRelease] = {}
        self._traces: dict[str, InferenceTrace] = {}

    def register_snapshot(self, snapshot: DatasetSnapshot) -> None:
        if snapshot.id in self._snapshots:
            raise ValueError(f"snapshot already registered: {snapshot.id}")
        self._snapshots[snapshot.id] = snapshot

    def register_release(self, release: ModelRelease) -> None:
        if release.id in self._releases:
            raise ValueError(f"model release already registered: {release.id}")
        missing = set(release.dataset_snapshot_ids) - set(self._snapshots)
        if missing:
            raise ValueError(f"model release references unknown data snapshots: {sorted(missing)}")
        self._releases[release.id] = release

    def promote(
        self,
        release_id: str,
        readiness: FoundationReadinessReport,
    ) -> ModelRelease:
        release = self._releases[release_id]
        if readiness.model_name != release.model_name:
            raise ValueError("readiness report model name does not match the release")
        if readiness.model_version != release.model_version:
            raise ValueError("readiness report version does not match the release")
        if readiness.context_of_use != release.context_of_use:
            raise ValueError("readiness report context of use does not match the release")
        if not readiness.passed:
            raise ValueError("a model with failed readiness gates cannot be promoted")
        promoted = release.model_copy(update={"status": "released"})
        self._releases[release_id] = promoted
        return promoted

    def resolve_for_agent(self, model_name: str, context_of_use: str) -> ModelRelease:
        candidates = [
            release
            for release in self._releases.values()
            if release.model_name == model_name
            and release.context_of_use == context_of_use
            and release.status == "released"
        ]
        if not candidates:
            raise KeyError("no released model exactly matches the requested context of use")
        return max(candidates, key=lambda item: item.created_at)

    def record_trace(self, trace: InferenceTrace) -> None:
        if trace.model_release_id not in self._releases:
            raise ValueError("inference trace references an unknown model release")
        if self._releases[trace.model_release_id].status != "released":
            raise ValueError("agents cannot invoke an unreleased model")
        if trace.id in self._traces:
            raise ValueError("inference trace already exists")
        self._traces[trace.id] = trace
