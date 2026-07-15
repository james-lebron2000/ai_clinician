from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta

from .schemas import (
    DEVELOPMENT_SPLITS,
    LOCKED_SPLITS,
    AccessTier,
    AlignmentEdge,
    AllowedUse,
    ConsentDirective,
    ConsentMode,
    DatasetManifest,
    DerivedArtifactPolicy,
    Modality,
    PairingLevel,
    RegistryAuditReport,
    RegistryFinding,
    SampleIndexRecord,
    SpecimenLineageEdge,
    SplitName,
)


class HumanDataRegistry:
    """Governed index for cross-scale data; raw data remain in approved storage."""

    def __init__(self) -> None:
        self._manifests: dict[str, DatasetManifest] = {}
        self._assets: dict[str, SampleIndexRecord] = {}
        self._consents: dict[tuple[str, str], list[ConsentDirective]] = defaultdict(list)
        self._lineage_edges: dict[str, SpecimenLineageEdge] = {}
        self._alignment_edges: dict[str, AlignmentEdge] = {}

    @property
    def manifests(self) -> list[DatasetManifest]:
        return list(self._manifests.values())

    @property
    def assets(self) -> list[SampleIndexRecord]:
        return list(self._assets.values())

    def register_dataset(self, manifest: DatasetManifest) -> None:
        if manifest.id in self._manifests:
            raise ValueError(f"dataset already registered: {manifest.id}")
        self._manifests[manifest.id] = manifest

    def register_asset(self, asset: SampleIndexRecord) -> None:
        if asset.id in self._assets:
            raise ValueError(f"asset already registered: {asset.id}")
        manifest = self._manifests.get(asset.dataset_id)
        if manifest is None:
            raise ValueError(f"unknown dataset: {asset.dataset_id}")
        if asset.modality not in manifest.modalities:
            raise ValueError("asset modality is not declared by its dataset")
        if manifest.modality_scales[asset.modality] is not asset.biological_scale:
            raise ValueError("asset biological scale differs from the governed modality route")
        if asset.site_id not in manifest.site_ids:
            raise ValueError("asset originates from an undeclared site")
        self._assets[asset.id] = asset

    def register_consent(self, directive: ConsentDirective) -> None:
        key = (directive.subject_key_namespace, directive.subject_key)
        if any(item.id == directive.id for item in self._consents[key]):
            raise ValueError(f"consent directive already registered: {directive.id}")
        self._consents[key].append(directive)

    def register_lineage_edge(self, edge: SpecimenLineageEdge) -> None:
        if edge.id in self._lineage_edges:
            raise ValueError(f"lineage edge already registered: {edge.id}")
        if edge.parent_id == edge.child_id:
            raise ValueError("lineage edge cannot point to itself")
        self._lineage_edges[edge.id] = edge

    def register_alignment_edge(self, edge: AlignmentEdge) -> None:
        if edge.id in self._alignment_edges:
            raise ValueError(f"alignment edge already registered: {edge.id}")
        left = self._assets.get(edge.left_asset_id)
        right = self._assets.get(edge.right_asset_id)
        if left is None or right is None:
            raise ValueError("alignment edge references an unknown asset")
        if left.dataset_id != edge.dataset_id or right.dataset_id != edge.dataset_id:
            raise ValueError("alignment edge dataset does not own both assets")
        if left.split is not right.split:
            raise ValueError("alignment cannot cross dataset splits")
        if left.modality is right.modality:
            raise ValueError("cross-modal alignment requires different modalities")
        if not self._is_governed_pair(
            edge.pairing_level,
            left,
            right,
            timedelta(seconds=edge.time_delta_seconds or 0),
        ):
            raise ValueError(
                "alignment evidence conflicts with subject/specimen/time identifiers"
            )
        if edge.pairing_level is PairingLevel.SAME_SUBJECT_TIME_WINDOW:
            assert left.acquisition_time is not None
            assert right.acquisition_time is not None
            observed_delta = abs(left.acquisition_time - right.acquisition_time).total_seconds()
            if abs(observed_delta - (edge.time_delta_seconds or 0)) > 1:
                raise ValueError("alignment time delta differs from asset timestamps")
        self._alignment_edges[edge.id] = edge

    def audit(
        self,
        *,
        compute_region: str,
        commercial_project: bool = False,
        model_export_planned: bool = False,
    ) -> RegistryAuditReport:
        findings: list[RegistryFinding] = []
        for manifest in self._manifests.values():
            if manifest.contains_direct_identifiers:
                findings.append(
                    RegistryFinding(
                        code="DIRECT_IDENTIFIERS",
                        severity="critical",
                        message=f"dataset {manifest.id} contains direct identifiers",
                    )
                )
            if compute_region not in manifest.permitted_compute_regions:
                findings.append(
                    RegistryFinding(
                        code="COMPUTE_REGION",
                        severity="critical",
                        message=(
                            f"dataset {manifest.id} cannot be processed in {compute_region}"
                        ),
                    )
                )
            if commercial_project and AllowedUse.COMMERCIAL_RND not in manifest.allowed_uses:
                findings.append(
                    RegistryFinding(
                        code="COMMERCIAL_USE",
                        severity="critical",
                        message=f"dataset {manifest.id} does not permit commercial R&D",
                    )
                )
            if (
                model_export_planned
                and manifest.derived_model_policy
                is not DerivedArtifactPolicy.EXPORT_ALLOWED
            ):
                findings.append(
                    RegistryFinding(
                        code="DERIVED_MODEL_EXPORT",
                        severity="critical",
                        message=(
                            f"dataset {manifest.id} does not pre-authorize exporting derived weights"
                        ),
                    )
                )

        split_by_group: dict[tuple[str, str], set[SplitName]] = defaultdict(set)
        assets_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
        split_by_subject: dict[tuple[str, str], set[SplitName]] = defaultdict(set)
        assets_by_subject: dict[tuple[str, str], list[str]] = defaultdict(list)
        sites_by_external_status: dict[str, set[str]] = defaultdict(set)

        for asset in self._assets.values():
            manifest = self._manifests[asset.dataset_id]
            group_key = (asset.dataset_id, asset.group_key)
            split_by_group[group_key].add(asset.split)
            assets_by_group[group_key].append(asset.id)
            if asset.subject_key:
                subject_key = (manifest.subject_key_namespace, asset.subject_key)
                split_by_subject[subject_key].add(asset.split)
                assets_by_subject[subject_key].append(asset.id)
            elif asset.split in LOCKED_SPLITS:
                findings.append(
                    RegistryFinding(
                        code="LOCKED_SUBJECT_UNKNOWN",
                        severity="critical",
                        message=f"locked asset {asset.id} lacks a subject key",
                        asset_ids=[asset.id],
                    )
                )
            if asset.direct_identifier_present:
                findings.append(
                    RegistryFinding(
                        code="ASSET_DIRECT_IDENTIFIER",
                        severity="critical",
                        message=f"asset {asset.id} contains a direct identifier",
                        asset_ids=[asset.id],
                    )
                )
            if asset.quality_status != "usable" and asset.split in DEVELOPMENT_SPLITS:
                findings.append(
                    RegistryFinding(
                        code="QUARANTINED_TRAINING_ASSET",
                        severity="critical",
                        message=f"non-usable asset {asset.id} is assigned to development",
                        asset_ids=[asset.id],
                    )
                )
            required_use = (
                AllowedUse.AI_TRAINING
                if asset.split in DEVELOPMENT_SPLITS
                else AllowedUse.AI_VALIDATION
            )
            if required_use not in manifest.allowed_uses:
                findings.append(
                    RegistryFinding(
                        code="USE_NOT_PERMITTED",
                        severity="critical",
                        message=(
                            f"dataset {manifest.id} does not permit {required_use.value} "
                            f"for asset {asset.id}"
                        ),
                        asset_ids=[asset.id],
                    )
                )
            if manifest.consent_mode is ConsentMode.INDIVIDUAL_DIRECTIVE:
                consent_key = (manifest.subject_key_namespace, asset.subject_key or "")
                directives = self._consents.get(consent_key, [])
                if not any(
                    directive.permits(required_use, asset.modality, compute_region)
                    for directive in directives
                ):
                    findings.append(
                        RegistryFinding(
                            code="INDIVIDUAL_CONSENT",
                            severity="critical",
                            message=f"asset {asset.id} lacks an active individual-use directive",
                            asset_ids=[asset.id],
                        )
                    )
            if (
                asset.biological_scale.value in {"cell", "tissue", "organ"}
                and not asset.lineage_node_id
            ):
                findings.append(
                    RegistryFinding(
                        code="MISSING_SPECIMEN_LINEAGE",
                        severity="warning",
                        message=f"asset {asset.id} has no specimen-lineage node",
                        asset_ids=[asset.id],
                    )
                )
            status = "external" if asset.split is SplitName.EXTERNAL_SITE_TEST else "development"
            sites_by_external_status[asset.site_id].add(status)

        for group, splits in split_by_group.items():
            if len(splits) > 1:
                findings.append(
                    RegistryFinding(
                        code="GROUP_SPLIT_LEAKAGE",
                        severity="critical",
                        message=f"independent group {group} spans splits: {sorted(s.value for s in splits)}",
                        asset_ids=assets_by_group[group],
                    )
                )
        for subject, splits in split_by_subject.items():
            if len(splits) > 1:
                findings.append(
                    RegistryFinding(
                        code="SUBJECT_SPLIT_LEAKAGE",
                        severity="critical",
                        message=f"subject {subject} spans splits: {sorted(s.value for s in splits)}",
                        asset_ids=assets_by_subject[subject],
                    )
                )
        for site, statuses in sites_by_external_status.items():
            if statuses == {"external", "development"}:
                findings.append(
                    RegistryFinding(
                        code="EXTERNAL_SITE_LEAKAGE",
                        severity="critical",
                        message=f"external validation site {site} also contributes development data",
                    )
                )

        if self._has_lineage_cycle():
            findings.append(
                RegistryFinding(
                    code="LINEAGE_CYCLE",
                    severity="critical",
                    message="specimen lineage contains a cycle",
                )
            )

        modality_counts = Counter(asset.modality.value for asset in self._assets.values())
        split_counts = Counter(asset.split.value for asset in self._assets.values())
        subject_keys = {
            (self._manifests[asset.dataset_id].subject_key_namespace, asset.subject_key)
            for asset in self._assets.values()
            if asset.subject_key
        }
        return RegistryAuditReport(
            passed=not any(item.severity == "critical" for item in findings),
            findings=findings,
            dataset_count=len(self._manifests),
            asset_count=len(self._assets),
            subject_count=len(subject_keys),
            modality_counts=dict(modality_counts),
            split_counts=dict(split_counts),
        )

    def _has_lineage_cycle(self) -> bool:
        graph: dict[str, list[str]] = defaultdict(list)
        for edge in self._lineage_edges.values():
            graph[edge.parent_id].append(edge.child_id)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(child) for child in graph.get(node, [])):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in list(graph))

    def alignment_pairs(
        self,
        left_modality: Modality,
        right_modality: Modality,
        *,
        maximum_time_delta: timedelta = timedelta(days=1),
        strong_contrastive_only: bool = True,
        splits: set[SplitName] | None = None,
    ) -> list[tuple[SampleIndexRecord, SampleIndexRecord]]:
        governed_splits = DEVELOPMENT_SPLITS if splits is None else splits
        pairs: list[tuple[SampleIndexRecord, SampleIndexRecord]] = []
        explicit = [
            edge
            for edge in self._alignment_edges.values()
            if {
                self._assets[edge.left_asset_id].modality,
                self._assets[edge.right_asset_id].modality,
            }
            == {left_modality, right_modality}
            and (not strong_contrastive_only or edge.eligible_for_strong_contrastive())
            and self._assets[edge.left_asset_id].split in governed_splits
        ]
        if explicit:
            oriented: list[tuple[SampleIndexRecord, SampleIndexRecord]] = []
            for edge in explicit:
                left = self._assets[edge.left_asset_id]
                right = self._assets[edge.right_asset_id]
                oriented.append((left, right) if left.modality is left_modality else (right, left))
            return oriented
        by_dataset: dict[str, list[SampleIndexRecord]] = defaultdict(list)
        for asset in self._assets.values():
            if asset.quality_status == "usable":
                by_dataset[asset.dataset_id].append(asset)
        for dataset_id, assets in by_dataset.items():
            manifest = self._manifests[dataset_id]
            if manifest.pairing_level is PairingLevel.COHORT_UNPAIRED:
                continue
            if strong_contrastive_only and manifest.pairing_level not in {
                PairingLevel.SAME_MEASUREMENT,
                PairingLevel.SAME_SPECIMEN,
            }:
                continue
            left = [asset for asset in assets if asset.modality is left_modality]
            right = [asset for asset in assets if asset.modality is right_modality]
            for left_asset in left:
                for right_asset in right:
                    if left_asset.split not in governed_splits:
                        continue
                    if left_asset.split is not right_asset.split:
                        continue
                    if self._is_governed_pair(
                        manifest.pairing_level,
                        left_asset,
                        right_asset,
                        maximum_time_delta,
                    ):
                        pairs.append((left_asset, right_asset))
        return pairs

    @staticmethod
    def _is_governed_pair(
        pairing_level: PairingLevel,
        left: SampleIndexRecord,
        right: SampleIndexRecord,
        maximum_time_delta: timedelta,
    ) -> bool:
        if pairing_level is PairingLevel.SAME_MEASUREMENT:
            return bool(left.pairing_key and left.pairing_key == right.pairing_key)
        if pairing_level is PairingLevel.SAME_SPECIMEN:
            return bool(left.specimen_key and left.specimen_key == right.specimen_key)
        if pairing_level is PairingLevel.SAME_SUBJECT_TIME_WINDOW:
            return bool(
                left.subject_key
                and left.subject_key == right.subject_key
                and left.acquisition_time
                and right.acquisition_time
                and abs(left.acquisition_time - right.acquisition_time) <= maximum_time_delta
            )
        if pairing_level is PairingLevel.SAME_SUBJECT:
            return bool(left.subject_key and left.subject_key == right.subject_key)
        return False
