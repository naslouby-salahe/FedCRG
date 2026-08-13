"""Build and freeze the R14 DIAD training-schema-only feature contract."""

from __future__ import annotations

from pathlib import Path

from fedcrg.artifacts.dataset import EligibilityManifestStore
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DatasetFeatureContractId, ExperimentId, PolicyId
from fedcrg.data.datasets.diad import DiadAdapter
from fedcrg.data.feature_sensitivity import NumericSafeFeatureContract, derive_numeric_safe_features


class BuildDiadFeatureSensitivityContract:
    """Derive R14 features strictly from frozen eligible clients' training rows."""

    def build(
        self,
        data_root: Path,
        eligibility_manifest: Path,
        output: Path,
        train_count: int = 2000,
    ) -> NumericSafeFeatureContract:
        eligible = EligibilityManifestStore().load(eligibility_manifest).eligible_clients
        if not eligible:
            raise ValueError("R14 requires a frozen non-empty DIAD eligibility list")
        adapter = DiadAdapter(data_root)
        discovered = set(adapter.discover_clients())
        missing = tuple(client for client in eligible if client not in discovered)
        if missing:
            raise ValueError(
                "Frozen eligible clients are absent from the acquired DIAD source: "
                + ", ".join(client.value for client in missing)
            )
        training_frames = {
            client: adapter.load_training_schema(client, train_count) for client in eligible
        }
        contract = derive_numeric_safe_features(training_frames)
        atomic_write_json(
            output,
            {
                "experiment": "R14",
                "source": "training_schema_only",
                "eligible_clients": [client.value for client in eligible],
                "feature_contract": contract,
            },
        )
        return contract


def r14_config(
    base_config: ExperimentConfig, contract: NumericSafeFeatureContract
) -> ExperimentConfig:
    """Create the R14-only derived config after the training-schema feature freeze.

    Rebuilds the full validated payload and revalidates through the model instead of
    an unchecked partial-field update, so the DIAD-training-schema cross-field
    invariants (feature_contract/feature_names/feature_count) are enforced.
    """
    payload = base_config.model_dump(mode="python")
    payload["id"] = ExperimentId.DIAD_FEATURE_SENSITIVITY
    dataset = dict(payload["dataset"])
    dataset["feature_contract"] = DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE
    dataset["feature_count"] = contract.dimension
    dataset["feature_names"] = contract.features
    dataset["calibration_seeds"] = (base_config.dataset.primary_calibration_seed,)
    payload["dataset"] = dataset
    detector = dict(payload["detector"])
    detector["hidden_dims"] = contract.encoder_hidden_dims
    payload["detector"] = detector
    payload["policies"] = (
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.SHRINKAGE,
        PolicyId.FEDCRG,
    )
    return ExperimentConfig.model_validate(payload)


class PrepareDiadFeatureSensitivity:
    """Derive/freeze R14 features and prepare the named DIAD role assignment."""

    def prepare(
        self,
        base_config: ExperimentConfig,
        data_root: Path,
        eligibility_manifest: Path,
        feature_manifest: Path,
    ) -> tuple[ExperimentConfig, Path]:
        from fedcrg.application.prepare_data import PrepareData
        from fedcrg.data.datasets.diad import DiadFeatureSensitivityAdapter

        contract = BuildDiadFeatureSensitivityContract().build(
            data_root,
            eligibility_manifest,
            feature_manifest,
            train_count=base_config.dataset.split.train_benign,
        )
        config = r14_config(base_config, contract)
        adapter = DiadFeatureSensitivityAdapter(data_root, contract.features)
        prepared = PrepareData().prepare(
            config,
            data_root,
            adapter_override=adapter,
        )
        return config, prepared
