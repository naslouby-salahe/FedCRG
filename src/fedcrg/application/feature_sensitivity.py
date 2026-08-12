"""Build and freeze the R14 DIAD training-schema-only feature contract."""

from __future__ import annotations

import json
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.ids import ClientId
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
        raw = json.loads(eligibility_manifest.read_text(encoding="utf-8"))
        eligible = tuple(ClientId(str(value)) for value in raw.get("eligible_clients", []))
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
            client: adapter.load_training_schema(client, train_count)
            for client in eligible
        }
        contract = derive_numeric_safe_features(training_frames)
        atomic_write_json(
            output,
            {
                "experiment": "R14",
                "source": "training_schema_only",
                "eligible_clients": [client.value for client in eligible],
                **contract.to_dict(),
            },
        )
        return contract


def r14_config(base_config, contract: NumericSafeFeatureContract):
    """Create the R14-only derived config after the training-schema feature freeze."""
    from fedcrg.core.enums import ExperimentId, PolicyId

    dataset = base_config.dataset.model_copy(
        update={
            "feature_count": contract.dimension,
            "calibration_seeds": (base_config.dataset.primary_calibration_seed,),
        }
    )
    detector = base_config.detector.model_copy(
        update={"hidden_dims": contract.encoder_hidden_dims}
    )
    return base_config.model_copy(
        update={
            "id": ExperimentId.DIAD_FEATURE_SENSITIVITY,
            "dataset": dataset,
            "detector": detector,
            "policies": (
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.LOCAL_QUANTILE,
                PolicyId.SHRINKAGE,
                PolicyId.FEDCRG,
            ),
        }
    )


class PrepareDiadFeatureSensitivity:
    """Derive/freeze R14 features and prepare the named DIAD role assignment."""

    def prepare(
        self,
        base_config,
        data_root: Path,
        eligibility_manifest: Path,
        feature_manifest: Path,
    ) -> tuple[object, Path]:
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
