"""Configuration for dataset identity, feature contracts, and split sizing."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from fedcrg.domain.constants import DIAD_EXPECTED_SOURCE_CLIENTS
from fedcrg.domain.enums import DatasetFeatureContractId, DatasetId


class SplitConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    train_benign: int = Field(gt=0)
    reference_benign: int = Field(gt=0)
    mismatch_benign: int = Field(gt=0)
    calibration_benign: int = Field(gt=0)
    benign_guard: int = Field(ge=0)
    min_benign_test: int = Field(gt=0)
    attack_dev: int = Field(gt=0)
    min_attack_test: int = Field(ge=0)
    min_attack_test_per_group: int = Field(gt=0)

    @property
    def reservoir_size(self) -> int:
        return (
            self.reference_benign
            + self.mismatch_benign
            + self.calibration_benign
            + self.benign_guard
        )

    @property
    def minimum_benign_rows(self) -> int:
        return self.train_benign + self.reservoir_size + self.min_benign_test


class DatasetConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    id: DatasetId
    feature_contract: DatasetFeatureContractId
    source_version: str
    parser_version: str = "1"
    feature_count: int = Field(gt=0)
    feature_names: tuple[str, ...] = ()
    expected_clients: int | None = Field(default=None, gt=0)
    expected_source_clients: int | None = Field(default=None, gt=0)
    minimum_clients: int = Field(default=1, gt=0)
    minimum_benign_rows: int | None = Field(default=None, gt=0)
    minimum_malicious_rows: int | None = Field(default=None, gt=0)
    split: SplitConfig
    calibration_seeds: tuple[int, ...]
    primary_calibration_seed: int
    expected_benign_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> DatasetConfig:
        if not self.calibration_seeds:
            raise ValueError("At least one calibration seed is required")
        if len(set(self.calibration_seeds)) != len(self.calibration_seeds):
            raise ValueError("Calibration seeds must be unique")
        if self.primary_calibration_seed not in self.calibration_seeds:
            raise ValueError("Primary calibration seed must be in calibration_seeds")
        if self.feature_names and len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique")

        if self.id is DatasetId.NBAIOT:
            if self.feature_contract is not DatasetFeatureContractId.NBAIOT_LOCKED_115:
                raise ValueError("N-BaIoT must use the locked 115-feature contract")
            if self.feature_count != 115 or self.expected_clients != 9:
                raise ValueError("N-BaIoT contract requires 115 features and nine clients")
            if len(self.expected_benign_counts) != 9:
                raise ValueError("N-BaIoT requires the nine-client benign-count cross-check ledger")

        elif self.id is DatasetId.DIAD:
            if self.feature_contract not in {
                DatasetFeatureContractId.DIAD_LOCKED_86,
                DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            }:
                raise ValueError("DIAD uses either the locked or training-derived feature contract")
            if self.expected_source_clients != DIAD_EXPECTED_SOURCE_CLIENTS:
                raise ValueError(
                    f"DIAD contract requires {DIAD_EXPECTED_SOURCE_CLIENTS} source identities"
                )
            if self.minimum_benign_rows != 7800 or self.minimum_malicious_rows != 1000:
                raise ValueError("DIAD eligibility counts must remain 7800 benign / 1000 malicious")
            if self.minimum_clients != 10:
                raise ValueError("DIAD external validation requires at least ten eligible clients")
            if self.feature_contract is DatasetFeatureContractId.DIAD_LOCKED_86:
                if self.feature_count != 86:
                    raise ValueError("Confirmatory DIAD requires exactly 86 model features")
                if self.feature_names:
                    raise ValueError("Locked DIAD feature names are owned by the dataset adapter")
            else:
                if not self.feature_names or len(self.feature_names) != self.feature_count:
                    raise ValueError(
                        "Training-derived DIAD requires a frozen feature-name list matching feature_count"
                    )

        elif self.id is DatasetId.SYNTHETIC:
            if self.feature_contract is not DatasetFeatureContractId.SYNTHETIC:
                raise ValueError("Synthetic experiments must use the synthetic feature contract")
        return self
