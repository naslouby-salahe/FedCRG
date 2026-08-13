"""Configuration for dataset identity, feature contracts, and split sizing."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

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
    parser_version: str
    feature_count: int = Field(gt=0)
    feature_names: tuple[str, ...] = ()
    expected_clients: int | None = Field(default=None, gt=0)
    expected_source_clients: int | None = Field(default=None, gt=0)
    minimum_clients: int = Field(gt=0)
    minimum_benign_rows: int | None = Field(default=None, gt=0)
    minimum_malicious_rows: int | None = Field(default=None, gt=0)
    split: SplitConfig
    calibration_seeds: tuple[int, ...]
    primary_calibration_seed: int
    expected_benign_counts: dict[str, int]

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
            if self.expected_clients is None or self.expected_clients < 1:
                raise ValueError("N-BaIoT must declare its nine-client contract")
            if len(self.expected_benign_counts) != self.expected_clients:
                raise ValueError("N-BaIoT expected-benign-count ledger must match the client count")

        elif self.id is DatasetId.DIAD:
            if self.feature_contract not in {
                DatasetFeatureContractId.DIAD_LOCKED_86,
                DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            }:
                raise ValueError("DIAD uses either the locked or training-derived feature contract")
            if self.expected_source_clients is None or self.expected_source_clients < 1:
                raise ValueError("DIAD must declare its source identity count")
            if self.minimum_benign_rows is None or self.minimum_malicious_rows is None:
                raise ValueError("DIAD eligibility counts must be declared")
            if self.feature_contract is DatasetFeatureContractId.DIAD_LOCKED_86:
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
