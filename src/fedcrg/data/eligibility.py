"""Pre-outcome natural-client eligibility evaluation."""

from __future__ import annotations

import numpy as np

from fedcrg.config.models import DatasetConfig
from fedcrg.core.enums import (
    DatasetFeatureContractId,
    DatasetId,
    EligibilityStatus,
    FailureCode,
)
from fedcrg.data.datasets.diad import DIAD_FEATURES
from fedcrg.data.models import ClientData, EligibilityRecord


_DIAD_PRECEDENCE = (
    FailureCode.ID_INVALID,
    FailureCode.FEATURE_MISSING,
    FailureCode.FINITE_RATE_FAIL,
    FailureCode.BENIGN_COUNT_LT_7800,
    FailureCode.MALICIOUS_COUNT_LT_1000,
    FailureCode.ATTACK_DEV_CAPACITY_LT_500,
)


class ClientEligibilityEvaluator:
    """Evaluate dataset-contract rules before detector or threshold outcomes exist."""

    @staticmethod
    def model_features(config: DatasetConfig) -> tuple[str, ...]:
        if config.id is not DatasetId.DIAD:
            return ()
        if config.feature_contract is DatasetFeatureContractId.DIAD_LOCKED_86:
            return DIAD_FEATURES
        if config.feature_contract is DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE:
            if not config.feature_names:
                raise ValueError("R14 DIAD feature contract is not frozen")
            return config.feature_names
        raise ValueError(f"Unsupported DIAD feature contract: {config.feature_contract.value}")

    def evaluate(self, data: ClientData, config: DatasetConfig) -> EligibilityRecord:
        benign_count = len(data.benign)
        malicious_count = len(data.attack)
        if config.id is not DatasetId.DIAD:
            return EligibilityRecord(
                client_id=data.client_id,
                status=EligibilityStatus.ELIGIBLE,
                benign_count=benign_count,
                malicious_count=malicious_count,
                attack_development_capacity=malicious_count,
                primary_code=None,
                secondary_codes=(),
                chronology=data.chronology,
            )

        model_features = self.model_features(config)
        violations: list[FailureCode] = []
        missing = [column for column in model_features if column not in data.benign.columns]
        if missing:
            violations.append(FailureCode.FEATURE_MISSING)

        train = data.benign.iloc[: config.split.train_benign]
        if len(train) >= config.split.train_benign and not missing:
            values = train.loc[:, list(model_features)].to_numpy(dtype=np.float64)
            finite_rates = np.isfinite(values).mean(axis=0)
            if np.any(finite_rates < 0.99):
                violations.append(FailureCode.FINITE_RATE_FAIL)

        if config.minimum_benign_rows is not None and benign_count < config.minimum_benign_rows:
            violations.append(FailureCode.BENIGN_COUNT_LT_7800)
        if config.minimum_malicious_rows is not None and malicious_count < config.minimum_malicious_rows:
            violations.append(FailureCode.MALICIOUS_COUNT_LT_1000)

        capacity = self.attack_development_capacity(
            data,
            config.split.min_attack_test_per_group,
        )
        if capacity < config.split.attack_dev:
            violations.append(FailureCode.ATTACK_DEV_CAPACITY_LT_500)

        ordered = tuple(code for code in _DIAD_PRECEDENCE if code in violations)
        primary = ordered[0] if ordered else None
        return EligibilityRecord(
            client_id=data.client_id,
            status=(EligibilityStatus.EXCLUDED if primary else EligibilityStatus.ELIGIBLE),
            benign_count=benign_count,
            malicious_count=malicious_count,
            attack_development_capacity=capacity,
            primary_code=primary,
            secondary_codes=ordered[1:],
            chronology=data.chronology,
        )

    @staticmethod
    def attack_development_capacity(data: ClientData, reserve_per_group: int) -> int:
        if data.attack.empty or "attack_group" not in data.attack.columns:
            return 0
        counts = data.attack["attack_group"].astype(str).value_counts()
        return int(
            sum(
                max(0, int(count) - min(reserve_per_group, int(count)))
                for count in counts
            )
        )
