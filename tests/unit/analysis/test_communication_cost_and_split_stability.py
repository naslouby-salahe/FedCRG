from __future__ import annotations

from fedcrg.config import Study
from fedcrg.experiments.analyses import (
    summarize_state_stability,
    summarize_threshold_stability,
)
from fedcrg.learning.federated import (
    model_communication,
    preprocessing_communication,
)
from fedcrg.thresholding.policies import threshold_policy_communication
from fedcrg.types import DecisionState, ExperimentId, PolicyId


def test_primary_model_communication_ledger() -> None:
    ledger = model_communication(client_count=9, rounds=30, trainable_parameters=36_626)
    assert ledger.model_payload_bytes == 146_504
    assert ledger.federation_bytes == 79_112_160


def test_preprocessing_communication_ledger() -> None:
    ledger = preprocessing_communication(client_count=9, feature_count=115)
    assert ledger.bytes_per_client == 1_840
    assert ledger.federation_upload_bytes == 16_560


def test_primary_policy_payload_ledger() -> None:
    config = Study.load().resolve(ExperimentId.PRIMARY_NBAIOT)
    rows = {item.policy: item for item in threshold_policy_communication(config, client_count=9)}
    assert rows[PolicyId.FEDCRG].upload_bytes_per_client == 4_000
    assert rows[PolicyId.GLOBAL_QUANTILE].upload_bytes_per_client == 44_000
    assert rows[PolicyId.SUPERVISED_F1].upload_bytes_per_client == 8_000


def test_stability_summaries_are_explicit() -> None:
    threshold = summarize_threshold_stability((1.0, 2.0, 3.0))
    assert threshold.count == 3
    assert threshold.iqr == 1.0
    states = summarize_state_stability(
        (
            DecisionState.REFERENCE_RETAINED,
            DecisionState.REFERENCE_RETAINED,
            DecisionState.PERSONALIZED,
            DecisionState.PERSONALIZED,
        )
    )
    assert states.transition_count == 1
    assert states.transition_frequency == 1 / 3
