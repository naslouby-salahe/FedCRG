"""Deterministic communication accounting for training, preprocessing, and threshold policies."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.constants import SUPERVISED_THRESHOLD_CANDIDATES
from fedcrg.core.enums import PolicyId

_FLOAT32_BYTES = 4
_FLOAT64_BYTES = 8
_INT64_BYTES = 8


@dataclass(frozen=True, slots=True)
class ModelCommunicationLedger:
    client_count: int
    rounds: int
    trainable_parameters: int
    model_payload_bytes: int
    per_client_bytes: int
    federation_bytes: int


@dataclass(frozen=True, slots=True)
class PreprocessingCommunicationLedger:
    client_count: int
    feature_count: int
    extrema_per_client: int
    bytes_per_client: int
    federation_upload_bytes: int


@dataclass(frozen=True, slots=True)
class PolicyCommunicationLedger:
    policy: PolicyId
    client_count: int
    upload_bytes_per_client: int
    federation_upload_bytes: int
    note: str


def model_communication(*, client_count: int, rounds: int, trainable_parameters: int) -> ModelCommunicationLedger:
    if client_count <= 0 or rounds <= 0 or trainable_parameters <= 0:
        raise ValueError("Model communication inputs must be positive")
    payload = trainable_parameters * _FLOAT32_BYTES
    per_client = 2 * rounds * payload
    federation = client_count * per_client
    return ModelCommunicationLedger(client_count, rounds, trainable_parameters, payload, per_client, federation)


def preprocessing_communication(*, client_count: int, feature_count: int) -> PreprocessingCommunicationLedger:
    if client_count <= 0 or feature_count <= 0:
        raise ValueError("Preprocessing communication inputs must be positive")
    extrema = 2 * feature_count
    per_client = extrema * _FLOAT64_BYTES
    return PreprocessingCommunicationLedger(client_count, feature_count, extrema, per_client, client_count * per_client)


def threshold_policy_communication(config: ExperimentConfig, *, client_count: int) -> tuple[PolicyCommunicationLedger, ...]:
    """Return scalar-payload uploads only, excluding headers/transport framing."""
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    split = config.dataset.split
    full_benign_budget = split.reference_benign + split.mismatch_benign + split.calibration_benign
    reference_bytes = split.reference_benign * _FLOAT64_BYTES
    full_score_bytes = full_benign_budget * _FLOAT64_BYTES
    summary_moment_bytes = 2 * (_INT64_BYTES + 2 * _FLOAT64_BYTES)
    candidate_f1_bytes = SUPERVISED_THRESHOLD_CANDIDATES * _FLOAT64_BYTES
    return (
        PolicyCommunicationLedger(PolicyId.FEDCRG, client_count, reference_bytes, client_count * reference_bytes, "Reference scores only, mismatch can be logged as a count and local readiness needs no raw-score upload."),
        PolicyCommunicationLedger(PolicyId.GLOBAL_QUANTILE, client_count, full_score_bytes, client_count * full_score_bytes, "Reference implementation uploads the complete R+G+C benign policy budget."),
        PolicyCommunicationLedger(PolicyId.THREE_SIGMA, client_count, full_score_bytes, client_count * full_score_bytes, "Naive score-upload reference implementation, optimized distributed moments are outside the locked comparator."),
        PolicyCommunicationLedger(PolicyId.SUMMARY_STATISTIC_SELECT, client_count, summary_moment_bytes + candidate_f1_bytes, client_count * (summary_moment_bytes + candidate_f1_bytes), "Two class moment triples plus one 1,000-value F1 vector per client."),
        PolicyCommunicationLedger(PolicyId.SUPERVISED_F1, client_count, candidate_f1_bytes, client_count * candidate_f1_bytes, "One 1,000-value F1 vector per client, candidate broadcasts are excluded."),
    )
