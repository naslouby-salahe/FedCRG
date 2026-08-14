import inspect

from fedcrg.thresholding.metrics import ClientEvaluation
from fedcrg.thresholding.policies import BenignPolicyEvidence


def test_protocol_evaluation_has_no_attack_or_label_argument() -> None:
    parameters = set(inspect.signature(ClientEvaluation.evaluate_client).parameters)
    forbidden = {"labels", "attack", "attack_scores", "attack_dev", "attack_test"}
    assert not (parameters & forbidden)


def test_benign_policy_evidence_cannot_carry_attack_arrays() -> None:
    parameters = inspect.signature(BenignPolicyEvidence.__init__).parameters
    assert set(parameters) == {
        "self",
        "client_id",
        "reference_scores",
        "mismatch_scores",
        "calibration_scores",
        "evaluation",
    }
