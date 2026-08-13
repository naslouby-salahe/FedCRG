import numpy as np
from fedcrg.config.models import ProtocolConfig
from fedcrg.core.enums import DecisionState
from fedcrg.protocol.service import FedCRGProtocol


def test_protocol_service_composes_reference_readiness_mismatch_and_decision() -> None:
    protocol = FedCRGProtocol(); config = ProtocolConfig(); reference = protocol.estimate_reference({"a": np.linspace(0, 1, 500), "b": np.linspace(0, 1, 500)}, config); calibration = np.linspace(0.0, 1.0, 2000); mismatch = np.full(3000, reference.value - 1.0); protocol.precompute_readiness(2000, config); result = protocol.evaluate_client("a", reference, calibration, mismatch, config); assert result.client_id == "a"; assert result.decision.state in {DecisionState.PERSONALIZED, DecisionState.REFERENCE_RETAINED, DecisionState.ASSUMPTION_VIOLATION}
