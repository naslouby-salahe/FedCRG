"""
FedCRG - Federated Calibration Readiness Gate

A post-training operating-point governance layer for federated IoT anomaly detection.
FedCRG is NOT the detector itself, but the admission protocol for client-specific
threshold personalization.

This package implements the FedCRG v2.0 protocol as specified in:
    docs/FedCRG Roadmap.md

Authoritative source: FedCRG Roadmap v2.0, Protocol date: 12 August 2026
"""

__version__ = "0.1.0"
__protocol_version__ = "2.0"

__all__ = ["__version__", "__protocol_version__"]
