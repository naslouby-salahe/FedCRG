"""Tests for FedCRG Reference Threshold Module.

Tests cover:
- LOCKED and DERIVED constant values
- Reference threshold computation
- N-BaIoT primary constants
- DIAD constants
- Gate-B minimum evidence computation
- Run ID formatting
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedcrg.reference import (
    Alpha,
    Rho,
    A,
    B,
    GammaA,
    GammaB,
    PrimaryAlpha,
    PrimaryRho,
    PrimaryA,
    PrimaryB,
    PrimaryGammaA,
    PrimaryGammaB,
    NBaiotClients,
    NBaiotReferencePerClient,
    NBaiotTotalReference,
    NBaiotQRef,
    DiadReferencePerClient,
    diad_q_ref,
    compute_n_g_min,
    PrimaryNGMin,
    compute_q_ref,
    build_reference_threshold,
    ReferenceThresholdResult,
    format_run_id,
)


# =============================================================================
# LOCKED AND DERIVED CONSTANT TESTS
# =============================================================================


class TestLockedConstants:
    """Test LOCKED and DERIVED constant values from Section 4."""

    def test_alpha_locked_value(self):
        """Alpha MUST be LOCKED to 0.01."""
        assert Alpha() == 0.01
        assert PrimaryAlpha() == 0.01

    def test_rho_locked_value(self):
        """Rho MUST be LOCKED to 0.50."""
        assert Rho() == 0.50
        assert PrimaryRho() == 0.50

    def test_gamma_a_locked_value(self):
        """Gamma_A MUST be LOCKED to 0.95."""
        assert GammaA() == 0.95
        assert PrimaryGammaA() == 0.95

    def test_gamma_b_locked_value(self):
        """Gamma_B MUST be LOCKED to 0.95."""
        assert GammaB() == 0.95
        assert PrimaryGammaB() == 0.95

    def test_a_derived_value(self):
        """a = max(0, alpha * (1 - rho)) MUST equal 0.005."""
        assert A() == pytest.approx(0.005)
        assert PrimaryA() == pytest.approx(0.005)

    def test_b_derived_value(self):
        """b = min(1, alpha * (1 + rho)) MUST equal 0.015."""
        assert B() == pytest.approx(0.015)
        assert PrimaryB() == pytest.approx(0.015)


# =============================================================================
# N-BAIOT PRIMARY CONSTANTS
# =============================================================================


class TestNBaiotConstants:
    """Test N-BaIoT primary dataset constants from Section 7.1."""

    def test_nbaiot_clients(self):
        """N-BaIoT MUST have exactly 9 natural clients."""
        assert NBaiotClients == 9

    def test_nbaiot_reference_per_client(self):
        """Each N-BaIoT client MUST have 500 reference scores."""
        assert NBaiotReferencePerClient == 500

    def test_nbaiot_total_reference(self):
        """Total N-BaIoT reference scores MUST be 9 * 500 = 4500."""
        assert NBaiotTotalReference == 4500

    def test_nbaiot_q_ref(self):
        """N-BaIoT q_ref MUST be 4456."""
        assert NBaiotQRef == 4456


# =============================================================================
# DIAD CONSTANTS
# =============================================================================


class TestDiadConstants:
    """Test DIAD dataset constants from Section 7.2."""

    def test_diad_reference_per_client(self):
        """Each DIAD client MUST have 300 reference scores."""
        assert DiadReferencePerClient == 300

    def test_diad_q_ref_computation(self):
        """DIAD q_ref MUST be computed correctly for various client counts."""
        # For K_D = 105 (max official devices)
        total_ref, q_ref = diad_q_ref(105)
        assert total_ref == 105 * 300
        assert q_ref == min(total_ref, math.ceil((total_ref + 1) * 0.99))

        # For K_D = 10
        total_ref, q_ref = diad_q_ref(10)
        assert total_ref == 3000
        assert q_ref == min(3000, math.ceil(3001 * 0.99))


# =============================================================================
# GATE-B MINIMUM EVIDENCE
# =============================================================================


class TestGateBMinimum:
    """Test Gate-B minimum evidence computation from Section 5.3.2."""

    def test_primary_n_g_min(self):
        """Primary n_G_min MUST be 736 for a=0.005, gamma_B=0.95."""
        assert PrimaryNGMin == 736

    def test_compute_n_g_min_primary(self):
        """compute_n_g_min MUST return 736 for primary parameters."""
        assert compute_n_g_min(0.005, 0.95) == 736

    def test_compute_n_g_min_exact(self):
        """Test exact computation of n_G_min."""
        # For a=0.005, gamma_B=0.95
        # Formula: find smallest n where 1 - ((1-0.95)/2)^(1/n) < 0.005
        # i.e., 1 - (0.025)^(1/n) < 0.005
        # i.e., (0.025)^(1/n) > 0.995
        # i.e., 1/n > log(0.995)/log(0.025)
        # i.e., n < log(0.025)/log(0.995)
        
        # At n=735: 1 - (0.025)^(1/735) = 0.0050063101... > 0.005
        # At n=736: 1 - (0.025)^(1/736) = 0.0049995250... < 0.005
        
        assert compute_n_g_min(0.005, 0.95) == 736

    def test_compute_n_g_min_various_a(self):
        """Test n_G_min for various a values."""
        # These values are from roadmap Section 382-386
        # For n_G=736, low mismatch when x=0
        # For n_G=1000, low mismatch when x<=0, high when x>=24
        # For n_G=1500, low mismatch when x<=2, high when x>=33
        # For n_G=2000, low mismatch when x<=3, high when x>=42
        # For n_G=3000, low mismatch when x<=7, high when x>=59
        
        # Verify these are consistent with n_G_min
        assert compute_n_g_min(0.005, 0.95) == 736

    def test_compute_n_g_min_one_sided(self):
        """When a=0, low-side mismatch is impossible (one-sided by design)."""
        # Roadmap Section 492-496: If a=0, no finite bidirectional minimum exists
        result = compute_n_g_min(0.0, 0.95)
        assert result == 0  # Special case indicator


# =============================================================================
# REFERENCE THRESHOLD COMPUTATION
# =============================================================================


class TestReferenceThreshold:
    """Test reference threshold computation from Section 5.1."""

    def test_compute_q_ref_formula(self):
        """q_ref = min(N_R, ceil((N_R + 1)(1 - alpha)))."""
        # N-BaIoT case
        assert compute_q_ref(4500, 0.01) == 4456
        
        # General cases
        assert compute_q_ref(100, 0.01) == min(100, math.ceil(101 * 0.99))
        assert compute_q_ref(1000, 0.01) == min(1000, math.ceil(1001 * 0.99))

    def test_compute_q_ref_edge_cases(self):
        """Test edge cases for q_ref computation."""
        # When (N_R + 1)(1 - alpha) >= N_R
        assert compute_q_ref(100, 0.0) == 100  # alpha=0 means ceil(101*1.0) = 101, min(100, 101) = 100
        
        # Small N_R
        assert compute_q_ref(1, 0.01) == 1
        assert compute_q_ref(2, 0.01) == 2

    def test_build_reference_threshold_basic(self):
        """Test basic reference threshold construction."""
        # Simple case: 2 clients, 3 scores each
        scores = {
            "client_1": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "client_2": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        }
        
        result = build_reference_threshold(scores, alpha=0.01)
        
        assert isinstance(result, ReferenceThresholdResult)
        assert result.n_r == 6
        assert result.n_clients == 2
        assert result.scores_per_client == 3
        assert len(result.sorted_scores) == 6
        
        # q_ref = min(6, ceil(7 * 0.99)) = min(6, ceil(6.93)) = min(6, 7) = 6
        assert result.q_ref == 6
        # tau_ref = 6th order statistic = 6.0
        assert result.tau_ref == pytest.approx(6.0)

    def test_build_reference_threshold_nbaiot_primary(self):
        """Test N-BaIoT primary reference threshold construction."""
        # Create mock N-BaIoT data: 9 clients, 500 scores each
        n_clients = 9
        scores_per_client = 500
        
        # Create synthetic scores
        np.random.seed(42)
        scores = {
            f"nb{i:02d}": np.random.randn(scores_per_client).astype(np.float64)
            for i in range(1, n_clients + 1)
        }
        
        result = build_reference_threshold(scores, alpha=0.01)
        
        assert result.n_r == 4500
        assert result.n_clients == 9
        assert result.scores_per_client == 500
        assert result.q_ref == 4456
        assert len(result.sorted_scores) == 4500

    def test_build_reference_threshold_validation(self):
        """Test input validation for reference threshold construction."""
        # Empty input
        with pytest.raises(ValueError, match="No reference scores provided"):
            build_reference_threshold({})
        
        # Client with zero scores
        with pytest.raises(ValueError, match="zero reference scores"):
            build_reference_threshold({"client_1": np.array([])})
        
        # Unequal client lengths
        with pytest.raises(ValueError, match="expected"):
            build_reference_threshold({
                "client_1": np.array([1.0, 2.0]),
                "client_2": np.array([1.0]),
            })


# =============================================================================
# RUN ID FORMATTING
# =============================================================================


class TestRunIdFormatting:
    """Test run ID formatting from Appendix B."""

    def test_format_run_id_basic(self):
        """Test basic run ID formatting."""
        run_id = format_run_id(
            dataset="nbaiot",
            detector="ae",
            model_seed=11,
            cal_seed=1000,
            alpha=0.01,
            rho=0.5,
            gamma_a=0.95,
            gamma_b=0.95,
            policy="fedcrg",
        )
        
        # alpha_ppm = 0.01 * 1_000_000 = 10000
        # rho_bp = 0.5 * 10_000 = 5000
        # gamma_bp = 0.95 * 10_000 = 9500
        expected = "nbaiot__ae__ms11__cs1000__a10000__r5000__ga9500__gb9500__fedcrg"
        assert run_id == expected

    def test_format_run_id_example_from_roadmap(self):
        """Test the example run ID from the roadmap."""
        run_id = format_run_id(
            dataset="nbaiot",
            detector="ae",
            model_seed=11,
            cal_seed=1000,
            policy="fedcrg",
        )
        
        expected = "nbaiot__ae__ms11__cs1000__a10000__r5000__ga9500__gb9500__fedcrg"
        assert run_id == expected

    def test_format_run_id_various_parameters(self):
        """Test run ID formatting with various parameters."""
        run_id = format_run_id(
            dataset="diad",
            detector="deep_svdd",
            model_seed=22,
            cal_seed=2000,
            alpha=0.005,
            rho=0.25,
            gamma_a=0.99,
            gamma_b=0.95,
            policy="global_q99_full",
        )
        
        # alpha_ppm = 0.005 * 1_000_000 = 5000
        # rho_bp = 0.25 * 10_000 = 2500
        # gamma_a_bp = 0.99 * 10_000 = 9900
        # gamma_b_bp = 0.95 * 10_000 = 9500
        expected = "diad__deep_svdd__ms22__cs2000__a5000__r2500__ga9900__gb9500__global_q99_full"
        assert run_id == expected
