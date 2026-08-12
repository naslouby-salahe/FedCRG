#!/usr/bin/env python
"""
Final Verification Script

This script performs a comprehensive verification of the FedCRG implementation
against the roadmap requirements to ensure completeness before final commit.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/naslouby/Projects/FedCRG")
sys.path.insert(0, str(PROJECT_ROOT))


def verify_core_constants():
    """Verify all LOCKED and DERIVED constants match roadmap"""
    print("\n=== Verifying Core Constants ===")
    
    from fedcrg.reference import (
        Alpha, Rho, A, B, GammaA, GammaB,
        NBaiotClients, NBaiotReferencePerClient, NBaiotTotalReference, NBaiotQRef,
        DiadReferencePerClient,
        PrimaryNGMin, compute_n_g_min,
    )
    
    # LOCKED values from roadmap Section 7-20
    checks = [
        ("Alpha", Alpha(), 0.01),
        ("Rho", Rho(), 0.50),
        ("GammaA", GammaA(), 0.95),
        ("GammaB", GammaB(), 0.95),
        ("NBaiotClients", NBaiotClients, 9),
        ("NBaiotReferencePerClient", NBaiotReferencePerClient, 500),
        ("DiadReferencePerClient", DiadReferencePerClient, 300),
    ]
    
    for name, actual, expected in checks:
        if actual != expected:
            print(f"FAIL: {name} = {actual}, expected {expected}")
            return False
    
    # DERIVED values
    if abs(A() - 0.005) > 1e-15:
        print(f"FAIL: A = {A()}, expected 0.005")
        return False
    if abs(B() - 0.015) > 1e-15:
        print(f"FAIL: B = {B()}, expected 0.015")
        return False
    if NBaiotTotalReference != 4500:
        print(f"FAIL: NBaiotTotalReference = {NBaiotTotalReference}, expected 4500")
        return False
    if NBaiotQRef != 4456:
        print(f"FAIL: NBaiotQRef = {NBaiotQRef}, expected 4456")
        return False
    if PrimaryNGMin != 736:
        print(f"FAIL: PrimaryNGMin = {PrimaryNGMin}, expected 736")
        return False
    
    print("PASS: All core constants verified")
    return True


def verify_gate_a_implementation():
    """Verify Gate A implementation"""
    print("\n=== Verifying Gate A Implementation ===")
    
    from fedcrg.gate_a import (
        compute_gate_a,
        GateAResult,
        GateATable,
        GateATableEntry,
        _compute_p_r,
        verify_gate_a_exact_values,
        precompute_primary_gate_a_table,
    )
    from fedcrg.reference import PrimaryAlpha, PrimaryRho, PrimaryA, PrimaryB, GammaA
    import numpy as np
    
    # Test exact values
    if not verify_gate_a_exact_values(tolerance=1e-10):
        print("FAIL: Gate A exact values verification failed")
        return False
    
    # Test precomputation
    precomputed = precompute_primary_gate_a_table()
    if 2000 not in precomputed:
        print("FAIL: Precomputed table missing n=2000")
        return False
    if precomputed[2000].rank_r != 1982:
        print(f"FAIL: Precomputed rank for n=2000 = {precomputed[2000].rank_r}, expected 1982")
        return False
    
    # Test compute_gate_a with real data
    np.random.seed(42)
    scores = np.random.randn(2000).astype(np.float64)
    result = compute_gate_a(scores, PrimaryAlpha(), PrimaryRho(), GammaA())
    if not result.ready:
        print("FAIL: Gate A should be READY for n=2000")
        return False
    
    print("PASS: Gate A implementation verified")
    return True


def verify_gate_b_implementation():
    """Verify Gate B implementation"""
    print("\n=== Verifying Gate B Implementation ===")
    
    from fedcrg.gate_b import (
        compute_gate_b,
        compute_clopper_pearson_interval,
        compute_clopper_pearson_lower,
        compute_clopper_pearson_upper,
        GateBResult,
    )
    from fedcrg.reference import PrimaryA, PrimaryB, GammaB, compute_n_g_min
    from fedcrg.states import GateBMismatchState
    import numpy as np
    
    # Test n_G_min
    if compute_n_g_min(PrimaryA(), GammaB()) != 736:
        print("FAIL: n_G_min computation failed")
        return False
    
    # Test Clopper-Pearson at boundaries
    # At n=736, x=0: U(0,736) should be < 0.005
    U = compute_clopper_pearson_upper(0, 736, 0.025)
    if U >= 0.005:
        print(f"FAIL: U(0,736) = {U}, should be < 0.005")
        return False
    
    # Test Gate B computation
    np.random.seed(42)
    gate_scores = np.random.randn(3000).astype(np.float64)
    tau_ref = 0.5
    result = compute_gate_b(gate_scores, tau_ref, PrimaryA(), PrimaryB(), GammaB())
    if result is None:
        print("FAIL: Gate B computation returned None")
        return False
    if not isinstance(result.mismatch_state, GateBMismatchState):
        print("FAIL: Gate B result has invalid mismatch_state")
        return False
    
    print("PASS: Gate B implementation verified")
    return True


def verify_reference_threshold():
    """Verify reference threshold implementation"""
    print("\n=== Verifying Reference Threshold Implementation ===")
    
    from fedcrg.reference import (
        build_reference_threshold,
        compute_q_ref,
        ReferenceThresholdResult,
    )
    import numpy as np
    
    # Test q_ref computation
    if compute_q_ref(4500, 0.01) != 4456:
        print("FAIL: q_ref(4500, 0.01) should be 4456")
        return False
    
    # Test reference threshold building
    mock_scores = {
        "client_0": np.random.randn(500).astype(np.float64),
        "client_1": np.random.randn(500).astype(np.float64),
    }
    result = build_reference_threshold(mock_scores, 0.01)
    if result is None:
        print("FAIL: build_reference_threshold returned None")
        return False
    if result.n_r != 1000:
        print(f"FAIL: Expected n_r=1000, got {result.n_r}")
        return False
    if result.n_clients != 2:
        print(f"FAIL: Expected n_clients=2, got {result.n_clients}")
        return False
    
    print("PASS: Reference threshold implementation verified")
    return True


def verify_states():
    """Verify state machine implementation"""
    print("\n=== Verifying State Machine Implementation ===")
    
    from fedcrg.states import (
        decide_fedcrg,
        get_state_from_conditions,
        FedCRGState,
        GateAMismatchState,
        GateBMismatchState,
        FedCRGDecision,
        ReasonCode,
    )
    from fedcrg.reference import ReferenceThresholdResult, A, B, GammaA, GammaB, compute_n_g_min
    from fedcrg.gate_a import GateAResult
    from fedcrg.gate_b import GateBResult
    import numpy as np
    
    # Create mock results
    mock_ref = ReferenceThresholdResult(
        tau_ref=0.5,
        q_ref=4456,
        n_r=4500,
        n_clients=9,
        scores_per_client=500,
        sorted_scores=np.array([], dtype=np.float64),
    )
    
    mock_gate_a = GateAResult(
        n=2000,
        rank=1982,
        coverage_probability=0.9805,
        ready=True,
        tau_local=0.45,
        tie_count=1,
        a=A(),
        b=B(),
        gamma_a=GammaA(),
        sorted_calibration_scores=None,
    )
    
    # Test all five states
    test_cases = [
        (FedCRGState.NO_MATERIAL_MISMATCH_DEMONSTRATED, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED, True, 1, 3000),
        (FedCRGState.LOCAL_PERSONALIZE, GateBMismatchState.HIGH_MISMATCH, True, 1, 3000),
        (FedCRGState.CALIBRATION_DEFICIT, GateBMismatchState.HIGH_MISMATCH, False, 1, 3000),
        (FedCRGState.GATE_B_INSUFFICIENT, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED, True, 1, 500),
        (FedCRGState.CALIBRATION_ASSUMPTION_VIOLATION, GateBMismatchState.HIGH_MISMATCH, True, 3, 3000),
    ]
    
    for expected_state, gate_b_state, gate_a_ready, tie_count, n_g in test_cases:
        mock_gate_a_test = GateAResult(
            n=2000,
            rank=1982,
            coverage_probability=0.9805,
            ready=gate_a_ready,
            tau_local=0.45,
            tie_count=tie_count,
            a=A(),
            b=B(),
            gamma_a=GammaA(),
            sorted_calibration_scores=None,
        )
        
        mock_gate_b_test = GateBResult(
            n=n_g,
            x=60,
            fpr_hat=0.02,
            cp_lower=0.018,
            cp_upper=0.022,
            p_low=0.0,
            p_high=1.0,
            mismatch_state=gate_b_state,
            a=A(),
            b=B(),
            gamma_b=GammaB(),
            n_g_min=736,
            gate_scores=None,
        )
        
        decision = decide_fedcrg(mock_ref, mock_gate_a_test, mock_gate_b_test)
        if decision.state != expected_state:
            print(f"FAIL: Expected {expected_state}, got {decision.state}")
            return False
    
    print("PASS: State machine implementation verified")
    return True


def verify_data_roles():
    """Verify DatasetRole implementation"""
    print("\n=== Verifying DatasetRole Implementation ===")
    
    from fedcrg.data.base import DatasetRole, FEDCRG_ROLES, BENIGN_ROLES, ATTACK_ROLES
    
    # Check all roles exist
    required_roles = ["R", "G", "C", "TRAIN", "TEST_BENIGN", "TEST_ATTACK", "DEV_ATTACK", "GUARD"]
    for role in required_roles:
        if not hasattr(DatasetRole, role):
            print(f"FAIL: DatasetRole.{role} missing")
            return False
    
    # Check FEDCRG_ROLES
    if len(FEDCRG_ROLES) != 5:
        print(f"FAIL: FEDCRG_ROLES should have 5 roles, has {len(FEDCRG_ROLES)}")
        return False
    
    # Check that FEDCRG_ROLES are all benign
    for role in FEDCRG_ROLES:
        if role not in BENIGN_ROLES:
            print(f"FAIL: FEDCRG_ROLES contains non-benign role: {role}")
            return False
    
    print("PASS: DatasetRole implementation verified")
    return True


def verify_experiments():
    """Verify experiment registry"""
    print("\n=== Verifying Experiment Registry ===")
    
    from fedcrg.experiments.registry import get_registry, ExperimentID, ExperimentType
    
    registry = get_registry()
    
    # Check S1-S6
    s_experiments = registry.list_s1_to_s6()
    if sorted(s_experiments) != ["S1", "S2", "S3", "S4", "S5", "S6"]:
        print(f"FAIL: S1-S6 mismatch: {s_experiments}")
        return False
    
    # Check R1-R14
    r_experiments = registry.list_r1_to_r14()
    expected_r = [f"R{i}" for i in range(1, 15)]
    if sorted(r_experiments) != sorted(expected_r):
        print(f"FAIL: R1-R14 mismatch: {r_experiments}")
        return False
    
    # Check all experiments have configs
    for exp_id in s_experiments + r_experiments:
        config = registry.get(exp_id)
        if config is None:
            print(f"FAIL: {exp_id} has no config")
            return False
    
    print("PASS: Experiment registry verified")
    return True


def verify_policies():
    """Verify policy registry"""
    print("\n=== Verifying Policy Registry ===")
    
    from fedcrg.config import PolicyID
    
    expected_policies = [
        "REF-Q99-R", "GLOBAL-Q99-FULL", "LOCAL-Q99-FULL",
        "GATE-A-ONLY", "GATE-B-ONLY", "SHRINKAGE",
        "FEDDETECT-3SIGMA", "DEV-F1-LG-SELECT", "LARIDI-STYLE-SS",
        "SUP-F1-1000", "ORACLE-TEST", "FEDCRG",
    ]
    
    actual_policies = [p.value for p in PolicyID]
    if sorted(actual_policies) != sorted(expected_policies):
        print(f"FAIL: Policy mismatch")
        return False
    
    if len(actual_policies) != 12:
        print(f"FAIL: Expected 12 policies, got {len(actual_policies)}")
        return False
    
    print("PASS: Policy registry verified")
    return True


def verify_cli_commands():
    """Verify CLI commands are reachable"""
    print("\n=== Verifying CLI Commands ===")
    
    import subprocess
    
    commands = [
        ["python", "-m", "fedcrg", "--help"],
        ["python", "-m", "fedcrg", "doctor", "--help"],
        ["python", "-m", "fedcrg", "synthetic", "--help"],
        ["python", "-m", "fedcrg", "data", "--help"],
        ["python", "-m", "fedcrg", "tables", "--help"],
        ["python", "-m", "fedcrg", "train", "--help"],
        ["python", "-m", "fedcrg", "score", "--help"],
        ["python", "-m", "fedcrg", "evaluate", "--help"],
        ["python", "-m", "fedcrg", "robustness", "--help"],
        ["python", "-m", "fedcrg", "benchmark", "--help"],
        ["python", "-m", "fedcrg", "report", "--help"],
        ["python", "-m", "fedcrg", "verify", "--help"],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=PROJECT_ROOT,
            )
            # Check if help text is present
            output = result.stdout + result.stderr
            if "help" not in output.lower() and result.returncode != 0:
                print(f"FAIL: CLI command not reachable: {' '.join(cmd)}")
                return False
        except Exception as e:
            print(f"FAIL: CLI command error: {' '.join(cmd)}: {e}")
            return False
    
    print("PASS: All CLI commands reachable")
    return True


def verify_matrix_files():
    """Verify matrix files exist and have content"""
    print("\n=== Verifying Matrix Files ===")
    
    matrix_dir = PROJECT_ROOT / "docs" / "matrix"
    expected_files = [
        "01_core_requirements.md",
        "02_statistical_core.md",
        "03_dataset_requirements.md",
        "04_training_requirements.md",
        "05_baseline_requirements.md",
        "06_metrics_requirements.md",
        "07_experiment_requirements.md",
        "08_implementation_requirements.md",
        "09_testing_requirements.md",
        "10_failure_claims.md",
    ]
    
    for file_name in expected_files:
        file_path = matrix_dir / file_name
        if not file_path.exists():
            print(f"FAIL: Missing matrix file: {file_name}")
            return False
        if file_path.stat().st_size < 100:
            print(f"FAIL: Matrix file too small: {file_name}")
            return False
    
    # Check matrix index
    matrix_index = PROJECT_ROOT / "docs" / "FedCRG Audit and Implementation Matrix.md"
    if not matrix_index.exists():
        print("FAIL: Missing matrix index")
        return False
    
    print("PASS: All matrix files verified")
    return True


def verify_audit_scripts():
    """Verify audit scripts exist"""
    print("\n=== Verifying Audit Scripts ===")
    
    scripts_dir = PROJECT_ROOT / "scripts"
    expected_scripts = [
        "audit_2_scientific_contract.py",
        "audit_3_experimental_completeness.py",
        "audit_4_implementability.py",
    ]
    
    for script_name in expected_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"FAIL: Missing audit script: {script_name}")
            return False
    
    print("PASS: All audit scripts verified")
    return True


def main():
    """Run all verification tests"""
    print("=" * 70)
    print("FINAL VERIFICATION: FedCRG Implementation Completeness")
    print("=" * 70)
    
    tests = [
        ("Core Constants", verify_core_constants),
        ("Gate A Implementation", verify_gate_a_implementation),
        ("Gate B Implementation", verify_gate_b_implementation),
        ("Reference Threshold", verify_reference_threshold),
        ("State Machine", verify_states),
        ("DatasetRole", verify_data_roles),
        ("Experiment Registry", verify_experiments),
        ("Policy Registry", verify_policies),
        ("CLI Commands", verify_cli_commands),
        ("Matrix Files", verify_matrix_files),
        ("Audit Scripts", verify_audit_scripts),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ FINAL VERIFICATION PASSED")
        print("\nAll core requirements verified:")
        print("  - All LOCKED and DERIVED constants match roadmap")
        print("  - Gate A and Gate B implementations verified")
        print("  - Reference threshold implementation verified")
        print("  - State machine implementation verified")
        print("  - DatasetRole implementation verified")
        print("  - Experiment registry (S1-S6, R1-R14) verified")
        print("  - Policy registry (12 policies) verified")
        print("  - All CLI commands reachable")
        print("  - All matrix files present")
        print("  - All audit scripts present")
        return 0
    else:
        print(f"\n✗ FINAL VERIFICATION FAILED: {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
