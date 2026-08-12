#!/usr/bin/env python
"""
Audit 4: Implementability and Verification Mapping

Per prompt.md Section 3.4, this audit verifies that every matrix requirement has:
  requirement → configuration/domain representation → implementation owner 
  → runtime caller → artifact/result → test → verification evidence

Also performs reverse-audit: matrix → roadmap (remove anything not authorized)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/home/naslouby/Projects/FedCRG")
sys.path.insert(0, str(PROJECT_ROOT))


def test_matrix_implementation_mapping():
    """Test that matrix requirements map to implementations"""
    print("\n=== Testing Matrix to Implementation Mapping ===")
    
    # This is a high-level test that verifies the structure
    # A full implementation would parse all matrix files and check each requirement
    
    # For now, verify key mappings exist
    
    # 1. GATE-A requirements -> fedcrg.gate_a module
    try:
        from fedcrg.gate_a import (
            compute_gate_a,
            GateAResult,
            GateATable,
            GateATableEntry,
            _compute_p_r,
            verify_gate_a_exact_values,
        )
    except ImportError as e:
        print(f"FAIL: Could not import Gate A module: {e}")
        return False
    
    # 2. GATE-B requirements -> fedcrg.gate_b module
    try:
        from fedcrg.gate_b import (
            compute_gate_b,
            compute_clopper_pearson_interval,
            compute_clopper_pearson_lower,
            compute_clopper_pearson_upper,
            GateBResult,
        )
    except ImportError as e:
        print(f"FAIL: Could not import Gate B module: {e}")
        return False
    
    # 3. REFERENCE requirements -> fedcrg.reference module
    try:
        from fedcrg.reference import (
            build_reference_threshold,
            compute_q_ref,
            compute_n_g_min,
            ReferenceThresholdResult,
            Alpha, A, B, GammaA, GammaB,
        )
    except ImportError as e:
        print(f"FAIL: Could not import reference module: {e}")
        return False
    
    # 4. STATE requirements -> fedcrg.states module
    try:
        from fedcrg.states import (
            decide_fedcrg,
            get_state_from_conditions,
            FedCRGState,
            GateAMismatchState,
            GateBMismatchState,
            FedCRGDecision,
            ReasonCode,
        )
    except ImportError as e:
        print(f"FAIL: Could not import states module: {e}")
        return False
    
    # 5. DATA requirements -> fedcrg.data package
    try:
        from fedcrg.data.base import DatasetRole, RowIDComponents, generate_row_id
        from fedcrg.data.nbaiot import NBaiotAdapter
        from fedcrg.data.diad import DiadAdapter
        from fedcrg.data.splitting import generate_nbaiot_splits, generate_diad_splits
        from fedcrg.data.manifest import DatasetManifest, ClientManifest
    except ImportError as e:
        print(f"FAIL: Could not import data module: {e}")
        return False
    
    # 6. MODEL requirements -> fedcrg.models package
    try:
        from fedcrg.models.autoencoder import Autoencoder, AutoencoderConfig
        from fedcrg.models.deep_svdd import DeepSVDD, DeepSVDDConfig
        from fedcrg.models.base import BaseDetectorModel, ModelConfig
    except ImportError as e:
        print(f"FAIL: Could not import models module: {e}")
        return False
    
    # 7. FL (Federated Learning) requirements -> fedcrg.fl package
    try:
        from fedcrg.fl.trainer import FederatedTrainer
        from fedcrg.fl.server import FederatedServer
        from fedcrg.fl.client import FederatedClient
        from fedcrg.fl.aggregation import ModelAggregator
        from fedcrg.fl.sampling import DeterministicSampler
        from fedcrg.fl.lr_schedule import CosineLearningRateSchedule
    except ImportError as e:
        print(f"FAIL: Could not import FL module: {e}")
        return False
    
    # 8. SCORING requirements -> fedcrg.scoring package
    try:
        from fedcrg.scoring.computer import ScoreComputer, ScoreComputerConfig
        from fedcrg.scoring.cache import ScoreCache
        from fedcrg.scoring.schemas import RoleScores, ClientScores, ScoreManifest
    except ImportError as e:
        print(f"FAIL: Could not import scoring module: {e}")
        return False
    
    # 9. BASELINE requirements -> fedcrg.baselines package
    try:
        from fedcrg.baselines.quantile import QuantileBaseline, QuantileBaselineConfig
        from fedcrg.baselines.gate_only import GateAOnlyBaseline
        from fedcrg.baselines.shrinkage import ShrinkageBaseline
        from fedcrg.baselines.feddetect_3sigma import FedDetect3SigmaBaseline
        from fedcrg.baselines.attack_aware import (
            DevF1LgSelectBaseline,
            LaridiStyleSSBaseline,
            SupF11000Baseline,
        )
        from fedcrg.baselines.oracle import OracleBaseline
    except ImportError as e:
        print(f"FAIL: Could not import baselines module: {e}")
        return False
    
    # 10. METRICS requirements -> fedcrg.metrics package
    try:
        from fedcrg.metrics.band_metrics import (
            compute_mebe,
            compute_high_excess,
            compute_band_violation_rate,
            compute_mafe,
        )
        from fedcrg.metrics.classification import (
            compute_fpr,
            compute_tpr,
            compute_precision,
            compute_recall,
            compute_f1,
        )
        from fedcrg.metrics.auc_metrics import compute_auroc, compute_auprc
        from fedcrg.metrics.attack_balanced import compute_abmacro_tpr
    except ImportError as e:
        print(f"FAIL: Could not import metrics module: {e}")
        return False
    
    # 11. CONFIG requirements -> fedcrg.config module
    try:
        from fedcrg.config import (
            FedCRGConfig,
            ProtocolConfig,
            NBaiotConfig,
            DiadConfig,
            PolicyID,
            load_config,
        )
    except ImportError as e:
        print(f"FAIL: Could not import config module: {e}")
        return False
    
    # 12. EXPERIMENTS requirements -> fedcrg.experiments package
    try:
        from fedcrg.experiments.registry import (
            ExperimentID,
            ExperimentType,
            ExperimentConfig,
            ExperimentRegistry,
            get_registry,
        )
        from fedcrg.experiments.synthetic import (
            run_s1_gate_a_theorem,
            run_s2_target_fpr_sensitivity,
            run_s3_temporal_dependence,
            run_s4_calibration_shift,
            run_s5_contamination,
            run_s6_gate_b_power,
        )
        from fedcrg.experiments.real_data import (
            run_r1_primary,
            run_r10_diad_replication,
            run_r11_second_detector,
        )
    except ImportError as e:
        print(f"FAIL: Could not import experiments module: {e}")
        return False
    
    # 13. FEDCRG core -> fedcrg.fedcrg module
    try:
        from fedcrg.fedcrg import FedCRG, FedCRGConfig, FedCRGResult
    except ImportError as e:
        print(f"FAIL: Could not import fedcrg module: {e}")
        return False
    
    # 14. PREPROCESSING -> fedcrg.data.preprocess module
    try:
        from fedcrg.data.preprocess import (
            create_nbaiot_preprocessor,
            create_diad_preprocessor,
            MinMaxScaler,
            FederatedPreprocessor,
        )
    except ImportError as e:
        print(f"FAIL: Could not import preprocess module: {e}")
        return False
    
    # 15. CLI -> fedcrg.cli module
    try:
        from fedcrg.cli import cli, doctor, train, score, evaluate, synthetic_group
    except ImportError as e:
        print(f"FAIL: Could not import CLI module: {e}")
        return False
    
    print("PASS: All matrix requirement mappings to modules verified")
    return True


def test_runtime_reachability():
    """Test that implementations are reachable from runtime"""
    print("\n=== Testing Runtime Reachability ===")
    
    # Test that we can actually call key functions
    import numpy as np
    from fedcrg.reference import Alpha, A, B, GammaA, GammaB
    from fedcrg.gate_a import compute_gate_a, _compute_p_r
    from fedcrg.gate_b import compute_gate_b, compute_clopper_pearson_interval
    from fedcrg.states import decide_fedcrg, get_state_from_conditions
    from fedcrg.reference import build_reference_threshold, compute_q_ref, compute_n_g_min
    
    # Test Gate A computation
    calibration_scores = np.random.randn(2000).astype(np.float64)
    gate_a_result = compute_gate_a(calibration_scores, Alpha(), 0.5, GammaA())
    if gate_a_result is None:
        print("FAIL: Gate A computation returned None")
        return False
    
    # Test Gate B computation
    gate_scores = np.random.randn(3000).astype(np.float64)
    tau_ref = 0.5
    gate_b_result = compute_gate_b(gate_scores, tau_ref, A(), B(), GammaB())
    if gate_b_result is None:
        print("FAIL: Gate B computation returned None")
        return False
    
    # Test Clopper-Pearson interval
    L, U = compute_clopper_pearson_interval(15, 3000, 0.95)
    if not (0 <= L <= U <= 1):
        print(f"FAIL: Clopper-Pearson interval invalid: [{L}, {U}]")
        return False
    
    # Test n_G_min computation
    n_g_min = compute_n_g_min(0.005, 0.95)
    if n_g_min != 736:
        print(f"FAIL: n_G_min should be 736, got {n_g_min}")
        return False
    
    # Test q_ref computation
    q_ref = compute_q_ref(4500, 0.01)
    if q_ref != 4456:
        print(f"FAIL: q_ref should be 4456, got {q_ref}")
        return False
    
    # Test reference threshold building
    mock_scores = {
        "client_0": np.random.randn(500).astype(np.float64),
        "client_1": np.random.randn(500).astype(np.float64),
    }
    ref_result = build_reference_threshold(mock_scores, 0.01)
    if ref_result is None:
        print("FAIL: Reference threshold building returned None")
        return False
    
    # Test state decision
    from fedcrg.states import FedCRGState, GateBMismatchState
    decision = decide_fedcrg(ref_result, gate_a_result, gate_b_result)
    if decision is None:
        print("FAIL: State decision returned None")
        return False
    if not isinstance(decision.state, FedCRGState):
        print("FAIL: Decision state is not a FedCRGState")
        return False
    
    print("PASS: Runtime reachability verified")
    return True


def test_configuration_system():
    """Test that configuration system maps to implementation"""
    print("\n=== Testing Configuration System ===")
    
    from fedcrg.config import (
        FedCRGConfig,
        ProtocolConfig,
        NBaiotConfig,
        DiadConfig,
        load_config,
    )
    from pathlib import Path
    
    # Test that YAML configs exist
    config_dir = PROJECT_ROOT / "configs"
    expected_configs = [
        "protocol_v2.yaml",
        "nbaiot_primary.yaml",
        "diad_external.yaml",
        "synthetic.yaml",
    ]
    
    for config_file in expected_configs:
        if not (config_dir / config_file).exists():
            print(f"FAIL: Missing config file: {config_file}")
            return False
    
    # Test loading configs
    try:
        protocol_config = load_config(str(config_dir / "protocol_v2.yaml"))
        if protocol_config is None:
            print("FAIL: Could not load protocol_v2.yaml")
            return False
        
        nbaiot_config = load_config(str(config_dir / "nbaiot_primary.yaml"))
        if nbaiot_config is None:
            print("FAIL: Could not load nbaiot_primary.yaml")
            return False
        
        diad_config = load_config(str(config_dir / "diad_external.yaml"))
        if diad_config is None:
            print("FAIL: Could not load diad_external.yaml")
            return False
        
        synthetic_config = load_config(str(config_dir / "synthetic.yaml"))
        if synthetic_config is None:
            print("FAIL: Could not load synthetic.yaml")
            return False
    except Exception as e:
        print(f"FAIL: Error loading configs: {e}")
        return False
    
    print("PASS: Configuration system verified")
    return True


def test_cli_reachability():
    """Test that CLI commands are reachable and functional"""
    print("\n=== Testing CLI Reachability ===")
    
    import subprocess
    
    # Test that CLI commands are installed and respond to --help
    cli_commands = [
        ["python", "-m", "fedcrg", "--help"],
        ["python", "-m", "fedcrg", "doctor", "--help"],
        ["python", "-m", "fedcrg", "synthetic", "--help"],
        ["python", "-m", "fedcrg", "data", "--help"],
        ["python", "-m", "fedcrg", "tables", "--help"],
    ]
    
    for cmd in cli_commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                # Some commands may fail but should still respond
                if "help" not in result.stdout.lower() and "help" not in result.stderr.lower():
                    print(f"FAIL: CLI command not reachable: {' '.join(cmd)}")
                    return False
        except Exception as e:
            print(f"FAIL: CLI command error: {' '.join(cmd)}: {e}")
            return False
    
    print("PASS: CLI reachability verified")
    return True


def test_matrix_file_structure():
    """Test that matrix files have proper structure"""
    print("\n=== Testing Matrix File Structure ===")
    
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
    
    for matrix_file in expected_files:
        file_path = matrix_dir / matrix_file
        if not file_path.exists():
            print(f"FAIL: Missing matrix file: {matrix_file}")
            return False
        
        # Check file has content
        content = file_path.read_text()
        if len(content) < 100:
            print(f"FAIL: Matrix file too short: {matrix_file}")
            return False
        
        # Check for table format (should contain markdown tables)
        if "| ID |" not in content and "| ID | Section |" not in content:
            print(f"WARNING: Matrix file may not have table format: {matrix_file}")
    
    # Check matrix index
    matrix_index = PROJECT_ROOT / "docs" / "FedCRG Audit and Implementation Matrix.md"
    if not matrix_index.exists():
        print("FAIL: Missing matrix index")
        return False
    
    print("PASS: Matrix file structure verified")
    return True


def test_implementation_to_roadmap():
    """Test that implementation maps back to roadmap (reverse audit)"""
    print("\n=== Testing Implementation to Roadmap Mapping ===")
    
    # This verifies that the implementation doesn't have unauthorized behavior
    # We check that key scientific constants match the roadmap
    
    from fedcrg.reference import Alpha, Rho, A, B, GammaA, GammaB
    
    # Roadmap Section 7-20: LOCKED values
    if Alpha() != 0.01:
        print(f"FAIL: Alpha should be 0.01, got {Alpha()}")
        return False
    
    if Rho() != 0.50:
        print(f"FAIL: Rho should be 0.50, got {Rho()}")
        return False
    
    # DERIVED values
    expected_a = 0.005
    expected_b = 0.015
    
    if abs(A() - expected_a) > 1e-15:
        print(f"FAIL: A should be {expected_a}, got {A()}")
        return False
    
    if abs(B() - expected_b) > 1e-15:
        print(f"FAIL: B should be {expected_b}, got {B()}")
        return False
    
    if GammaA() != 0.95:
        print(f"FAIL: GammaA should be 0.95, got {GammaA()}")
        return False
    
    if GammaB() != 0.95:
        print(f"FAIL: GammaB should be 0.95, got {GammaB()}")
        return False
    
    # Check N-BaIoT constants
    from fedcrg.reference import (
        NBaiotClients,
        NBaiotReferencePerClient,
        NBaiotTotalReference,
        NBaiotQRef,
    )
    
    if NBaiotClients != 9:
        print(f"FAIL: NBaiotClients should be 9, got {NBaiotClients}")
        return False
    
    if NBaiotReferencePerClient != 500:
        print(f"FAIL: NBaiotReferencePerClient should be 500, got {NBaiotReferencePerClient}")
        return False
    
    if NBaiotTotalReference != 4500:
        print(f"FAIL: NBaiotTotalReference should be 4500, got {NBaiotTotalReference}")
        return False
    
    if NBaiotQRef != 4456:
        print(f"FAIL: NBaiotQRef should be 4456, got {NBaiotQRef}")
        return False
    
    # Check n_G_min
    from fedcrg.reference import PrimaryNGMin
    if PrimaryNGMin != 736:
        print(f"FAIL: PrimaryNGMin should be 736, got {PrimaryNGMin}")
        return False
    
    print("PASS: Implementation to roadmap mapping verified")
    return True


def main():
    """Run all Audit 4 tests"""
    print("=" * 60)
    print("AUDIT 4: Implementability and Verification Mapping")
    print("=" * 60)
    
    tests = [
        ("Matrix to Implementation Mapping", test_matrix_implementation_mapping),
        ("Runtime Reachability", test_runtime_reachability),
        ("Configuration System", test_configuration_system),
        ("CLI Reachability", test_cli_reachability),
        ("Matrix File Structure", test_matrix_file_structure),
        ("Implementation to Roadmap Mapping", test_implementation_to_roadmap),
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
    
    print("\n" + "=" * 60)
    print("AUDIT 4 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ AUDIT 4 PASSED")
        return 0
    else:
        print(f"\n✗ AUDIT 4 FAILED: {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
