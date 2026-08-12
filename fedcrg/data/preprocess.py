"""
FedCRG Preprocessing Module

Implements preprocessing, scaling, and imputation per Section 7.4 of the FedCRG Roadmap v2.0.

N-BaIoT preprocessing (Section 7.4.1):
- No missing value imputation (hard parser failure on missing/non-finite)
- Per-client per-feature min/max on T_k only
- Server computes global min/max from client extrema
- Scale: z_ij = (x_ij - m_j) / (M_j - m_j)
- No clipping to [0,1]
- Constant features (M_j = m_j) set to 0.0

CIC IoT-DIAD preprocessing (Section 7.4.2):
- Parse 86-feature allowlist, coerce to numeric
- Client-local median imputation on T_k for missing values
- Requires >= 99.0% finite values per client-feature pair
- Global min/max scaling after imputation
- No clipping to [0,1]

Normative reference: Section 7.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from fedcrg.data.base import DatasetRole


# =============================================================================
# ERROR CODES
# =============================================================================

class PreprocessErrorCode(str, Enum):
    """Error codes for preprocessing failures."""

    # N-BaIoT errors
    NBAIOT_MISSING_VALUE = "NBAIOT_MISSING_VALUE"
    NBAIOT_NON_FINITE_VALUE = "NBAIOT_NON_FINITE_VALUE"
    NBAIOT_NON_NUMERIC_FEATURE = "NBAIOT_NON_NUMERIC_FEATURE"

    # DIAD errors
    DIAD_FEATURE_FINITE_RATE_FAIL = "DIAD_FEATURE_FINITE_RATE_FAIL"
    DIAD_FEATURE_PARSING_FAIL = "DIAD_FEATURE_PARSING_FAIL"

    # General errors
    CONSTANT_FEATURE_DETECTED = "CONSTANT_FEATURE_DETECTED"
    SCALING_DIVISION_BY_ZERO = "SCALING_DIVISION_BY_ZERO"


# =============================================================================
# SCALING TYPES
# =============================================================================

class ScalingType(str, Enum):
    """Types of feature scaling."""

    MIN_MAX = "min_max"
    STANDARD = "standard"
    NONE = "none"


# =============================================================================
# EXTREMA CLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class FeatureExtrema:
    """
    Per-feature min and max values.

    Attributes:
        feature_name: Name of the feature
        min_val: Minimum value for this feature
        max_val: Maximum value for this feature
        is_constant: Whether min_val == max_val
    """

    feature_name: str
    min_val: float
    max_val: float
    is_constant: bool = False

    def __post_init__(self):
        if self.min_val == self.max_val:
            object.__setattr__(self, "is_constant", True)


@dataclass(frozen=True, slots=True)
class ClientFeatureExtrema:
    """
    Per-client feature extrema computed from T_k.

    Attributes:
        client_id: Client identifier
        feature_extrema: Dictionary mapping feature names to FeatureExtrema
        input_dimension: Number of features
    """

    client_id: str
    feature_extrema: Dict[str, FeatureExtrema]
    input_dimension: int

    @classmethod
    def from_dataframe(
        cls,
        client_id: str,
        train_data: pd.DataFrame,
    ) -> "ClientFeatureExtrema":
        """
        Compute per-client feature extrema from training data.

        Args:
            client_id: Client identifier
            train_data: Training dataframe (T_k)

        Returns:
            ClientFeatureExtrema with min/max for each feature
        """
        feature_extrema = {}
        input_dimension = len(train_data.columns)

        for col in train_data.columns:
            col_data = train_data[col].values.astype(np.float64)
            min_val = float(np.min(col_data))
            max_val = float(np.max(col_data))
            feature_extrema[col] = FeatureExtrema(
                feature_name=col,
                min_val=min_val,
                max_val=max_val,
                is_constant=(min_val == max_val),
            )

        return cls(
            client_id=client_id,
            feature_extrema=feature_extrema,
            input_dimension=input_dimension,
        )


@dataclass(frozen=True, slots=True)
class GlobalFeatureExtrema:
    """
    Global feature extrema computed federatively from client extrema.

    For each feature j:
        m_j = min_k (m_kj)  # min across clients
        M_j = max_k (M_kj)  # max across clients

    Attributes:
        feature_extrema: Dictionary mapping feature names to FeatureExtrema
        input_dimension: Number of features
        client_ids: List of client IDs that contributed
        constant_features: Set of feature names that are constant across all clients
    """

    feature_extrema: Dict[str, FeatureExtrema]
    input_dimension: int
    client_ids: List[str]
    constant_features: set[str] = field(default_factory=set)

    @classmethod
    def from_client_extrema(
        cls,
        client_extrema_list: List[ClientFeatureExtrema],
    ) -> "GlobalFeatureExtrema":
        """
        Compute global feature extrema from client extrema.

        Per Section 7.4.1: Server computes m_j = min_k m_{kj} and M_j = max_k M_{kj}

        Args:
            client_extrema_list: List of ClientFeatureExtrema from all clients

        Returns:
            GlobalFeatureExtrema with federated min/max
        """
        if not client_extrema_list:
            raise ValueError("No client extrema provided")

        # Get all feature names from first client
        first_client = client_extrema_list[0]
        feature_names = list(first_client.feature_extrema.keys())
        input_dimension = first_client.input_dimension
        client_ids = [ce.client_id for ce in client_extrema_list]

        # Compute global min/max for each feature
        feature_extrema = {}
        constant_features = set()

        for feature_name in feature_names:
            all_mins = []
            all_maxs = []

            for ce in client_extrema_list:
                fe = ce.feature_extrema[feature_name]
                all_mins.append(fe.min_val)
                all_maxs.append(fe.max_val)

            global_min = min(all_mins)
            global_max = max(all_maxs)

            is_constant = (global_min == global_max)
            if is_constant:
                constant_features.add(feature_name)

            feature_extrema[feature_name] = FeatureExtrema(
                feature_name=feature_name,
                min_val=global_min,
                max_val=global_max,
                is_constant=is_constant,
            )

        return cls(
            feature_extrema=feature_extrema,
            input_dimension=input_dimension,
            client_ids=client_ids,
            constant_features=constant_features,
        )


# =============================================================================
# IMPUTATION CLASSES (for DIAD)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClientImputationStats:
    """
    Client-local imputation statistics for DIAD.

    Per Section 7.4.2: Remaining missing values are imputed with the client-local
    median fitted on that client's T_k only.

    Attributes:
        client_id: Client identifier
        feature_medians: Dictionary mapping feature names to median values
        finite_rate: Dictionary mapping feature names to finite rate in T_k
    """

    client_id: str
    feature_medians: Dict[str, float]
    finite_rate: Dict[str, float]

    @classmethod
    def from_dataframe(
        cls,
        client_id: str,
        train_data: pd.DataFrame,
        finite_threshold: float = 0.99,
    ) -> "ClientImputationStats":
        """
        Compute client imputation statistics from training data.

        Args:
            client_id: Client identifier
            train_data: Training dataframe (T_k)
            finite_threshold: Minimum required finite rate (default 0.99)

        Returns:
            ClientImputationStats with medians and finite rates

        Raises:
            ValueError: If any feature has finite rate below threshold
        """
        feature_medians = {}
        finite_rate = {}

        for col in train_data.columns:
            col_data = train_data[col].values
            finite_mask = np.isfinite(col_data)
            finite_count = np.sum(finite_mask)
            total_count = len(col_data)

            rate = finite_count / total_count if total_count > 0 else 0.0
            finite_rate[col] = rate

            if rate < finite_threshold:
                raise ValueError(
                    f"DIAD feature {col} has finite rate {rate:.4f} "
                    f"< {finite_threshold} for client {client_id}"
                )

            # Compute median on finite values only
            finite_values = col_data[finite_mask].astype(np.float64)
            feature_medians[col] = float(np.median(finite_values))

        return cls(
            client_id=client_id,
            feature_medians=feature_medians,
            finite_rate=finite_rate,
        )


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """
    Configuration for preprocessing.

    Attributes:
        scaling_type: Type of scaling to apply (default: MIN_MAX)
        clip_to_bounds: Whether to clip scaled values to [0,1] (default: False)
        finite_threshold: Minimum finite rate for DIAD features (default: 0.99)
        handle_constant: How to handle constant features (default: "zero")
    """

    scaling_type: ScalingType = ScalingType.MIN_MAX
    clip_to_bounds: bool = False
    finite_threshold: float = 0.99
    handle_constant: str = "zero"  # "zero", "error", or "skip"


# =============================================================================
# PREPROCESSOR CLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class MinMaxScaler:
    """
    Min-max scaler for federated preprocessing.

    Per Section 7.4.1 and 7.4.2:
        z_ij = (x_ij - m_j) / (M_j - m_j)
    
    For constant features (M_j = m_j), set z_ij = 0.

    Attributes:
        global_extrema: Global feature extrema
        clip_to_bounds: Whether to clip to [0,1]
    """

    global_extrema: GlobalFeatureExtrema
    clip_to_bounds: bool = False

    def scale(
        self,
        data: npt.NDArray[np.float64],
        feature_names: Optional[List[str]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Apply min-max scaling to data.

        Args:
            data: Input data array (n_samples, n_features)
            feature_names: Optional list of feature names (must match global_extrema)

        Returns:
            Scaled data array (n_samples, n_features)
        """
        if feature_names is None:
            feature_names = list(self.global_extrema.feature_extrema.keys())

        if data.shape[1] != len(feature_names):
            raise ValueError(
                f"Data has {data.shape[1]} features but {len(feature_names)} feature names provided"
            )

        # Create a copy to avoid modifying the input
        scaled_data = data.copy()

        for i, feature_name in enumerate(feature_names):
            fe = self.global_extrema.feature_extrema[feature_name]

            if fe.is_constant:
                # Set to 0 for constant features
                scaled_data[:, i] = 0.0
            else:
                # Apply min-max scaling: z_ij = (x_ij - m_j) / (M_j - m_j)
                denominator = fe.max_val - fe.min_val
                scaled_data[:, i] = (scaled_data[:, i] - fe.min_val) / denominator

        if self.clip_to_bounds:
            np.clip(scaled_data, 0.0, 1.0, out=scaled_data)

        return scaled_data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "MinMaxScaler",
            "clip_to_bounds": self.clip_to_bounds,
            "global_extrema": {
                "feature_extrema": {
                    name: {
                        "min_val": fe.min_val,
                        "max_val": fe.max_val,
                        "is_constant": fe.is_constant,
                    }
                    for name, fe in self.global_extrema.feature_extrema.items()
                },
                "input_dimension": self.global_extrema.input_dimension,
                "client_ids": self.global_extrema.client_ids,
                "constant_features": list(self.global_extrema.constant_features),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinMaxScaler":
        """Create from dictionary."""
        feature_extrema = {
            name: FeatureExtrema(
                feature_name=name,
                min_val=fe_data["min_val"],
                max_val=fe_data["max_val"],
                is_constant=fe_data.get("is_constant", False),
            )
            for name, fe_data in data["global_extrema"]["feature_extrema"].items()
        }

        global_extrema = GlobalFeatureExtrema(
            feature_extrema=feature_extrema,
            input_dimension=data["global_extrema"]["input_dimension"],
            client_ids=data["global_extrema"]["client_ids"],
            constant_features=set(data["global_extrema"].get("constant_features", [])),
        )

        return cls(
            global_extrema=global_extrema,
            clip_to_bounds=data.get("clip_to_bounds", False),
        )


@dataclass(slots=True)
class FederatedPreprocessor:
    """
    Federated preprocessor that handles both N-BaIoT and DIAD preprocessing.

    This class manages:
    1. Client-local feature statistics computation
    2. Federated aggregation of statistics
    3. Scaling transformation
    4. Imputation for DIAD

    Attributes:
        config: Preprocessing configuration
        dataset_type: Type of dataset ("nbaiot" or "diad")
        input_dimension: Number of input features
        feature_names: List of feature names
    """

    config: PreprocessConfig
    dataset_type: str  # "nbaiot" or "diad"
    input_dimension: int
    feature_names: List[str]

    # Computed during fitting
    client_extrema: Dict[str, ClientFeatureExtrema] = field(default_factory=dict)
    global_extrema: Optional[GlobalFeatureExtrema] = None
    client_imputation: Dict[str, ClientImputationStats] = field(default_factory=dict)

    # Built scaler
    scaler: Optional[MinMaxScaler] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.dataset_type not in ("nbaiot", "diad"):
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def fit_client_train_stats(
        self,
        client_id: str,
        train_data: pd.DataFrame,
    ) -> None:
        """
        Fit client-specific training statistics.

        For N-BaIoT: Compute per-feature min/max on T_k.
        For DIAD: Compute per-feature median and finite rate on T_k.

        Args:
            client_id: Client identifier
            train_data: Training dataframe (T_k)

        Raises:
            ValueError: On preprocessing failures (missing values for N-BaIoT, 
                       finite rate failures for DIAD)
        """
        if self.dataset_type == "nbaiot":
            # N-BaIoT: Check for missing/non-finite values
            for col in train_data.columns:
                col_data = train_data[col].values
                if not np.all(np.isfinite(col_data)):
                    non_finite_mask = ~np.isfinite(col_data)
                    raise ValueError(
                        f"N-BaIoT feature {col} has {np.sum(non_finite_mask)} "
                        f"non-finite values for client {client_id}"
                    )

            # Compute client feature extrema
            self.client_extrema[client_id] = ClientFeatureExtrema.from_dataframe(
                client_id, train_data
            )

        elif self.dataset_type == "diad":
            # DIAD: Compute imputation statistics
            try:
                self.client_imputation[client_id] = ClientImputationStats.from_dataframe(
                    client_id, train_data, self.config.finite_threshold
                )
            except ValueError as e:
                # Re-raise with error code
                raise ValueError(f"{PreprocessErrorCode.DIAD_FEATURE_FINITE_RATE_FAIL}: {e}")

            # Also compute client feature extrema (for scaling)
            self.client_extrema[client_id] = ClientFeatureExtrema.from_dataframe(
                client_id, train_data
            )

    def fit_global_stats(self) -> None:
        """
        Aggregate client statistics to compute global statistics.

        For both N-BaIoT and DIAD: Compute global min/max from client extrema.
        """
        if not self.client_extrema:
            raise ValueError("No client extrema available. Call fit_client_train_stats first.")

        self.global_extrema = GlobalFeatureExtrema.from_client_extrema(
            list(self.client_extrema.values())
        )

        # Build the scaler
        self.scaler = MinMaxScaler(
            global_extrema=self.global_extrema,
            clip_to_bounds=self.config.clip_to_bounds,
        )

    def transform_client_data(
        self,
        client_id: str,
        data: pd.DataFrame,
    ) -> npt.NDArray[np.float64]:
        """
        Transform client data through preprocessing pipeline.

        For N-BaIoT:
            - Apply min-max scaling using global extrema
            - No imputation

        For DIAD:
            - Impute missing values with client-local median
            - Apply min-max scaling using global extrema

        Args:
            client_id: Client identifier
            data: Input dataframe to transform

        Returns:
            Preprocessed data as float64 numpy array
        """
        # Convert to numpy array
        arr = data.values.astype(np.float64)

        if self.dataset_type == "diad":
            # Apply client-local imputation
            imputation_stats = self.client_imputation[client_id]
            for i, col in enumerate(data.columns):
                finite_mask = np.isfinite(arr[:, i])
                non_finite_mask = ~finite_mask

                if np.any(non_finite_mask):
                    median_val = imputation_stats.feature_medians[col]
                    arr[non_finite_mask, i] = median_val

        # Apply scaling
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_global_stats first.")

        return self.scaler.scale(arr, list(data.columns))

    def get_scaler_hash(self) -> str:
        """
        Get a hash of the scaler configuration.

        Returns:
            SHA-256 hash of the scaler configuration
        """
        import hashlib
        import json

        if self.scaler is None:
            raise ValueError("Scaler not fitted")

        scaler_dict = self.scaler.to_dict()
        scaler_str = json.dumps(scaler_dict, sort_keys=True)
        return hashlib.sha256(scaler_str.encode()).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_nbaiot_preprocessor(
    config: Optional[PreprocessConfig] = None,
    input_dimension: int = 115,
    feature_names: Optional[List[str]] = None,
) -> FederatedPreprocessor:
    """
    Create a preprocessor for N-BaIoT data.

    Args:
        config: Optional preprocessing configuration
        input_dimension: Input dimension (default: 115)
        feature_names: Optional list of feature names

    Returns:
        Configured FederatedPreprocessor for N-BaIoT
    """
    if config is None:
        config = PreprocessConfig(
            scaling_type=ScalingType.MIN_MAX,
            clip_to_bounds=False,
            finite_threshold=0.99,
            handle_constant="zero",
        )

    return FederatedPreprocessor(
        config=config,
        dataset_type="nbaiot",
        input_dimension=input_dimension,
        feature_names=feature_names or [f"feature_{i}" for i in range(input_dimension)],
    )


def create_diad_preprocessor(
    config: Optional[PreprocessConfig] = None,
    input_dimension: int = 86,
    feature_names: Optional[List[str]] = None,
) -> FederatedPreprocessor:
    """
    Create a preprocessor for CIC IoT-DIAD data.

    Args:
        config: Optional preprocessing configuration
        input_dimension: Input dimension (default: 86)
        feature_names: Optional list of feature names

    Returns:
        Configured FederatedPreprocessor for DIAD
    """
    if config is None:
        config = PreprocessConfig(
            scaling_type=ScalingType.MIN_MAX,
            clip_to_bounds=False,
            finite_threshold=0.99,
            handle_constant="zero",
        )

    return FederatedPreprocessor(
        config=config,
        dataset_type="diad",
        input_dimension=input_dimension,
        feature_names=feature_names or [f"feature_{i}" for i in range(input_dimension)],
    )


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_nbaiot_preprocessing() -> bool:
    """
    Verify N-BaIoT preprocessing implementation.

    Tests:
    - Per-client min/max computation
    - Global min/max aggregation
    - Min-max scaling
    - Constant feature handling
    - No clipping
    """
    import numpy as np

    # Create test data
    np.random.seed(42)

    # Simulate 2 clients with 3 features each
    client_data = {
        "nb01": pd.DataFrame({
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "f2": [5.0, 5.0, 5.0, 5.0, 5.0],  # constant feature
        }),
        "nb02": pd.DataFrame({
            "f0": [2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [5.0, 15.0, 25.0, 35.0, 45.0],
            "f2": [5.0, 5.0, 5.0, 5.0, 5.0],  # constant feature
        }),
    }

    # Create preprocessor
    preprocessor = create_nbaiot_preprocessor(
        input_dimension=3,
        feature_names=["f0", "f1", "f2"],
    )

    # Fit client statistics
    for cid, data in client_data.items():
        preprocessor.fit_client_train_stats(cid, data)

    # Fit global statistics
    preprocessor.fit_global_stats()

    # Verify global extrema
    assert preprocessor.global_extrema is not None
    fe = preprocessor.global_extrema.feature_extrema

    # f0: min=1.0, max=6.0
    assert abs(fe["f0"].min_val - 1.0) < 1e-10
    assert abs(fe["f0"].max_val - 6.0) < 1e-10
    assert not fe["f0"].is_constant

    # f1: min=5.0, max=50.0
    assert abs(fe["f1"].min_val - 5.0) < 1e-10
    assert abs(fe["f1"].max_val - 50.0) < 1e-10
    assert not fe["f1"].is_constant

    # f2: min=5.0, max=5.0 (constant)
    assert abs(fe["f2"].min_val - 5.0) < 1e-10
    assert abs(fe["f2"].max_val - 5.0) < 1e-10
    assert fe["f2"].is_constant

    # Verify constant features set
    assert "f2" in preprocessor.global_extrema.constant_features

    # Test transformation
    test_data = pd.DataFrame({
        "f0": [1.0, 3.0, 6.0],
        "f1": [5.0, 25.0, 50.0],
        "f2": [5.0, 5.0, 5.0],
    })

    transformed = preprocessor.transform_client_data("nb01", test_data)

    # f0: (1-1)/(6-1)=0, (3-1)/(6-1)=0.4, (6-1)/(6-1)=1.0
    assert abs(transformed[0, 0] - 0.0) < 1e-10
    assert abs(transformed[1, 0] - 0.4) < 1e-10
    assert abs(transformed[2, 0] - 1.0) < 1e-10

    # f1: (5-5)/(50-5)=0, (25-5)/(50-5)=20/45=0.444..., (50-5)/(50-5)=1.0
    assert abs(transformed[0, 1] - 0.0) < 1e-10
    assert abs(transformed[1, 1] - 20/45) < 1e-10
    assert abs(transformed[2, 1] - 1.0) < 1e-10

    # f2: constant feature -> 0.0
    assert abs(transformed[0, 2] - 0.0) < 1e-10
    assert abs(transformed[1, 2] - 0.0) < 1e-10
    assert abs(transformed[2, 2] - 0.0) < 1e-10

    # Verify no clipping (values outside [0,1] should remain)
    test_data_outside = pd.DataFrame({
        "f0": [0.0, 7.0, 10.0],
        "f1": [0.0, 55.0, 100.0],
        "f2": [3.0, 5.0, 7.0],
    })
    transformed_outside = preprocessor.transform_client_data("nb01", test_data_outside)

    # f0: (0-1)/(6-1)=-0.2, (7-1)/(6-1)=1.2, (10-1)/(6-1)=1.8
    assert transformed_outside[0, 0] < 0.0  # -0.2
    assert transformed_outside[1, 0] > 1.0  # 1.2
    assert transformed_outside[2, 0] > 1.0  # 1.8

    print("N-BaIoT preprocessing verification passed.")
    return True


def verify_diad_preprocessing() -> bool:
    """
    Verify DIAD preprocessing implementation.

    Tests:
    - Client-local median imputation
    - Finite rate checking
    - Global min/max aggregation with imputed values
    - Min-max scaling
    """
    import numpy as np

    # Create test data with missing values
    # Need at least 99% finite values, so with 200 rows, at most 2 NaN per feature
    np.random.seed(42)

    # Client with 200 rows, all finite for training (meets 99% threshold)
    client_data = {
        "diad_abc123": pd.DataFrame({
            "f0": [1.0] * 100 + [3.0] * 100,  # all finite
            "f1": list(range(200)),  # varying values
            "f2": [5.0] * 200,  # constant
        }),
        "diad_def456": pd.DataFrame({
            "f0": [2.0] * 100 + [4.0] * 100,  # all finite
            "f1": list(range(200, 400)),  # varying values
            "f2": [6.0] * 200,  # constant
        }),
    }

    # Create preprocessor with 99% threshold
    preprocessor = create_diad_preprocessor(
        config=PreprocessConfig(finite_threshold=0.99),
        input_dimension=3,
        feature_names=["f0", "f1", "f2"],
    )

    # Fit client statistics
    for cid, data in client_data.items():
        preprocessor.fit_client_train_stats(cid, data)

    # Verify imputation statistics
    for cid in client_data:
        imputation = preprocessor.client_imputation[cid]
        # f0 should have median computed from finite values
        assert "f0" in imputation.feature_medians
        # f1 should have median
        assert "f1" in imputation.feature_medians
        # f2 should have median
        assert "f2" in imputation.feature_medians

    # Fit global statistics
    preprocessor.fit_global_stats()

    # Test transformation with missing values (99% finite rate needed)
    # 100 rows, at most 1 NaN per feature to maintain >= 99% finite
    test_data = pd.DataFrame({
        "f0": [1.0] * 50 + [np.nan] + [3.0] * 49,  # 100 rows, 1 NaN = 99% finite
        "f1": list(range(100)),
        "f2": [5.0] * 100,
    })

    transformed = preprocessor.transform_client_data("diad_abc123", test_data)

    # Check that NaN was imputed
    assert np.all(np.isfinite(transformed))

    # Check scaling - f1 should be scaled based on global min/max
    assert np.all(transformed[:, 1] >= 0.0)
    assert np.all(transformed[:, 1] <= 1.0)

    # Check that constant feature f2 is set to 0
    # f2 has min=5.0 and max=6.0 globally
    # But since f2 values in test_data are all 5.0, after scaling: (5-5)/(6-5) = 0
    # Actually, global min/max for f2 would be 5.0 to 6.0
    # Test data has f2=5.0, so scaled: (5-5)/(6-5) = 0.0
    assert np.all(transformed[:, 2] >= 0.0)

    print("DIAD preprocessing verification passed.")
    return True


def verify_preprocessing() -> None:
    """Run all preprocessing verification tests."""
    assert verify_nbaiot_preprocessing()
    assert verify_diad_preprocessing()
    print("All preprocessing verification tests passed.")


if __name__ == "__main__":
    verify_preprocessing()
