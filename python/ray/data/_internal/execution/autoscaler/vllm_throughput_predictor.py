"""vLLM Throughput Predictor using Gaussian Process Regression.

This module provides throughput prediction for vLLM inference operators in Ray Data
pipelines. It uses Gaussian Process (GP) regression to model the relationship between
workload features (input/output token lengths) and throughput, enabling more accurate
throughput estimation for MILP-based autoscaling.

Key features:
- Anomaly detection to filter samples limited by upstream sending rate
- Online GP model with sliding window for continuous learning
- Cold-start handling with EMA fallback
- Feature normalization using z-score standardization
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _check_sklearn_dependency():
    """Check if sklearn is available. Only called when the feature is enabled."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        return GaussianProcessRegressor, Matern, WhiteKernel, StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for vLLM throughput prediction. "
            "Please install it with: pip install scikit-learn"
        )


@dataclass
class ThroughputSample:
    """A single observation sample for throughput prediction.

    Attributes:
        timestamp: Unix timestamp when the sample was collected.
        mean_input_length: Mean input token count in the observation window.
        std_input_length: Standard deviation of input token counts.
        p50_input_length: 50th percentile (median) of input token counts.
        p95_input_length: 95th percentile of input token counts.
        mean_output_length: Mean output token count in the observation window.
        std_output_length: Standard deviation of output token counts.
        p50_output_length: 50th percentile (median) of output token counts.
        p95_output_length: 95th percentile of output token counts.
        observed_throughput: Observed throughput (rows/second) in the window.
        gpu_utilization: GPU utilization ratio [0, 1] during the window.
        queue_length: Average pending request queue length during the window.
        num_actors: Number of active actors during the observation.
    """

    timestamp: float
    mean_input_length: float
    std_input_length: float
    p50_input_length: float
    p95_input_length: float
    mean_output_length: float
    std_output_length: float
    p50_output_length: float
    p95_output_length: float
    observed_throughput: float
    gpu_utilization: float
    queue_length: float
    num_actors: int = 1

    def get_feature_vector(self) -> np.ndarray:
        """Extract the 8-dimensional feature vector for GP input."""
        return np.array(
            [
                self.mean_input_length,
                self.std_input_length,
                self.p50_input_length,
                self.p95_input_length,
                self.mean_output_length,
                self.std_output_length,
                self.p50_output_length,
                self.p95_output_length,
            ]
        )


class FeatureExtractor:
    """Extract workload features from a list of request token lengths."""

    @staticmethod
    def extract_features(
        input_lengths: List[int], output_lengths: List[int]
    ) -> Dict[str, float]:
        """Extract statistical features from token length lists.

        Args:
            input_lengths: List of input token counts.
            output_lengths: List of output token counts.

        Returns:
            Dictionary containing mean, std, p50, p95 for both input and output.
        """
        if not input_lengths or not output_lengths:
            return None

        input_arr = np.array(input_lengths)
        output_arr = np.array(output_lengths)

        return {
            "mean_input_length": float(np.mean(input_arr)),
            "std_input_length": float(np.std(input_arr)),
            "p50_input_length": float(np.percentile(input_arr, 50)),
            "p95_input_length": float(np.percentile(input_arr, 95)),
            "mean_output_length": float(np.mean(output_arr)),
            "std_output_length": float(np.std(output_arr)),
            "p50_output_length": float(np.percentile(output_arr, 50)),
            "p95_output_length": float(np.percentile(output_arr, 95)),
        }


class SampleValidator:
    """Validate samples to filter out those limited by upstream sending rate."""

    # Thresholds for anomaly detection
    GPU_UTIL_THRESHOLD = 0.7  # Minimum GPU utilization for valid sample
    QUEUE_LENGTH_THRESHOLD = 1.0  # Minimum queue length for valid sample
    GP_RESIDUAL_THRESHOLD = 2.0  # Max z-score for GP residual filtering

    def __init__(
        self,
        gpu_util_threshold: float = GPU_UTIL_THRESHOLD,
        queue_length_threshold: float = QUEUE_LENGTH_THRESHOLD,
        gp_residual_threshold: float = GP_RESIDUAL_THRESHOLD,
    ):
        self.gpu_util_threshold = gpu_util_threshold
        self.queue_length_threshold = queue_length_threshold
        self.gp_residual_threshold = gp_residual_threshold

    def is_valid_sample(self, sample: ThroughputSample) -> bool:
        """Check if a sample is valid based on GPU utilization and queue length.

        A sample is considered valid (not limited by upstream) if:
        1. GPU utilization >= threshold, OR
        2. Queue length >= threshold

        Args:
            sample: The throughput sample to validate.

        Returns:
            True if the sample is valid, False otherwise.
        """
        return (
            sample.gpu_utilization >= self.gpu_util_threshold
            or sample.queue_length >= self.queue_length_threshold
        )

    def filter_by_gp_residual(
        self,
        sample: ThroughputSample,
        predicted_mean: float,
        predicted_std: float,
    ) -> bool:
        """Secondary filtering using GP prediction residual.

        Filters out samples where the observed throughput deviates too much
        from the GP prediction (potential outliers).

        Args:
            sample: The throughput sample.
            predicted_mean: GP predicted mean throughput.
            predicted_std: GP predicted standard deviation.

        Returns:
            True if the sample passes the residual check, False otherwise.
        """
        if predicted_std <= 0:
            return True

        residual = abs(sample.observed_throughput - predicted_mean)
        z_score = residual / predicted_std

        return z_score <= self.gp_residual_threshold


class OnlineGPModel:
    """Online Gaussian Process model with sliding window for throughput prediction.

    Uses Matérn 5/2 kernel which is suitable for modeling non-smooth functions.
    The model maintains a sliding window of samples and refits periodically.
    """

    # Default hyperparameters
    WINDOW_SIZE = 100  # Maximum number of samples to keep
    MIN_SAMPLES_FOR_FIT = 30  # Minimum samples required to fit GP
    REFIT_INTERVAL = 10  # Refit GP every N new samples

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        min_samples_for_fit: int = MIN_SAMPLES_FOR_FIT,
        refit_interval: int = REFIT_INTERVAL,
    ):
        self.window_size = window_size
        self.min_samples_for_fit = min_samples_for_fit
        self.refit_interval = refit_interval

        # Lazy load sklearn components
        self._gp = None
        self._scaler = None
        self._sklearn_loaded = False

        # Sample storage
        self._samples: List[ThroughputSample] = []
        self._samples_since_last_fit = 0
        self._is_fitted = False

    def _ensure_sklearn(self):
        """Lazy load sklearn components."""
        if not self._sklearn_loaded:
            GaussianProcessRegressor, Matern, WhiteKernel, StandardScaler = (
                _check_sklearn_dependency()
            )

            # Matérn 5/2 kernel + white noise for numerical stability
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=0.1, noise_level_bounds=(1e-5, 1e1)
            )

            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            )
            self._scaler = StandardScaler()
            self._sklearn_loaded = True

    def add_sample(self, sample: ThroughputSample):
        """Add a new sample to the model.

        Args:
            sample: The throughput sample to add.
        """
        self._samples.append(sample)
        self._samples_since_last_fit += 1

        # Maintain sliding window
        if len(self._samples) > self.window_size:
            self._samples = self._samples[-self.window_size :]

        # Check if we should refit
        if (
            len(self._samples) >= self.min_samples_for_fit
            and self._samples_since_last_fit >= self.refit_interval
        ):
            self._fit()

    def _fit(self):
        """Fit the GP model on current samples."""
        self._ensure_sklearn()

        if len(self._samples) < self.min_samples_for_fit:
            logger.debug(
                f"Not enough samples to fit GP: {len(self._samples)} < "
                f"{self.min_samples_for_fit}"
            )
            return

        # Extract features and targets
        X = np.array([s.get_feature_vector() for s in self._samples])
        y = np.array([s.observed_throughput for s in self._samples])

        # Normalize features
        X_normalized = self._scaler.fit_transform(X)

        # Fit GP
        try:
            self._gp.fit(X_normalized, y)
            self._is_fitted = True
            self._samples_since_last_fit = 0
            logger.debug(f"GP model fitted with {len(self._samples)} samples")
        except Exception as e:
            logger.warning(f"Failed to fit GP model: {e}")
            self._is_fitted = False

    def predict(
        self, features: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """Predict throughput for given workload features.

        Args:
            features: 8-dimensional feature vector.

        Returns:
            Tuple of (predicted_mean, predicted_std), or (None, None) if not fitted.
        """
        if not self._is_fitted:
            return None, None

        self._ensure_sklearn()

        try:
            X_normalized = self._scaler.transform(features.reshape(1, -1))
            mean, std = self._gp.predict(X_normalized, return_std=True)
            return float(mean[0]), float(std[0])
        except Exception as e:
            logger.warning(f"GP prediction failed: {e}")
            return None, None

    @property
    def is_fitted(self) -> bool:
        """Check if the model is fitted and ready for prediction."""
        return self._is_fitted

    @property
    def num_samples(self) -> int:
        """Get the current number of samples."""
        return len(self._samples)


class VLLMThroughputPredictor:
    """Main class for vLLM throughput prediction.

    Integrates sample collection, validation, GP modeling, and EMA fallback
    to provide accurate throughput predictions for vLLM operators.
    """

    # EMA smoothing factor for cold-start fallback
    EMA_ALPHA = 0.3

    def __init__(
        self,
        gpu_util_threshold: float = SampleValidator.GPU_UTIL_THRESHOLD,
        queue_length_threshold: float = SampleValidator.QUEUE_LENGTH_THRESHOLD,
        gp_residual_threshold: float = SampleValidator.GP_RESIDUAL_THRESHOLD,
        window_size: int = OnlineGPModel.WINDOW_SIZE,
        min_samples_for_gp: int = OnlineGPModel.MIN_SAMPLES_FOR_FIT,
    ):
        """Initialize the throughput predictor.

        Args:
            gpu_util_threshold: Minimum GPU utilization for valid samples.
            queue_length_threshold: Minimum queue length for valid samples.
            gp_residual_threshold: Maximum z-score for GP residual filtering.
            window_size: Maximum number of samples to keep in GP model.
            min_samples_for_gp: Minimum samples required before using GP.
        """
        self._validator = SampleValidator(
            gpu_util_threshold=gpu_util_threshold,
            queue_length_threshold=queue_length_threshold,
            gp_residual_threshold=gp_residual_threshold,
        )
        self._gp_model = OnlineGPModel(
            window_size=window_size,
            min_samples_for_fit=min_samples_for_gp,
        )
        self._min_samples_for_gp = min_samples_for_gp

        # EMA state for cold-start fallback
        self._ema_throughput: Optional[float] = None

        # Statistics
        self._total_samples = 0
        self._valid_samples = 0

        # Current workload features (updated with each sample)
        self._current_features: Optional[np.ndarray] = None

    def add_sample(self, sample: ThroughputSample) -> bool:
        """Add a new observation sample.

        Args:
            sample: The throughput sample to add.

        Returns:
            True if the sample was considered valid, False otherwise.
        """
        self._total_samples += 1

        # Update current features
        self._current_features = sample.get_feature_vector()

        # Always update EMA (for cold-start fallback)
        self._update_ema(sample.observed_throughput)

        # Validate sample using GPU utilization and queue length
        is_valid = self._validator.is_valid_sample(sample)

        if not is_valid:
            logger.debug(
                f"Sample rejected: gpu_util={sample.gpu_utilization:.2f}, "
                f"queue_len={sample.queue_length:.1f}"
            )
            return False

        # If GP is fitted, apply secondary residual filtering
        if self._gp_model.is_fitted:
            pred_mean, pred_std = self._gp_model.predict(self._current_features)
            if pred_mean is not None and pred_std is not None:
                if not self._validator.filter_by_gp_residual(
                    sample, pred_mean, pred_std
                ):
                    logger.debug(
                        f"Sample rejected by GP residual filter: "
                        f"observed={sample.observed_throughput:.2f}, "
                        f"predicted={pred_mean:.2f}±{pred_std:.2f}"
                    )
                    return False

        # Sample is valid, add to GP model
        self._gp_model.add_sample(sample)
        self._valid_samples += 1
        logger.debug(
            f"Valid sample added: throughput={sample.observed_throughput:.2f}, "
            f"total_valid={self._valid_samples}"
        )
        return True

    def _update_ema(self, throughput: float):
        """Update EMA throughput estimate.

        Args:
            throughput: The observed throughput value.
        """
        if self._ema_throughput is None:
            self._ema_throughput = throughput
        else:
            self._ema_throughput = (
                self.EMA_ALPHA * throughput
                + (1 - self.EMA_ALPHA) * self._ema_throughput
            )

    def predict(
        self, features: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], str]:
        """Predict throughput for given or current workload features.

        Args:
            features: Optional 8-dimensional feature vector.
                     If None, uses the most recent features.

        Returns:
            Tuple of (predicted_throughput, prediction_source).
            prediction_source is either "gp" or "ema".
            Returns (None, "none") if prediction is not available.
        """
        if features is None:
            features = self._current_features

        # Try GP prediction first
        if self._gp_model.is_fitted and features is not None:
            pred_mean, pred_std = self._gp_model.predict(features)
            if pred_mean is not None:
                logger.debug(
                    f"GP prediction: {pred_mean:.2f}±{pred_std:.2f}"
                )
                return pred_mean, "gp"

        # Fall back to EMA
        if self._ema_throughput is not None:
            logger.debug(f"EMA fallback: {self._ema_throughput:.2f}")
            return self._ema_throughput, "ema"

        return None, "none"

    def get_unit_throughput(
        self, features: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """Get the unit throughput prediction for MILP solver.

        This is the main interface for DS2 autoscaler integration.

        Args:
            features: Optional workload features. If None, uses current features.

        Returns:
            Predicted unit throughput, or None if not available.
        """
        throughput, source = self.predict(features)
        if throughput is not None:
            logger.debug(f"Unit throughput ({source}): {throughput:.2f}")
        return throughput

    @property
    def is_ready(self) -> bool:
        """Check if the predictor has enough data for prediction."""
        return self._ema_throughput is not None

    @property
    def is_gp_ready(self) -> bool:
        """Check if the GP model is ready for prediction."""
        return self._gp_model.is_fitted

    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics for debugging."""
        return {
            "total_samples": self._total_samples,
            "valid_samples": self._valid_samples,
            "gp_samples": self._gp_model.num_samples,
            "gp_fitted": self._gp_model.is_fitted,
            "ema_throughput": self._ema_throughput,
        }


class VLLMMetricsCollector:
    """Collector for vLLM operator metrics.

    Aggregates token length statistics and other metrics from vLLM outputs
    over a collection window, then creates ThroughputSamples for the predictor.
    """

    # Default collection window size (number of requests)
    DEFAULT_WINDOW_SIZE = 100

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        """Initialize the metrics collector.

        Args:
            window_size: Number of requests to collect before creating a sample.
        """
        self.window_size = window_size

        # Token length buffers for current window
        self._input_lengths: List[float] = []
        self._output_lengths: List[float] = []

        # Last collected metrics
        self._last_gpu_util: float = 0.0
        self._last_queue_length: float = 0.0

        # Timestamps for throughput calculation
        self._window_start_time: Optional[float] = None
        self._total_requests_in_window: int = 0

    def add_request(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        gpu_utilization: Optional[float] = None,
        queue_length: Optional[float] = None,
    ):
        """Add a completed request to the collector.

        Args:
            num_input_tokens: Number of input tokens for this request.
            num_output_tokens: Number of output tokens generated.
            gpu_utilization: Current GPU utilization (0-1), if available.
            queue_length: Current queue length, if available.
        """
        if self._window_start_time is None:
            self._window_start_time = time.time()

        self._input_lengths.append(float(num_input_tokens))
        self._output_lengths.append(float(num_output_tokens))
        self._total_requests_in_window += 1

        if gpu_utilization is not None:
            self._last_gpu_util = gpu_utilization
        if queue_length is not None:
            self._last_queue_length = queue_length

    def is_window_ready(self) -> bool:
        """Check if enough requests have been collected to create a sample."""
        return len(self._input_lengths) >= self.window_size

    def create_sample(
        self,
        num_actors: int,
        gpu_utilization: Optional[float] = None,
        queue_length: Optional[float] = None,
    ) -> Optional[ThroughputSample]:
        """Create a ThroughputSample from collected metrics.

        Args:
            num_actors: Current number of active actors.
            gpu_utilization: Override GPU utilization value.
            queue_length: Override queue length value.

        Returns:
            A ThroughputSample, or None if not enough data.
        """
        if len(self._input_lengths) == 0:
            return None

        # Use provided values or fall back to last collected
        gpu_util = gpu_utilization if gpu_utilization is not None else self._last_gpu_util
        q_len = queue_length if queue_length is not None else self._last_queue_length

        # Calculate throughput (requests per second per actor)
        elapsed_time = time.time() - (self._window_start_time or time.time())
        if elapsed_time > 0 and num_actors > 0:
            observed_throughput = self._total_requests_in_window / elapsed_time / num_actors
        else:
            observed_throughput = 0.0

        # Extract features using FeatureExtractor
        features = FeatureExtractor.extract_features(
            self._input_lengths, self._output_lengths
        )

        # Create sample
        sample = ThroughputSample(
            timestamp=time.time(),
            mean_input_length=features[0],
            std_input_length=features[1],
            p50_input_length=features[2],
            p95_input_length=features[3],
            mean_output_length=features[4],
            std_output_length=features[5],
            p50_output_length=features[6],
            p95_output_length=features[7],
            observed_throughput=observed_throughput,
            gpu_utilization=gpu_util,
            queue_length=q_len,
            num_actors=num_actors,
        )

        # Reset for next window
        self._reset_window()

        return sample

    def _reset_window(self):
        """Reset the collection window."""
        self._input_lengths = []
        self._output_lengths = []
        self._window_start_time = None
        self._total_requests_in_window = 0

    def get_current_features(self) -> Optional[np.ndarray]:
        """Get feature vector from current partial window.

        Useful for prediction when window is not yet complete.
        """
        if len(self._input_lengths) == 0:
            return None

        return FeatureExtractor.extract_features(
            self._input_lengths, self._output_lengths
        )


def estimate_gpu_utilization(
    num_tasks_in_flight: int,
    max_tasks_per_actor: int,
    num_actors: int,
) -> float:
    """Estimate GPU utilization based on task queue depth.

    This is a heuristic when actual GPU metrics are not available.

    Args:
        num_tasks_in_flight: Total tasks currently in flight.
        max_tasks_per_actor: Maximum tasks per actor.
        num_actors: Number of running actors.

    Returns:
        Estimated GPU utilization (0-1).
    """
    if num_actors == 0 or max_tasks_per_actor == 0:
        return 0.0

    max_capacity = num_actors * max_tasks_per_actor
    utilization = num_tasks_in_flight / max_capacity

    # Clamp to [0, 1] and apply a slight boost since
    # tasks in flight usually means GPU is working
    return min(1.0, utilization * 1.2)
