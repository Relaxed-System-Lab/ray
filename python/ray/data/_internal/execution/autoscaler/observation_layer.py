import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    np = None
    GaussianProcessRegressor = None
    Matern = None
    C = None
    _SKLEARN_AVAILABLE = False


@dataclass
class ObservationState:
    ema_throughput: float = 0.0
    last_estimate: float = 0.0
    last_bytes_in: int = 0
    last_bytes_out: int = 0
    last_rows_out: int = 0
    last_queue_size: Optional[float] = None
    last_queue_time: Optional[float] = None
    samples: Deque[Tuple[List[float], float]] = field(default_factory=deque)
    gp: Optional[Any] = None
    gp_fitted: bool = False
    feature_dim: Optional[int] = None


class VLLMObservationLayer:
    """Observation layer for vLLM operators (Trident-style throughput estimation)."""

    def __init__(
        self,
        *,
        ema_alpha: float,
        min_samples: int = 6,
        max_samples: int = 50,
        utilization_threshold: float = 0.6,
        queue_rel_change_threshold: float = 0.5,
        queue_abs_change_threshold: float = 50.0,
        z_threshold: float = 2.5,
        min_throughput: float = 0.001,
    ):
        self._ema_alpha = ema_alpha
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._utilization_threshold = utilization_threshold
        self._queue_rel_change_threshold = queue_rel_change_threshold
        self._queue_abs_change_threshold = queue_abs_change_threshold
        self._z_threshold = z_threshold
        self._min_throughput = min_throughput

        self._states: Dict[Any, ObservationState] = {}
        self._logged_gp_unavailable = False

    def reset(self, op: Any, *, queue_size: float) -> None:
        metrics = op._metrics
        state = ObservationState(samples=deque(maxlen=self._max_samples))
        state.last_bytes_in = metrics.bytes_task_inputs_processed
        state.last_bytes_out = metrics.bytes_task_outputs_generated
        state.last_rows_out = metrics.rows_task_outputs_generated
        state.last_queue_size = queue_size
        state.last_queue_time = time.time()
        self._states[op] = state

    def observe(
        self,
        *,
        op: Any,
        raw_throughput: float,
        delta_rows: float,
        delta_wall_time: float,
        queue_size: float,
        pool_util: float,
        avg_rows_per_bundle: float,
    ) -> float:
        state = self._get_or_create_state(op, queue_size)
        metrics = op._metrics
        now = time.time()

        delta_bytes_in = metrics.bytes_task_inputs_processed - state.last_bytes_in
        delta_bytes_out = metrics.bytes_task_outputs_generated - state.last_bytes_out
        delta_rows_out = metrics.rows_task_outputs_generated - state.last_rows_out

        state.last_bytes_in = metrics.bytes_task_inputs_processed
        state.last_bytes_out = metrics.bytes_task_outputs_generated
        state.last_rows_out = metrics.rows_task_outputs_generated

        avg_bytes_in = delta_bytes_in / delta_rows if delta_rows > 0 else 0.0
        avg_bytes_out = delta_bytes_out / delta_rows_out if delta_rows_out > 0 else 0.0

        features = self._build_features(
            avg_bytes_in=avg_bytes_in,
            avg_bytes_out=avg_bytes_out,
            queue_size=queue_size,
            avg_rows_per_bundle=avg_rows_per_bundle,
        )

        valid_signal = self._passes_signal_filter(
            delta_rows=delta_rows,
            delta_wall_time=delta_wall_time,
            pool_util=pool_util,
            queue_size=queue_size,
            state=state,
        )

        valid_model = True
        if valid_signal and state.gp_fitted:
            valid_model = self._passes_model_filter(state, features, raw_throughput)

        if valid_signal and valid_model:
            self._update_ema(state, raw_throughput)
            self._add_observation(state, features, raw_throughput)
            self._fit_gp(state)

        state.last_queue_size = queue_size
        state.last_queue_time = now

        estimate = self._predict_throughput(state, features, raw_throughput)
        return max(estimate, self._min_throughput)

    def _get_or_create_state(self, op: Any, queue_size: float) -> ObservationState:
        state = self._states.get(op)
        if state is None:
            metrics = op._metrics
            state = ObservationState(samples=deque(maxlen=self._max_samples))
            state.last_bytes_in = metrics.bytes_task_inputs_processed
            state.last_bytes_out = metrics.bytes_task_outputs_generated
            state.last_rows_out = metrics.rows_task_outputs_generated
            state.last_queue_size = queue_size
            state.last_queue_time = time.time()
            self._states[op] = state
        return state

    def _build_features(
        self,
        *,
        avg_bytes_in: float,
        avg_bytes_out: float,
        queue_size: float,
        avg_rows_per_bundle: float,
    ) -> List[float]:
        return [
            math.log1p(max(avg_bytes_in, 0.0)),
            math.log1p(max(avg_bytes_out, 0.0)),
            math.log1p(max(queue_size, 0.0)),
            math.log1p(max(avg_rows_per_bundle, 0.0)),
        ]

    def _passes_signal_filter(
        self,
        *,
        delta_rows: float,
        delta_wall_time: float,
        pool_util: float,
        queue_size: float,
        state: ObservationState,
    ) -> bool:
        if delta_rows <= 0 or delta_wall_time <= 0:
            return False
        if pool_util < self._utilization_threshold:
            return False
        if state.last_queue_size is not None:
            queue_delta = queue_size - state.last_queue_size
            rel_change = abs(queue_delta) / max(1.0, state.last_queue_size)
            if (
                rel_change >= self._queue_rel_change_threshold
                and abs(queue_delta) >= self._queue_abs_change_threshold
            ):
                return False
        return True

    def _passes_model_filter(
        self,
        state: ObservationState,
        features: List[float],
        throughput: float,
    ) -> bool:
        if not state.gp_fitted or state.gp is None:
            return True
        if not _SKLEARN_AVAILABLE or np is None:
            return True
        try:
            mean, std = state.gp.predict(np.array([features]), return_std=True)
            std_val = float(std[0])
            if std_val <= 0:
                return True
            z_score = abs(throughput - float(mean[0])) / std_val
            return z_score <= self._z_threshold
        except Exception:
            logger.debug("GP prediction failed during model-based filtering.")
            return True

    def _update_ema(self, state: ObservationState, throughput: float) -> None:
        if state.ema_throughput <= 0:
            state.ema_throughput = throughput
        else:
            state.ema_throughput = (
                self._ema_alpha * throughput + (1.0 - self._ema_alpha) * state.ema_throughput
            )

    def _add_observation(
        self,
        state: ObservationState,
        features: List[float],
        throughput: float,
    ) -> None:
        state.samples.append((list(features), float(throughput)))

    def _fit_gp(self, state: ObservationState) -> None:
        if not _SKLEARN_AVAILABLE or np is None:
            if not self._logged_gp_unavailable:
                logger.warning(
                    "scikit-learn not available; observation layer will use EMA only."
                )
                self._logged_gp_unavailable = True
            state.gp_fitted = False
            return
        if len(state.samples) < self._min_samples:
            state.gp_fitted = False
            return
        feature_dim = len(state.samples[0][0])
        if state.gp is None or state.feature_dim != feature_dim:
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=[1.0] * feature_dim, nu=2.5
            )
            state.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=2,
            )
            state.feature_dim = feature_dim
        X = np.array([s[0] for s in state.samples], dtype=float)
        y = np.array([s[1] for s in state.samples], dtype=float)
        try:
            state.gp.fit(X, y)
            state.gp_fitted = True
        except Exception:
            logger.debug("Failed to fit GP model for observation layer.")
            state.gp_fitted = False

    def _predict_throughput(
        self,
        state: ObservationState,
        features: List[float],
        raw_throughput: float,
    ) -> float:
        if state.gp_fitted and state.gp is not None and _SKLEARN_AVAILABLE and np is not None:
            try:
                mean = state.gp.predict(np.array([features]))[0]
                if math.isfinite(mean):
                    state.last_estimate = float(mean)
                    return state.last_estimate
            except Exception:
                logger.debug("GP prediction failed; falling back to EMA/raw.")
        if state.ema_throughput > 0:
            state.last_estimate = state.ema_throughput
            return state.last_estimate
        if state.last_estimate > 0:
            return state.last_estimate
        state.last_estimate = raw_throughput
        return raw_throughput
