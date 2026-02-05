import copy
import logging
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import ray
from ray.data.block import BlockAccessor

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
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)


@dataclass
class WorkloadFeatures:
    mean_input_tokens: float
    var_input_tokens: float
    mean_output_tokens: float
    var_output_tokens: float

    def to_vector(self, *, log_scale: bool = True) -> List[float]:
        values = [
            self.mean_input_tokens,
            self.var_input_tokens,
            self.mean_output_tokens,
            self.var_output_tokens,
        ]
        if log_scale:
            return [math.log1p(max(v, 0.0)) for v in values]
        return values


class ClusterStatus(Enum):
    PENDING = "pending"
    TUNING = "tuning"
    TUNED = "tuned"


class ConfigApplyScope(Enum):
    PROBE = "probe"
    ROLLOUT = "rollout"


@dataclass
class WorkloadCluster:
    cluster_id: int
    centroid: List[float]
    count: float = 0.0
    status: ClusterStatus = ClusterStatus.PENDING
    config: Optional[Dict[str, Any]] = None
    pending_config: Optional[Dict[str, Any]] = None
    pending_observation_count: int = -1
    needs_rollout: bool = False
    last_tuned_centroid: Optional[List[float]] = None
    last_tuned_time: Optional[float] = None
    last_updated_time: Optional[float] = None
    optimizer: Optional["VLLMConfigOptimizer"] = None


@dataclass
class SwitchDecision:
    cluster_id: int
    config: Dict[str, Any]
    reason: str
    scope: ConfigApplyScope


@dataclass
class RolloutState:
    cluster_id: int
    config: Dict[str, Any]
    remaining: int
    last_update_time: float = 0.0


class VLLMConfigOptimizer:
    """Lightweight Bayesian optimizer inspired by SCOOT/HEBO.

    Falls back to random search if sklearn is unavailable or insufficient data.
    """

    def __init__(
        self,
        *,
        space: Optional[List[Dict[str, Any]]] = None,
        random_seed: Optional[int] = None,
        min_random: int = 5,
        candidates: int = 128,
    ):
        self._random = random.Random(random_seed)
        self._space = space or self._default_space()
        self._min_random = min_random
        self._candidates = candidates
        self._X: List[List[float]] = []
        self._y: List[float] = []
        self._gp: Optional[Any] = None
        self._feature_dim: Optional[int] = None
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_objective: Optional[float] = None
        self._observed: set = set()

    def observe(self, config: Dict[str, Any], objective: float) -> None:
        normalized = self.normalize_config(config)
        vec = self._vectorize(normalized)
        self._X.append(vec)
        self._y.append(objective)
        self._observed.add(tuple(vec))
        if self._best_objective is None or objective > self._best_objective:
            self._best_objective = objective
            self._best_config = copy.deepcopy(normalized)

    def suggest(self) -> Dict[str, Any]:
        if len(self._X) < self._min_random or not _SKLEARN_AVAILABLE or np is None:
            return self._sample_random()

        self._fit_gp()
        if self._gp is None:
            return self._sample_random()

        best_y = max(self._y) if self._y else 0.0
        best_candidate: Optional[Dict[str, Any]] = None
        best_ei = -float("inf")
        for _ in range(self._candidates):
            candidate = self._sample_random()
            vec_list = self._vectorize(candidate)
            if tuple(vec_list) in self._observed:
                continue
            vec = np.array([vec_list], dtype=float)
            try:
                mean, std = self._gp.predict(vec, return_std=True)
            except Exception:
                continue
            mu = float(mean[0])
            sigma = float(std[0])
            ei = self._expected_improvement(mu, sigma, best_y)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        return best_candidate or self._sample_random()

    @property
    def best_config(self) -> Optional[Dict[str, Any]]:
        if self._best_config is None:
            return None
        return copy.deepcopy(self._best_config)

    @property
    def num_observations(self) -> int:
        return len(self._y)

    def reset(self) -> None:
        self._X.clear()
        self._y.clear()
        self._gp = None
        self._feature_dim = None
        self._best_config = None
        self._best_objective = None
        self._observed.clear()

    def normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for spec in self._space:
            name = spec["name"]
            if name in config:
                normalized[name] = config[name]
            else:
                normalized[name] = self._default_for_spec(spec)
        return normalized

    def _default_for_spec(self, spec: Dict[str, Any]) -> Any:
        if "default" in spec:
            return spec["default"]
        typ = spec["type"]
        if typ == "int":
            return int(spec["lb"])
        if typ == "float":
            return float(spec["lb"])
        if typ == "bool":
            return False
        if typ == "cat":
            categories = list(spec.get("categories") or [])
            return categories[0] if categories else None
        return None

    def _fit_gp(self) -> None:
        if not _SKLEARN_AVAILABLE or np is None:
            self._gp = None
            return
        if len(self._X) < self._min_random:
            self._gp = None
            return
        feature_dim = len(self._X[0])
        if self._gp is None or self._feature_dim != feature_dim:
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=[1.0] * feature_dim, nu=2.5
            )
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=1,
            )
            self._feature_dim = feature_dim
        X = np.array(self._X, dtype=float)
        y = np.array(self._y, dtype=float)
        try:
            self._gp.fit(X, y)
        except Exception:
            self._gp = None

    def _expected_improvement(self, mu: float, sigma: float, best: float) -> float:
        if sigma <= 1e-8:
            return 0.0
        z = (mu - best) / sigma
        return (mu - best) * self._phi(z) + sigma * self._phi_pdf(z)

    @staticmethod
    def _phi(z: float) -> float:
        # Standard normal CDF
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @staticmethod
    def _phi_pdf(z: float) -> float:
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)

    def _sample_random(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for spec in self._space:
            name = spec["name"]
            typ = spec["type"]
            if typ == "int":
                config[name] = self._random.randint(int(spec["lb"]), int(spec["ub"]))
            elif typ == "float":
                lb = float(spec["lb"])
                ub = float(spec["ub"])
                config[name] = self._random.uniform(lb, ub)
            elif typ == "bool":
                config[name] = bool(self._random.getrandbits(1))
            elif typ == "cat":
                config[name] = self._random.choice(list(spec["categories"]))
            else:
                raise ValueError(f"Unknown parameter type: {typ}")
        return config

    def _vectorize(self, config: Dict[str, Any]) -> List[float]:
        vec: List[float] = []
        for spec in self._space:
            name = spec["name"]
            typ = spec["type"]
            value = config.get(name)
            if typ == "int":
                vec.append(float(value))
            elif typ == "float":
                vec.append(float(value))
            elif typ == "bool":
                vec.append(1.0 if value else 0.0)
            elif typ == "cat":
                categories = list(spec["categories"])
                vec.append(float(categories.index(value)) if value in categories else 0.0)
            else:
                vec.append(0.0)
        return vec

    @staticmethod
    def _default_space() -> List[Dict[str, Any]]:
        # Inspired by SCOOT's vLLM tuning space.
        return [
            {"name": "max_num_seqs", "type": "int", "lb": 4, "ub": 64},
            {"name": "max_num_batched_tokens", "type": "int", "lb": 1024, "ub": 32768},
            {"name": "block_size", "type": "int", "lb": 8, "ub": 64},
            {"name": "enable_chunked_prefill", "type": "bool"},
            {"name": "enable_prefix_caching", "type": "bool"},
            {"name": "disable_custom_all_reduce", "type": "bool"},
        ]


class VLLMWorkloadFeatureExtractor:
    """Extract vLLM workload features from operator queues or metrics."""

    def __init__(
        self,
        *,
        max_blocks: int = 64,
        max_rows: int = 2048,
        seen_cache: int = 1024,
        log_interval_s: float = 30.0,
    ):
        self._max_blocks = max_blocks
        self._max_rows = max_rows
        self._seen_cache = seen_cache
        self._log_interval_s = log_interval_s
        self._seen_ids: Dict[Any, Deque[str]] = defaultdict(lambda: deque(maxlen=seen_cache))
        self._seen_set: Dict[Any, set] = defaultdict(set)
        self._last_log_time: Dict[Any, float] = defaultdict(lambda: 0.0)
        self._last_log_source: Dict[Any, str] = {}
        self._last_missing_log_time: Dict[Any, float] = defaultdict(lambda: 0.0)

    def reset(self, op: Any) -> None:
        if op in self._seen_ids:
            self._seen_ids.pop(op, None)
        if op in self._seen_set:
            self._seen_set.pop(op, None)
        self._last_log_time.pop(op, None)
        self._last_log_source.pop(op, None)

    def extract(self, op: Any, op_state: Any) -> Optional[WorkloadFeatures]:
        metrics_features = self._extract_from_metrics(op)
        if metrics_features is not None:
            self._maybe_log_features(op, metrics_features, source="metrics")
            return metrics_features
        bundles = list(self._iter_output_bundles(op, op_state))
        if not bundles:
            self._maybe_log_missing(op, op_state, reason="empty_output_queue")
            return None

        input_stats = RunningStats()
        output_stats = RunningStats()
        seen_ids = self._seen_ids[op]
        seen_set = self._seen_set[op]

        rows_collected = 0
        blocks_checked = 0
        for bundle in bundles:
            for block_ref, _ in bundle.blocks:
                block_id = self._block_ref_id(block_ref)
                if block_id in seen_set:
                    continue
                seen_set.add(block_id)
                seen_ids.append(block_id)
                while len(seen_set) > self._seen_cache:
                    old = seen_ids.popleft()
                    seen_set.discard(old)
                try:
                    block = ray.get(block_ref)
                except Exception:
                    continue
                rows_collected += self._accumulate_from_block(
                    block, input_stats, output_stats, self._max_rows - rows_collected
                )
                blocks_checked += 1
                if rows_collected >= self._max_rows or blocks_checked >= self._max_blocks:
                    break
            if rows_collected >= self._max_rows or blocks_checked >= self._max_blocks:
                break

        if input_stats.count == 0 and output_stats.count == 0:
            self._maybe_log_missing(op, op_state, reason="no_token_fields")
            return None

        features = WorkloadFeatures(
            mean_input_tokens=input_stats.mean,
            var_input_tokens=input_stats.variance(),
            mean_output_tokens=output_stats.mean,
            var_output_tokens=output_stats.variance(),
        )
        self._maybe_log_features(op, features, source="queue")
        return features

    def _maybe_log_features(
        self, op: Any, features: WorkloadFeatures, *, source: str
    ) -> None:
        now = time.time()
        last = self._last_log_time[op]
        if now - last < self._log_interval_s and self._last_log_source.get(op) == source:
            return
        self._last_log_time[op] = now
        self._last_log_source[op] = source
        logger.info(
            "vLLM features (%s) for %s: mean_in=%.2f var_in=%.2f mean_out=%.2f var_out=%.2f",
            source,
            getattr(op, "name", op),
            features.mean_input_tokens,
            features.var_input_tokens,
            features.mean_output_tokens,
            features.var_output_tokens,
        )

    def _maybe_log_missing(self, op: Any, op_state: Any, *, reason: str) -> None:
        now = time.time()
        last = self._last_missing_log_time[op]
        if now - last < self._log_interval_s:
            return
        self._last_missing_log_time[op] = now
        output_queue = getattr(op_state, "output_queue", None)
        op_queue = getattr(op, "_output_queue", None)

        output_bundles = None
        output_blocks = None
        if output_queue is not None:
            try:
                output_bundles = len(output_queue)
            except Exception:
                output_bundles = None
            output_blocks = getattr(output_queue, "num_blocks", None)

        op_queue_size = None
        if op_queue is not None:
            task_outputs = getattr(op_queue, "_task_outputs", None)
            if isinstance(task_outputs, dict):
                try:
                    op_queue_size = sum(len(v) for v in task_outputs.values())
                except Exception:
                    op_queue_size = None
            else:
                queue_deque = getattr(op_queue, "_queue", None)
                if queue_deque is not None:
                    try:
                        op_queue_size = len(queue_deque)
                    except Exception:
                        op_queue_size = None

        logger.info(
            "vLLM feature snapshot missing for %s (reason=%s). output_queue_bundles=%s "
            "output_queue_blocks=%s op_queue_size=%s",
            getattr(op, "name", op),
            reason,
            output_bundles,
            output_blocks,
            op_queue_size,
        )

    def _extract_from_metrics(self, op: Any) -> Optional[WorkloadFeatures]:
        metrics = getattr(op, "metrics", None)
        if metrics is None:
            metrics = getattr(op, "_metrics", None)
        if metrics is None:
            return None
        extra = getattr(metrics, "extra_metrics", None)
        if not extra:
            return None
        try:
            mean_in = extra["vllm_input_tokens_mean"]
            var_in = extra["vllm_input_tokens_var"]
            mean_out = extra["vllm_output_tokens_mean"]
            var_out = extra["vllm_output_tokens_var"]
        except KeyError:
            return None
        return WorkloadFeatures(
            mean_input_tokens=float(mean_in),
            var_input_tokens=float(var_in),
            mean_output_tokens=float(mean_out),
            var_output_tokens=float(var_out),
        )

    def _iter_output_bundles(self, op: Any, op_state: Any) -> Iterable[Any]:
        """Best-effort snapshot of output bundles without mutating queues."""
        # Try op_state output queue first (buffer between operators).
        queue = getattr(op_state, "output_queue", None)
        bundles = self._snapshot_opbuffer_queue(queue)
        if bundles:
            return bundles

        # Fallback to operator-level output queue (per-task ordering).
        op_queue = getattr(op, "_output_queue", None)
        bundles = self._snapshot_operator_queue(op_queue)
        if bundles:
            return bundles

        return []

    def _snapshot_opbuffer_queue(self, queue: Any) -> List[Any]:
        if queue is None:
            return []
        lock = getattr(queue, "_lock", None)
        bundle_queue = getattr(queue, "_queue", None)
        if lock is not None:
            with lock:
                return self._snapshot_bundle_queue(bundle_queue)
        return self._snapshot_bundle_queue(bundle_queue)

    def _snapshot_operator_queue(self, queue: Any) -> List[Any]:
        if queue is None:
            return []
        # Ordered output queue: sample from per-task deques.
        task_outputs = getattr(queue, "_task_outputs", None)
        if isinstance(task_outputs, dict):
            bundles: List[Any] = []
            for outputs in task_outputs.values():
                try:
                    for bundle in list(outputs):
                        bundles.append(bundle)
                        if len(bundles) >= self._max_blocks:
                            return bundles
                except Exception:
                    continue
            return bundles
        # Unordered output queue: deque of bundles.
        queue_deque = getattr(queue, "_queue", None)
        if queue_deque is not None:
            try:
                return list(queue_deque)[: self._max_blocks]
            except Exception:
                return []
        return []

    def _snapshot_bundle_queue(self, bundle_queue: Any) -> List[Any]:
        if bundle_queue is None:
            return []
        # Best-effort snapshot without mutating the queue.
        head = getattr(bundle_queue, "_head", None)
        if head is None:
            peek = getattr(bundle_queue, "peek", None)
            if callable(peek):
                bundle = peek()
                return [bundle] if bundle is not None else []
            return []
        bundles: List[Any] = []
        node = head
        while node is not None and len(bundles) < self._max_blocks:
            bundles.append(node.value)
            node = node.next
        return bundles

    def _accumulate_from_block(
        self,
        block: Any,
        input_stats: RunningStats,
        output_stats: RunningStats,
        row_budget: int,
    ) -> int:
        accessor = BlockAccessor.for_block(block)
        rows_iter = accessor.iter_rows(public_row_format=True)
        rows_used = 0
        for row in rows_iter:
            for in_tokens, out_tokens in self._iter_token_pairs(row):
                if in_tokens is not None:
                    input_stats.update(in_tokens)
                if out_tokens is not None:
                    output_stats.update(out_tokens)
                rows_used += 1
                if rows_used >= row_budget:
                    break
            if rows_used >= row_budget:
                break
        return rows_used

    def _iter_token_pairs(
        self, row: Any
    ) -> Iterable[Tuple[Optional[float], Optional[float]]]:
        payload = row
        if isinstance(row, dict) and "__data" in row:
            payload = row.get("__data")

        if isinstance(payload, (list, tuple)):
            for item in payload:
                yield self._extract_tokens_from_payload(item)
            return

        yield self._extract_tokens_from_payload(payload)

    def _extract_tokens_from_payload(
        self, payload: Any
    ) -> Tuple[Optional[float], Optional[float]]:
        in_tokens = self._read_token_value(payload, "num_input_tokens", "prompt_token_ids")
        out_tokens = self._read_token_value(payload, "num_generated_tokens", "generated_tokens")
        return in_tokens, out_tokens

    def _read_token_value(
        self,
        payload: Any,
        count_key: str,
        list_key: str,
    ) -> Optional[float]:
        value = None
        if isinstance(payload, dict):
            value = payload.get(count_key)
            if value is None and list_key in payload:
                try:
                    value = len(payload.get(list_key) or [])
                except Exception:
                    value = None
        else:
            if hasattr(payload, count_key):
                value = getattr(payload, count_key)
            elif hasattr(payload, list_key):
                try:
                    value = len(getattr(payload, list_key) or [])
                except Exception:
                    value = None
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value

    def _block_ref_id(self, block_ref: Any) -> str:
        if hasattr(block_ref, "hex"):
            try:
                return block_ref.hex()
            except Exception:
                pass
        return str(block_ref)


class VLLMAdaptationLayer:
    """Adaptation layer for vLLM operators (online clustering + BO tuning)."""

    def __init__(
        self,
        *,
        max_clusters: int = 6,
        distance_threshold: float = 1.2,
        min_samples: int = 5,
        min_cluster_fraction: float = 0.1,
        centroid_drift_threshold: float = 0.4,
        decay_gamma: float = 0.995,
        decay_interval_s: float = 300.0,
        match_window: int = 50,
        switch_consistency: float = 0.7,
        cooldown_s: float = 60.0,
        tuning_cooldown_s: float = 60.0,
        bo_steps_required: int = 5,
        rollout_interval_s: float = 30.0,
        prune_threshold: float = 0.1,
        cluster_log_interval_s: float = 60.0,
        feature_extractor: Optional[VLLMWorkloadFeatureExtractor] = None,
    ):
        self._max_clusters = max_clusters
        self._distance_threshold = distance_threshold
        self._min_samples = min_samples
        self._min_cluster_fraction = min_cluster_fraction
        self._centroid_drift_threshold = centroid_drift_threshold
        self._decay_gamma = decay_gamma
        self._decay_interval_s = decay_interval_s
        self._match_window = match_window
        self._switch_consistency = switch_consistency
        self._cooldown_s = cooldown_s
        self._tuning_cooldown_s = tuning_cooldown_s
        self._bo_steps_required = bo_steps_required
        self._rollout_interval_s = rollout_interval_s
        self._prune_threshold = prune_threshold
        self._cluster_log_interval_s = cluster_log_interval_s
        self._feature_extractor = feature_extractor or VLLMWorkloadFeatureExtractor()

        self._clusters: Dict[Any, List[WorkloadCluster]] = defaultdict(list)
        self._match_history: Dict[Any, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self._match_window)
        )
        self._active_cluster: Dict[Any, Optional[int]] = defaultdict(lambda: None)
        self._last_switch_time: Dict[Any, float] = defaultdict(lambda: 0.0)
        self._last_decay_time: Dict[Any, float] = defaultdict(lambda: 0.0)
        self._last_tune_time: Dict[Any, float] = defaultdict(lambda: 0.0)
        self._last_cluster_log_time: Dict[Any, float] = defaultdict(lambda: 0.0)
        self._observation_count: Dict[Any, int] = defaultdict(int)
        self._rollout_state: Dict[Any, Optional[RolloutState]] = defaultdict(
            lambda: None
        )
        self._next_cluster_id: Dict[Any, int] = defaultdict(int)
        self._lock = threading.Lock()

    def reset(self, op: Any) -> None:
        with self._lock:
            self._clusters.pop(op, None)
            self._match_history.pop(op, None)
            self._active_cluster.pop(op, None)
            self._last_switch_time.pop(op, None)
            self._last_decay_time.pop(op, None)
            self._last_tune_time.pop(op, None)
            self._next_cluster_id.pop(op, None)
            self._observation_count.pop(op, None)
            self._rollout_state.pop(op, None)
        self._feature_extractor.reset(op)

    def extract_features(self, op: Any, op_state: Any) -> Optional[WorkloadFeatures]:
        return self._feature_extractor.extract(op, op_state)

    def observe(
        self,
        *,
        op: Any,
        features: WorkloadFeatures,
        throughput: Optional[float] = None,
    ) -> Optional[SwitchDecision]:
        now = time.time()
        feature_vec = features.to_vector(log_scale=True)

        with self._lock:
            self._maybe_decay(op, now)
            cluster = self._assign_cluster(op, feature_vec, now)
            self._match_history[op].append(cluster.cluster_id)
            if throughput is not None:
                self._observation_count[op] += 1
                self._maybe_record_probe_result(op, cluster, throughput, now)

            self._maybe_enter_tuning(op, cluster, now)

            decision = self._maybe_probe(op, cluster, now)
            if decision is None:
                decision = self._maybe_switch(op, now)
            self._maybe_log_cluster_state(op, now)

        return decision

    def apply_config(self, op: Any, decision: SwitchDecision) -> bool:
        config = decision.config
        if not config:
            return False
        apply_fn = getattr(op, "apply_adaptive_config", None)
        if callable(apply_fn):
            try:
                applied = bool(apply_fn(config))
            except Exception:
                logger.exception("Failed to apply adaptive config for %s", getattr(op, "name", op))
                return False
        else:
            logger.info(
                "Adaptive config ready for %s but no apply hook found: %s",
                getattr(op, "name", op),
                config,
            )
            return False

        if not applied:
            return False

        now = time.time()
        with self._lock:
            cluster = self._get_cluster(op, decision.cluster_id)
            if decision.scope == ConfigApplyScope.PROBE:
                if cluster is None:
                    return applied
                cluster.pending_config = dict(config)
                cluster.pending_observation_count = self._observation_count[op]
                logger.info(
                    "vLLM adaptation: probe config applied for %s cluster %s",
                    getattr(op, "name", op),
                    decision.cluster_id,
                )
            elif decision.scope == ConfigApplyScope.ROLLOUT:
                state = self._rollout_state.get(op)
                if state is not None and state.cluster_id == decision.cluster_id:
                    state.remaining -= 1
                    state.last_update_time = now
                    if state.remaining <= 0:
                        self._rollout_state[op] = None
                        if cluster is not None:
                            cluster.needs_rollout = False
                        logger.info(
                            "vLLM adaptation: rollout completed for %s cluster %s",
                            getattr(op, "name", op),
                            decision.cluster_id,
                        )
        return applied

    def confirm_switch(self, op: Any, decision: SwitchDecision) -> None:
        if decision.scope == ConfigApplyScope.PROBE:
            return
        self._active_cluster[op] = decision.cluster_id
        self._last_switch_time[op] = time.time()

    def _assign_cluster(
        self, op: Any, feature_vec: List[float], now: float
    ) -> WorkloadCluster:
        clusters = self._clusters[op]
        if not clusters:
            cluster = self._create_cluster(op, feature_vec, now)
            clusters.append(cluster)
            logger.info(
                "vLLM adaptation: created initial cluster %s for %s",
                cluster.cluster_id,
                getattr(op, "name", op),
            )
            return cluster

        best_cluster = None
        best_distance = float("inf")
        for cluster in clusters:
            dist = self._distance(feature_vec, cluster.centroid)
            if dist < best_distance:
                best_distance = dist
                best_cluster = cluster

        if best_cluster is None or best_distance > self._distance_threshold:
            cluster = self._create_cluster(op, feature_vec, now)
            clusters.append(cluster)
            logger.info(
                "vLLM adaptation: created new cluster %s for %s (distance=%.3f)",
                cluster.cluster_id,
                getattr(op, "name", op),
                best_distance,
            )
            if len(clusters) > self._max_clusters:
                self._merge_closest_clusters(clusters)
            return cluster

        # Update centroid incrementally
        best_cluster.count += 1.0
        best_cluster.centroid = self._update_centroid(
            best_cluster.centroid, feature_vec, best_cluster.count
        )
        best_cluster.last_updated_time = now
        return best_cluster

    def _create_cluster(
        self, op: Any, feature_vec: List[float], now: float
    ) -> WorkloadCluster:
        cluster_id = self._next_cluster_id[op]
        self._next_cluster_id[op] += 1
        optimizer = VLLMConfigOptimizer()
        return WorkloadCluster(
            cluster_id=cluster_id,
            centroid=list(feature_vec),
            count=1.0,
            status=ClusterStatus.PENDING,
            last_updated_time=now,
            optimizer=optimizer,
        )

    def _merge_closest_clusters(self, clusters: List[WorkloadCluster]) -> None:
        if len(clusters) < 2:
            return
        best_pair = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self._distance(clusters[i].centroid, clusters[j].centroid)
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (i, j)
        if best_pair is None:
            return
        i, j = best_pair
        cluster_a = clusters[i]
        cluster_b = clusters[j]
        total = cluster_a.count + cluster_b.count
        if total <= 0:
            return
        merged_centroid = [
            (cluster_a.centroid[k] * cluster_a.count + cluster_b.centroid[k] * cluster_b.count)
            / total
            for k in range(len(cluster_a.centroid))
        ]
        cluster_a.centroid = merged_centroid
        cluster_a.count = total
        if cluster_b.status == ClusterStatus.TUNED:
            cluster_a.status = cluster_b.status
            cluster_a.config = cluster_b.config
            cluster_a.needs_rollout = cluster_b.needs_rollout
            cluster_a.last_tuned_centroid = cluster_b.last_tuned_centroid
            cluster_a.last_tuned_time = cluster_b.last_tuned_time
        clusters.pop(j)

    def _should_trigger_tuning(
        self, op: Any, cluster: WorkloadCluster, now: float
    ) -> bool:
        if cluster.status == ClusterStatus.TUNING:
            return False
        if now - self._last_tune_time[op] < self._tuning_cooldown_s:
            return False
        total = sum(c.count for c in self._clusters[op])
        if total <= 0:
            return False
        share = cluster.count / total
        if cluster.count < self._min_samples or share < self._min_cluster_fraction:
            return False
        if cluster.status == ClusterStatus.PENDING:
            return True
        if cluster.status == ClusterStatus.TUNED:
            if cluster.last_tuned_centroid is None:
                return True
            drift = self._distance(cluster.centroid, cluster.last_tuned_centroid)
            return drift >= self._centroid_drift_threshold
        return False

    def _maybe_enter_tuning(self, op: Any, cluster: WorkloadCluster, now: float) -> None:
        if not self._should_trigger_tuning(op, cluster, now):
            return
        if cluster.optimizer is None:
            cluster.optimizer = VLLMConfigOptimizer()
        elif cluster.status == ClusterStatus.TUNED:
            cluster.optimizer.reset()
        cluster.status = ClusterStatus.TUNING
        cluster.pending_config = None
        cluster.pending_observation_count = -1
        cluster.needs_rollout = False
        self._last_tune_time[op] = now
        logger.info(
            "vLLM adaptation: tuning triggered for %s cluster %s",
            getattr(op, "name", op),
            cluster.cluster_id,
        )

    def _maybe_record_probe_result(
        self, op: Any, cluster: WorkloadCluster, throughput: float, now: float
    ) -> None:
        if cluster.status != ClusterStatus.TUNING:
            return
        if cluster.pending_config is None or cluster.pending_observation_count < 0:
            return
        if self._observation_count[op] <= cluster.pending_observation_count:
            return
        if cluster.optimizer is None:
            return
        cluster.optimizer.observe(cluster.pending_config, throughput)
        logger.info(
            "vLLM adaptation: probe result for %s cluster %s (step %s/%s, throughput=%.4f)",
            getattr(op, "name", op),
            cluster.cluster_id,
            cluster.optimizer.num_observations,
            self._bo_steps_required,
            throughput,
        )
        cluster.pending_config = None
        cluster.pending_observation_count = -1
        if cluster.optimizer.num_observations >= self._bo_steps_required:
            best_config = cluster.optimizer.best_config
            if best_config:
                cluster.config = best_config
                cluster.status = ClusterStatus.TUNED
                cluster.needs_rollout = True
                cluster.last_tuned_centroid = list(cluster.centroid)
                cluster.last_tuned_time = now
                logger.info(
                    "vLLM adaptation: tuning completed for %s cluster %s after %s steps. best=%s",
                    getattr(op, "name", op),
                    cluster.cluster_id,
                    cluster.optimizer.num_observations,
                    best_config,
                )

    def _maybe_probe(
        self, op: Any, cluster: WorkloadCluster, now: float
    ) -> Optional[SwitchDecision]:
        if cluster.status != ClusterStatus.TUNING:
            return None
        if cluster.pending_config is not None:
            return None
        if cluster.optimizer is None:
            return None
        if cluster.optimizer.num_observations >= self._bo_steps_required:
            return None
        candidate = cluster.optimizer.suggest()
        reason = (
            f"bo probe step {cluster.optimizer.num_observations + 1}"
            f"/{self._bo_steps_required}"
        )
        return SwitchDecision(
            cluster_id=cluster.cluster_id,
            config=candidate,
            reason=reason,
            scope=ConfigApplyScope.PROBE,
        )

    def _maybe_switch(self, op: Any, now: float) -> Optional[SwitchDecision]:
        rollout = self._rollout_state.get(op)
        if rollout is not None:
            cluster = self._get_cluster(op, rollout.cluster_id)
            if cluster is None or cluster.config != rollout.config:
                self._rollout_state[op] = None
            elif rollout.remaining <= 0:
                self._rollout_state[op] = None
            elif now - rollout.last_update_time >= self._rollout_interval_s:
                return SwitchDecision(
                    cluster_id=rollout.cluster_id,
                    config=rollout.config,
                    reason=f"rollout remaining={rollout.remaining}",
                    scope=ConfigApplyScope.ROLLOUT,
                )

        if now - self._last_switch_time[op] < self._cooldown_s:
            return None
        history = self._match_history[op]
        if not history:
            return None
        freq: Dict[int, int] = defaultdict(int)
        for cid in history:
            freq[cid] += 1
        dominant = max(freq.items(), key=lambda item: item[1])
        dominant_id, count = dominant
        if count / len(history) < self._switch_consistency:
            return None
        cluster = self._get_cluster(op, dominant_id)
        if cluster is None or cluster.status != ClusterStatus.TUNED or not cluster.config:
            return None
        active = self._active_cluster[op]
        if active == dominant_id and not cluster.needs_rollout:
            return None

        rollout_count = self._get_actor_count(op)
        if rollout_count <= 0:
            return None
        rollout_state = RolloutState(
            cluster_id=dominant_id,
            config=dict(cluster.config),
            remaining=rollout_count,
            last_update_time=0.0,
        )
        self._rollout_state[op] = rollout_state
        reason = (
            f"dominant cluster {dominant_id} with {count}/{len(history)} matches; "
            f"rollout={rollout_count}"
        )
        logger.info(
            "vLLM adaptation: rollout start for %s -> cluster %s (%s)",
            getattr(op, "name", op),
            dominant_id,
            reason,
        )
        return SwitchDecision(
            cluster_id=dominant_id,
            config=rollout_state.config,
            reason=reason,
            scope=ConfigApplyScope.ROLLOUT,
        )

    def _get_actor_count(self, op: Any) -> int:
        info_fn = getattr(op, "get_actor_info", None)
        if callable(info_fn):
            try:
                info = info_fn()
                running = getattr(info, "running", None)
                if isinstance(running, int):
                    return running
            except Exception:
                pass
        actor_pool = getattr(op, "_actor_pool", None)
        if actor_pool is not None:
            try:
                return int(actor_pool.num_running_actors())
            except Exception:
                return 0
        return 0

    def _get_cluster(self, op: Any, cluster_id: int) -> Optional[WorkloadCluster]:
        for cluster in self._clusters[op]:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def _maybe_decay(self, op: Any, now: float) -> None:
        last = self._last_decay_time[op]
        if now - last < self._decay_interval_s:
            return
        for cluster in self._clusters[op]:
            cluster.count *= self._decay_gamma
        self._clusters[op] = [
            c for c in self._clusters[op] if c.count >= self._prune_threshold
        ]
        self._last_decay_time[op] = now

    def _maybe_log_cluster_state(self, op: Any, now: float) -> None:
        last = self._last_cluster_log_time[op]
        if now - last < self._cluster_log_interval_s:
            return
        self._last_cluster_log_time[op] = now
        clusters = self._clusters[op]
        if not clusters:
            logger.info("vLLM adaptation: cluster counts for %s: none", getattr(op, "name", op))
            return
        summary = ", ".join(
            f"{c.cluster_id}:{c.count:.2f}({c.status.value})" for c in clusters
        )
        logger.info(
            "vLLM adaptation: cluster counts for %s: %s",
            getattr(op, "name", op),
            summary,
        )

    @staticmethod
    def _distance(vec_a: List[float], vec_b: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))

    @staticmethod
    def _update_centroid(
        centroid: List[float], vec: List[float], count: float
    ) -> List[float]:
        if count <= 0:
            return list(vec)
        alpha = 1.0 / max(count, 1.0)
        return [c + alpha * (v - c) for c, v in zip(centroid, vec)]
