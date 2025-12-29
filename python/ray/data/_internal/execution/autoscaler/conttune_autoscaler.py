"""
ContTune Autoscaler - Conservative Bayesian Optimization based autoscaler.

This implements the Small Phase of the ContTune algorithm, which uses
Conservative Bayesian Optimization (CBO) to tune operator parallelism.

The Big Phase is not implemented as Ray Data doesn't have a backpressure concept.

Reference: ContTune paper - Continuous Tuning for Stream Processing Systems.
"""
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import ray
from .autoscaler import Autoscaler
from .autoscaling_actor_pool import ActorPoolScalingRequest
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = logging.getLogger(__name__)


@dataclass
class HistoricalObservation:
    """Stores historical observations for an operator."""
    parallelism: int
    processing_ability: float  # rows/s
    timestamp: float


class OperatorHistory:
    """Manages historical observations for an operator with Top-K technique."""

    def __init__(self, k: int = 10):
        self.k = k
        self.observations: Dict[int, List[HistoricalObservation]] = defaultdict(list)

    def add_observation(self, parallelism: int, processing_ability: float, timestamp: float):
        """Add a new observation."""
        obs = HistoricalObservation(parallelism, processing_ability, timestamp)
        self.observations[parallelism].append(obs)
        # Keep only K most recent observations per parallelism level
        if len(self.observations[parallelism]) > self.k:
            self.observations[parallelism] = sorted(
                self.observations[parallelism],
                key=lambda x: x.timestamp,
                reverse=True
            )[:self.k]

    def get_mean_processing_ability(self, parallelism: int) -> Optional[float]:
        """Get mean-reverted processing ability for a parallelism level."""
        if parallelism not in self.observations or not self.observations[parallelism]:
            return None
        abilities = [obs.processing_ability for obs in self.observations[parallelism]]
        return float(np.mean(abilities))

    def get_all_observations(self) -> List[Tuple[int, float]]:
        """Get all observations as (parallelism, processing_ability) pairs."""
        result = []
        for parallelism in self.observations:
            pa = self.get_mean_processing_ability(parallelism)
            if pa is not None:
                result.append((parallelism, pa))
        return sorted(result, key=lambda x: x[0])

    def get_max_parallelism(self) -> int:
        """Get maximum observed parallelism."""
        if not self.observations:
            return 1
        return max(self.observations.keys())

    def get_observed_parallelisms(self) -> List[int]:
        """Get list of observed parallelism levels."""
        return sorted(self.observations.keys())


class ConservativeBayesianOptimization:
    """
    Conservative Bayesian Optimization (CBO) for parallelism tuning.
    Combines Gaussian Process with conservative exploration using DS2-like methods.
    """

    def __init__(self, alpha: int = 3, max_parallelism: int = 90):
        self.alpha = alpha  # Threshold for scoring function
        self.max_parallelism = max_parallelism

        # GP configuration
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True
        )
        self.is_fitted = False

    def fit(self, history: OperatorHistory):
        """Fit GP model on historical observations."""
        observations = history.get_all_observations()
        if len(observations) < 2:
            self.is_fitted = False
            return

        X = np.array([[p] for p, _ in observations])
        y = np.array([pa for _, pa in observations])

        self.gp.fit(X, y)
        self.is_fitted = True

    def predict(self, parallelism: int) -> Tuple[float, float]:
        """Predict processing ability with mean and std."""
        if not self.is_fitted:
            return 0.0, float('inf')

        X = np.array([[parallelism]])
        mean, std = self.gp.predict(X, return_std=True)
        return float(mean[0]), float(std[0])

    def acquisition_function(self, parallelism: int, upstream_rate: float,
                             p_max: int) -> float:
        """
        Custom acquisition function (Equation 5 in paper):
        arg max (p* - p_i) * I(μ(p_i) - λ_i)
        """
        mean, _ = self.predict(parallelism)

        # Indicator function: 1 if mean >= upstream_rate, 0 otherwise
        if mean >= upstream_rate:
            return p_max - parallelism
        return -float('inf')

    def get_acquisition_suggestion(self, upstream_rate: float, p_max: int) -> int:
        """Find parallelism that maximizes acquisition function."""
        best_parallelism = p_max
        best_value = -float('inf')

        for p in range(1, p_max + 1):
            value = self.acquisition_function(p, upstream_rate, p_max)
            if value > best_value:
                best_value = value
                best_parallelism = p

        return best_parallelism

    def get_nearest_distance(self, parallelism: int, history: OperatorHistory) -> int:
        """Calculate nearest distance to observed parallelism levels."""
        observed = history.get_observed_parallelisms()
        if not observed:
            return float('inf')

        distances = [abs(parallelism - p) for p in observed]
        return min(distances)

    def ds2_linear_suggestion(self, upstream_rate: float, current_parallelism: int,
                              current_processing_ability: float) -> int:
        """
        DS2-style linear scaling suggestion (conservative exploration).
        Based on Equation 7 from DS2 paper.
        """
        if current_processing_ability <= 0:
            return current_parallelism

        # Estimate required parallelism
        # Note: processing_ability is already the total processing capacity,
        # so we directly use upstream_rate / processing_ability
        required = int(math.ceil(upstream_rate / current_processing_ability))
        return max(1, min(required, self.max_parallelism))

    def suggest(self, upstream_rate: float, current_parallelism: int,
                current_processing_ability: float, history: OperatorHistory,
                p_max: int) -> int:
        """
        Main CBO suggestion method.
        Balances fast exploitation (GP) and conservative exploration (DS2).
        """
        self.fit(history)

        # Get acquisition-based suggestion
        if self.is_fitted:
            p_acq = self.get_acquisition_suggestion(upstream_rate, p_max)
        else:
            p_acq = p_max

        # Calculate nearest distance (scoring function)
        d_nearest = self.get_nearest_distance(p_acq, history)

        # Decision: use acquisition or DS2-based suggestion
        if d_nearest <= self.alpha:
            # Fast exploitation: use GP-based suggestion
            return p_acq
        else:
            # Conservative exploration: use DS2-like linear method
            return self.ds2_linear_suggestion(
                upstream_rate, current_parallelism, current_processing_ability
            )


class ContTuneAutoscaler(Autoscaler):
    """
    ContTune autoscaler implementation using Conservative Bayesian Optimization.

    This implements the Small Phase of the ContTune algorithm.
    Big Phase is not implemented as Ray Data doesn't have backpressure concept.

    Key assumptions:
    1. Linear pipeline (not DAG)
    2. Only considers ActorPoolMapOperator
    3. Source rate = first ActorPoolMapOperator's output rate
    4. First ActorPoolMapOperator's parallelism is not modified
    """

    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 60

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
        alpha: int = 3,  # Threshold for CBO scoring function
        max_parallelism: int = 90,
        k: int = 10,  # Number of recent observations to keep per parallelism level
        use_incremental_output_rate: bool = False,
    ):
        super().__init__(topology, resource_manager, execution_id)
        self._last_scaling_time = time.time()
        self._alpha = alpha
        self._max_parallelism = max_parallelism
        self._k = k
        # If True, use incremental output_rate (delta_rows / observation_interval)
        # If False (default), use cumulative output_rate (total_rows / total_time)
        self._use_incremental_output_rate = use_incremental_output_rate
        # Track start time for cumulative output_rate calculation
        self._start_time: float = time.time()
        # Track last observation time and metrics for incremental output_rate calculation
        self._last_observation_time: float = time.time()
        # Dict: op_name -> last_rows_output
        self._last_op_metrics: dict = {}

        # Per-operator state
        self._operator_histories: Dict[str, OperatorHistory] = {}
        self._operator_cbo: Dict[str, ConservativeBayesianOptimization] = {}
        self._p_max_global: int = 1  # Maximum parallelism observed across all operators

    def try_trigger_scaling(self):
        """Try to trigger ContTune autoscaling."""
        self._conttune_scaling()

    def on_executor_shutdown(self):
        """Called when the executor is shutting down."""
        logger.info("ContTune autoscaler shutting down.")

    def get_total_resources(self) -> ExecutionResources:
        cluster_res = ray.cluster_resources()
        if "NPU" in cluster_res:
            cluster_res["GPU"] = cluster_res["NPU"]
        return ExecutionResources.from_resource_dict(cluster_res)

    def _get_or_create_history(self, operator_name: str) -> OperatorHistory:
        """Get or create history tracker for an operator."""
        if operator_name not in self._operator_histories:
            self._operator_histories[operator_name] = OperatorHistory(k=self._k)
        return self._operator_histories[operator_name]

    def _get_or_create_cbo(self, operator_name: str) -> ConservativeBayesianOptimization:
        """Get or create CBO instance for an operator."""
        if operator_name not in self._operator_cbo:
            self._operator_cbo[operator_name] = ConservativeBayesianOptimization(
                alpha=self._alpha,
                max_parallelism=self._max_parallelism
            )
        return self._operator_cbo[operator_name]

    def _conttune_scaling(self):
        """Perform ContTune autoscaling (Small Phase only)."""
        now = time.time()
        if now - self._last_scaling_time < self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            logger.debug(
                f"Skipping ContTune autoscaling: only {now - self._last_scaling_time:.1f}s "
                f"since last scaling (min gap: {self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS}s)"
            )
            return

        # Collect ActorPoolMapOperators in order (linear pipeline)
        operators: List[ActorPoolMapOperator] = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                operators.append(op)

        n = len(operators)
        if n == 0:
            logger.debug("No ActorPoolMapOperators found. Skipping ContTune autoscaling.")
            return

        # Check if all operators have processed data
        if not all(
            op._metrics.block_generation_time > 0
            and op._metrics.rows_task_inputs_processed > 0
            and op._metrics.rows_task_outputs_generated > 0
            for op in operators
        ):
            logger.debug("Not all operators have processed data yet. Skipping ContTune autoscaling.")
            return

        # Collect metrics and compute optimal parallelism
        metrics = self._collect_operator_metrics(operators)
        optimal_parallelism = self._compute_conttune_parallelism(metrics, now)

        logger.info(f"ContTune autoscaling: target parallelism = {optimal_parallelism}")

        # Apply scaling (skip the first operator)
        for i, op in enumerate(operators):
            if i == 0:
                # First operator's parallelism is not modified
                continue
            target = optimal_parallelism[i]
            self._scale_operator(op, target)

        self._last_scaling_time = now

    def _collect_operator_metrics(self, operators: List[ActorPoolMapOperator]) -> List[dict]:
        """Collect metrics for each operator.

        For ContTune, we need:
        - output_rate: rows output per second
          - If use_incremental_output_rate: delta_rows / observation_interval
          - Otherwise (default): total_rows / total_time_since_start
        - processing_ability: total processing capacity (rows/s) at current parallelism
        - current_parallelism: current number of actors
        - rows_input: total input rows processed
        - rows_output: total output rows generated
        - operator_name: name for history tracking

        Returns:
            List of metrics dict for each operator.
        """
        now = time.time()
        total_time = now - self._start_time
        observation_interval = now - self._last_observation_time

        metrics = []
        for op in operators:
            # Get current parallelism from actor pool
            actor_pools = op.get_autoscaling_actor_pools()
            current_parallelism = actor_pools[0].current_size() if actor_pools else 1

            # Get current metric values
            rows_output = op._metrics.rows_task_outputs_generated
            rows_input = op._metrics.rows_task_inputs_processed
            wall_time = op._metrics.block_generation_time

            # Calculate output_rate
            if self._use_incremental_output_rate:
                # Incremental: delta_rows / observation_interval
                last_rows_output = self._last_op_metrics.get(op.name, 0)
                delta_rows_output = rows_output - last_rows_output
                output_rate = delta_rows_output / observation_interval if observation_interval > 0 else 0.0
                # Update last observed rows_output for next call
                self._last_op_metrics[op.name] = rows_output
            else:
                # Cumulative (default): total_rows / total_time_since_start
                output_rate = rows_output / total_time if total_time > 0 else 0.0

            # Processing ability = total rows processed / wall_time (this is total capacity)
            processing_ability = rows_input / wall_time if wall_time > 0 else 0.0

            metrics.append({
                "output_rate": output_rate,
                "processing_ability": processing_ability,
                "current_parallelism": current_parallelism,
                "rows_input": rows_input,
                "rows_output": rows_output,
                "operator_name": op.name,
            })

            logger.info(
                f"Operator {op.name}: output_rate={output_rate:.2f}, "
                f"processing_ability={processing_ability:.2f}, "
                f"current_parallelism={current_parallelism}, "
                f"rows_input={rows_input}, rows_output={rows_output}"
            )

        # Update last observation time (for incremental mode)
        self._last_observation_time = now

        return metrics



    def _compute_conttune_parallelism(self, metrics: List[dict], timestamp: float) -> List[int]:
        """Compute optimal parallelism using ContTune's Conservative Bayesian Optimization.

        For each operator (except the first):
        1. Add current observation to history
        2. Use CBO to suggest optimal parallelism based on upstream rate
        3. If GP suggestion is too far from observed data, fall back to DS2

        Args:
            metrics: List of metrics dict for each operator.
            timestamp: Current timestamp for history tracking.

        Returns:
            List of optimal parallelism for each operator.
        """
        n = len(metrics)
        optimal = []

        # Source rate = first operator's output rate
        source_rate = metrics[0]["output_rate"]

        # First operator keeps its current parallelism
        optimal.append(metrics[0]["current_parallelism"])

        # Update p_max_global based on first operator
        self._p_max_global = max(self._p_max_global, metrics[0]["current_parallelism"])

        # Track cumulative selectivity for computing upstream rate
        output_rate_star = [source_rate]

        for i in range(1, n):
            m = metrics[i]
            operator_name = m["operator_name"]
            current_parallelism = m["current_parallelism"]
            processing_ability = m["processing_ability"]
            rows_input = m["rows_input"]
            rows_output = m["rows_output"]

            # Get or create history and CBO for this operator
            history = self._get_or_create_history(operator_name)
            cbo = self._get_or_create_cbo(operator_name)

            # Add current observation to history
            if processing_ability > 0:
                history.add_observation(current_parallelism, processing_ability, timestamp)

            # Update global p_max
            self._p_max_global = max(self._p_max_global, history.get_max_parallelism())

            # Upstream rate for this operator = previous operator's output_rate_star
            upstream_rate = output_rate_star[i - 1]

            # Get CBO suggestion
            pi = cbo.suggest(
                upstream_rate=upstream_rate,
                current_parallelism=current_parallelism,
                current_processing_ability=processing_ability,
                history=history,
                p_max=self._p_max_global
            )

            # Clamp to valid range
            pi = max(1, min(pi, self._max_parallelism))
            optimal.append(pi)

            # Compute this operator's output_rate_star for downstream
            # selectivity = rows_output / rows_input
            if rows_input > 0:
                selectivity = rows_output / rows_input
                output_rate_star.append(selectivity * upstream_rate)
            else:
                output_rate_star.append(0.0)

            logger.debug(
                f"Operator {i} ({operator_name}): upstream_rate={upstream_rate:.2f}, "
                f"processing_ability={processing_ability:.2f}, "
                f"selectivity={rows_output/rows_input if rows_input > 0 else 0:.4f}, "
                f"optimal_parallelism={pi}"
            )

        return optimal

    def _scale_operator(self, op: ActorPoolMapOperator, target_concurrency: int):
        """Scale an operator's actor pool to match the target concurrency.

        Args:
            op: The ActorPoolMapOperator to scale.
            target_concurrency: The target number of actors (concurrency level).
        """
        actor_pools = op.get_autoscaling_actor_pools()
        if not actor_pools:
            logger.warning(f"Operator {op.name} has no autoscaling actor pools.")
            return

        # Assume single actor pool per operator
        actor_pool = actor_pools[0]
        current_size = actor_pool.current_size()

        delta = target_concurrency - current_size
        logger.info(
            f"ContTune - Operator {op.name}: current_size={current_size}, "
            f"target_concurrency={target_concurrency}, delta={delta}"
        )

        if delta == 0:
            logger.debug(
                f"Operator {op.name}: no scaling needed "
                f"(current={current_size}, target={target_concurrency})"
            )
            return

        if delta > 0:
            # Scale up
            logger.info(
                f"Operator {op.name}: scaling up by {delta} actors "
                f"(current={current_size}, target={target_concurrency})"
            )
            actor_pool.scale(
                ActorPoolScalingRequest(
                    delta=delta,
                    reason=f"ContTune autoscaling to target concurrency {target_concurrency}"
                )
            )
        else:
            # Scale down
            num_to_remove = abs(delta)
            logger.info(
                f"Operator {op.name}: scaling down by {num_to_remove} actors "
                f"(current={current_size}, target={target_concurrency})"
            )

            # Use my_scale_down if available
            if hasattr(actor_pool, 'my_scale_down'):
                num_removed, num_marked = actor_pool.my_scale_down(
                    target_num_actors=num_to_remove,
                    forced=False
                )
                logger.info(
                    f"Operator {op.name}: scale down result - "
                    f"removed {num_removed} actors immediately, "
                    f"marked {num_marked} actors for removal"
                )
            else:
                # Fallback to regular scale method
                actor_pool.scale(
                    ActorPoolScalingRequest(
                        delta=-num_to_remove,
                        reason=f"ContTune autoscaling to target concurrency {target_concurrency}"
                    )
                )