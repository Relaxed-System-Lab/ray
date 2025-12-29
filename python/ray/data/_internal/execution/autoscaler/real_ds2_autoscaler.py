"""
Real DS2 Autoscaler - Original DS2 algorithm implementation.

This implements the original DS2 algorithm without resource constraints.
It uses an online algorithm based on source rate to compute the required
parallelism for each operator.

Reference: DS2 paper - linearity-based scaling for streaming dataflow systems.
"""
import logging
import math
import time
from typing import TYPE_CHECKING, List

import ray
from .autoscaler import Autoscaler
from .autoscaling_actor_pool import ActorPoolScalingRequest
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = logging.getLogger(__name__)


class RealDS2Autoscaler(Autoscaler):
    """
    Original DS2 autoscaler implementation.
    
    Based on the DS2 paper: linearity-based scaling.
    
    Key assumptions:
    1. Linear pipeline (not DAG)
    2. Only considers ActorPoolMapOperator
    3. Source rate = first ActorPoolMapOperator's output rate
    4. First ActorPoolMapOperator's parallelism is not modified
    5. Processing capacity scales linearly with parallelism
    """
    
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 60
    
    # EMA smoothing factor (alpha) for throughput calculation.
    EMA_ALPHA = 0.5

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
        max_parallelism: int = 90,
        use_incremental_output_rate: bool = False,
    ):
        super().__init__(topology, resource_manager, execution_id)
        self._last_scaling_time = time.time()
        self._max_parallelism = max_parallelism
        # If True, use incremental output_rate (delta_rows / observation_interval)
        # If False (default), use cumulative output_rate (total_rows / total_time)
        self._use_incremental_output_rate = use_incremental_output_rate
        # Track start time for cumulative output_rate calculation
        self._start_time: float = time.time()
        # Track last observation time and metrics for incremental output_rate calculation
        self._last_observation_time: float = time.time()
        # Dict: op_name -> last_rows_output
        self._last_op_metrics: dict = {}

    def try_trigger_scaling(self):
        """Try to trigger DS2 autoscaling."""
        self._ds2_scaling()

    def on_executor_shutdown(self):
        """Called when the executor is shutting down."""
        logger.info("Real DS2 autoscaler shutting down.")

    def get_total_resources(self) -> ExecutionResources:
        cluster_res = ray.cluster_resources()
        if "NPU" in cluster_res:
            cluster_res["GPU"] = cluster_res["NPU"]
        return ExecutionResources.from_resource_dict(cluster_res)

    def _ds2_scaling(self):
        """Perform original DS2 autoscaling.
        
        DS2 Algorithm (Equation 7 from paper):
        For each operator i (except source):
            1. Get upstream output rate (which becomes this operator's input rate)
            2. Calculate processing ability per instance = total_processing_ability / current_parallelism
            3. Optimal parallelism = ceil(upstream_rate / processing_ability_per_instance)
        """
        now = time.time()
        if now - self._last_scaling_time < self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            logger.debug(
                f"Skipping DS2 autoscaling: only {now - self._last_scaling_time:.1f}s "
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
            logger.debug("No ActorPoolMapOperators found. Skipping DS2 autoscaling.")
            return

        # Check if all operators have processed data
        if not all(
            op._metrics.block_generation_time > 0
            and op._metrics.rows_task_inputs_processed > 0
            and op._metrics.rows_task_outputs_generated > 0
            for op in operators
        ):
            logger.debug("Not all operators have processed data yet. Skipping DS2 autoscaling.")
            return

        # Collect metrics for each operator
        # For DS2, we need: output_rate, input_rate (upstream_data_rate), processing_ability, current_parallelism
        metrics = self._collect_operator_metrics(operators)
        
        # Compute optimal parallelism using DS2 algorithm
        optimal_parallelism = self._compute_ds2_parallelism(metrics)

        logger.info(f"Real DS2 autoscaling: target parallelism = {optimal_parallelism}")

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

        For DS2, we need:
        - output_rate: rows output per second
          - If use_incremental_output_rate: delta_rows / observation_interval
          - Otherwise (default): total_rows / total_time_since_start
        - processing_ability: total processing capacity (rows/s) at current parallelism
        - current_parallelism: current number of actors
        - rows_input: total input rows processed
        - rows_output: total output rows generated

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

    def _compute_ds2_parallelism(self, metrics: List[dict]) -> List[int]:
        """Compute optimal parallelism using original DS2 algorithm.

        DS2 Algorithm (based on Equation 7 from paper):
        1. Source rate = first operator's output rate
        2. For each subsequent operator i:
           - upstream_rate = selectivity_chain * source_rate
           - processing_ability_per_instance = processing_ability / current_parallelism
           - optimal_parallelism = ceil(upstream_rate / processing_ability_per_instance)

        Args:
            metrics: List of metrics dict for each operator.

        Returns:
            List of optimal parallelism for each operator.
        """
        n = len(metrics)
        optimal = []

        # Source rate = first operator's output rate
        source_rate = metrics[0]["output_rate"]

        # First operator keeps its current parallelism
        optimal.append(metrics[0]["current_parallelism"])

        # Track cumulative selectivity for computing upstream rate
        # output_rate_star[i] represents the "true" output rate at operator i
        # considering the source rate and selectivity chain
        output_rate_star = [source_rate]

        for i in range(1, n):
            m = metrics[i]
            current_parallelism = m["current_parallelism"]
            processing_ability = m["processing_ability"]
            rows_input = m["rows_input"]
            rows_output = m["rows_output"]

            # Upstream rate for this operator = previous operator's output_rate_star
            upstream_rate = output_rate_star[i - 1]

            # Processing ability per instance
            if current_parallelism > 0 and processing_ability > 0:
                # pa_per_instance = processing_ability / current_parallelism

                # Optimal parallelism (Equation 7)
                pi = int(math.ceil(upstream_rate / processing_ability))
                pi = max(1, min(pi, self._max_parallelism))
            else:
                pi = current_parallelism

            optimal.append(pi)

            # Compute this operator's output_rate_star for downstream
            # selectivity = rows_output / rows_input
            if rows_input > 0:
                selectivity = rows_output / rows_input
                output_rate_star.append(selectivity * upstream_rate)
            else:
                # output_rate_star.append(0.0)
                raise ValueError(f"Operator {i} has zero input rows, cannot compute selectivity.")

            logger.debug(
                f"Operator {i}: upstream_rate={upstream_rate:.2f}, "
                f"processing_ability={processing_ability}, "
                f"selectivity={rows_output/rows_input}, "
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
            f"Real DS2 - Operator {op.name}: current_size={current_size}, "
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
                    reason=f"Real DS2 autoscaling to target concurrency {target_concurrency}"
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
                        reason=f"Real DS2 autoscaling to target concurrency {target_concurrency}"
                    )
                )
