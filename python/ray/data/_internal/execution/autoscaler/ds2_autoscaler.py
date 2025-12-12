import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import ray
from .autoscaler import Autoscaler
from .autoscaling_actor_pool import ActorPoolScalingRequest
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator
from ray.data._internal.execution.autoscaler.ds2_milp_solver import milp_solver

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology
    from ray.data._internal.execution.autoscaler.vllm_throughput_predictor import (
        VLLMThroughputPredictor,
    )

logger = logging.getLogger(__name__)


class DS2Autoscaler(Autoscaler):
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 60

    # EMA smoothing factor (alpha) for throughput calculation.
    # Higher values (closer to 1.0) give more weight to recent observations.
    # Lower values (closer to 0.0) give more weight to historical data.
    EMA_ALPHA = 0.5

    # Name pattern to identify vLLM operators
    VLLM_OPERATOR_NAME_PATTERN = "vLLM"

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
    ):
        super().__init__(topology, resource_manager, execution_id)

        # Last time when DS2 scaling was triggered.
        self._last_scaling_time = time.time()

        # vLLM throughput predictors (one per vLLM operator)
        # Only initialized when enable_vllm_throughput_prediction is True
        self._vllm_predictors: Dict[str, "VLLMThroughputPredictor"] = {}
        self._vllm_collectors: Dict[str, Any] = {}  # VLLMMetricsCollector instances
        self._vllm_prediction_enabled = self._init_vllm_prediction()

    def try_trigger_scaling(self):
        """Try to trigger DS2 autoscaling."""
        self.ds2_scaling()

    def on_executor_shutdown(self):
        """Called when the executor is shutting down.

        DS2 autoscaler doesn't need to clean up external resources since it
        directly manages actor pools without using Ray's autoscaler.
        """
        logger.info("DS2 autoscaler shutting down.")

    def get_total_resources(self) -> ExecutionResources:
        cluster_res = ray.cluster_resources()
        # NPU priority: if NPU exists, use it as GPU
        if "NPU" in cluster_res:
            cluster_res["GPU"] = cluster_res["NPU"]
        return ExecutionResources.from_resource_dict(cluster_res)

    def ds2_scaling(self):
        """Perform DS2 autoscaling based on MILP solver results.

        This method:
        1. Checks if enough time has passed since last scaling
        2. Collects metrics from all ActorPoolMapOperators
        3. Calls MILP solver to get optimal concurrency for each operator
        4. Scales each operator's actor pool to match the target concurrency
        """
        # Check frequency limit
        now = time.time()
        if now - self._last_scaling_time < self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            logger.debug(
                f"Skipping DS2 autoscaling: only {now - self._last_scaling_time:.1f}s "
                f"since last scaling (min gap: {self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS}s)"
            )
            return

        # First, check if all operators have started processing
        # Use total cumulative values (without updating EMA) to avoid polluting EMA during cold start
        wall_time_list = self.get_wall_time()
        logger.debug(f"Total wall time list: {wall_time_list}")

        n = len(wall_time_list)
        if n == 0:
            logger.debug("No ActorPoolMapOperators found. Skipping DS2 autoscaling.")
            return

        # Check if all operators have processed data
        all_work = all(wall_time > 0 for wall_time in wall_time_list)

        if not all_work:
            logger.debug(
                "Not all operators have processed data yet. Skipping DS2 autoscaling."
            )
            return

        # Collect vLLM throughput samples (if enabled)
        self._collect_vllm_samples()

        # All operators are working, now collect EMA-smoothed metrics
        # This updates the EMA values and last snapshots
        ema_wall_time_list = self.get_ema_wall_time()
        logger.debug(f"EMA wall time list: {ema_wall_time_list}")
        ema_num_processed_rows_list = self.get_ema_num_processed_rows()
        logger.debug(f"EMA num processed rows list: {ema_num_processed_rows_list}")

        per_actor_resource_usage_list = self.get_per_actor_resource_usage()
        logger.debug(f"Per actor resource usage list: {per_actor_resource_usage_list}")
        total_resources = self.get_total_resources()
        logger.debug(f"Total resources: {total_resources}")

        # Calculate unit throughput for each operator
        # For vLLM operators with prediction enabled, use the predictor
        # Otherwise, use the standard EMA-based calculation
        unit_throughput_list = []
        op_index = 0
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                wall_time = ema_wall_time_list[op_index]
                num_rows = ema_num_processed_rows_list[op_index]

                if wall_time <= 0:
                    # This should not happen since we already checked all_work
                    logger.warning(
                        f"EMA wall time is 0 for operator {op.name}. "
                        f"Skipping DS2 autoscaling."
                    )
                    raise ValueError("EMA wall time is 0 for an operator")

                ut = self._get_unit_throughput_for_op(op, wall_time, num_rows)
                unit_throughput_list.append(ut)
                op_index += 1

        # Extract CPU and GPU usage
        cpu_usage_list = []
        for per_actor_resource_usage in per_actor_resource_usage_list:
            if per_actor_resource_usage._cpu is None:
                cpu_usage_list.append(0)
            else:
                cpu_usage_list.append(per_actor_resource_usage._cpu)

        gpu_usage_list = []
        for per_actor_resource_usage in per_actor_resource_usage_list:
            if per_actor_resource_usage._gpu is None:
                gpu_usage_list.append(0)
            else:
                gpu_usage_list.append(per_actor_resource_usage._gpu)

        # Get D_o (total rows output generated)
        D_o = None
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                D_o = op._metrics.rows_task_outputs_generated

        N_cpu = total_resources._cpu
        N_gpu = total_resources._gpu

        # Call MILP solver to get optimal concurrency
        concurrency_list = milp_solver(
            n, unit_throughput_list, cpu_usage_list, gpu_usage_list,
            ema_num_processed_rows_list, D_o, N_cpu, N_gpu,
        )

        if concurrency_list is None:
            logger.warning("MILP solver returned None. Skipping DS2 autoscaling.")
            raise ValueError("MILP solver returned None.")

        logger.info(
            f"DS2 autoscaling: target concurrency = {concurrency_list}, "
            f"total resources: CPU={N_cpu}, GPU={N_gpu}"
        )

        # Apply scaling to each operator
        op_index = 0
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                if op_index >= len(concurrency_list):
                    logger.warning(
                        f"Not enough concurrency values from MILP solver. "
                        f"Expected at least {op_index + 1}, got {len(concurrency_list)}"
                    )
                    break

                target_concurrency = concurrency_list[op_index]
                self._scale_operator(op, target_concurrency)

                op_index += 1

        # Update last scaling time
        self._last_scaling_time = now

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
            f"Operator {op.name}: current_size={current_size}, "
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
                    reason=f"DS2 autoscaling to target concurrency {target_concurrency}"
                )
            )
        else:
            # Scale down using my_scale_down
            num_to_remove = abs(delta)
            logger.info(
                f"Operator {op.name}: scaling down by {num_to_remove} actors "
                f"(current={current_size}, target={target_concurrency})"
            )

            # Use my_scale_down to ensure actors are removed
            # forced=False means we wait for tasks to complete before killing actors
            # Check if actor_pool has my_scale_down method (it's only in _ActorPool)
            if hasattr(actor_pool, 'my_scale_down'):
                num_removed, num_marked = actor_pool.my_scale_down(  # type: ignore
                    target_num_actors=num_to_remove,
                    forced=False
                )

                logger.info(
                    f"Operator {op.name}: scale down result - "
                    f"removed {num_removed} actors immediately, "
                    f"marked {num_marked} actors for removal"
                )
            else:
                raise ValueError(
                    f"Actor pool for operator {op.name} does not support my_scale_down method.")
                # Fallback to regular scale method
                # logger.warning(
                #     f"Operator {op.name}: actor pool does not support my_scale_down, "
                #     f"using regular scale method"
                # )
                # actor_pool.scale(
                #     ActorPoolScalingRequest(
                #         delta=-num_to_remove,
                #         reason=f"DS2 autoscaling to target concurrency {target_concurrency}"
                #     )
                # )

    def get_wall_time(self) -> List[float]:
        """Get total cumulative wall time for each operator.

        Returns the total block_generation_time (cumulative since operator start).
        This is used to check if operators have started processing (value > 0).
        Does NOT update any metrics.

        Returns:
            List of total wall time values for each ActorPoolMapOperator.
        """
        wall_time_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                wall_time_list.append(op._metrics.block_generation_time)
        return wall_time_list

    def get_num_processed_rows(self) -> List[int]:
        """Get total cumulative number of processed rows for each operator.

        Returns the total rows_task_inputs_processed (cumulative since operator start).
        This is used to check if operators have started processing (value > 0).
        Does NOT update any metrics.

        Returns:
            List of total processed row counts for each ActorPoolMapOperator.
        """
        num_processed_rows_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                num_processed_rows_list.append(op._metrics.rows_task_inputs_processed)
        return num_processed_rows_list

    def get_ema_wall_time(self) -> List[float]:
        """Get EMA-smoothed wall time for each operator and update metrics.

        Computes the exponential moving average (EMA) of wall time deltas
        to smooth out short-term fluctuations and provide a more stable
        estimate of operator throughput.

        This method should only be called after all operators have started
        processing (i.e., after all_work check passes).

        Returns:
            List of EMA-smoothed wall time values for each ActorPoolMapOperator.
        """
        wall_time_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                current_time = op._metrics.block_generation_time
                last_time = op._metrics.last_block_generation_time

                # Calculate the time delta since last call
                time_delta = current_time - last_time

                # Apply EMA smoothing
                if op._metrics.ema_wall_time == 0.0:
                    # First time: initialize EMA with current delta
                    ema_time = time_delta
                else:
                    # EMA formula: EMA_new = α * current + (1 - α) * EMA_old
                    ema_time = (
                        self.EMA_ALPHA * time_delta +
                        (1 - self.EMA_ALPHA) * op._metrics.ema_wall_time
                    )

                # Update metrics
                op._metrics.ema_wall_time = ema_time
                op._metrics.last_block_generation_time = current_time

                wall_time_list.append(ema_time)

        return wall_time_list

    def get_ema_num_processed_rows(self) -> List[int]:
        """Get EMA-smoothed number of processed rows for each operator and update metrics.

        Computes the exponential moving average (EMA) of processed rows deltas
        to smooth out short-term fluctuations and provide a more stable
        estimate of operator throughput.

        This method should only be called after all operators have started
        processing (i.e., after all_work check passes).

        Returns:
            List of EMA-smoothed processed row counts for each ActorPoolMapOperator.
        """
        num_processed_rows_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                current_rows = op._metrics.rows_task_inputs_processed
                last_rows = op._metrics.last_rows_task_inputs_processed

                # Calculate the rows delta since last call
                rows_delta = current_rows - last_rows

                # Apply EMA smoothing
                if op._metrics.ema_processed_rows == 0.0:
                    # First time: initialize EMA with current delta
                    ema_rows = float(rows_delta)
                else:
                    # EMA formula: EMA_new = α * current + (1 - α) * EMA_old
                    ema_rows = (
                        self.EMA_ALPHA * rows_delta +
                        (1 - self.EMA_ALPHA) * op._metrics.ema_processed_rows
                    )

                # Update metrics
                op._metrics.ema_processed_rows = ema_rows
                op._metrics.last_rows_task_inputs_processed = current_rows

                # Return as integer (rounded)
                num_processed_rows_list.append(int(round(ema_rows)))

        return num_processed_rows_list

    def get_per_actor_resource_usage(self) -> List[ExecutionResources]:
        per_actor_resource_usage_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                per_actor_resource_usage_list.append(
                    op.get_per_actor_resource_usage()
                )
        return per_actor_resource_usage_list

    # ========== vLLM Throughput Prediction Methods ==========

    def _init_vllm_prediction(self) -> bool:
        """Initialize vLLM throughput prediction if enabled.

        Returns:
            True if vLLM prediction is enabled, False otherwise.
        """
        from ray.data.context import DataContext

        ctx = DataContext.get_current()
        if not ctx.enable_vllm_throughput_prediction:
            return False

        # Lazy import to avoid sklearn dependency when feature is disabled
        from ray.data._internal.execution.autoscaler.vllm_throughput_predictor import (
            VLLMThroughputPredictor,
            VLLMMetricsCollector,
        )

        # Initialize predictors and collectors for each vLLM operator
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                if self._is_vllm_operator(op):
                    self._vllm_predictors[op.name] = VLLMThroughputPredictor()
                    self._vllm_collectors[op.name] = VLLMMetricsCollector()
                    logger.info(
                        f"Initialized vLLM throughput predictor for operator: {op.name}"
                    )

        if self._vllm_predictors:
            logger.info(
                f"vLLM throughput prediction enabled for {len(self._vllm_predictors)} "
                f"operators"
            )
            return True
        else:
            logger.debug("No vLLM operators found, prediction disabled")
            return False

    def _is_vllm_operator(self, op: ActorPoolMapOperator) -> bool:
        """Check if an operator is a vLLM operator by its name.

        Args:
            op: The operator to check.

        Returns:
            True if the operator is a vLLM operator, False otherwise.
        """
        return self.VLLM_OPERATOR_NAME_PATTERN in op.name

    def _get_vllm_predictor(
        self, op: ActorPoolMapOperator
    ) -> Optional["VLLMThroughputPredictor"]:
        """Get the vLLM throughput predictor for an operator.

        Args:
            op: The operator.

        Returns:
            The predictor if available, None otherwise.
        """
        if not self._vllm_prediction_enabled:
            return None
        return self._vllm_predictors.get(op.name)

    def _get_unit_throughput_for_op(
        self,
        op: ActorPoolMapOperator,
        ema_wall_time: float,
        ema_num_rows: int,
    ) -> float:
        """Get unit throughput for an operator.

        For vLLM operators with prediction enabled, uses the predictor.
        Otherwise, uses the standard EMA-based calculation.

        Args:
            op: The operator.
            ema_wall_time: EMA-smoothed wall time.
            ema_num_rows: EMA-smoothed number of processed rows.

        Returns:
            The unit throughput value.
        """
        # Default calculation
        if ema_wall_time > 0:
            default_ut = ema_num_rows / ema_wall_time
        else:
            default_ut = 0.0

        # Check if vLLM prediction is available
        predictor = self._get_vllm_predictor(op)
        if predictor is None:
            return default_ut

        # Try to get predicted throughput
        predicted_ut = predictor.get_unit_throughput()
        if predicted_ut is not None:
            logger.debug(
                f"Operator {op.name}: using predicted UT={predicted_ut:.2f} "
                f"(default would be {default_ut:.2f})"
            )
            return predicted_ut

        # Fall back to default EMA calculation
        logger.debug(
            f"Operator {op.name}: prediction not ready, using default UT={default_ut:.2f}"
        )
        return default_ut

    def _collect_vllm_samples(self):
        """Collect throughput samples from vLLM operators.

        This method should be called periodically to feed the predictor with
        new observations. Uses EMA-based throughput and heuristic metrics
        when actual vLLM metrics are not available.
        """
        if not self._vllm_prediction_enabled:
            return

        from ray.data._internal.execution.autoscaler.vllm_throughput_predictor import (
            ThroughputSample,
            estimate_gpu_utilization,
        )

        for op in self._topology:
            if not isinstance(op, ActorPoolMapOperator):
                continue
            if not self._is_vllm_operator(op):
                continue

            predictor = self._vllm_predictors.get(op.name)
            if predictor is None:
                continue

            # Get operator metrics
            actor_pool = op._actor_pool
            num_actors = actor_pool.num_running_actors()
            if num_actors == 0:
                continue

            num_tasks_in_flight = actor_pool.num_tasks_in_flight()
            max_tasks_per_actor = actor_pool.max_tasks_in_flight_per_actor()

            # Estimate GPU utilization from task queue depth
            gpu_util = estimate_gpu_utilization(
                num_tasks_in_flight, max_tasks_per_actor, num_actors
            )

            # Queue length normalized by number of actors
            queue_length = (
                num_tasks_in_flight / num_actors if num_actors > 0 else 0.0
            )

            # Get throughput from operator metrics
            metrics = op._metrics
            wall_time = metrics.wall_clock_time
            rows_processed = metrics.rows_task_outputs_generated

            if wall_time <= 0 or rows_processed <= 0:
                continue

            # Calculate observed throughput (rows per second per actor)
            observed_throughput = rows_processed / wall_time / num_actors

            # Create sample with heuristic token lengths
            # In production, these would come from actual vLLM metrics
            import time as time_module

            sample = ThroughputSample(
                timestamp=time_module.time(),
                # Heuristic token lengths (typical for chat/instruction models)
                mean_input_length=512.0,
                std_input_length=256.0,
                p50_input_length=384.0,
                p95_input_length=1024.0,
                mean_output_length=256.0,
                std_output_length=128.0,
                p50_output_length=192.0,
                p95_output_length=512.0,
                observed_throughput=observed_throughput,
                gpu_utilization=gpu_util,
                queue_length=queue_length,
                num_actors=num_actors,
            )

            is_valid = predictor.add_sample(sample)
            logger.debug(
                f"Operator {op.name}: collected sample "
                f"(valid={is_valid}, throughput={observed_throughput:.2f}, "
                f"gpu_util={gpu_util:.2f}, queue_len={queue_length:.1f})"
            )

    def get_vllm_predictor_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all vLLM predictors for debugging.

        Returns:
            Dictionary mapping operator names to their predictor stats.
        """
        stats = {}
        for name, predictor in self._vllm_predictors.items():
            stats[name] = predictor.get_stats()
        return stats