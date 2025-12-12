import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import ray
from .autoscaler import Autoscaler
from .autoscaling_actor_pool import ActorPoolScalingRequest
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator
from ray.data._internal.execution.autoscaler.ds2_milp_solver import milp_solver
from ray.data._internal.execution.autoscaler.ds2_milp_solvers import (
    milp_solver_queue_digestion,
    milp_solver_relative_deviation,
    milp_solver_time_unified,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology

logger = logging.getLogger(__name__)


class SolverType(Enum):
    """Enum for different MILP solver types."""
    # Original solver without queue size consideration
    BASIC = "basic"
    # Algorithm 1: Queue digestion priority - larger queue leads to more parallelism
    QUEUE_DIGESTION = "queue_digestion"
    # Algorithm 2: Relative deviation with weights - maintain queue in target range
    RELATIVE_DEVIATION = "relative_deviation"
    # Algorithm 3: Time scale unified - maintain queue using unified time scale
    TIME_UNIFIED = "time_unified"


class DS2Autoscaler(Autoscaler):
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 60

    # EMA smoothing factor (alpha) for throughput calculation.
    # Higher values (closer to 1.0) give more weight to recent observations.
    # Lower values (closer to 0.0) give more weight to historical data.
    EMA_ALPHA = 0.5

    # Default time horizon for planning (seconds)
    DEFAULT_TIME_HORIZON = 60.0

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
        solver_type: SolverType = SolverType.BASIC,
        solver_weight: float = 1.0,  # Weight parameter for all solver algorithms
        time_horizon: Optional[float] = None,  # Planning time horizon
        target_queue_sizes: Optional[List[float]] = None,  # Target queue sizes (required for non-BASIC solvers)
    ):
        super().__init__(topology, resource_manager, execution_id)

        # Last time when DS2 scaling was triggered.
        self._last_scaling_time = time.time()

        # Solver configuration
        self._solver_type = solver_type
        self._solver_weight = solver_weight
        self._time_horizon = time_horizon or self.DEFAULT_TIME_HORIZON
        self._target_queue_sizes = target_queue_sizes

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

        # Calculate unit throughput for each operator using EMA values
        unit_throughput_list = []
        for wall_time, num_rows in zip(ema_wall_time_list, ema_num_processed_rows_list):
            if wall_time > 0:
                unit_throughput_list.append(num_rows / wall_time)
            else:
                # This should not happen since we already checked all_work
                # But keep it for safety
                logger.warning(
                    f"EMA wall time is 0 for an operator. This should not happen. "
                    f"Skipping DS2 autoscaling."
                )
                raise ValueError("EMA wall time is 0 for an operator")

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
        D_o: float = 0.0
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                D_o = float(op._metrics.rows_task_outputs_generated)

        N_cpu = total_resources._cpu or 0.0
        N_gpu = total_resources._gpu or 0.0

        # Call appropriate MILP solver based on solver_type
        concurrency_list = self._call_solver(
            n=n,
            unit_throughput_list=unit_throughput_list,
            cpu_usage_list=cpu_usage_list,
            gpu_usage_list=gpu_usage_list,
            ema_num_processed_rows_list=ema_num_processed_rows_list,
            ema_wall_time_list=ema_wall_time_list,
            D_o=D_o,
            N_cpu=N_cpu,
            N_gpu=N_gpu,
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

    def _call_solver(
        self,
        n: int,
        unit_throughput_list: List[float],
        cpu_usage_list: List[float],
        gpu_usage_list: List[float],
        ema_num_processed_rows_list: List[int],
        ema_wall_time_list: List[float],
        D_o: float,
        N_cpu: float,
        N_gpu: float,
    ) -> Optional[List[int]]:
        """Call the appropriate MILP solver based on solver_type.

        Args:
            n: Number of operators
            unit_throughput_list: Unit throughput for each operator
            cpu_usage_list: CPU usage per actor for each operator
            gpu_usage_list: GPU usage per actor for each operator
            ema_num_processed_rows_list: EMA processed rows (D_i) for each operator
            ema_wall_time_list: EMA wall time for each operator (for weight calculation)
            D_o: Total output rows generated
            N_cpu: Total available CPU
            N_gpu: Total available GPU

        Returns:
            List of target concurrency for each operator, or None if solver failed.
        """
        D_i = [float(x) for x in ema_num_processed_rows_list]

        if self._solver_type == SolverType.BASIC:
            # Original solver without queue size consideration
            return milp_solver(
                n, unit_throughput_list, cpu_usage_list, gpu_usage_list,
                ema_num_processed_rows_list, D_o, N_cpu, N_gpu,
            )

        elif self._solver_type == SolverType.QUEUE_DIGESTION:
            # Algorithm 1: Queue digestion priority
            Q = self.get_queue_sizes()
            Q_target = self.get_target_queue_sizes(n)

            # Ensure lists have correct length
            if len(Q) < n:
                Q = Q + [0.0] * (n - len(Q))

            logger.debug(
                f"Queue digestion solver: Q={Q}, Q_target={Q_target}, "
                f"T={self._time_horizon}, beta={self._solver_weight}"
            )

            return milp_solver_queue_digestion(
                n=n,
                UT=unit_throughput_list,
                u=cpu_usage_list,
                g=gpu_usage_list,
                D_i=D_i,
                D_o=D_o,
                N_cpu=N_cpu,
                N_gpu=N_gpu,
                Q=Q,
                Q_target=Q_target,
                T=self._time_horizon,
                beta=self._solver_weight,
            )

        elif self._solver_type == SolverType.RELATIVE_DEVIATION:
            # Algorithm 2: Relative deviation with weights
            B_current = self.get_buffer_sizes()
            B_target = self.get_target_buffer_sizes(n)

            # Ensure lists have correct length
            if len(B_current) < n:
                B_current = B_current + [0.0] * (n - len(B_current))

            logger.debug(
                f"Relative deviation solver: B_current={B_current}, B_target={B_target}, "
                f"T={self._time_horizon}, alpha={self._solver_weight}"
            )

            return milp_solver_relative_deviation(
                n=n,
                UT=unit_throughput_list,
                u=cpu_usage_list,
                g=gpu_usage_list,
                D_i=D_i,
                D_o=D_o,
                N_cpu=N_cpu,
                N_gpu=N_gpu,
                B_current=B_current,
                B_target=B_target,
                T=self._time_horizon,
                alpha=self._solver_weight,
                ema_wall_time=ema_wall_time_list,
            )

        elif self._solver_type == SolverType.TIME_UNIFIED:
            # Algorithm 3: Time scale unified
            B_current = self.get_buffer_sizes()
            B_target = self.get_target_buffer_sizes(n)

            # Ensure lists have correct length
            if len(B_current) < n:
                B_current = B_current + [0.0] * (n - len(B_current))

            logger.debug(
                f"Time unified solver: B_current={B_current}, B_target={B_target}, "
                f"T={self._time_horizon}, alpha={self._solver_weight}"
            )

            return milp_solver_time_unified(
                n=n,
                UT=unit_throughput_list,
                u=cpu_usage_list,
                g=gpu_usage_list,
                D_i=D_i,
                D_o=D_o,
                N_cpu=N_cpu,
                N_gpu=N_gpu,
                B_current=B_current,
                B_target=B_target,
                T=self._time_horizon,
                alpha=self._solver_weight,
            )

        else:
            logger.error(f"Unknown solver type: {self._solver_type}")
            return None

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

    def _estimate_avg_rows_per_bundle(self, op: ActorPoolMapOperator) -> float:
        """Estimate average rows per bundle for an operator.

        Uses metrics from processed inputs to estimate. If no data is available,
        returns a default value of 1.0 (treating bundles as rows).

        Args:
            op: The operator to estimate for.

        Returns:
            Estimated average rows per bundle.
        """
        metrics = op.metrics
        # Use processed inputs to estimate avg rows per bundle
        if metrics.num_task_inputs_processed > 0 and metrics.rows_task_inputs_processed > 0:
            return metrics.rows_task_inputs_processed / metrics.num_task_inputs_processed
        # Fallback: if no processed data yet, return 1.0 (bundle count = row count estimate)
        return 1.0

    def get_queue_sizes(self) -> List[float]:
        """Get current queue sizes in rows for each operator.

        This iterates through all bundles in the queues to get the exact row count.
        The queue includes both external input queues and internal operator queues.

        This ensures the queue size unit (rows) matches the throughput unit (rows/s).

        Returns:
            List of queue sizes in rows for each ActorPoolMapOperator.
            Falls back to estimated value if exact row count is unavailable.
        """
        queue_sizes = []
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator):
                # Try to get exact row count by iterating bundles
                exact_rows = op_state.total_enqueued_input_rows()
                if exact_rows is not None:
                    queue_sizes.append(float(exact_rows))
                else:
                    # Fallback to estimation if any bundle has unknown row count
                    num_bundles = op_state.total_enqueued_input_bundles()
                    avg_rows_per_bundle = self._estimate_avg_rows_per_bundle(op)
                    estimated_rows = float(num_bundles) * avg_rows_per_bundle
                    queue_sizes.append(estimated_rows)
        return queue_sizes

    def get_buffer_sizes(self) -> List[float]:
        """Get current input buffer sizes in rows for each operator.

        The buffer B_i represents all inputs (in rows) waiting to be processed
        by operator i, including both external queues and internal queues.

        This is equivalent to get_queue_sizes() but conceptually represents
        the buffer between operators in the pipeline.

        Returns:
            List of input buffer sizes in rows for each ActorPoolMapOperator.
        """
        # Buffer size is the same as queue size (both in rows)
        return self.get_queue_sizes()

    def get_target_queue_sizes(self, n: int) -> List[float]:
        """Get target queue sizes for each operator.

        Args:
            n: Number of operators

        Returns:
            List of target queue sizes for each operator.

        Raises:
            ValueError: If target_queue_sizes was not specified during initialization.
        """
        if self._target_queue_sizes is None:
            raise ValueError(
                "target_queue_sizes must be specified during DS2Autoscaler initialization "
                "when using non-BASIC solver types. Please provide target_queue_sizes parameter."
            )

        # Extend or truncate to match number of operators
        if len(self._target_queue_sizes) >= n:
            return self._target_queue_sizes[:n]
        else:
            # Pad with last value
            last_val = self._target_queue_sizes[-1] if self._target_queue_sizes else 1.0
            return self._target_queue_sizes + [last_val] * (n - len(self._target_queue_sizes))

    def get_target_buffer_sizes(self, n: int) -> List[float]:
        """Get target buffer sizes for each operator.

        Target buffer B_target_i represents the desired buffer size between
        operator i-1 and operator i (input queue for operator i).

        Args:
            n: Number of operators

        Returns:
            List of target buffer sizes for each operator.
        """
        return self.get_target_queue_sizes(n)