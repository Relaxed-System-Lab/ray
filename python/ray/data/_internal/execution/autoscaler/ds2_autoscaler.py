import logging
import time
from typing import TYPE_CHECKING, List

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

logger = logging.getLogger(__name__)


class DS2Autoscaler(Autoscaler):
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 60

    # EMA smoothing factor (alpha) for throughput calculation.
    # Higher values (closer to 1.0) give more weight to recent observations.
    # Lower values (closer to 0.0) give more weight to historical data.
    EMA_ALPHA = 0.5

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
    ):
        super().__init__(topology, resource_manager, execution_id)

        # Last time when DS2 scaling was triggered.
        self._last_scaling_time = 0

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
        return ExecutionResources.from_resource_dict(ray.cluster_resources()) 

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

        # Collect metrics from all ActorPoolMapOperators
        wall_time_list = self.get_wall_time()
        logger.debug(f"Wall time list: {wall_time_list}")
        num_processed_rows_list = self.get_num_processed_rows()
        logger.debug(f"Num processed rows list: {num_processed_rows_list}")
        per_actor_resource_usage_list = self.get_per_actor_resource_usage()
        logger.debug(f"Per actor resource usage list: {per_actor_resource_usage_list}")
        total_resources = self.get_total_resources()
        logger.debug(f"Total resources: {total_resources}")
        n = len(wall_time_list)

        if n == 0:
            logger.debug("No ActorPoolMapOperators found. Skipping DS2 autoscaling.")
            return

        # Calculate unit throughput for each operator
        unit_throughput_list = []
        all_work = True
        for wall_time, num_rows in zip(wall_time_list, num_processed_rows_list):
            if wall_time > 0:
                unit_throughput_list.append(num_rows / wall_time)
            else:
                all_work = False

        if not all_work:
            logger.debug(
                "Not all operators have processed data yet. Skipping DS2 autoscaling."
            )
            return
        logger.info(
            f"DS2 autoscaling: unit throughput = {unit_throughput_list}, "
            f"num processed rows = {num_processed_rows_list}"
            f"wall times = {wall_time_list}"
        )

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
            num_processed_rows_list, D_o, N_cpu, N_gpu,
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
        """Get smoothed wall time for each operator using EMA.

        Computes the exponential moving average (EMA) of wall time deltas
        to smooth out short-term fluctuations and provide a more stable
        estimate of operator throughput.

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

    def get_num_processed_rows(self) -> List[int]:
        """Get smoothed number of processed rows for each operator using EMA.

        Computes the exponential moving average (EMA) of processed rows deltas
        to smooth out short-term fluctuations and provide a more stable
        estimate of operator throughput.

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
    