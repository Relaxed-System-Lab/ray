import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import ray
from .autoscaler import Autoscaler
from .autoscaling_actor_pool import ActorPoolScalingRequest
from .adaptation_layer import ConfigApplyScope, VLLMAdaptationLayer
from .observation_layer import VLLMObservationLayer
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

    # Gap time (seconds) after all operators are initialized before starting autoscaling.
    # This allows the system to stabilize after initialization.
    INITIALIZATION_GAP_TIME = 60.0

    # EMA smoothing factor (alpha) for throughput calculation.
    # Higher values (closer to 1.0) give more weight to recent observations.
    # Lower values (closer to 0.0) give more weight to historical data.
    EMA_ALPHA = 0.5

    # Default time horizon for planning (seconds)
    DEFAULT_TIME_HORIZON = 60.0
    MIN_THROUGHPUT_FALLBACK = 0.001
    MIN_D_I_FALLBACK = 1.0

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

        # Initialization tracking
        self._all_operators_initialized = False
        self._initialization_complete_time: Optional[float] = None
        self._observation_layer = VLLMObservationLayer(ema_alpha=self.EMA_ALPHA)
        # Fast tuning defaults: fewer samples and BO steps, tighter cooldowns.
        self._adaptation_layer = VLLMAdaptationLayer(
            min_samples=3,
            tuning_cooldown_s=30.0,
            bo_steps_required=3,
            rollout_interval_s=20.0,
        )

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
        1. Checks if all operators have been initialized
        2. Waits for INITIALIZATION_GAP_TIME after initialization completes
        3. Checks if enough time has passed since last scaling
        4. Collects metrics from all ActorPoolMapOperators
        5. Calls MILP solver to get optimal concurrency for each operator
        6. Scales each operator's actor pool to match the target concurrency
        """
        now = time.time()

        # Step 1: Check if all operators have been initialized
        if not self._all_operators_initialized:
            wall_time_list = self.get_wall_time()
            processed_rows_list = self.get_num_processed_rows()

            n = len(wall_time_list)
            if n == 0:
                logger.debug("No ActorPoolMapOperators found. Skipping DS2 autoscaling.")
                return

            # Check if all operators have processed data (initialization complete)
            all_initialized = all(wall_time > 0 and processed_rows > 0
                                 for wall_time, processed_rows in zip(wall_time_list, processed_rows_list))

            if not all_initialized:
                logger.debug(
                    "Not all operators have been initialized yet. "
                    f"Wall time: {wall_time_list}, Processed rows: {processed_rows_list}"
                )
                return

            # All operators are now initialized
            self._all_operators_initialized = True
            self._initialization_complete_time = now

            # Initialize baseline metrics for all operators
            self._initialize_baseline_metrics()

            logger.info(
                f"All operators initialized at {now}. "
                f"Will start autoscaling after {self.INITIALIZATION_GAP_TIME}s gap time."
            )
            return

        # Step 2: Wait for gap time after initialization
        assert self._initialization_complete_time is not None
        time_since_init = now - self._initialization_complete_time
        if time_since_init < self.INITIALIZATION_GAP_TIME:
            logger.debug(
                f"Skipping DS2 autoscaling: only {time_since_init:.1f}s since initialization "
                f"(gap time: {self.INITIALIZATION_GAP_TIME}s)"
            )
            return

        # Step 3: Check frequency limit
        if now - self._last_scaling_time < self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            logger.debug(
                f"Skipping DS2 autoscaling: only {now - self._last_scaling_time:.1f}s "
                f"since last scaling (min gap: {self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS}s)"
            )
            return

        # Step 4: Collect incremental metrics (delta since last snapshot)
        delta_wall_time_list = self.get_delta_wall_time()
        logger.info(f"Delta wall time list: {delta_wall_time_list}")
        delta_num_processed_rows_list = self.get_delta_num_processed_rows()
        logger.info(f"Delta num processed rows list: {delta_num_processed_rows_list}")

        # Adjust delta_wall_time for operators with "Preprocess" in their name
        op_index = 0
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
                if "Preprocess" in op.name:
                    original_delta_wall_time = delta_wall_time_list[op_index]
                    delta_wall_time_list[op_index] = original_delta_wall_time * 2
                    logger.info(
                        f"Operator {op.name} contains 'Preprocess', "
                        f"multiplying delta_wall_time by 2: {original_delta_wall_time} -> {delta_wall_time_list[op_index]}"
                    )
                op_index += 1
        logger.info(f"Processed delta wall time list: {delta_wall_time_list}")

        per_actor_resource_usage_list = self.get_per_actor_resource_usage()
        logger.info(f"Per actor resource usage list: {per_actor_resource_usage_list}")
        total_resources = self.get_total_resources()
        logger.info(f"Total resources: {total_resources}")

        # Calculate unit throughput for each operator using delta values.
        # vLLM operators use the observation layer to estimate sustainable throughput.
        unit_throughput_list: List[float] = []
        op_index = 0
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
                wall_time = delta_wall_time_list[op_index]
                num_rows = delta_num_processed_rows_list[op_index]
                raw_throughput = self._compute_raw_throughput(wall_time, num_rows)

                if self._is_vllm_op(op):
                    queue_size = self._get_queue_size_rows(op, op_state)
                    avg_rows_per_bundle = self._estimate_avg_rows_per_bundle(op)
                    observed = self._observation_layer.observe(
                        op=op,
                        raw_throughput=raw_throughput,
                        delta_rows=num_rows,
                        delta_wall_time=wall_time,
                        queue_size=queue_size,
                        pool_util=self._get_pool_util(op),
                        avg_rows_per_bundle=avg_rows_per_bundle,
                    )
                    unit_throughput_list.append(observed)
                    features = self._adaptation_layer.extract_features(op, op_state)
                    if features is not None:
                        decision = self._adaptation_layer.observe(
                            op=op,
                            features=features,
                            throughput=observed,
                        )
                        if decision is not None:
                            applied = self._adaptation_layer.apply_config(op, decision)
                            if applied:
                                if decision.scope == ConfigApplyScope.ROLLOUT:
                                    self._adaptation_layer.confirm_switch(op, decision)
                                    self._observation_layer.reset(op, queue_size=queue_size)
                                logger.info(
                                    "vLLM adaptation applied for %s (cluster %s, scope=%s).",
                                    op.name,
                                    decision.cluster_id,
                                    decision.scope.value,
                                )
                            else:
                                logger.info(
                                    "vLLM adaptation candidate skipped for %s (cluster %s, scope=%s).",
                                    op.name,
                                    decision.cluster_id,
                                    decision.scope.value,
                                )
                else:
                    unit_throughput_list.append(raw_throughput)

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
        D_o: float = 0.0
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                D_o = float(op._metrics.rows_task_outputs_generated)

        N_cpu = total_resources._cpu or 0.0
        N_gpu = total_resources._gpu or 0.0

        n = len(delta_wall_time_list)

        # Call appropriate MILP solver based on solver_type
        concurrency_list = self._call_solver(
            n=n,
            unit_throughput_list=unit_throughput_list,
            cpu_usage_list=cpu_usage_list,
            gpu_usage_list=gpu_usage_list,
            delta_num_processed_rows_list=delta_num_processed_rows_list,
            delta_wall_time_list=delta_wall_time_list,
            D_o=D_o,
            N_cpu=N_cpu,
            N_gpu=N_gpu,
        )

        if concurrency_list is None:
            logger.warning("MILP solver returned None. Skipping DS2 autoscaling.")
            return

        RESERVE_CPU_FRACTION = 0.05
        # Post-processing: scale up CPU-only operators to maximize CPU utilization
        concurrency_list = self._postprocess_scale_cpu_operators(
            concurrency_list=concurrency_list,
            cpu_usage_list=cpu_usage_list,
            gpu_usage_list=gpu_usage_list,
            N_cpu=N_cpu*(1 - RESERVE_CPU_FRACTION),
        )

        logger.info(
            f"DS2 autoscaling: target concurrency = {concurrency_list}, "
            f"total resources: CPU={N_cpu}, GPU={N_gpu}"
        )

        # Apply scaling to each operator
        op_index = 0
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator):
                # For completed operators, set concurrency to 0
                if op.completed() or (
                    op._inputs_complete and op_state.total_enqueued_input_bundles() == 0
                ):
                    logger.info(f"Operator {op.name} is completed, scaling down to 0.")
                    self._scale_operator(op, target_concurrency=0)
                    continue

                if op_index >= len(concurrency_list):
                    logger.warning(
                        f"Not enough concurrency values from MILP solver. "
                        f"Expected at least {op_index + 1}, got {len(concurrency_list)}"
                    )
                    break

                target_concurrency = concurrency_list[op_index]

                # If operator name contains "BreakByBlock", multiply concurrency by 4
                if "BreakByBlock" in op.name:
                    target_concurrency = target_concurrency * 4
                    logger.info(
                        f"Operator {op.name} contains 'BreakByBlock', "
                        f"multiplying concurrency by 4: {concurrency_list[op_index]} -> {target_concurrency}"
                    )

                # If operator name contains "VideoCaptionVLLM", ensure concurrency >= 6
                if "VideoCaptionVLLM" in op.name:
                    if target_concurrency < 6:
                        original_concurrency = target_concurrency
                        target_concurrency = 6
                        logger.info(
                            f"Operator {op.name} contains 'VideoCaptionVLLM', "
                            f"enforcing minimum concurrency of 6: {original_concurrency} -> {target_concurrency}"
                        )

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
        delta_num_processed_rows_list: List[float],
        delta_wall_time_list: List[float],
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
            delta_num_processed_rows_list: Delta processed rows (D_i) for each operator
            delta_wall_time_list: Delta wall time for each operator (for weight calculation)
            D_o: Total output rows generated
            N_cpu: Total available CPU
            N_gpu: Total available GPU

        Returns:
            List of target concurrency for each operator, or None if solver failed.
        """
        D_i = [float(x) for x in delta_num_processed_rows_list]
        D_i_safe = [di if di > 0 else self.MIN_D_I_FALLBACK for di in D_i]
        if D_i != D_i_safe:
            logger.warning(
                "Some D_i values are <= 0; using fallback %s for solver stability. D_i=%s",
                self.MIN_D_I_FALLBACK,
                D_i,
            )
        logger.info(f"n={n}, ut={unit_throughput_list}, cpu_usage={cpu_usage_list},"
                    f"gpu_usage={gpu_usage_list},"
                    f"D_i={D_i_safe}, D_o={D_o}, N_cpu={N_cpu}, N_gpu={N_gpu}")

        if self._solver_type == SolverType.BASIC:
            # Original solver without queue size consideration

            return milp_solver(
                n, unit_throughput_list, cpu_usage_list, gpu_usage_list,
                D_i_safe, D_o, N_cpu, N_gpu,
            )

        elif self._solver_type == SolverType.QUEUE_DIGESTION:
            # Algorithm 1: Queue digestion priority (Q_target = 0)
            Q = self.get_queue_sizes()

            # Ensure lists have correct length
            if len(Q) < n:
                Q = Q + [0.0] * (n - len(Q))

            logger.debug(
                f"Queue digestion solver: Q={Q}, "
                f"T={self._time_horizon}, beta={self._solver_weight}"
            )

            return milp_solver_queue_digestion(
                n=n,
                UT=unit_throughput_list,
                u=cpu_usage_list,
                g=gpu_usage_list,
                D_i=D_i_safe,
                D_o=D_o,
                N_cpu=N_cpu,
                N_gpu=N_gpu,
                Q=Q,
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
                D_i=D_i_safe,
                D_o=D_o,
                N_cpu=N_cpu,
                N_gpu=N_gpu,
                B_current=B_current,
                B_target=B_target,
                T=self._time_horizon,
                alpha=self._solver_weight,
                ema_wall_time=delta_wall_time_list,
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
                D_i=D_i_safe,
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

    def _is_vllm_op(self, op: ActorPoolMapOperator) -> bool:
        """Return True if operator name contains vLLM (case-insensitive)."""
        return "vllm" in op.name.lower()

    def _compute_raw_throughput(self, wall_time: float, num_rows: float) -> float:
        """Compute raw throughput with a small fallback to avoid division by zero."""
        if wall_time > 0:
            return num_rows / wall_time
        logger.warning(
            f"Delta wall time is 0 for an operator (delta_rows={num_rows}). "
            f"Using fallback throughput of {self.MIN_THROUGHPUT_FALLBACK} rows/s."
        )
        return self.MIN_THROUGHPUT_FALLBACK

    def _get_queue_size_rows(
        self,
        op: ActorPoolMapOperator,
        op_state: "OpState",
    ) -> float:
        exact_rows = op_state.total_enqueued_input_rows()
        if exact_rows is not None:
            return float(exact_rows)
        num_bundles = op_state.total_enqueued_input_bundles()
        avg_rows_per_bundle = self._estimate_avg_rows_per_bundle(op)
        return float(num_bundles) * avg_rows_per_bundle

    def _get_pool_util(self, op: ActorPoolMapOperator) -> float:
        """Best-effort pool utilization for observation layer filtering."""
        if hasattr(op, "get_pool_util"):
            try:
                return op.get_pool_util()  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Failed to fetch pool util via ActorPoolMapOperator.", exc_info=True)
        try:
            actor_pools = op.get_autoscaling_actor_pools()
            if actor_pools:
                return actor_pools[0].get_pool_util()
        except Exception:
            logger.debug("Failed to fetch pool util via actor pools.", exc_info=True)
        # Fallback: treat as fully utilized to avoid over-filtering.
        return 1.0


    def _postprocess_scale_cpu_operators(
        self,
        concurrency_list: List[int],
        cpu_usage_list: List[float],
        gpu_usage_list: List[float],
        N_cpu: float,
    ) -> List[int]:
        """Post-process MILP results to maximize CPU utilization for CPU-only operators.

        This method:
        1. Classifies operators into GPU/NPU operators and CPU-only operators
        2. Calculates CPU resources consumed by GPU/NPU operators
        3. Proportionally scales up CPU-only operators to use remaining CPU resources

        Args:
            concurrency_list: List of concurrency values from MILP solver
            cpu_usage_list: CPU usage per actor for each operator
            gpu_usage_list: GPU usage per actor for each operator
            N_cpu: Total available CPU resources

        Returns:
            Updated concurrency list with scaled CPU-only operators
        """
        n = len(concurrency_list)
        if n == 0:
            return concurrency_list

        # Step 1: Classify operators into GPU/NPU and CPU-only
        gpu_operator_indices = []
        cpu_only_operator_indices = []

        for i in range(n):
            if gpu_usage_list[i] > 0:
                gpu_operator_indices.append(i)
            else:
                cpu_only_operator_indices.append(i)

        logger.debug(
            f"Post-processing: GPU/NPU operators: {gpu_operator_indices}, "
            f"CPU-only operators: {cpu_only_operator_indices}"
        )

        # If no CPU-only operators, nothing to scale
        if not cpu_only_operator_indices:
            logger.debug("No CPU-only operators found. Skipping CPU scaling.")
            return concurrency_list

        # Step 2: Calculate CPU resources consumed by GPU/NPU operators
        gpu_operators_cpu_usage = 0.0
        for i in gpu_operator_indices:
            gpu_operators_cpu_usage += concurrency_list[i] * cpu_usage_list[i]

        logger.debug(
            f"GPU/NPU operators CPU usage: {gpu_operators_cpu_usage}, "
            f"Total CPU: {N_cpu}"
        )

        # Step 3: Calculate remaining CPU for CPU-only operators
        remaining_cpu = N_cpu - gpu_operators_cpu_usage
        if remaining_cpu <= 0:
            logger.warning(
                f"No remaining CPU after GPU/NPU operators. "
                f"GPU/NPU CPU usage: {gpu_operators_cpu_usage}, Total CPU: {N_cpu}"
            )
            return concurrency_list

        # Step 4: Calculate current CPU usage by CPU-only operators
        current_cpu_only_usage = 0.0
        for i in cpu_only_operator_indices:
            current_cpu_only_usage += concurrency_list[i] * cpu_usage_list[i]

        if current_cpu_only_usage <= 0:
            logger.debug("CPU-only operators have zero CPU usage. Skipping scaling.")
            return concurrency_list

        # Step 5: Calculate scaling factor
        # We want to scale up proportionally so that total CPU-only usage = remaining_cpu
        scaling_factor = remaining_cpu / current_cpu_only_usage

        logger.debug(
            f"CPU scaling: remaining_cpu={remaining_cpu}, "
            f"current_cpu_only_usage={current_cpu_only_usage}, "
            f"scaling_factor={scaling_factor}"
        )

        # Only scale up (factor > 1), don't scale down
        if scaling_factor <= 1.0:
            logger.debug(
                f"Scaling factor {scaling_factor} <= 1.0. "
                f"No need to scale up CPU-only operators."
            )
            return concurrency_list

        # Step 6: Apply scaling to CPU-only operators
        result = list(concurrency_list)
        for i in cpu_only_operator_indices:
            old_concurrency = result[i]
            # Scale and round to nearest integer, ensure at least 1
            new_concurrency = max(1, int(round(old_concurrency * scaling_factor)))
            result[i] = new_concurrency

            logger.debug(
                f"CPU-only operator {i}: concurrency {old_concurrency} -> {new_concurrency} "
                f"(factor={scaling_factor:.2f})"
            )

        logger.info(
            f"Post-processing complete: original={list(concurrency_list)}, "
            f"scaled={result}, scaling_factor={scaling_factor:.2f}"
        )

        return result

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

    def _is_op_completed(self, op: "PhysicalOperator", op_state: "OpState") -> bool:
        """Check if an operator has completed processing.

        An operator is considered completed if:
        - op.completed() returns True, OR
        - All inputs are complete AND no bundles are enqueued

        This is consistent with the logic in default_autoscaler.py.
        """
        return op.completed() or (
            op._inputs_complete and op_state.total_enqueued_input_bundles() == 0
        )

    def _initialize_baseline_metrics(self):
        """Initialize baseline metrics for all operators after initialization completes.

        This sets the last_block_generation_time and last_rows_task_inputs_processed
        to the current values, so that subsequent delta calculations start from this point.
        """
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
                op._metrics.last_block_generation_time = op._metrics.block_generation_time
                op._metrics.last_rows_task_inputs_processed = op._metrics.rows_task_inputs_processed
                if self._is_vllm_op(op):
                    queue_size = self._get_queue_size_rows(op, op_state)
                    self._observation_layer.reset(op, queue_size=queue_size)
                    self._adaptation_layer.reset(op)
        logger.info("Baseline metrics initialized for all operators.")

    def get_wall_time(self) -> List[float]:
        """Get total cumulative wall time for each operator.

        Returns the total block_generation_time (cumulative since operator start).
        This is used to check if operators have started processing (value > 0).
        Does NOT update any metrics.

        Returns:
            List of total wall time values for each ActorPoolMapOperator.
        """
        wall_time_list = []
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
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
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
                num_processed_rows_list.append(op._metrics.rows_task_inputs_processed)
        return num_processed_rows_list

    def get_delta_wall_time(self) -> List[float]:
        """Get delta wall time for each operator since last call and update baseline.

        Computes the wall time delta since the last snapshot and updates the snapshot.
        This should only be called after all operators have been initialized.

        Returns:
            List of delta wall time values for each ActorPoolMapOperator.
        """
        delta_wall_time_list = []
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
                current_time = op._metrics.block_generation_time
                last_time = op._metrics.last_block_generation_time

                # Calculate the time delta since last call
                time_delta = current_time - last_time

                # Update the snapshot for next call
                op._metrics.last_block_generation_time = current_time

                delta_wall_time_list.append(time_delta)

        return delta_wall_time_list

    def get_delta_num_processed_rows(self) -> List[float]:
        """Get delta processed rows for each operator since last call and update baseline.

        Computes the processed rows delta since the last snapshot and updates the snapshot.
        This should only be called after all operators have been initialized.

        Returns:
            List of delta processed row counts for each ActorPoolMapOperator.
        """
        delta_num_processed_rows_list = []
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
                current_rows = op._metrics.rows_task_inputs_processed
                last_rows = op._metrics.last_rows_task_inputs_processed

                # Calculate the rows delta since last call
                rows_delta = current_rows - last_rows

                # Update the snapshot for next call
                op._metrics.last_rows_task_inputs_processed = current_rows

                delta_num_processed_rows_list.append(float(rows_delta))

        return delta_num_processed_rows_list



    def get_per_actor_resource_usage(self) -> List[ExecutionResources]:
        per_actor_resource_usage_list = []
        for op, op_state in self._topology.items():
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
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
            if isinstance(op, ActorPoolMapOperator) and not self._is_op_completed(op, op_state):
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
