"""Placement-aware DS2 autoscaler for Ray Data.

This module implements a placement-aware autoscaler that optimizes operator
placement across heterogeneous cluster nodes. It extends the basic DS2 autoscaler
to consider:
- Per-node resource constraints (CPU, memory, GPU)
- Data flow routing between operators across nodes
- Network egress constraints
- Migration costs for changing placements

See milp_placement_design.md for the complete mathematical formulation.
"""

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import ray
from .autoscaler import Autoscaler
from .ds2_placement_milp_solver import (
    NodeResources,
    OperatorResources,
    PlacementSolverInput,
    PlacementSolverOutput,
    milp_placement_solver,
)
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
    _ActorPool,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology

logger = logging.getLogger(__name__)


# Default migration cost parameters
DEFAULT_BASE_START_COST = 2.0  # seconds for basic actor startup
DEFAULT_BASE_STOP_COST = 0.5   # seconds for graceful shutdown
DEFAULT_GPU_START_PENALTY = 10.0  # additional seconds per GPU
DEFAULT_GPU_STOP_PENALTY = 1.0    # additional seconds per GPU
DEFAULT_MEMORY_START_PENALTY_PER_GB = 0.5  # additional seconds per GB memory


class DS2PlacementAutoscaler(Autoscaler):
    """Placement-aware DS2 autoscaler.

    This autoscaler optimizes both operator parallelism AND placement across
    heterogeneous cluster nodes. It uses a MILP solver to maximize throughput
    while respecting per-node resource constraints and minimizing network
    traffic and migration costs.
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
        epsilon_1: float = 0.001,  # Network penalty weight
        epsilon_2: float = 0.001,  # Migration penalty weight
        # Migration cost overrides (optional)
        c_start_overrides: Optional[Dict[str, float]] = None,  # op_name -> cost
        c_stop_overrides: Optional[Dict[str, float]] = None,   # op_name -> cost
    ):
        """Initialize the placement-aware DS2 autoscaler.

        Args:
            topology: The topology of the data pipeline.
            resource_manager: The resource manager for the execution.
            execution_id: Unique identifier for this execution.
            epsilon_1: Weight for network egress penalty in objective function.
            epsilon_2: Weight for migration cost penalty in objective function.
            c_start_overrides: Optional dict mapping operator names to startup costs.
            c_stop_overrides: Optional dict mapping operator names to stop costs.
        """
        super().__init__(topology, resource_manager, execution_id)

        # Last time when scaling was triggered.
        self._last_scaling_time = time.time()

        # Solver configuration
        self._epsilon_1 = epsilon_1
        self._epsilon_2 = epsilon_2

        # Migration cost overrides
        self._c_start_overrides = c_start_overrides or {}
        self._c_stop_overrides = c_stop_overrides or {}

        # Cache for node resources (refreshed on each scaling attempt)
        self._node_resources_cache: Optional[List[NodeResources]] = None
        self._node_id_to_index: Dict[str, int] = {}

    def try_trigger_scaling(self):
        """Try to trigger placement-aware autoscaling."""
        self._placement_aware_scaling()

    def on_executor_shutdown(self):
        """Called when the executor is shutting down."""
        logger.info("Placement-aware DS2 autoscaler shutting down.")

    def get_total_resources(self) -> ExecutionResources:
        cluster_res = ray.cluster_resources()
        # NPU priority: if NPU exists, use it as GPU
        if "NPU" in cluster_res:
            cluster_res["GPU"] = cluster_res["NPU"]
        return ExecutionResources.from_resource_dict(cluster_res)

    def _get_cluster_nodes(self) -> List[NodeResources]:
        """Get resource information for all cluster nodes.

        Returns:
            List of NodeResources for each node in the cluster.
        """
        nodes = []
        self._node_id_to_index = {}

        for node in ray.nodes():
            if not node.get("Alive", False):
                continue

            node_id = node["NodeID"]
            resources = node.get("Resources", {})

            # Extract CPU, memory, GPU
            cpu = resources.get("CPU", 0.0)
            # Memory is in bytes in Ray
            memory = resources.get("memory", 0.0)
            # NPU priority: if NPU exists, use it as GPU
            gpu = resources.get("NPU", resources.get("GPU", 0.0))

            self._node_id_to_index[node_id] = len(nodes)
            nodes.append(NodeResources(
                node_id=node_id,
                cpu=cpu,
                memory=memory,
                gpu=gpu,
            ))

        return nodes

    def _get_actor_pool_operators(self) -> List[ActorPoolMapOperator]:
        """Get all ActorPoolMapOperators from the topology."""
        operators = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                operators.append(op)
        return operators

    def _get_operator_resources(
        self, operators: List[ActorPoolMapOperator]
    ) -> List[OperatorResources]:
        """Get resource requirements for each operator.

        Args:
            operators: List of ActorPoolMapOperator instances.

        Returns:
            List of OperatorResources for each operator.
        """
        result = []
        for op in operators:
            per_actor = op.get_per_actor_resource_usage()
            result.append(OperatorResources(
                cpu=per_actor._cpu or 0.0,
                memory=per_actor._memory or 0.0,
                gpu=per_actor._gpu or 0.0,
            ))
        return result

    def _get_throughput_per_instance(
        self, operators: List[ActorPoolMapOperator]
    ) -> List[float]:
        """Get throughput per instance (UT_i) for each operator.

        Uses EMA-smoothed metrics for stable estimates.

        Args:
            operators: List of ActorPoolMapOperator instances.

        Returns:
            List of throughput values (records/second per instance).
        """
        result = []
        for op in operators:
            # Get EMA wall time and processed rows
            current_time = op._metrics.block_generation_time
            last_time = op._metrics.last_block_generation_time
            time_delta = current_time - last_time

            current_rows = op._metrics.rows_task_inputs_processed
            last_rows = op._metrics.last_rows_task_inputs_processed
            rows_delta = current_rows - last_rows

            # Apply EMA smoothing
            if op._metrics.ema_wall_time == 0.0:
                ema_time = time_delta
            else:
                ema_time = (
                    self.EMA_ALPHA * time_delta +
                    (1 - self.EMA_ALPHA) * op._metrics.ema_wall_time
                )

            if op._metrics.ema_processed_rows == 0.0:
                ema_rows = float(rows_delta)
            else:
                ema_rows = (
                    self.EMA_ALPHA * rows_delta +
                    (1 - self.EMA_ALPHA) * op._metrics.ema_processed_rows
                )

            # Update metrics
            op._metrics.ema_wall_time = ema_time
            op._metrics.last_block_generation_time = current_time
            op._metrics.ema_processed_rows = ema_rows
            op._metrics.last_rows_task_inputs_processed = current_rows

            # Calculate throughput per instance
            if ema_time > 0:
                # Total throughput / number of instances
                actor_pools = op.get_autoscaling_actor_pools()
                num_instances = actor_pools[0].current_size() if actor_pools else 1
                total_throughput = ema_rows / ema_time
                throughput_per_instance = total_throughput / max(num_instances, 1)
            else:
                throughput_per_instance = 0.0

            result.append(throughput_per_instance)

        return result

    def _get_expansion_factors(
        self, operators: List[ActorPoolMapOperator]
    ) -> List[float]:
        """Get input expansion factors (D_i) for each operator.

        D_i represents the cumulative data volume at operator i's input
        relative to the original pipeline input.

        Args:
            operators: List of ActorPoolMapOperator instances.

        Returns:
            List of expansion factors.
        """
        result = []
        for op in operators:
            # D_i = rows_task_inputs_processed for operator i
            d_i = float(op._metrics.rows_task_inputs_processed)
            if d_i <= 0:
                d_i = 1.0  # Default to 1.0 if no data processed yet
            result.append(d_i)
        return result

    def _get_output_expansion_factor(
        self, operators: List[ActorPoolMapOperator]
    ) -> float:
        """Get the final output expansion factor (D_o).

        D_o represents the total output volume of the pipeline.

        Args:
            operators: List of ActorPoolMapOperator instances.

        Returns:
            Output expansion factor.
        """
        if not operators:
            return 1.0
        # D_o is the output of the last operator
        last_op = operators[-1]
        d_o = float(last_op._metrics.rows_task_outputs_generated)
        if d_o <= 0:
            d_o = 1.0
        return d_o

    def _get_output_data_sizes(
        self, operators: List[ActorPoolMapOperator]
    ) -> List[float]:
        """Get output data sizes (s_i) in MB per record for each operator.

        Args:
            operators: List of ActorPoolMapOperator instances.

        Returns:
            List of output data sizes in MB per record.
        """
        result = []
        for op in operators:
            bytes_out = op._metrics.bytes_task_outputs_generated
            rows_out = op._metrics.rows_task_outputs_generated
            if rows_out > 0:
                # Convert bytes to MB
                s_i = (bytes_out / rows_out) / (1024 * 1024)
            else:
                s_i = 0.0  # No data yet
            result.append(s_i)
        return result

    def _estimate_default_migration_costs(
        self, op: ActorPoolMapOperator
    ) -> tuple[float, float]:
        """Estimate default migration costs based on resource requirements.

        Args:
            op: The operator to estimate costs for.

        Returns:
            Tuple of (c_start, c_stop) in seconds.
        """
        per_actor = op.get_per_actor_resource_usage()
        num_gpus = per_actor._gpu or 0.0
        memory_bytes = per_actor._memory or 0.0
        memory_gb = memory_bytes / (1024 ** 3)

        c_start = (
            DEFAULT_BASE_START_COST +
            num_gpus * DEFAULT_GPU_START_PENALTY +
            memory_gb * DEFAULT_MEMORY_START_PENALTY_PER_GB
        )
        c_stop = DEFAULT_BASE_STOP_COST + num_gpus * DEFAULT_GPU_STOP_PENALTY

        return c_start, c_stop

    def _get_migration_costs(
        self, operators: List[ActorPoolMapOperator]
    ) -> tuple[List[float], List[float]]:
        """Get migration costs for all operators.

        Uses adaptive estimation: observed values if available, otherwise
        resource-based defaults. Manual overrides take highest priority.

        Args:
            operators: List of ActorPoolMapOperator instances.

        Returns:
            Tuple of (c_start_list, c_stop_list).
        """
        c_start_list = []
        c_stop_list = []

        for op in operators:
            # Get actor pool for observed metrics
            actor_pools = op.get_autoscaling_actor_pools()
            actor_pool = actor_pools[0] if actor_pools else None

            # Priority 1: Manual override
            if op.name in self._c_start_overrides:
                c_start = self._c_start_overrides[op.name]
            # Priority 2: Observed average (if available)
            elif actor_pool and hasattr(actor_pool, 'get_observed_startup_cost'):
                observed = actor_pool.get_observed_startup_cost()
                if observed is not None:
                    c_start = observed
                else:
                    # Priority 3: Resource-based default
                    c_start, _ = self._estimate_default_migration_costs(op)
            else:
                c_start, _ = self._estimate_default_migration_costs(op)

            # Same logic for c_stop
            if op.name in self._c_stop_overrides:
                c_stop = self._c_stop_overrides[op.name]
            elif actor_pool and hasattr(actor_pool, 'get_observed_shutdown_cost'):
                observed = actor_pool.get_observed_shutdown_cost()
                if observed is not None:
                    c_stop = observed
                else:
                    _, c_stop = self._estimate_default_migration_costs(op)
            else:
                _, c_stop = self._estimate_default_migration_costs(op)

            c_start_list.append(c_start)
            c_stop_list.append(c_stop)

        return c_start_list, c_stop_list

    def _get_current_placement(
        self,
        operators: List[ActorPoolMapOperator],
        nodes: List[NodeResources],
    ) -> List[List[int]]:
        """Get current placement matrix x_bar[i][k].

        Args:
            operators: List of operators.
            nodes: List of cluster nodes.

        Returns:
            2D list where x_bar[i][k] = current instances of operator i on node k.
        """
        K = len(nodes)
        placement = []

        for op in operators:
            actor_pools = op.get_autoscaling_actor_pools()
            if not actor_pools:
                placement.append([0] * K)
                continue

            actor_pool = actor_pools[0]
            if not hasattr(actor_pool, 'get_actor_count_by_node'):
                placement.append([0] * K)
                continue

            counts_by_node = actor_pool.get_actor_count_by_node()

            # Convert to list indexed by node index
            op_placement = [0] * K
            for node_id, count in counts_by_node.items():
                if node_id in self._node_id_to_index:
                    idx = self._node_id_to_index[node_id]
                    op_placement[idx] = count

            placement.append(op_placement)

        return placement

    def _check_all_operators_working(
        self, operators: List[ActorPoolMapOperator]
    ) -> bool:
        """Check if all operators have started processing.

        Args:
            operators: List of operators to check.

        Returns:
            True if all operators have processed data, False otherwise.
        """
        for op in operators:
            if op._metrics.block_generation_time <= 0:
                return False
        return True



    def _placement_aware_scaling(self):
        """Perform placement-aware autoscaling.

        This method:
        1. Checks if enough time has passed since last scaling
        2. Collects metrics from all ActorPoolMapOperators
        3. Gets cluster node information
        4. Calls MILP solver to get optimal placement
        5. Applies the placement changes
        """
        # Check frequency limit
        now = time.time()
        if now - self._last_scaling_time < self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            logger.debug(
                f"Skipping placement-aware scaling: only {now - self._last_scaling_time:.1f}s "
                f"since last scaling (min gap: {self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS}s)"
            )
            return

        # Get operators
        operators = self._get_actor_pool_operators()
        n = len(operators)
        if n == 0:
            logger.debug("No ActorPoolMapOperators found. Skipping scaling.")
            return

        # Check if all operators have started processing
        if not self._check_all_operators_working(operators):
            logger.debug(
                "Not all operators have processed data yet. Skipping scaling."
            )
            return

        # Get cluster nodes
        nodes = self._get_cluster_nodes()
        K = len(nodes)
        if K == 0:
            logger.warning("No alive nodes found in cluster. Skipping scaling.")
            return

        logger.debug(f"Found {n} operators and {K} nodes")

        # Collect metrics
        UT = self._get_throughput_per_instance(operators)
        D_i = self._get_expansion_factors(operators)
        D_o = self._get_output_expansion_factor(operators)
        s = self._get_output_data_sizes(operators)
        op_resources = self._get_operator_resources(operators)
        c_start, c_stop = self._get_migration_costs(operators)
        x_bar = self._get_current_placement(operators, nodes)

        # Build solver input
        solver_input = PlacementSolverInput(
            n=n,
            K=K,
            UT=UT,
            D_i=D_i,
            D_o=D_o,
            s=s,
            op_resources=op_resources,
            node_resources=nodes,
            x_bar=x_bar,
            c_start=c_start,
            c_stop=c_stop,
            epsilon_1=self._epsilon_1,
            epsilon_2=self._epsilon_2,
        )

        logger.debug(
            f"Solver input: n={n}, K={K}, UT={UT}, "
            f"epsilon_1={self._epsilon_1}, epsilon_2={self._epsilon_2}"
        )

        # Call solver
        result = milp_placement_solver(solver_input)

        if result is None:
            logger.warning("MILP placement solver returned None. Skipping scaling.")
            return

        logger.info(
            f"Placement solver result: status={result.status}, "
            f"throughput={result.throughput:.2f}, "
            f"egress={result.egress_max:.2f}, migration={result.migration_cost:.2f}"
        )

        # Apply placement changes
        self._apply_placement(operators, nodes, result)

        # Update last scaling time
        self._last_scaling_time = now

    def _apply_placement(
        self,
        operators: List[ActorPoolMapOperator],
        nodes: List[NodeResources],
        result: PlacementSolverOutput,
    ):
        """Apply the placement solution to the operators.

        Args:
            operators: List of operators.
            nodes: List of cluster nodes.
            result: The solver output with optimal placement.
        """
        for i, op in enumerate(operators):
            actor_pools = op.get_autoscaling_actor_pools()
            if not actor_pools:
                logger.warning(f"Operator {op.name} has no actor pools.")
                continue

            actor_pool = actor_pools[0]

            # Check if actor pool supports node-aware scaling
            if not hasattr(actor_pool, 'scale_on_node'):
                logger.warning(
                    f"Actor pool for {op.name} doesn't support node-aware scaling."
                )
                continue

            # Get current placement
            current_by_node = {}
            if hasattr(actor_pool, 'get_actor_count_by_node'):
                current_by_node = actor_pool.get_actor_count_by_node()

            # Apply changes for each node
            for k, node in enumerate(nodes):
                target = result.x[i][k]
                current = current_by_node.get(node.node_id, 0)
                delta = target - current

                if delta > 0:
                    # Scale up on this node
                    logger.info(
                        f"Operator {op.name}: scaling up {delta} actors on node {node.node_id}"
                    )
                    actor_pool.scale_on_node(
                        node_id=node.node_id,
                        num_actors=delta,
                        reason=f"Placement optimization (target={target})",
                    )
                elif delta < 0:
                    # Scale down on this node
                    num_to_remove = abs(delta)
                    logger.info(
                        f"Operator {op.name}: scaling down {num_to_remove} actors "
                        f"on node {node.node_id}"
                    )
                    actor_pool.scale_down_on_node(
                        node_id=node.node_id,
                        num_to_remove=num_to_remove,
                        forced=False,
                    )