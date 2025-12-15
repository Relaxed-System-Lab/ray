"""Placement-aware MILP solver for DS2 autoscaler.

This module implements the placement-aware optimization model that considers:
- Heterogeneous cluster nodes with different CPU/memory/GPU capacities
- Operator placement decisions (x_{i,k}) on specific nodes
- Data flow routing (w_{i,k,l}) between operators across nodes
- Network egress constraints (E_max)
- Migration costs (M) for changing placements

See milp_placement_design.md for the complete mathematical formulation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pulp

logger = logging.getLogger(__name__)


@dataclass
class NodeResources:
    """Resource capacity of a single cluster node."""

    node_id: str
    cpu: float  # Number of CPU cores
    memory: float  # Memory in bytes
    gpu: float  # Number of GPUs


@dataclass
class OperatorResources:
    """Resource requirements for a single operator instance."""

    cpu: float  # CPU cores per instance
    memory: float  # Memory per instance in bytes
    gpu: float  # GPUs per instance


@dataclass
class PlacementSolverInput:
    """Input parameters for the placement-aware MILP solver."""

    # Number of operators
    n: int
    # Number of nodes
    K: int
    # Per-operator throughput: UT_i (records/second per instance)
    UT: List[float]
    # Per-operator input expansion factor: D_i
    D_i: List[float]
    # Final output expansion factor: D_o
    D_o: float
    # Per-operator output data size: s_i (MB per record)
    s: List[float]
    # Per-operator resource requirements (length n)
    op_resources: List[OperatorResources]
    # List of node resources (length K)
    node_resources: List[NodeResources]
    # Current placement: x_bar[i][k] = current instances of operator i on node k
    x_bar: List[List[int]]
    # Per-operator startup cost: c_start[i] (seconds)
    c_start: List[float]
    # Per-operator stop cost: c_stop[i] (seconds)
    c_stop: List[float]
    # Weight for network egress penalty
    epsilon_1: float = 0.001
    # Weight for migration cost penalty
    epsilon_2: float = 0.001


@dataclass
class PlacementSolverOutput:
    """Output from the placement-aware MILP solver."""

    # Optimal placement: x[i][k] = instances of operator i on node k
    x: List[List[int]]
    # Total parallelism per operator: p[i]
    p: List[int]
    # Optimal throughput (records/second of original input)
    throughput: float
    # Peak egress traffic (MB/s)
    egress_max: float
    # Total migration cost (seconds)
    migration_cost: float
    # Data flow routing: w[i][k][l] = flow units from op i on node k to op i+1 on node l
    w: List[List[List[int]]]
    # Solver status
    status: str


def milp_placement_solver(input_params: PlacementSolverInput) -> Optional[PlacementSolverOutput]:
    """Solve the placement-aware MILP optimization problem.

    This solver maximizes throughput while considering:
    - Per-node resource constraints (CPU, memory, GPU)
    - Data flow conservation between adjacent operators
    - Network egress constraints
    - Migration costs from current placement

    Args:
        input_params: PlacementSolverInput containing all problem parameters.

    Returns:
        PlacementSolverOutput with optimal placement, or None if infeasible.
    """
    n = input_params.n
    K = input_params.K

    if n == 0 or K == 0:
        logger.warning("Empty problem: n=%d, K=%d", n, K)
        return None

    # Validate input dimensions
    assert len(input_params.op_resources) == n
    assert len(input_params.UT) == n
    assert len(input_params.D_i) == n
    assert len(input_params.s) == n
    assert len(input_params.c_start) == n
    assert len(input_params.c_stop) == n
    assert len(input_params.x_bar) == n
    assert len(input_params.node_resources) == K
    for i in range(n):
        assert len(input_params.x_bar[i]) == K

    # Create the problem
    prob = pulp.LpProblem("PlacementAwareDS2", pulp.LpMaximize)

    # === Decision Variables ===

    # Throughput (continuous, non-negative)
    tau = pulp.LpVariable("tau", lowBound=0, cat=pulp.LpContinuous)

    # Peak egress traffic (continuous, non-negative)
    E_max_var = pulp.LpVariable("E_max", lowBound=0, cat=pulp.LpContinuous)

    # Migration cost (continuous, non-negative)
    M_var = pulp.LpVariable("M", lowBound=0, cat=pulp.LpContinuous)

    # Parallelism per operator: p[i] (positive integer)
    p = [pulp.LpVariable(f"p_{i}", lowBound=1, cat=pulp.LpInteger) for i in range(n)]

    # Placement: x[i][k] = instances of operator i on node k (non-negative integer)
    x = [
        [pulp.LpVariable(f"x_{i}_{k}", lowBound=0, cat=pulp.LpInteger) for k in range(K)]
        for i in range(n)
    ]

    # Data flow: w[i][k][l] = flow from operator i on node k to operator i+1 on node l
    # Only needed for i in [0, n-2] (n-1 flows between n operators)
    w = [
        [
            [pulp.LpVariable(f"w_{i}_{k}_{l}", lowBound=0, cat=pulp.LpInteger) for l in range(K)]
            for k in range(K)
        ]
        for i in range(n - 1)
    ]

    # Migration deltas: delta_plus[i][k], delta_minus[i][k]
    delta_plus = [
        [pulp.LpVariable(f"delta_plus_{i}_{k}", lowBound=0, cat=pulp.LpInteger) for k in range(K)]
        for i in range(n)
    ]
    delta_minus = [
        [pulp.LpVariable(f"delta_minus_{i}_{k}", lowBound=0, cat=pulp.LpInteger) for k in range(K)]
        for i in range(n)
    ]

    # === Objective Function ===
    # max tau - epsilon_1 * E_max - epsilon_2 * M
    prob += tau - input_params.epsilon_1 * E_max_var - input_params.epsilon_2 * M_var, "Objective"

    # === Constraints ===

    # Shorthand for parameters
    UT = input_params.UT
    D = input_params.D_i
    D_o = input_params.D_o
    s = input_params.s
    c_start = input_params.c_start
    c_stop = input_params.c_stop
    x_bar = input_params.x_bar

    # 5.1 Throughput constraints: tau <= (D_o / D_i) * p_i * UT_i
    for i in range(n):
        if D[i] > 0 and UT[i] > 0:
            # tau <= (D_o / D_i) * p_i * UT_i
            prob += tau <= (D_o / D[i]) * p[i] * UT[i], f"Throughput_Op_{i}"

    # 5.2 Instance allocation consistency: sum_k x[i][k] = p[i]
    for i in range(n):
        prob += pulp.lpSum(x[i][k] for k in range(K)) == p[i], f"Allocation_{i}"

    # 5.3 Node resource capacity constraints
    for k in range(K):
        node = input_params.node_resources[k]
        ops = input_params.op_resources

        # CPU constraint
        prob += (
            pulp.lpSum(ops[i].cpu * x[i][k] for i in range(n)) <= node.cpu,
            f"CPU_Node_{k}",
        )

        # Memory constraint (convert to same units)
        prob += (
            pulp.lpSum(ops[i].memory * x[i][k] for i in range(n)) <= node.memory,
            f"Memory_Node_{k}",
        )

        # GPU constraint
        prob += (
            pulp.lpSum(ops[i].gpu * x[i][k] for i in range(n)) <= node.gpu,
            f"GPU_Node_{k}",
        )

    # 5.4 Data flow conservation constraints
    for i in range(n - 1):
        # Outflow from operator i on node k: sum_l w[i][k][l] = x[i][k]
        for k in range(K):
            prob += (
                pulp.lpSum(w[i][k][l] for l in range(K)) == x[i][k],
                f"Outflow_Op_{i}_Node_{k}",
            )

        # Inflow to operator i+1 on node l: sum_k w[i][k][l] = x[i+1][l]
        for l in range(K):
            prob += (
                pulp.lpSum(w[i][k][l] for k in range(K)) == x[i + 1][l],
                f"Inflow_Op_{i+1}_Node_{l}",
            )

    # 5.5 Node egress traffic constraints
    # sum_{i=0}^{n-2} (x[i][k] - w[i][k][k]) * UT[i] * s[i] <= E_max
    for k in range(K):
        egress_terms = []
        for i in range(n - 1):
            # x[i][k] - w[i][k][k] is the amount sent to other nodes
            # Multiply by throughput and data size
            egress_terms.append((x[i][k] - w[i][k][k]) * UT[i] * s[i])
        prob += pulp.lpSum(egress_terms) <= E_max_var, f"Egress_Node_{k}"

    # 5.6 Migration variable constraints: x[i][k] = x_bar[i][k] + delta_plus[i][k] - delta_minus[i][k]
    for i in range(n):
        for k in range(K):
            prob += (
                x[i][k] == x_bar[i][k] + delta_plus[i][k] - delta_minus[i][k],
                f"Migration_Op_{i}_Node_{k}",
            )

    # 5.7 Migration cost definition: M = sum_i sum_k (c_start[i] * delta_plus[i][k] + c_stop[i] * delta_minus[i][k])
    migration_terms = []
    for i in range(n):
        for k in range(K):
            migration_terms.append(c_start[i] * delta_plus[i][k])
            migration_terms.append(c_stop[i] * delta_minus[i][k])
    prob += M_var == pulp.lpSum(migration_terms), "Migration_Cost_Definition"

    # === Solve ===
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    except Exception as e:
        logger.error("MILP solver failed: %s", e)
        return None

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        logger.warning("MILP solver status: %s (not optimal)", status)
        if status == "Infeasible":
            return None

    # === Extract solution ===
    x_result = [[int(pulp.value(x[i][k]) or 0) for k in range(K)] for i in range(n)]
    p_result = [int(pulp.value(p[i]) or 0) for i in range(n)]
    throughput_result = float(pulp.value(tau) or 0)
    egress_max_result = float(pulp.value(E_max_var) or 0)
    migration_cost_result = float(pulp.value(M_var) or 0)

    # Extract data flow
    w_result = [
        [[int(pulp.value(w[i][k][l]) or 0) for l in range(K)] for k in range(K)]
        for i in range(n - 1)
    ]

    return PlacementSolverOutput(
        x=x_result,
        p=p_result,
        throughput=throughput_result,
        egress_max=egress_max_result,
        migration_cost=migration_cost_result,
        w=w_result,
        status=status,
    )

