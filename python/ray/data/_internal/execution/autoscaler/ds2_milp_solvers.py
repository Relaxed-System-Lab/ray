"""
MILP Solvers for DS2 Autoscaler with Queue Size Consideration.

This module implements three different MILP solver variants that consider queue size:
1. Queue Digestion Priority: Maximize throughput + β * queue digestion rate
2. Relative Deviation with Weights: Maintain queue in target range using weighted relative deviation
3. Time Scale Unified: Maintain queue in target range using unified time scale
"""

from typing import List, Optional, Tuple
from pulp import (
    LpProblem, LpMaximize, LpMinimize, LpVariable, LpStatus, LpStatusOptimal,
    lpSum, PULP_CBC_CMD
)


def milp_solver_queue_digestion(
    n: int,
    UT: List[float],  # Unit throughput for each operator
    u: List[float],   # CPU usage per actor for each operator
    g: List[float],   # GPU usage per actor for each operator
    D_i: List[float], # Input data volume processed by each operator
    D_o: float,       # Final output data volume
    N_cpu: float,     # Total available CPU
    N_gpu: float,     # Total available GPU
    Q: List[float],   # Current queue size for each operator
    T: float,         # Time horizon for planning
    beta: float = 1.0,  # Weight for queue digestion term
) -> Optional[List[int]]:
    """
    Algorithm 1: Queue Digestion Priority MILP Solver (Simplified with Q_target = 0).

    Objective: max τ/τ_ref + β * Σd_i/D_ref

    Where:
    - τ is system throughput
    - d_i is queue digestion amount for operator i
    - τ_ref is reference throughput for normalization
    - D_ref = Σ Q_i (total queue size, since Q_target = 0)

    The larger the queue size, the more parallelism is allocated.
    Goal is to drain the queues to 0.
    """
    if n == 0:
        return []

    # --- Calculate reference values for normalization ---
    # p_max for each operator (theoretical max parallelism under resource constraints)
    p_max = []
    for i in range(n):
        cpu_limit = N_cpu / u[i] if u[i] > 0 else float('inf')
        gpu_limit = N_gpu / g[i] if g[i] > 0 else float('inf')
        p_max.append(min(cpu_limit, gpu_limit))

    # τ_ref = min_i(D_o/D_i * UT_i * p_i_max)
    tau_ref_candidates = []
    for i in range(n):
        if D_i[i] > 0:
            scaling_factor = D_o / D_i[i]
            tau_ref_candidates.append(scaling_factor * UT[i] * p_max[i])
    tau_ref = min(tau_ref_candidates) if tau_ref_candidates else 1.0
    tau_ref = max(tau_ref, 1e-6)  # Avoid division by zero

    # D_ref = Σ Q_i (with Q_target = 0)
    D_ref = sum(Q[i] for i in range(n))
    D_ref = max(D_ref, 1e-6)  # Avoid division by zero

    # --- Variables ---
    p = LpVariable.dicts("p", range(n), lowBound=1, cat='Integer')
    tau = LpVariable("tau", lowBound=0)
    d = LpVariable.dicts("d", range(n), lowBound=0)  # Queue digestion amount

    # --- Problem ---
    prob = LpProblem("Queue_Digestion_Priority", LpMaximize)

    # --- Objective: max τ/τ_ref + β * Σd_i/D_ref ---
    # Note: PuLP doesn't support LpVariable / float, so we multiply by inverse
    prob += tau * (1.0 / tau_ref) + beta * lpSum([d[i] for i in range(n)]) * (1.0 / D_ref)

    # --- Constraints ---
    # 1. Throughput constraint: τ + (D_o/D_i) * d_i/T <= (D_o/D_i) * p_i * UT_i
    for i in range(n):
        if D_i[i] > 0:
            scaling_factor = D_o / D_i[i]
            prob += tau + scaling_factor * d[i] * (1.0 / T) <= scaling_factor * p[i] * UT[i], \
                f"Throughput_Constraint_{i}"

    # 2. Queue digestion bound: 0 <= d_i <= Q_i (with Q_target = 0)
    for i in range(n):
        prob += d[i] <= Q[i], f"Queue_Digestion_Upper_Bound_{i}"
    
    # 3. Resource constraints
    prob += lpSum([u[i] * p[i] for i in range(n)]) <= N_cpu, "CPU_Constraint"
    prob += lpSum([g[i] * p[i] for i in range(n)]) <= N_gpu, "GPU_Constraint"
    
    # --- Solve ---
    prob.solve(PULP_CBC_CMD(msg=0))
    
    if prob.status == LpStatusOptimal:
        return [max(1, int(p[i].varValue)) for i in range(n)]
    return None


def milp_solver_relative_deviation(
    n: int,
    UT: List[float],  # Unit throughput for each operator
    u: List[float],   # CPU usage per actor for each operator
    g: List[float],   # GPU usage per actor for each operator
    D_i: List[float], # Input data volume processed by each operator (D_i/D_{i-1} ratio)
    D_o: float,       # Final output data volume
    N_cpu: float,     # Total available CPU
    N_gpu: float,     # Total available GPU
    B_current: List[float],  # Current buffer size for each operator (starting from op 2)
    B_target: List[float],   # Target buffer size for each operator
    T: float,         # Time horizon for planning
    alpha: float = 1.0,  # Weight for deviation penalty
    w: Optional[List[float]] = None,  # Weights for each operator (optional)
    tau_ref: Optional[float] = None,  # Reference throughput for normalization
    ema_wall_time: Optional[List[float]] = None,  # EMA wall time for weight calculation
) -> Optional[List[int]]:
    """
    Algorithm 2: Relative Deviation with Weights MILP Solver.

    Objective: max τ/τ_ref - α * Σw_i * (δ_i+ + δ_i-) / B_target_i

    This algorithm maintains queue size within a target range using
    weighted relative deviation penalty.

    Args:
        ema_wall_time: EMA wall time for each operator. If provided, weights are
            calculated based on running time (longer running operators get higher weight).
    """
    if n == 0:
        return []

    # --- Default weights based on ema_wall_time (longer running ops get higher weight) ---
    if w is None:
        if ema_wall_time is not None and len(ema_wall_time) == n:
            # Use ema_wall_time for weight calculation
            total_time = sum(t for t in ema_wall_time if t > 0)
            if total_time > 0:
                w = [t / total_time if t > 0 else 1 / n for t in ema_wall_time]
            else:
                w = [1 / n for _ in range(n)]
        else:
            # Fallback: equal weights
            w = [1 / n for _ in range(n)]
    
    # --- Calculate tau_ref if not provided ---
    if tau_ref is None:
        # Use max possible throughput as reference
        tau_ref_candidates = []
        for i in range(n):
            if D_i[i] > 0:
                cpu_limit = N_cpu / u[i] if u[i] > 0 else float('inf')
                gpu_limit = N_gpu / g[i] if g[i] > 0 else float('inf')
                p_max = min(cpu_limit, gpu_limit)
                scaling_factor = D_o / D_i[i]
                tau_ref_candidates.append(scaling_factor * UT[i] * p_max)
        tau_ref = min(tau_ref_candidates) if tau_ref_candidates else 1.0
    tau_ref = max(tau_ref, 1e-6)

    # --- Variables ---
    p = LpVariable.dicts("p", range(n), lowBound=1, cat='Integer')
    tau = LpVariable("tau", lowBound=0)
    # δ+ and δ- for buffer deviation (for operators 2 to n, index 1 to n-1)
    delta_plus = LpVariable.dicts("delta_plus", range(1, n), lowBound=0)
    delta_minus = LpVariable.dicts("delta_minus", range(1, n), lowBound=0)

    # --- Problem ---
    prob = LpProblem("Relative_Deviation_Weighted", LpMaximize)

    # --- Objective: max τ/τ_ref - α * Σw_i * (δ_i+ + δ_i-) / B_target_i ---
    # Note: PuLP doesn't support LpVariable / float, so we multiply by inverse
    deviation_term = lpSum([
        w[i] * (delta_plus[i] + delta_minus[i]) * (1.0 / max(B_target[i], 1e-6))
        for i in range(1, n)
    ])
    prob += tau * (1.0 / tau_ref) - alpha * deviation_term

    # --- Constraints ---
    # 1. Throughput constraint: τ <= (D_o/D_i) * p_i * UT_i
    for i in range(n):
        if D_i[i] > 0:
            scaling_factor = D_o / D_i[i]
            prob += tau <= scaling_factor * p[i] * UT[i], f"Throughput_Constraint_{i}"

    # 2. Resource constraints
    prob += lpSum([u[i] * p[i] for i in range(n)]) <= N_cpu, "CPU_Constraint"
    prob += lpSum([g[i] * p[i] for i in range(n)]) <= N_gpu, "GPU_Constraint"

    # 3. Buffer balance constraint for i = 2 to n (index 1 to n-1)
    # B_current + T * (D_i/D_{i-1} * p_{i-1} * UT_{i-1} - p_i * UT_i) - B_target = δ+ - δ-
    for i in range(1, n):
        if D_i[i-1] > 0:
            input_rate_ratio = D_i[i] / D_i[i-1]  # D_i / D_{i-1}
            production_rate = input_rate_ratio * p[i-1] * UT[i-1]
            consumption_rate = p[i] * UT[i]
            buffer_change = T * (production_rate - consumption_rate)
            prob += (B_current[i] + buffer_change - B_target[i] ==
                    delta_plus[i] - delta_minus[i]), f"Buffer_Balance_{i}"

    # --- Solve ---
    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status == LpStatusOptimal:
        return [max(1, int(p[i].varValue)) for i in range(n)]
    return None


def milp_solver_time_unified(
    n: int,
    UT: List[float],  # Unit throughput for each operator
    u: List[float],   # CPU usage per actor for each operator
    g: List[float],   # GPU usage per actor for each operator
    D_i: List[float], # Input data volume processed by each operator
    D_o: float,       # Final output data volume
    N_cpu: float,     # Total available CPU
    N_gpu: float,     # Total available GPU
    B_current: List[float],  # Current buffer size for each operator
    B_target: List[float],   # Target buffer size for each operator
    T: float,         # Time horizon for planning
    alpha: float = 1.0,  # Weight for deviation penalty
) -> Optional[List[int]]:
    """
    Algorithm 3: Time Scale Unified MILP Solver.

    Objective: max τ - α * Σ(δ_i+ + δ_i-) / T

    This algorithm maintains queue size within a target range.
    By dividing deviation by T, the term (δ/T) has the unit of "records/sec",
    which is the same as τ. So α represents: how much throughput loss
    corresponds to one unit of buffer deviation rate.
    """
    if n == 0:
        return []

    # --- Variables ---
    p = LpVariable.dicts("p", range(n), lowBound=1, cat='Integer')
    tau = LpVariable("tau", lowBound=0)
    # δ+ and δ- for buffer deviation (for operators 2 to n, index 1 to n-1)
    delta_plus = LpVariable.dicts("delta_plus", range(1, n), lowBound=0)
    delta_minus = LpVariable.dicts("delta_minus", range(1, n), lowBound=0)

    # --- Problem ---
    prob = LpProblem("Time_Scale_Unified", LpMaximize)

    # --- Objective: max τ - α * Σ(δ_i+ + δ_i-) / T ---
    # Note: PuLP doesn't support LpVariable / float, so we multiply by inverse
    deviation_term = lpSum([
        (delta_plus[i] + delta_minus[i]) * (1.0 / T)
        for i in range(1, n)
    ])
    prob += tau - alpha * deviation_term

    # --- Constraints ---
    # 1. Throughput constraint: τ <= (D_o/D_i) * p_i * UT_i
    for i in range(n):
        if D_i[i] > 0:
            scaling_factor = D_o / D_i[i]
            prob += tau <= scaling_factor * p[i] * UT[i], f"Throughput_Constraint_{i}"

    # 2. Resource constraints
    prob += lpSum([u[i] * p[i] for i in range(n)]) <= N_cpu, "CPU_Constraint"
    prob += lpSum([g[i] * p[i] for i in range(n)]) <= N_gpu, "GPU_Constraint"

    # 3. Buffer balance constraint for i = 2 to n (index 1 to n-1)
    # B_current + T * (D_i/D_{i-1} * p_{i-1} * UT_{i-1} - p_i * UT_i) - B_target = δ+ - δ-
    for i in range(1, n):
        if D_i[i-1] > 0:
            input_rate_ratio = D_i[i] / D_i[i-1]  # D_i / D_{i-1}
            production_rate = input_rate_ratio * p[i-1] * UT[i-1]
            consumption_rate = p[i] * UT[i]
            buffer_change = T * (production_rate - consumption_rate)
            prob += (B_current[i] + buffer_change - B_target[i] ==
                    delta_plus[i] - delta_minus[i]), f"Buffer_Balance_{i}"

    # --- Solve ---
    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status == LpStatusOptimal:
        return [max(1, int(p[i].varValue)) for i in range(n)]
    return None


if __name__ == "__main__":
    """Test the three MILP solvers with sample data."""

    # --- Test Parameters ---
    n = 7  # Total number of operations

    UT = [3.97, 113.58, 1.2, 3.28, 2.57, 4.65, 0.0289]  # Unit throughput
    u = [1.529, 1.685, 10.07, 2.1, 1.53, 1.52, 10.86]   # CPU usage per actor
    g = [0, 1, 0, 1, 0, 1, 0]  # GPU usage per actor

    D_i = [91, 91, 91, 33, 20, 20, 8]  # Input data volume
    D_o = 8  # Final output data volume

    N_cpu = 512    # Total available CPU
    N_gpu = 8      # Total available GPUs

    # Queue/Buffer parameters
    Q = [100, 200, 150, 80, 60, 40, 20]  # Current queue size (Q_target = 0)

    B_current = [0, 200, 150, 80, 60, 40, 20]  # Current buffer (first is ignored)
    B_target = [0, 100, 75, 40, 30, 20, 10]    # Target buffer

    T = 60.0  # Time horizon (seconds)

    print("=" * 60)
    print("Testing MILP Solvers with Queue Size Consideration")
    print("=" * 60)

    # Test Algorithm 1: Queue Digestion Priority (Q_target = 0)
    print("\n--- Algorithm 1: Queue Digestion Priority (Q_target = 0) ---")
    result1 = milp_solver_queue_digestion(
        n, UT, u, g, D_i, D_o, N_cpu, N_gpu, Q, T, beta=1.0
    )
    print(f"Result: {result1}")

    # Test Algorithm 2: Relative Deviation with Weights
    print("\n--- Algorithm 2: Relative Deviation with Weights ---")
    result2 = milp_solver_relative_deviation(
        n, UT, u, g, D_i, D_o, N_cpu, N_gpu, B_current, B_target, T, alpha=1.0
    )
    print(f"Result: {result2}")

    # Test Algorithm 3: Time Scale Unified
    print("\n--- Algorithm 3: Time Scale Unified ---")
    result3 = milp_solver_time_unified(
        n, UT, u, g, D_i, D_o, N_cpu, N_gpu, B_current, B_target, T, alpha=1.0
    )
    print(f"Result: {result3}")

    print("\n" + "=" * 60)
    print("All tests completed!")
