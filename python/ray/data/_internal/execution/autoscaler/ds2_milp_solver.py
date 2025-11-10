from pulp import *

def milp_solver(n, UT, u, n_i, D_i, D_o, N_cpu, N_gpu):
    # --- Variables ---
    p = LpVariable.dicts("p", range(1, n + 1), lowBound=1, cat='Integer')
    tau = LpVariable("tau", lowBound=0)

    # 1. Initialize the optimization problem
    prob = LpProblem("Scaled_Pipeline_Throughput_Optimization", LpMaximize)
    # --- Objective Function (Stage 1) ---
    prob += tau, "Maximize_System_Throughput"

    # --- Constraints ---
    # 1. Scaled Throughput Constraint
    for i in range(1, n + 1):
        scaling_factor = D_o / D_i[i-1]  # 改为 i-1 来访问 list
        prob += tau <= scaling_factor * p[i] * UT[i-1], f"Scaled_Throughput_Constraint_Op_{i}"  # 改为 i-1

    # 2. Resource Constraints
    prob += lpSum([u[i-1] * p[i] for i in range(1, n + 1)]) <= N_cpu, "Total_CPU_Core_Constraint"  # 改为 i-1
    prob += lpSum([n_i[i-1] * p[i] for i in range(1, n + 1)]) <= N_gpu, "Total_GPU_Constraint"  # 改为 i-1
    for i in range(2, n):
        prob += p[1] * UT[0] * D_i[i]  <= p[i + 1] * UT[i] * D_i[0], f"Process_Rate_Constraint_{i}_{i+1}"  # 改为使用 list 索引

    # --- Solve Stage 1: Find Optimal Throughput ---
    prob.solve(PULP_CBC_CMD(msg=0))

    print("--- Stage 1: Maximizing Throughput ---")
    print(f"Status: {LpStatus[prob.status]}")

    # Check if an optimal solution was found
    if prob.status == LpStatusOptimal:
        # Get the optimal throughput value
        optimal_tau = tau.varValue
        print(f"\n✅ Optimal Throughput Found: τ = {optimal_tau:.2f} items/sec")

        # --- Stage 2: Minimize Parallelism for the Optimal Throughput ---
        print("\n--- Stage 2: Minimizing Parallelism for Optimal Throughput ---")

        # Add a new constraint to fix the throughput to the optimal value found in stage 1.
        # We add a small tolerance to avoid potential floating-point precision issues.
        prob += tau >= optimal_tau - 1e-6, "Fix_Optimal_Throughput_With_Tolerance"

        # Change the objective function to minimize the sum of parallelisms
        prob.setObjective(lpSum([p[i] for i in range(1, n + 1)]))
        prob.sense = LpMinimize # Set the problem to a minimization problem

        # Solve the modified problem
        prob.solve(PULP_CBC_CMD(msg=0))

        print(f"Status: {LpStatus[prob.status]}")
        if prob.status == LpStatusOptimal:
            print("\n✅ Final Optimal Solution with Minimized Parallelism:")
            for i in range(1, n + 1):
                print(f"  - Parallelism for Operation {i} (p_{i}): {int(p[i].varValue)}")

            total_parallelism = sum(p[i].varValue for i in range(1, n + 1))
            print(f"\nMinimized Total Parallelism: {int(total_parallelism)}")
            print(f"Maximized System Throughput (τ): {tau.varValue:.2f} items/sec")

            # Optional: Print resource utilization to verify the solution
            total_cpu_used = sum(u[i-1] * p[i].varValue for i in range(1, n + 1))  # 改为 i-1
            total_gpu_used = sum(n_i[i-1] * p[i].varValue for i in range(1, n + 1))  # 改为 i-1

            print("\n📊 Final Resource Utilization:")
            print(f"  - CPU Cores: {total_cpu_used:.2f}/{N_cpu} ({(total_cpu_used/N_cpu)*100:.2f}%)")
            print(f"  - GPUs: {total_gpu_used:.0f}/{N_gpu} ({(total_gpu_used/N_gpu)*100:.2f}%)")
        else:
            print("\n❌ No optimal solution could be found for minimizing parallelism.")
        concurrency = [int(p[i].varValue) for i in range(1, n + 1)]
        return concurrency

    else:
        print("\n❌ No optimal solution could be found in Stage 1.")

if __name__ == "__main__":
    # --- Parameters ---
    # Using the second set of UT values from your example
    n = 7  # Total number of operations
    
    # 改为 list，索引从 0 开始
    UT = [3.97, 113.58, 1.2, 3.28, 2.57, 4.65, 0.0289]  # Unit parallelism throughput
    u = [1.529, 1.685, 10.07, 2.1, 1.53, 1.52, 10.86]  # Unit parallelism CPU utilization
    n_i = [0, 1, 0, 1, 0, 1, 0]  # Unit parallelism GPU requirement

    # Data volume parameters
    D_i = [91, 91, 91, 33, 20, 20, 8]  # Input data volume for operator i
    D_o = 8  # Final output data volume

    # Total available resources
    N_cpu = 512    # Total available CPU cores
    M_cpu = 1024 * 1024  # Total available CPU memory (MB)
    N_gpu = 8       # Total available GPUs

    concurrency = milp_solver(n, UT, u, n_i, D_i, D_o, N_cpu, N_gpu)
    print(concurrency)