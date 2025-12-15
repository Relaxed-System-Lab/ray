"""Unit tests for the placement-aware MILP solver.

These tests verify the correctness of the placement optimization model
including resource constraints, data flow conservation, and migration costs.
"""

import pytest

from ray.data._internal.execution.autoscaler.ds2_placement_milp_solver import (
    NodeResources,
    OperatorResources,
    PlacementSolverInput,
    PlacementSolverOutput,
    milp_placement_solver,
)


class TestPlacementMILPSolver:
    """Test cases for the placement-aware MILP solver."""

    def test_single_operator_single_node(self):
        """Test basic case: single operator on single node."""
        solver_input = PlacementSolverInput(
            n=1,
            K=1,
            UT=[10.0],  # 10 records/second per instance
            D_i=[100.0],  # 100 input records
            D_o=100.0,  # 100 output records (no expansion)
            s=[0.001],  # 1KB per record
            op_resources=[OperatorResources(cpu=1.0, memory=1e9, gpu=0.0)],
            node_resources=[NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=0.0)],
            x_bar=[[0]],  # No current placement
            c_start=[2.0],
            c_stop=[0.5],
            epsilon_1=0.001,
            epsilon_2=0.001,
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        assert result.status == "Optimal"
        # Should place at least 1 instance
        assert sum(result.x[0]) >= 1
        # Total parallelism should match placement
        assert result.p[0] == sum(result.x[0])
        # Throughput should be positive
        assert result.throughput > 0

    def test_resource_constraints_cpu(self):
        """Test that CPU constraints are respected."""
        solver_input = PlacementSolverInput(
            n=1,
            K=1,
            UT=[10.0],
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=4.0, memory=1e9, gpu=0.0)],  # 4 CPU per instance
            node_resources=[NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=0.0)],  # 8 CPU total
            x_bar=[[0]],
            c_start=[2.0],
            c_stop=[0.5],
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # Should place at most 2 instances (8 CPU / 4 CPU per instance)
        assert result.x[0][0] <= 2

    def test_resource_constraints_gpu(self):
        """Test that GPU constraints are respected."""
        solver_input = PlacementSolverInput(
            n=1,
            K=1,
            UT=[10.0],
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=1.0, memory=1e9, gpu=1.0)],  # 1 GPU per instance
            node_resources=[NodeResources(node_id="node1", cpu=32.0, memory=64e9, gpu=4.0)],  # 4 GPUs
            x_bar=[[0]],
            c_start=[12.0],  # Higher startup cost for GPU
            c_stop=[1.5],
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # Should place at most 4 instances (4 GPUs / 1 GPU per instance)
        assert result.x[0][0] <= 4

    def test_multi_node_placement(self):
        """Test placement across multiple nodes."""
        solver_input = PlacementSolverInput(
            n=1,
            K=2,
            UT=[10.0],
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=2.0, memory=1e9, gpu=0.0)],
            node_resources=[
                NodeResources(node_id="node1", cpu=4.0, memory=8e9, gpu=0.0),  # Can fit 2
                NodeResources(node_id="node2", cpu=4.0, memory=8e9, gpu=0.0),  # Can fit 2
            ],
            x_bar=[[0, 0]],
            c_start=[2.0],
            c_stop=[0.5],
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # Total placement should be sum across nodes
        assert result.p[0] == result.x[0][0] + result.x[0][1]
        # Each node should respect its CPU limit
        assert result.x[0][0] <= 2
        assert result.x[0][1] <= 2

    def test_data_flow_conservation(self):
        """Test data flow conservation between operators."""
        solver_input = PlacementSolverInput(
            n=2,
            K=2,
            UT=[10.0, 10.0],
            D_i=[100.0, 100.0],
            D_o=100.0,
            s=[0.001, 0.001],
            op_resources=[
                OperatorResources(cpu=2.0, memory=1e9, gpu=0.0),
                OperatorResources(cpu=2.0, memory=1e9, gpu=0.0),
            ],
            node_resources=[
                NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=0.0),
                NodeResources(node_id="node2", cpu=8.0, memory=16e9, gpu=0.0),
            ],
            x_bar=[[0, 0], [0, 0]],
            c_start=[2.0, 2.0],
            c_stop=[0.5, 0.5],
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # Data flow should be conserved
        # For each node k: sum_l w[0][k][l] == x[0][k] (outflow from op 0)
        for k in range(2):
            outflow = sum(result.w[0][k])
            assert outflow == result.x[0][k], f"Outflow mismatch at node {k}"

        # For each node l: sum_k w[0][k][l] == x[1][l] (inflow to op 1)
        for l in range(2):
            inflow = sum(result.w[0][k][l] for k in range(2))
            assert inflow == result.x[1][l], f"Inflow mismatch at node {l}"

    def test_migration_cost_from_existing_placement(self):
        """Test that migration cost is calculated correctly."""
        # Start with 2 instances on node 0
        solver_input = PlacementSolverInput(
            n=1,
            K=2,
            UT=[10.0],
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=2.0, memory=1e9, gpu=0.0)],
            node_resources=[
                NodeResources(node_id="node1", cpu=4.0, memory=8e9, gpu=0.0),
                NodeResources(node_id="node2", cpu=4.0, memory=8e9, gpu=0.0),
            ],
            x_bar=[[2, 0]],  # Currently 2 on node1, 0 on node2
            c_start=[2.0],
            c_stop=[0.5],
            epsilon_2=0.001,  # Small migration penalty
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # Migration cost should be non-negative
        assert result.migration_cost >= 0

    def test_heterogeneous_nodes(self):
        """Test placement on heterogeneous nodes with different resources."""
        solver_input = PlacementSolverInput(
            n=1,
            K=2,
            UT=[10.0],
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=2.0, memory=1e9, gpu=1.0)],
            node_resources=[
                NodeResources(node_id="cpu_node", cpu=16.0, memory=32e9, gpu=0.0),  # No GPU
                NodeResources(node_id="gpu_node", cpu=8.0, memory=16e9, gpu=4.0),   # Has GPU
            ],
            x_bar=[[0, 0]],
            c_start=[12.0],
            c_stop=[1.5],
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # All instances should be on the GPU node since operator requires GPU
        assert result.x[0][0] == 0, "Should not place on CPU-only node"
        assert result.x[0][1] >= 1, "Should place on GPU node"

    def test_empty_problem(self):
        """Test handling of empty problem (no operators or nodes)."""
        # No operators
        solver_input = PlacementSolverInput(
            n=0,
            K=1,
            UT=[],
            D_i=[],
            D_o=1.0,
            s=[],
            op_resources=[],
            node_resources=[NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=0.0)],
            x_bar=[],
            c_start=[],
            c_stop=[],
        )

        result = milp_placement_solver(solver_input)
        assert result is None

    def test_pipeline_with_expansion(self):
        """Test pipeline where data expands between operators."""
        solver_input = PlacementSolverInput(
            n=2,
            K=1,
            UT=[10.0, 5.0],  # Second operator is slower
            D_i=[100.0, 200.0],  # Data doubles after first operator
            D_o=200.0,
            s=[0.001, 0.002],  # Output size also increases
            op_resources=[
                OperatorResources(cpu=1.0, memory=1e9, gpu=0.0),
                OperatorResources(cpu=2.0, memory=2e9, gpu=0.0),
            ],
            node_resources=[NodeResources(node_id="node1", cpu=32.0, memory=64e9, gpu=0.0)],
            x_bar=[[0], [0]],
            c_start=[2.0, 2.0],
            c_stop=[0.5, 0.5],
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        assert result.status == "Optimal"
        # Both operators should have instances
        assert result.p[0] >= 1
        assert result.p[1] >= 1

    def test_egress_constraint(self):
        """Test that egress traffic is tracked."""
        solver_input = PlacementSolverInput(
            n=2,
            K=2,
            UT=[10.0, 10.0],
            D_i=[100.0, 100.0],
            D_o=100.0,
            s=[1.0, 1.0],  # 1 MB per record (large data)
            op_resources=[
                OperatorResources(cpu=2.0, memory=1e9, gpu=0.0),
                OperatorResources(cpu=2.0, memory=1e9, gpu=0.0),
            ],
            node_resources=[
                NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=0.0),
                NodeResources(node_id="node2", cpu=8.0, memory=16e9, gpu=0.0),
            ],
            x_bar=[[0, 0], [0, 0]],
            c_start=[2.0, 2.0],
            c_stop=[0.5, 0.5],
            epsilon_1=0.001,  # Network penalty
        )

        result = milp_placement_solver(solver_input)

        assert result is not None
        # Egress should be tracked (may be 0 if all data stays local)
        assert result.egress_max >= 0


class TestPlacementSolverEdgeCases:
    """Edge case tests for the placement solver."""

    def test_infeasible_resource_requirements(self):
        """Test handling of infeasible resource requirements."""
        solver_input = PlacementSolverInput(
            n=1,
            K=1,
            UT=[10.0],
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=100.0, memory=1e12, gpu=10.0)],  # Huge requirements
            node_resources=[NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=1.0)],  # Small node
            x_bar=[[0]],
            c_start=[2.0],
            c_stop=[0.5],
        )

        result = milp_placement_solver(solver_input)

        # Should return None for infeasible problem
        assert result is None

    def test_zero_throughput_operator(self):
        """Test handling of operator with zero throughput."""
        solver_input = PlacementSolverInput(
            n=1,
            K=1,
            UT=[0.0],  # Zero throughput
            D_i=[100.0],
            D_o=100.0,
            s=[0.001],
            op_resources=[OperatorResources(cpu=1.0, memory=1e9, gpu=0.0)],
            node_resources=[NodeResources(node_id="node1", cpu=8.0, memory=16e9, gpu=0.0)],
            x_bar=[[0]],
            c_start=[2.0],
            c_stop=[0.5],
        )

        result = milp_placement_solver(solver_input)

        # Should still return a result (solver handles this gracefully)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

