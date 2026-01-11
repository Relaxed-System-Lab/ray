from typing import TYPE_CHECKING

from .autoscaler import Autoscaler
from .autoscaling_actor_pool import AutoscalingActorPool
from .default_autoscaler import DefaultAutoscaler

if TYPE_CHECKING:
    from ..resource_manager import ResourceManager
    from ..streaming_executor_state import Topology
    from ray.data.context import AutoscalingConfig


def create_autoscaler(
    topology: "Topology",
    resource_manager: "ResourceManager",
    config: "AutoscalingConfig",
    *,
    execution_id: str
) -> Autoscaler:
    """Create an autoscaler based on the config.

    Args:
        topology: The streaming topology.
        resource_manager: The resource manager.
        config: The autoscaling configuration.
        execution_id: The execution ID.

    Returns:
        The autoscaler instance.

    Supported autoscaler types:
        - "default": DefaultAutoscaler (utilization-based scaling)
        - "ds2": DS2Autoscaler (DS2 algorithm with various solvers)
        - "real_ds2": RealDS2Autoscaler (closer to original DS2 paper)
        - "conttune": ContTuneAutoscaler (Conservative Bayesian Optimization)
    """
    autoscaler_type = config.autoscaler_type

    if autoscaler_type == "default":
        return DefaultAutoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
            config=config,
        )
    elif autoscaler_type == "ds2":
        from .ds2_autoscaler import DS2Autoscaler, SolverType

        # Map string solver type to enum
        solver_type_map = {
            "BASIC": SolverType.BASIC,
            "QUEUE_DIGESTION": SolverType.QUEUE_DIGESTION,
            "RELATIVE_DEVIATION": SolverType.RELATIVE_DEVIATION,
            "TIME_UNIFIED": SolverType.TIME_UNIFIED,
        }
        solver_type = solver_type_map.get(config.ds2_solver_type, SolverType.BASIC)

        return DS2Autoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
            solver_type=solver_type,
            solver_weight=config.ds2_solver_weight,
            time_horizon=config.ds2_time_horizon,
            target_queue_sizes=config.ds2_target_queue_sizes,
        )
    elif autoscaler_type == "real_ds2":
        from .real_ds2_autoscaler import RealDS2Autoscaler

        return RealDS2Autoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
            max_parallelism=config.max_parallelism,
            use_incremental_output_rate=config.use_incremental_output_rate,
        )
    elif autoscaler_type == "conttune":
        from .conttune_autoscaler import ContTuneAutoscaler

        return ContTuneAutoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
            alpha=config.conttune_alpha,
            max_parallelism=config.max_parallelism,
            k=config.conttune_k,
            use_incremental_output_rate=config.use_incremental_output_rate,
        )
    else:
        raise ValueError(
            f"Unknown autoscaler type: {autoscaler_type}. "
            f"Supported types: 'default', 'ds2', 'real_ds2', 'conttune'"
        )


# Lazy import for DS2Autoscaler to avoid circular import at module level
def __getattr__(name):
    if name == "DS2Autoscaler":
        from .ds2_autoscaler import DS2Autoscaler
        return DS2Autoscaler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Autoscaler",
    "DefaultAutoscaler",
    "DS2Autoscaler",
    "create_autoscaler",
    "AutoscalingActorPool",
]
