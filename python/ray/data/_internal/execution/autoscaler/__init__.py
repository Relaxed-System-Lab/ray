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
    # Import DS2Autoscaler here to avoid circular import
    from .ds2_autoscaler import DS2Autoscaler, SolverType

    # Use DS2Autoscaler instead of DefaultAutoscaler
    # basic
    return DS2Autoscaler(
        topology,
        resource_manager,
        execution_id=execution_id,
    )

    # queue_digestion
    # return DS2Autoscaler(
    #     topology,
    #     resource_manager,
    #     execution_id=execution_id,
    #     solver_type=SolverType.QUEUE_DIGESTION,
    #     solver_weight=2,
    # )

    # relative_deviation
    # return DS2Autoscaler(
    #     topology,
    #     resource_manager,
    #     execution_id=execution_id,
    #     solver_type=SolverType.RELATIVE_DEVIATION,
    #     solver_weight=2,
    #     target_queue_sizes=[20.0] * 9,
    # )

    # time unified
    # return DS2Autoscaler(
    #     topology,
    #     resource_manager,
    #     execution_id=execution_id,
    #     solver_type=SolverType.TIME_UNIFIED,
    #     solver_weight=2,
    #     target_queue_sizes=[256.0] * 9,
    # )


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
