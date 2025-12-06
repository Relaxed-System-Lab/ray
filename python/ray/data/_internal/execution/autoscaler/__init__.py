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
    from .ds2_autoscaler import DS2Autoscaler

    # Use DS2Autoscaler instead of DefaultAutoscaler
    return DS2Autoscaler(
        topology,
        resource_manager,
        execution_id=execution_id,
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
