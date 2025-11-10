from typing import TYPE_CHECKING

from .autoscaler import Autoscaler
from .autoscaling_actor_pool import AutoscalingActorPool
from .default_autoscaler import DefaultAutoscaler
from .ds2_autoscaler import DS2Autoscaler

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
    # Use DS2Autoscaler instead of DefaultAutoscaler
    return DS2Autoscaler(
        topology,
        resource_manager,
        execution_id=execution_id,
    )


__all__ = [
    "Autoscaler",
    "DefaultAutoscaler",
    "DS2Autoscaler",
    "create_autoscaler",
    "AutoscalingActorPool",
]
