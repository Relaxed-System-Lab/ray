import logging
from typing import TYPE_CHECKING, List

import ray
from .autoscaler import Autoscaler
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator
from ray.data._internal.execution.autoscaler.ds2_milp_solver import milp_solver

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology

logger = logging.getLogger(__name__)


class DS2Autoscaler(Autoscaler):
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 60

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
    ):
        super().__init__(topology, resource_manager, execution_id)

        # Last time when a request was sent to Ray's autoscaler.
        self._last_request_time = 0

    def try_trigger_scaling(self):
        pass

    def on_executor_shutdown(self):
        pass

    def get_total_resources(self) -> ExecutionResources:
        return ExecutionResources.from_resource_dict(ray.cluster_resources()) 

    def ds2_scaling(self):
        wall_time_list = self.get_wall_time()
        num_processed_rows_list = self.get_num_processed_rows()
        per_actor_resource_usage_list = self.get_per_actor_resource_usage()
        total_resources = self.get_total_resources()
        n = len(wall_time_list)
        unit_throughput_list = []
        
        all_work = True
        for wall_time, num_rows in zip(
            wall_time_list, num_processed_rows_list
        ):
            if wall_time > 0:
                unit_throughput_list.append(num_rows / wall_time)
            else:
                all_work = False
        if not all_work:
            logger.info(
                "Not all operators have processed data yet. Skipping DS2 autoscaling."
            )
            return
        cpu_usage_list = []
        for per_actor_resource_usage in per_actor_resource_usage_list:
            if per_actor_resource_usage._cpu is None:
                raise ValueError("CPU usage cannot be None")
            else:
                cpu_usage_list.append(per_actor_resource_usage._cpu)
        gpu_usage_list = []
        for per_actor_resource_usage in per_actor_resource_usage_list:
            if per_actor_resource_usage._gpu is None:
                gpu_usage_list.append(0)
            else:
                gpu_usage_list.append(per_actor_resource_usage._gpu)
        D_o = None
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                D_o = op._metrics.rows_task_outputs_generated
        N_cpu = total_resources._cpu
        N_gpu = total_resources._gpu
        concurrency = milp_solver(
            n, unit_throughput_list, cpu_usage_list, gpu_usage_list,
            num_processed_rows_list, D_o, N_cpu, N_gpu,
        )



    def get_wall_time(self) -> List[float]:
        wall_time_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                wall_time_list.append(op._metrics.block_generation_time)
        return wall_time_list

    def get_num_processed_rows(self) -> List[int]:
        num_processed_rows_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                num_processed_rows_list.append(op._metrics.rows_task_inputs_processed)
        return num_processed_rows_list
    
    def get_per_actor_resource_usage(self) -> List[ExecutionResources]:
        per_actor_resource_usage_list = []
        for op in self._topology:
            if isinstance(op, ActorPoolMapOperator):
                per_actor_resource_usage_list.append(
                    op.get_per_actor_resource_usage()
                )
        return per_actor_resource_usage_list
    