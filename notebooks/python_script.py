from pathlib import Path

from nuplan.planning.utils.multithreading.worker_ray import RayDistributed

from asim.dataset.dataset_specific.nuplan.nuplan_data_processor import NuplanDataProcessor

worker = RayDistributed(threads_per_node=16)
dataset = NuplanDataProcessor(splits=["nuplan_val"], output_path=Path("/home/daniel/asim_workspace/data"))
dataset.convert(worker)
