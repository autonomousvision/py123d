from pathlib import Path

from nuplan.planning.utils.multithreading.worker_ray import RayDistributed

from asim.dataset.dataset_specific.nuplan.data_conversion import NuPlanDataset

worker = RayDistributed(threads_per_node=16)
dataset = NuPlanDataset(splits=["nuplan_val"], output_path=Path("/home/daniel/asim_workspace/data"))
dataset.convert(worker)
