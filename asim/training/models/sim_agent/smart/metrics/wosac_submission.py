# import tarfile
# from pathlib import Path
# from typing import Dict, List

# import hydra
# from omegaconf import ListConfig
# from torch import Tensor
# from torchmetrics.metric import Metric
# from waymo_open_dataset.protos import sim_agents_submission_pb2

# from src.utils import RankedLogger
# from src.utils.wosac_utils import get_scenario_id_int_tensor

# log = RankedLogger(__name__, rank_zero_only=False)


# class WOSACSubmission(Metric):
#     def __init__(
#         self,
#         is_active: bool,
#         method_name: str,
#         authors: ListConfig[str],
#         affiliation: str,
#         description: str,
#         method_link: str,
#         account_name: str,
#     ) -> None:
#         super().__init__()
#         self.is_active = is_active
#         if self.is_active:
#             self.method_name = method_name
#             self.authors = authors
#             self.affiliation = affiliation
#             self.description = description
#             self.method_link = method_link
#             self.account_name = account_name
#             self.buffer_scenario_rollouts = []
#             self.i_file = 0
#             self.submission_dir = (
#                 hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
#             )
#             self.submission_dir = Path(self.submission_dir) / "wosac_submission"
#             self.submission_dir.mkdir(exist_ok=True)
#             self.submission_scenario_id = []

#             self.data_keys = [
#                 "scenario_id",
#                 "agent_id",
#                 "agent_batch",
#                 "pred_traj",
#                 "pred_z",
#                 "pred_head",
#             ]
#             for k in self.data_keys:
#                 self.add_state(k, default=[], dist_reduce_fx="cat")

#     def update(
#         self,
#         scenario_id: List[str],
#         agent_id: List[List[float]],
#         agent_batch: Tensor,
#         pred_traj: Tensor,
#         pred_z: Tensor,
#         pred_head: Tensor,
#         global_rank: int,
#     ) -> None:
#         _device = pred_traj.device
#         self.agent_id.append(agent_id)
#         self.scenario_id.append(get_scenario_id_int_tensor(scenario_id, _device))
#         self.pred_traj.append(pred_traj)
#         self.pred_z.append(pred_z)
#         self.pred_head.append(pred_head)

#         batch_size = len(scenario_id)
#         self.agent_batch.append(agent_batch + batch_size * global_rank)

#     def compute(self) -> Dict[str, Tensor]:
#         return {k: getattr(self, k) for k in self.data_keys}

#     def aggregate_rollouts(
#         self, scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]
#     ) -> None:
#         for rollout in scenario_rollouts:
#             if rollout.scenario_id not in self.submission_scenario_id:
#                 self.submission_scenario_id.append(rollout.scenario_id)
#                 self.buffer_scenario_rollouts.append(rollout)
#                 if len(self.buffer_scenario_rollouts) > 300:
#                     self._save_shard()

#     def save_sub_file(self) -> None:
#         self._save_shard()
#         self.i_file = 0
#         tar_file_name = self.submission_dir.as_posix() + ".tar.gz"

#         log.info(f"Saving wosac submission files to {tar_file_name}")

#         shard_files = sorted([p.as_posix() for p in self.submission_dir.glob("*")])
#         with tarfile.open(tar_file_name, "w:gz") as tar:
#             for output_filename in shard_files:
#                 tar.add(
#                     output_filename,
#                     arcname=output_filename + f"-of-{len(shard_files):05d}",
#                 )
#         log.info(f"DONE: Saved wosac submission files to {tar_file_name}")

#     def _save_shard(self) -> None:
#         shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
#             scenario_rollouts=self.buffer_scenario_rollouts,
#             submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
#             account_name=self.account_name,
#             unique_method_name=self.method_name,
#             authors=self.authors,
#             affiliation=self.affiliation,
#             description=self.description,
#             method_link=self.method_link,
#             uses_lidar_data=False,
#             uses_camera_data=False,
#             uses_public_model_pretraining=False,
#             num_model_parameters="7M",
#             acknowledge_complies_with_closed_loop_requirement=True,
#         )
#         output_filename = self.submission_dir / f"submission.binproto-{self.i_file:05d}"
#         log.info(f"Saving wosac submission files to {output_filename}")
#         with open(output_filename, "wb") as f:
#             f.write(shard_submission.SerializeToString())
#         self.i_file += 1
#         self.buffer_scenario_rollouts = []
