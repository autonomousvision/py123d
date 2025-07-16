from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import HeteroData

from asim.common.datatypes.detection.detection import BoxDetection, BoxDetectionSE2
from asim.common.geometry.base import StateSE2
from asim.common.geometry.bounding_box.bounding_box import BoundingBoxSE2
from asim.common.geometry.transform.se2_array import convert_relative_to_absolute_point_2d_array
from asim.common.geometry.utils import normalize_angle
from asim.dataset.arrow.conversion import BoxDetectionWrapper, DetectionType
from asim.dataset.maps.abstract_map import AbstractMap
from asim.dataset.scene.abstract_scene import AbstractScene
from asim.simulation.agents.abstract_agents import AbstractAgents
from asim.training.feature_builder.smart_feature_builder import SMARTFeatureBuilder
from asim.training.models.sim_agent.smart.datamodules.target_builder import _numpy_dict_to_torch
from asim.training.models.sim_agent.smart.smart import SMART
from asim.training.models.sim_agent.smart.smart_config import SMARTConfig


class SMARTAgents(AbstractAgents):

    # Whether the agent class requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scene: bool = True

    def __init__(self) -> None:
        """
        Initialize the constant velocity agents.
        """
        super().__init__()
        self._timestep_s: float = 0.1
        self._current_iteration: int = 0
        self._map_api: AbstractMap = None

        checkpoint_path = Path(
            "/home/daniel/asim_workspace/exp/smart_mini_run/2025.06.23.20.45.20/checkpoints/epoch_050.ckpt"
        )
        # checkpoint_path = Path("/home/daniel/epoch_050.ckpt")
        # checkpoint_path = Path("/home/daniel/epoch_027.ckpt")
        # checkpoint_path = Path("/home/daniel/epoch_008.ckpt")
        config = SMARTConfig(
            hidden_dim=64,
            num_freq_bands=64,
            num_heads=4,
            head_dim=8,
            dropout=0.1,
            hist_drop_prob=0.1,
            num_map_layers=2,
            num_agent_layers=4,
            pl2pl_radius=10,
            pl2a_radius=20,
            a2a_radius=20,
            time_span=20,
            num_historical_steps=11,
            num_future_steps=90,
        )

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._smart_model = SMART.load_from_checkpoint(
            checkpoint_path, config=config, strict=False, map_location=self._device
        )
        self._smart_model.eval()
        self._smart_model.to(self._device)

        self._smart_model.encoder.agent_encoder.num_future_steps = 150
        self._smart_model.validation_rollout_sampling.num_k = 1

        self._initial_box_detections: Optional[BoxDetectionWrapper] = None
        self._agent_indices: List[int] = []

    @abstractmethod
    def reset(
        self,
        map_api: AbstractMap,
        target_agents: List[BoxDetection],
        non_target_agents: List[BoxDetection],
        scene: Optional[AbstractScene] = None,
    ) -> List[BoxDetection]:
        assert scene is not None
        self._current_iteration = 0

        feature_builder = SMARTFeatureBuilder()
        features = feature_builder.build_features(scene)
        self._agent_indices = features["agent"]["id"].tolist()
        _numpy_dict_to_torch(features, device="cpu")
        torch_features = HeteroData(features)
        from torch_geometric.loader import DataLoader

        # If you have a dataset
        dataset = [torch_features]  # List with single sample
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch.to(self._device)
                pred_traj, pred_z, pred_head = self._smart_model.test_step(batch, 0)
                break

        origin = scene.get_ego_vehicle_state_at_iteration(0).bounding_box.center.state_se2

        self._pred_traj = convert_relative_to_absolute_point_2d_array(origin, pred_traj.cpu().numpy())
        self._pred_z = pred_z.cpu().numpy()
        self._pred_head = normalize_angle(pred_head.cpu().numpy() + origin.yaw)

        self._initial_box_detections = scene.get_box_detections_at_iteration(0)

        # self._initial_target_agents = [copy.deepcopy(agent) for agent in target_agents]
        return target_agents

    def step(self, non_target_agents: List[BoxDetection]):

        # (16, 10, 80, 2)
        pred_traj = self._pred_traj[:, 0]
        pred_head = self._pred_head[:, 0]

        current_target_agents = []
        for agent_idx, agent_id in enumerate(self._agent_indices):
            if agent_id == -1:
                continue

            initial_agent = self._initial_box_detections[agent_id]
            if initial_agent.metadata.detection_type != DetectionType.VEHICLE:
                continue

            new_center = StateSE2(
                x=pred_traj[agent_idx, self._current_iteration, 0],
                y=pred_traj[agent_idx, self._current_iteration, 1],
                yaw=pred_head[agent_idx, self._current_iteration],
            )
            propagated_bounding_box = BoundingBoxSE2(
                new_center,
                initial_agent.bounding_box_se2.length,
                initial_agent.bounding_box_se2.width,
            )
            new_velocity = initial_agent.velocity
            propagated_agent: BoxDetectionSE2 = BoxDetectionSE2(
                metadata=initial_agent.metadata,
                bounding_box_se2=propagated_bounding_box,
                velocity=new_velocity,
            )
            current_target_agents.append(propagated_agent)

        self._current_iteration += 1
        return current_target_agents
