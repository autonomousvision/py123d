from typing import Any, Dict

import numpy as np
import torch
from scipy.interpolate import interp1d


def get_polylines_from_polygon(polygon: np.ndarray) -> np.ndarray:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(start - end)
        unit_vec = (end - start) / length
        pl = []
        for i in range(int(length) + 1):  # 4.5 -> 5 [0,1,2,3,4]
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])
        pl.append([end[0], end[1], end[2]])
        return np.array(pl)

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])
    return np.concatenate([pl1, pl1[::-1], pl2, pl2[::-1]], axis=0)


def _interplating_polyline(polylines, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = []
    polylines_list = []
    euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
    euclidean_dists = np.concatenate([[0], euclidean_dists])
    breakpoints = np.where(euclidean_dists > 3)[0]
    breakpoints = np.concatenate([[0], breakpoints, [polylines.shape[0]]])
    for i in range(1, breakpoints.shape[0]):
        start = breakpoints[i - 1]
        end = breakpoints[i]
        dist_along_path_list.append(np.cumsum(euclidean_dists[start:end]) - euclidean_dists[start])
        polylines_list.append(polylines[start:end])

    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = dist_along_path_list[idx]
        polylines_cur = polylines_list[idx]
        # Create interpolation functions for x and y coordinates
        fxy = interp1d(dist_along_path, polylines_cur, axis=0)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])

        # Combine the new x and y coordinates into a single array
        new_polylines = fxy(new_dist_along_path)
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(
            new_polylines[1:, 1] - new_polylines[:-1, 1],
            new_polylines[1:, 0] - new_polylines[:-1, 0],
        )
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size :]
            last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0).to(torch.float32)
    else:
        multi_polylines_list = None
    return multi_polylines_list


def preprocess_map(map_data: Dict[str, Any]) -> Dict[str, Any]:
    pt2pl = map_data[("map_point", "to", "map_polygon")]["edge_index"]
    split_polyline_type = []
    split_polyline_pos = []
    split_polyline_theta = []
    split_polygon_type = []
    split_light_type = []

    for i in sorted(torch.unique(pt2pl[1])):
        index = pt2pl[0, pt2pl[1] == i]
        if len(index) <= 2:
            continue

        polygon_type = map_data["map_polygon"]["type"][i]
        light_type = map_data["map_polygon"]["light_type"][i]
        cur_type = map_data["map_point"]["type"][index]
        cur_pos = map_data["map_point"]["position"][index, :2]

        # assert len(np.unique(cur_type)) == 1

        split_polyline = _interplating_polyline(cur_pos.numpy())
        if split_polyline is None:
            continue
        split_polyline_pos.append(split_polyline[..., :2])
        split_polyline_theta.append(split_polyline[..., 2])
        split_polyline_type.append(cur_type[0].repeat(split_polyline.shape[0]))
        split_polygon_type.append(polygon_type.repeat(split_polyline.shape[0]))
        split_light_type.append(light_type.repeat(split_polyline.shape[0]))

    data = {}
    if len(split_polyline_pos) == 0:  # add dummy empty map
        data["map_save"] = {
            # 6e4 such that it's within the range of float16.
            "traj_pos": torch.zeros([1, 3, 2], dtype=torch.float32) + 6e4,
            "traj_theta": torch.zeros([1], dtype=torch.float32),
        }
        data["pt_token"] = {
            "type": torch.tensor([0], dtype=torch.uint8),
            "pl_type": torch.tensor([0], dtype=torch.uint8),
            "light_type": torch.tensor([0], dtype=torch.uint8),
            "num_nodes": 1,
        }
    else:
        data["map_save"] = {
            "traj_pos": torch.cat(split_polyline_pos, dim=0),  # [num_nodes, 3, 2]
            "traj_theta": torch.cat(split_polyline_theta, dim=0)[:, 0],  # [num_nodes]
        }
        data["pt_token"] = {
            "type": torch.cat(split_polyline_type, dim=0),  # [num_nodes], uint8
            "pl_type": torch.cat(split_polygon_type, dim=0),  # [num_nodes], uint8
            "light_type": torch.cat(split_light_type, dim=0),  # [num_nodes], uint8
            "num_nodes": data["map_save"]["traj_pos"].shape[0],
        }
    return data
