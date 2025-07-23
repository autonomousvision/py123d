import pickle
from pathlib import Path

import lightning as L
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from d123.training.models.sim_agent.smart.datasets.scalable_dataset import MultiDataset
from d123.training.models.sim_agent.smart.tokens.token_processor import TokenProcessor
from d123.training.models.sim_agent.smart.utils.geometry import wrap_angle
from d123.training.models.sim_agent.smart.utils.rollout import cal_polygon_contour, transform_to_local


def Kdisk_cluster(
    X,  # [n_trajs, 4, 2], bbox of the last point of the segment
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 6, 3], the complete segment
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if i == 0:
            choice_index = 0  # always include [0, 0, 0]
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()
        x0 = X[choice_index]
        # res_mask = torch.sum((X - x0) ** 2, dim=[1, 2]) / 4.0 > (tol**2)
        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]
        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"{i=}, {remain=:.2f}%, {n_inside=}")

    return torch.cat(ret_traj_list, dim=0)  # [N, 6, 3]


if __name__ == "__main__":
    L.seed_everything(seed=2, workers=True)
    n_trajs = 2048 * 100  # 2e5
    load_data_from_file = True
    data_cache_path = Path("/root/.cache/SMART")
    out_file_name = "agent_vocab_555_s2.pkl"
    tol_dist = [0.05, 0.05, 0.05]  # veh, ped, cyc

    # ! don't change these params
    shift = 5  # motion token time dimension
    num_cluster = 2048  # vocabulary size
    n_step = 91
    data_file_path = data_cache_path / "kdisk_trajs.pkl"
    if load_data_from_file:
        with open(data_file_path, "rb") as f:
            data = pickle.load(f)
    else:
        trajs = [
            torch.zeros([1, 6, 3], dtype=torch.float32),  # veh
            torch.zeros([1, 6, 3], dtype=torch.float32),  # ped
            torch.zeros([1, 6, 3], dtype=torch.float32),  # cyc
        ]
        dataloader = DataLoader(
            dataset=MultiDataset(raw_dir=data_cache_path / "training", transform=lambda x: HeteroData(x)),
            batch_size=8,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

        with tqdm(
            total=len(dataloader),
            desc=f"n_trajs={n_trajs}",
            postfix={"n_veh": 0, "n_ped": 0, "n_cyc": 0},
        ) as pbar:

            for data in dataloader:
                valid_mask = data["agent"]["valid_mask"]
                data["agent"]["heading"] = TokenProcessor._clean_heading(valid_mask, data["agent"]["heading"])

                for i_ag in range(valid_mask.shape[0]):
                    if valid_mask[i_ag, :].sum() < 30:
                        continue
                    for t in range(0, n_step - shift, shift):
                        if valid_mask[i_ag, t] and valid_mask[i_ag, t + shift]:
                            _type = data["agent"]["type"][i_ag]
                            if trajs[_type].shape[0] < n_trajs:
                                pos = data["agent"]["position"][i_ag, t : t + shift + 1, :2]
                                head = data["agent"]["heading"][i_ag, t : t + shift + 1]
                                pos, head = transform_to_local(
                                    pos_global=pos.unsqueeze(0),  # [1, 6, 2]
                                    head_global=head.unsqueeze(0),  # [1, 6]
                                    pos_now=pos[[0]],  # [1, 2]
                                    head_now=head[[0]],  # [1]
                                )
                                head = wrap_angle(head)
                                to_add = torch.cat([pos, head.unsqueeze(-1)], dim=-1)

                                if not (((trajs[_type] - to_add).abs().sum([1, 2]) < 1e-2).any()):
                                    trajs[_type] = torch.cat([trajs[_type], to_add], dim=0)
                pbar.update(1)
                pbar.set_postfix(
                    n_veh=trajs[0].shape[0],
                    n_ped=trajs[1].shape[0],
                    n_cyc=trajs[2].shape[0],
                )
                if trajs[0].shape[0] == n_trajs and trajs[1].shape[0] == n_trajs and trajs[2].shape[0] == n_trajs:
                    break

        # [n_trajs, shift+1, [relative_x, relative_y, relative_theta]]
        data = {"veh": trajs[0], "ped": trajs[1], "cyc": trajs[2]}

        with open(data_file_path, "wb") as f:
            pickle.dump(data, f)

    res = {"token_all": {}}

    for k, v in data.items():
        if k == "veh":
            width_length = torch.tensor([2.0, 4.8])
        elif k == "ped":
            width_length = torch.tensor([1.0, 1.0])
        elif k == "cyc":
            width_length = torch.tensor([1.0, 2.0])
        width_length = width_length.unsqueeze(0)  # [1, 2]

        contour = cal_polygon_contour(pos=v[:, -1, :2], head=v[:, -1, 2], width_length=width_length)  # [n_trajs, 4, 2]

        if k == "veh":
            tol = tol_dist[0]
        elif k == "ped":
            tol = tol_dist[1]
        elif k == "cyc":
            tol = tol_dist[2]
        print(k, tol)
        ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol, a_pos=v)
        ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

        contour = cal_polygon_contour(
            pos=ret_traj[:, :, :2],  # [N, 6, 2]
            head=ret_traj[:, :, 2],  # [N, 6]
            width_length=width_length.unsqueeze(0),
        )
        res["token_all"][k] = contour.numpy()

    with open(Path(__file__).resolve().parent / out_file_name, "wb") as f:
        pickle.dump(res, f)
