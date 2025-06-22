import math

import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from asim.training.models.sim_agent.smart.metrics import CrossEntropy, TokenCls, minADE
from asim.training.models.sim_agent.smart.modules.smart_decoder import SMARTDecoder
from asim.training.models.sim_agent.smart.smart_config import SMARTConfig
from asim.training.models.sim_agent.smart.tokens.token_processor import TokenProcessor

# from asim.training.models.sim_agent.smart.utils.finetune import set_model_for_finetuning

# from src.utils.vis_waymo import VisWaymo
# from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):
    def __init__(self, model_config: SMARTConfig) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(
            map_token_file=model_config.map_token_file,
            agent_token_file=model_config.agent_token_file,
            map_token_sampling=model_config.map_token_sampling,
            agent_token_sampling=model_config.agent_token_sampling,
        )

        self.encoder = SMARTDecoder(model_config=model_config, n_token_agent=self.token_processor.n_token_agent)
        # set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        # self.wosac_metrics = WOSACMetrics("val_closed")
        # self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(
            use_gt_raw=model_config.use_gt_raw,
            gt_thresh_scale_length=model_config.gt_thresh_scale_length,
            label_smoothing=model_config.label_smoothing,
            rollout_as_gt=model_config.rollout_as_gt,
        )

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        # self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=data["agent"]["train_mask"],  # [n_agent]
            current_epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(tokenized_map, tokenized_agent, self.validation_rollout_sampling)
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]
            self.minADE.update(
                pred=pred_traj,
                target=data["agent"]["position"][:, self.num_historical_steps :, : pred_traj.shape[-1]],
                target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
            )

            # TODO: delete
            # ! WOSAC
            # scenario_rollouts = None
            # if self.wosac_submission.is_active:  # ! save WOSAC submission
            #     self.wosac_submission.update(
            #         scenario_id=data["scenario_id"],
            #         agent_id=data["agent"]["id"],
            #         agent_batch=data["agent"]["batch"],
            #         pred_traj=pred_traj,
            #         pred_z=pred_z,
            #         pred_head=pred_head,
            #         global_rank=self.global_rank,
            #     )
            #     _gpu_dict_sync = self.wosac_submission.compute()
            #     if self.global_rank == 0:
            #         for k in _gpu_dict_sync.keys():  # single gpu fix
            #             if type(_gpu_dict_sync[k]) is list:
            #                 _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            #         scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            #         self.wosac_submission.aggregate_rollouts(scenario_rollouts)
            #     self.wosac_submission.reset()

            # else:  # ! compute metrics, disable if save WOSAC submission
            #     self.minADE.update(
            #         pred=pred_traj,
            #         target=data["agent"]["position"][:, self.num_historical_steps :, : pred_traj.shape[-1]],
            #         target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
            #     )

            # TODO: delete
            # # WOSAC metrics
            # if batch_idx < self.n_batch_wosac_metric:
            #     device = pred_traj.device
            #     scenario_rollouts = get_scenario_rollouts(
            #         scenario_id=get_scenario_id_int_tensor(
            #             data["scenario_id"], device
            #         ),
            #         agent_id=data["agent"]["id"],
            #         agent_batch=data["agent"]["batch"],
            #         pred_traj=pred_traj,
            #         pred_z=pred_z,
            #         pred_head=pred_head,
            #     )
            #     self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # TODO: delete
            # ! visualization
            # if self.global_rank == 0 and batch_idx < self.n_vis_batch:
            #     if scenario_rollouts is not None:
            #         for _i_sc in range(self.n_vis_scenario):
            #             _vis = VisWaymo(
            #                 scenario_path=data["tfrecord_path"][_i_sc],
            #                 save_dir=self.video_dir
            #                 / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
            #             )
            #             _vis.save_video_scenario_rollout(
            #                 scenario_rollouts[_i_sc], self.n_vis_rollout
            #             )
            #             for _path in _vis.video_paths:
            #                 self.logger.log_video(
            #                     "/".join(_path.split("/")[-3:]), [_path]
            #                 )

    def on_validation_epoch_end(self):
        # TODO: delete
        # if self.val_closed_loop:
        #     if not self.wosac_submission.is_active:
        #         epoch_wosac_metrics = self.wosac_metrics.compute()
        #         epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
        #         if self.global_rank == 0:
        #             epoch_wosac_metrics["epoch"] = (
        #                 self.log_epoch if self.log_epoch >= 0 else self.current_epoch
        #             )
        #             self.logger.log_metrics(epoch_wosac_metrics)

        #         self.wosac_metrics.reset()
        #         self.minADE.reset()

        #     if self.global_rank == 0:
        #         if self.wosac_submission.is_active:
        #             self.wosac_submission.save_sub_file()
        pass

    def configure_optimizers(self):

        # TODO: add to hydra config
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return self.lr_min_ratio + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps) / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(tokenized_map, tokenized_agent, self.validation_rollout_sampling)
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

        # TODO: delete
        # ! WOSAC submission save
        # self.wosac_submission.update(
        #     scenario_id=data["scenario_id"],
        #     agent_id=data["agent"]["id"],
        #     agent_batch=data["agent"]["batch"],
        #     pred_traj=pred_traj,
        #     pred_z=pred_z,
        #     pred_head=pred_head,
        #     global_rank=self.global_rank,
        # )
        # _gpu_dict_sync = self.wosac_submission.compute()
        # if self.global_rank == 0:
        #     for k in _gpu_dict_sync.keys():  # single gpu fix
        #         if type(_gpu_dict_sync[k]) is list:
        #             _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
        #     scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
        #     self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        # self.wosac_submission.reset()

    def on_test_epoch_end(self):
        # if self.global_rank == 0:
        #     self.wosac_submission.save_sub_file()
        pass
