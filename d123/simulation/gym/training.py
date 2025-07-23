"""
NOTE @DanielDauner:
This file needs refactoring. The training loop is specific to the default environment.
I will leave it for the initial code release but hope to find time to fix it.
"""

import datetime
import gc
import logging
import math
import os
import random
import re
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import schedulefree
import torch
import wandb
from carl_nuplan.common.logging import suppress_info_logs
from carl_nuplan.planning.script.builders.observation_building_builder import build_observation_builder
from carl_nuplan.planning.script.builders.reward_builder_builder import build_reward_builder
from carl_nuplan.planning.script.builders.scenario_sampler_builder import build_scenario_sampler
from carl_nuplan.planning.script.builders.simulation_building_builder import build_simulation_builder
from carl_nuplan.planning.script.builders.trajectory_building_builder import build_trajectory_builder
from gymnasium.envs.registration import register
from omegaconf import DictConfig, OmegaConf
from pytictoc import TicToc
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from d123.simulation.gym.policy.ppo.ppo_config import GlobalConfig
from d123.simulation.gym.policy.ppo.ppo_model import PPOPolicy

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options("json", sort_keys=True, indent=4)

logger = logging.getLogger(__name__)

REWARD_LOGGING: bool = True
COMFORT_LOGGING: bool = True


def save(model, optimizer, config, folder, model_file, optimizer_file):
    model_file = os.path.join(folder, model_file)
    torch.save(model.module.state_dict(), model_file)

    if optimizer is not None:
        optimizer_file = os.path.join(folder, optimizer_file)
        torch.save(optimizer.state_dict(), optimizer_file)

    json_config = jsonpickle.encode(config)
    with open(
        os.path.join(folder, "config_pickle.json"),
        "wt",
        encoding="utf-8",
    ) as f2:
        f2.write(json_config)


def make_env(cfg: DictConfig, config: GlobalConfig):
    @suppress_info_logs
    def thunk(idx: int = 0):

        scenario_sampler = build_scenario_sampler(cfg)
        simulation_builder = build_simulation_builder(cfg)
        trajectory_builder = build_trajectory_builder(cfg)
        observation_builder = build_observation_builder(cfg)
        reward_builder = build_reward_builder(cfg)

        env = gym.make(
            "EnvironmentWrapper-v0",
            scenario_sampler=scenario_sampler,
            simulation_builder=simulation_builder,
            trajectory_builder=trajectory_builder,
            observation_builder=observation_builder,
            reward_builder=reward_builder,
            terminate_on_failure=cfg.debug,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        if config.normalize_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=config.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def run_training(cfg: DictConfig) -> None:

    register(
        id="EnvironmentWrapper-v0",
        entry_point="d123.simulation.gym.environment.environment_wrapper:EnvironmentWrapper",
        max_episode_steps=None,
    )
    config = GlobalConfig()

    # Torchrun initialization
    # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
    rank = int(os.environ["RANK"])  # Rank across all processes
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank on Node
    world_size = int(os.environ["WORLD_SIZE"])  # Number of processes

    logger.info(f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}")

    local_batch_size = cfg.total_batch_size // world_size
    local_bs_per_env = local_batch_size // cfg.num_envs_per_gpu
    local_minibatch_size = cfg.total_minibatch_size // world_size
    num_minibatches = local_batch_size // local_minibatch_size

    run_name = f"{cfg.experiment_name}__{cfg.seed}"
    if rank == 0:
        exp_folder = os.path.join(cfg.output_dir, f"{cfg.experiment_name}")
        wandb_folder = os.path.join(exp_folder, "wandb")

        Path(exp_folder).mkdir(parents=True, exist_ok=True)
        Path(wandb_folder).mkdir(parents=True, exist_ok=True)

        if cfg.track:

            wandb.init(
                project=cfg.wandb_project_name,
                entity=cfg.wandb_entity,
                sync_tensorboard=True,
                # config=vars(cfg), # FIXME
                name=run_name,
                monitor_gym=False,
                allow_val_change=True,
                save_code=False,
                mode="online",
                resume="auto",
                dir=wandb_folder,
                settings=wandb.Settings(
                    _disable_stats=True, _disable_meta=True
                ),  # Can get large if we log all the cpu cores.
            )

        writer = SummaryWriter(exp_folder)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(cfg, resolve=True).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger.info(f"Is cuda available?: {torch.cuda.is_available()}")
    if cfg.train_gpu_ids is None:
        cfg.train_gpu_ids = list(range(torch.cuda.device_count()))

    # Load the config before overwriting values with current arguments
    if cfg.load_file is not None:
        load_folder = Path(cfg.load_file).parent.resolve()
        with open(os.path.join(load_folder, "config_pickle.json"), "rt", encoding="utf-8") as f:
            json_config = f.read()
        # 4 ms, might need to move outside the agent.
        loaded_config = jsonpickle.decode(json_config)
        # Overwrite all properties that were set in the saved config.
        config.__dict__.update(loaded_config.__dict__)

    # Configure config. Converts all arguments into config attributes
    config.initialize(**OmegaConf.to_container(cfg, resolve=True))

    if config.use_dd_ppo_preempt:
        # Compute unique port within machine based on experiment name and seed.
        experiment_id = int(re.findall(r"\d+", cfg.experiment_uid)[0])
        tcp_store_port = ((experiment_id * 1000) % 65534) + int(cfg.seed) + 5000
        # We use gloo, because nccl crashes when using multiple processes per GPU.
        num_rollouts_done_store = torch.distributed.TCPStore("127.0.0.1", tcp_store_port, world_size, rank == 0)
        torch.distributed.init_process_group(
            backend="gloo" if cfg.cpu_collect else "nccl",
            store=num_rollouts_done_store,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=15),
        )
        num_rollouts_done_store.set("num_done", "0")
        logger.info(f"Rank:{rank}, TCP_Store_Port: {tcp_store_port}")
    else:
        torch.distributed.init_process_group(
            backend="gloo" if cfg.cpu_collect else "nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=15),
        )

    device = (
        # torch.device(f"cuda:{cfg.train_gpu_ids[rank]}")
        torch.device(f"cuda:{cfg.train_gpu_ids[rank]}")
        if torch.cuda.is_available() and cfg.cuda
        else torch.device("cpu")
    )

    if torch.cuda.is_available() and cfg.cuda:
        torch.cuda.device(device)

    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.benchmark
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    # torch.set_float32_matmul_precision(config.matmul_precision)

    if rank == 0:
        json_config = jsonpickle.encode(config)
        with open(os.path.join(exp_folder, "config_pickle.json"), "w") as f2:
            f2.write(json_config)

    # NOTE: need to update the config with the argparse arguments before creating the gym environment because the gym env
    if cfg.debug:
        env = gym.vector.SyncVectorEnv([make_env(cfg=cfg, config=config) for _ in range(cfg.num_envs_per_gpu)])
    else:
        env = gym.vector.AsyncVectorEnv(
            [make_env(cfg=cfg, config=config) for _ in range(cfg.num_envs_per_gpu)],
            copy=False,
        )
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = PPOPolicy(env.single_observation_space, env.single_action_space, config=config).to(device)

    if config.compile_model:
        agent = torch.compile(agent)

    start_step = 0
    if cfg.load_file is not None:
        load_file_name = os.path.basename(cfg.load_file)
        algo_step = re.findall(r"\d+", load_file_name)
        if len(algo_step) > 0:
            start_step = int(algo_step[0]) + 1  # That step was already finished.
            logger.info(f"Start training from step: {start_step}")
        agent.load_state_dict(torch.load(cfg.load_file, map_location=device), strict=True)

    agent = torch.nn.parallel.DistributedDataParallel(
        agent,
        device_ids=None,
        output_device=None,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # If we are resuming training use last learning rate from config.
    # If we start a fresh training set the current learning rate according to arguments.
    if cfg.load_file is None:
        config.current_learning_rate = cfg.learning_rate

    # if rank == 0:
    #   model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
    #   num_params = sum(np.prod(p.size()) for p in model_parameters)
    #
    #   logger.info('Total trainable parameters: ', num_params)
    if cfg.schedule_free:
        optimizer = schedulefree.AdamWScheduleFree(
            agent.parameters(),
            lr=config.current_learning_rate,
            betas=tuple(cfg.adam_betas),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )

    elif config.weight_decay > 0.0:
        optimizer = optim.AdamW(
            agent.parameters(),
            lr=config.current_learning_rate,
            betas=tuple(cfg.adam_betas),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            agent.parameters(),
            betas=tuple(cfg.adam_betas),
            lr=config.current_learning_rate,
            eps=config.adam_eps,
        )

    # Load optimizer
    if cfg.load_file is not None:
        optimizer.load_state_dict(torch.load(cfg.load_file.replace("model_", "optimizer_"), map_location=device))

        if rank == 0:
            writer.add_scalar("charts/restart", 1, config.global_step)  # Log that a restart happened

    if config.cpu_collect:
        device = "cpu"

    # ALGO Logic: Storage setup
    obs = {
        "bev_semantics": torch.zeros(
            (local_bs_per_env, cfg.num_envs_per_gpu) + env.single_observation_space.spaces["bev_semantics"].shape,
            dtype=torch.uint8,
            device=device,
        ),
        "measurements": torch.zeros(
            (local_bs_per_env, cfg.num_envs_per_gpu) + env.single_observation_space.spaces["measurements"].shape,
            device=device,
        ),
        "value_measurements": torch.zeros(
            (local_bs_per_env, cfg.num_envs_per_gpu) + env.single_observation_space.spaces["value_measurements"].shape,
            device=device,
        ),
    }
    actions = torch.zeros(
        (local_bs_per_env, cfg.num_envs_per_gpu) + env.single_action_space.shape,
        device=device,
    )
    old_mus = torch.zeros(
        (local_bs_per_env, cfg.num_envs_per_gpu) + env.single_action_space.shape,
        device=device,
    )
    old_sigmas = torch.zeros(
        (local_bs_per_env, cfg.num_envs_per_gpu) + env.single_action_space.shape,
        device=device,
    )
    logprobs = torch.zeros((local_bs_per_env, cfg.num_envs_per_gpu), device=device)
    rewards = torch.zeros((local_bs_per_env, cfg.num_envs_per_gpu), device=device)
    dones = torch.zeros((local_bs_per_env, cfg.num_envs_per_gpu), device=device)
    values = torch.zeros((local_bs_per_env, cfg.num_envs_per_gpu), device=device)
    exp_n_steps = np.zeros((local_bs_per_env, cfg.num_envs_per_gpu), dtype=np.int32)
    exp_suggest = np.zeros((local_bs_per_env, cfg.num_envs_per_gpu), dtype=np.int32)

    # TRY NOT TO MODIFY: start the game
    reset_obs = env.reset(seed=[cfg.seed + rank * cfg.num_envs_per_gpu + i for i in range(cfg.num_envs_per_gpu)])
    next_obs = {
        "bev_semantics": torch.tensor(reset_obs[0]["bev_semantics"], device=device, dtype=torch.uint8),
        "measurements": torch.tensor(reset_obs[0]["measurements"], device=device, dtype=torch.float32),
        "value_measurements": torch.tensor(reset_obs[0]["value_measurements"], device=device, dtype=torch.float32),
    }
    next_done = torch.zeros(cfg.num_envs_per_gpu, device=device)
    next_lstm_state = (
        torch.zeros(
            config.num_lstm_layers,
            cfg.num_envs_per_gpu,
            config.features_dim,
            device=device,
        ),
        torch.zeros(
            config.num_lstm_layers,
            cfg.num_envs_per_gpu,
            config.features_dim,
            device=device,
        ),
    )
    num_updates = cfg.total_timesteps // cfg.total_batch_size
    local_processed_samples = 0
    start_time = time.time()
    agent.train()  # TODO change train and eval

    if rank == 0:
        avg_returns = deque(maxlen=100)

    # if config.use_hl_gauss_value_loss:
    #   hl_gauss_bins = rl_u.hl_gaus_bins(config.hl_gauss_vmin, config.hl_gauss_vmax, config.hl_gauss_bucket_size, device)

    for update in tqdm(range(start_step, num_updates), disable=rank != 0):
        if cfg.debug:
            print("WARNING: DEBUG MODE")

        if config.cpu_collect:
            device = "cpu"
            agent.to(device)
        # Free all data from last interation.
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.disable()

        if config.use_dd_ppo_preempt:
            num_rollouts_done_store.set("num_done", "0")

        # Buffers we use to store returns and aggregate them later to rank 0 for logging.
        total_returns = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
        total_lengths = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
        num_total_returns = torch.zeros(world_size, device=device, dtype=torch.int32, requires_grad=False)

        # reward
        if REWARD_LOGGING:
            reward_progress = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_red_light = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_collision = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_off_road = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_lane_distance = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_too_fast = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_off_route = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_comfort = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            reward_ttc = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)

        if COMFORT_LOGGING:
            comfort_lon_acceleration = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            comfort_lat_acceleration = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            comfort_jerk_metric = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            comfort_lon_jerk_metric = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            comfort_yaw_accel = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
            comfort_yaw_rate = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)

        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # Annealing the rate if instructed to do so.
        if cfg.schedule_free:
            optimizer.eval()
        else:
            if config.lr_schedule == "linear":
                frac = 1.0 - (update - 1.0) / num_updates
                config.current_learning_rate = frac * config.learning_rate
            elif config.lr_schedule == "step":
                frac = update / num_updates
                lr_multiplier = 1.0
                for change_percentage in config.lr_schedule_step_perc:
                    if frac > change_percentage:
                        lr_multiplier *= config.lr_schedule_step_factor
                config.current_learning_rate = lr_multiplier * config.learning_rate
            elif config.lr_schedule == "cosine":
                frac = update / num_updates
                config.current_learning_rate = 0.5 * config.learning_rate * (1 + math.cos(frac * math.pi))
            elif config.lr_schedule == "cosine_restart":
                frac = update / (num_updates + 1)  # + 1 so it doesn't become 100 %
                for idx, frac_restart in enumerate(config.lr_schedule_cosine_restarts):
                    if frac >= frac_restart:
                        current_idx = idx
                base_frac = config.lr_schedule_cosine_restarts[current_idx]
                length_current_interval = (
                    config.lr_schedule_cosine_restarts[current_idx + 1]
                    - config.lr_schedule_cosine_restarts[current_idx]
                )
                frac_current_iter = (frac - base_frac) / length_current_interval
                config.current_learning_rate = 0.5 * config.learning_rate * (1 + math.cos(frac_current_iter * math.pi))

            for param_group in optimizer.param_groups:
                param_group["lr"] = config.current_learning_rate

        t0 = TicToc()  # Data collect
        t1 = TicToc()  # Forward pass
        t2 = TicToc()  # Env step
        t3 = TicToc()  # Pre-processing
        t4 = TicToc()  # Train inter
        t5 = TicToc()  # Logging
        t0.tic()
        inference_times = []
        env_times = []
        for step in range(0, local_bs_per_env):
            config.global_step += 1 * world_size * cfg.num_envs_per_gpu
            local_processed_samples += 1 * world_size * cfg.num_envs_per_gpu

            obs["bev_semantics"][step] = next_obs["bev_semantics"]
            obs["measurements"][step] = next_obs["measurements"]
            obs["value_measurements"][step] = next_obs["value_measurements"]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                t1.tic()
                (
                    action,
                    logprob,
                    _,
                    value,
                    _,
                    mu,
                    sigma,
                    _,
                    _,
                    _,
                    next_lstm_state,
                ) = agent.forward(next_obs, lstm_state=next_lstm_state, done=next_done)
                if config.use_hl_gauss_value_loss:
                    # value_pdf = F.softmax(value, dim=1)
                    # value = torch.sum(value_pdf * hl_gauss_bins.unsqueeze(0), dim=1)
                    pass
                inference_times.append(t1.tocvalue())
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            old_mus[step] = mu
            old_sigmas[step] = sigma

            # TRY NOT TO MODIFY: execute the game and log data.
            t2.tic()
            next_obs, reward, termination, truncation, info = env.step(action.cpu().numpy())
            env_times.append(t2.tocvalue())

            if rank == 0:
                if "timing" in info.keys():
                    if info["timing"][0] is not None:
                        for key_, value_ in info["timing"][0].items():
                            tab_ = "time_step" if "step" in key_ else "time_reset"
                            writer.add_scalar(f"{tab_}/{key_}", np.mean(value_), config.global_step)

            done = np.logical_or(termination, truncation)  # Not treated separately in original PPO
            rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32)
            next_done = torch.tensor(done, device=device, dtype=torch.float32)
            next_obs = {
                "bev_semantics": torch.tensor(next_obs["bev_semantics"], device=device, dtype=torch.uint8),
                "measurements": torch.tensor(next_obs["measurements"], device=device, dtype=torch.float32),
                "value_measurements": torch.tensor(next_obs["value_measurements"], device=device, dtype=torch.float32),
            }

            if "final_info" in info.keys():

                for idx, single_info in enumerate(info["final_info"]):

                    if config.use_exploration_suggest:
                        # Exploration loss
                        exp_n_steps[step, idx] = single_info["n_steps"]
                        exp_suggest[step, idx] = single_info["suggest"]

                    # Sum up total returns and how often the env was reset during this iteration.
                    if single_info is not None:
                        if "episode" in single_info.keys():
                            print(
                                f"rank: {rank}, config.global_step={config.global_step}, episodic_return={single_info['episode']['r']}"
                            )
                            total_returns[rank] += single_info["episode"]["r"].item()
                            total_lengths[rank] += single_info["episode"]["l"].item()
                            num_total_returns[rank] += 1

                        if REWARD_LOGGING and "reward" in single_info.keys():
                            reward_progress[rank] += single_info["reward"]["reward_progress"]
                            reward_red_light[rank] += single_info["reward"]["reward_red_light"]
                            reward_collision[rank] += single_info["reward"]["reward_collision"]
                            reward_off_road[rank] += single_info["reward"]["reward_off_road"]
                            reward_lane_distance[rank] += single_info["reward"]["reward_lane_distance"]
                            reward_too_fast[rank] += single_info["reward"]["reward_too_fast"]
                            reward_off_route[rank] += single_info["reward"]["reward_off_route"]
                            reward_comfort[rank] += single_info["reward"]["reward_comfort"]
                            reward_ttc[rank] += single_info["reward"]["reward_ttc"]

                        if COMFORT_LOGGING and "comfort" in single_info.keys():
                            comfort_lon_acceleration[rank] += single_info["comfort"]["comfort_lon_acceleration"]
                            comfort_lat_acceleration[rank] += single_info["comfort"]["comfort_lat_acceleration"]
                            comfort_jerk_metric[rank] += single_info["comfort"]["comfort_jerk_metric"]
                            comfort_lon_jerk_metric[rank] += single_info["comfort"]["comfort_lon_jerk_metric"]
                            comfort_yaw_accel[rank] += single_info["comfort"]["comfort_yaw_accel"]
                            comfort_yaw_rate[rank] += single_info["comfort"]["comfort_yaw_rate"]

            if config.use_dd_ppo_preempt:
                num_done = int(num_rollouts_done_store.get("num_done"))
                min_steps = int(config.dd_ppo_min_perc * local_bs_per_env)
                if (num_done / world_size) > config.dd_ppo_preempt_threshold and step > min_steps:
                    logger.info(f"Rank:{rank}, Preempt at step: {step}, Num done: {num_done}")
                    break  # End data collection early the other workers are finished.

        t0.toc(msg=f"Rank:{rank}, Data collection.")
        print(f"Rank:{rank}, Avg forward time {sum(inference_times)}")
        print(f"Rank:{rank}, Avg env time {sum(env_times)}")
        t3.tic()

        if config.use_dd_ppo_preempt:
            num_rollouts_done_store.add("num_done", 1)

        # In case of a dd-ppo preempt this can be smaller than local batch size
        num_collected_steps = step + 1

        # bootstrap value if not done
        with torch.no_grad():
            # if config.use_hl_gauss_value_loss:
            #   next_value = agent.module.get_value(next_obs, next_lstm_state, next_done)
            #   value_pdf = F.softmax(next_value, dim=1)
            #   next_value = torch.sum(value_pdf * hl_gauss_bins.unsqueeze(0), dim=1)
            if config.use_hl_gauss_value_loss:
                False
            else:
                next_value = agent.module.get_value(next_obs, next_lstm_state, next_done).squeeze(1)
            if cfg.gae:
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0.0
                for t in reversed(range(num_collected_steps)):
                    if t == local_bs_per_env - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards, device=device)
                for t in reversed(range(num_collected_steps)):
                    if t == local_bs_per_env - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + cfg.gamma * nextnonterminal * next_return
                advantages = returns - values

        if config.cpu_collect:
            device = (
                torch.device(f"cuda:{cfg.train_gpu_ids}")
                if torch.cuda.is_available() and cfg.cuda
                else torch.device("cpu")
            )
            agent.to(device)

        b_exploration_suggests = np.zeros((num_collected_steps, cfg.num_envs_per_gpu), dtype=np.int32)
        if config.use_exploration_suggest:
            for step in range(num_collected_steps):
                n_steps = exp_n_steps[step][0]  # TODO
                if n_steps > 0:
                    n_start = max(0, step - n_steps)
                    b_exploration_suggests[n_start:step] = exp_suggest[step]

        if config.use_world_model_loss:  # TODO
            b_wm_added_index = np.zeros(num_collected_steps, dtype=np.int32)
            b_world_model_mask = torch.zeros(
                num_collected_steps,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            invalid_frames = config.num_future_prediction
            for step in reversed(range(num_collected_steps)):
                if invalid_frames <= 0:
                    b_wm_added_index[step] = config.num_future_prediction
                    b_world_model_mask[step] = 1.0
                else:
                    invalid_frames -= 1

                if dones[step]:
                    invalid_frames = 5

        b_obs = {
            "bev_semantics": obs["bev_semantics"][:num_collected_steps].reshape(
                (-1,) + env.single_observation_space.spaces["bev_semantics"].shape
            ),
            "measurements": obs["measurements"][:num_collected_steps].reshape(
                (-1,) + env.single_observation_space.spaces["measurements"].shape
            ),
            "value_measurements": obs["value_measurements"][:num_collected_steps].reshape(
                (-1,) + env.single_observation_space.spaces["value_measurements"].shape
            ),
        }
        b_logprobs = logprobs[:num_collected_steps].reshape(-1)
        b_actions = actions[:num_collected_steps].reshape((-1,) + env.single_action_space.shape)
        b_dones = dones[:num_collected_steps].reshape(-1)  # TODO check if pre-emption trick causes problems with LSTM.
        b_advantages = advantages[:num_collected_steps].reshape(-1)
        b_returns = returns[:num_collected_steps].reshape(-1)
        b_values = values[:num_collected_steps].reshape(-1)
        b_old_mus = old_mus[:num_collected_steps].reshape((-1,) + env.single_action_space.shape)
        b_old_sigmas = old_sigmas[:num_collected_steps].reshape((-1,) + env.single_action_space.shape)

        # When the data was collected on the CPU, move it to GPU before training
        if config.cpu_collect:
            b_obs["bev_semantics"] = b_obs["bev_semantics"].to(device)
            b_obs["measurements"] = b_obs["measurements"].to(device)
            b_obs["value_measurements"] = b_obs["value_measurements"].to(device)
            b_logprobs = b_logprobs.to(device)
            b_actions = b_actions.to(device)
            b_dones = b_dones.to(device)
            b_advantages = b_advantages.to(device)
            b_returns = b_returns.to(device)
            b_values = b_values.to(device)
            b_old_mus = b_old_mus.to(device)
            b_old_sigmas = b_old_sigmas.to(device)

        # Aggregate returns to GPU 0 for logging and storing the best model.
        # Gloo doesn't support AVG, so we implement it via sum / num returns
        torch.distributed.all_reduce(total_returns, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_lengths, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_total_returns, op=torch.distributed.ReduceOp.SUM)

        # reward
        if REWARD_LOGGING:
            torch.distributed.all_reduce(reward_progress, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_red_light, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_collision, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_off_road, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_lane_distance, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_too_fast, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_off_route, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_comfort, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(reward_ttc, op=torch.distributed.ReduceOp.SUM)

        if COMFORT_LOGGING:
            torch.distributed.all_reduce(comfort_lon_acceleration, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(comfort_lat_acceleration, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(comfort_jerk_metric, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(comfort_lon_jerk_metric, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(comfort_yaw_accel, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(comfort_yaw_rate, op=torch.distributed.ReduceOp.SUM)

        if rank == 0:
            num_total_returns_all_processes = torch.sum(num_total_returns)
            # Only can log return if there was any episode that finished
            if num_total_returns_all_processes > 0:
                total_returns_all_processes = torch.sum(total_returns)
                total_lengths_all_processes = torch.sum(total_lengths)
                avg_return = total_returns_all_processes / num_total_returns_all_processes
                avg_return = avg_return.item()
                avg_length = total_lengths_all_processes / num_total_returns_all_processes
                avg_length = avg_length.item()

                avg_returns.append(avg_return)
                windowed_avg_return = sum(avg_returns) / len(avg_returns)

                writer.add_scalar("charts/episodic_return", avg_return, config.global_step)
                writer.add_scalar(
                    "charts/windowed_avg_return",
                    windowed_avg_return,
                    config.global_step,
                )
                writer.add_scalar("charts/episodic_length", avg_length, config.global_step)

                if REWARD_LOGGING:
                    reward_progress = torch.sum(reward_progress) / num_total_returns_all_processes
                    reward_red_light = torch.sum(reward_red_light) / num_total_returns_all_processes
                    reward_collision = torch.sum(reward_collision) / num_total_returns_all_processes
                    reward_off_road = torch.sum(reward_off_road) / num_total_returns_all_processes
                    reward_lane_distance = torch.sum(reward_lane_distance) / num_total_returns_all_processes
                    reward_too_fast = torch.sum(reward_too_fast) / num_total_returns_all_processes
                    reward_off_route = torch.sum(reward_off_route) / num_total_returns_all_processes
                    reward_comfort = torch.sum(reward_comfort) / num_total_returns_all_processes
                    reward_ttc = torch.sum(reward_ttc) / num_total_returns_all_processes

                    writer.add_scalar("reward/progress", reward_progress.item(), config.global_step)
                    writer.add_scalar("reward/red_light", reward_red_light.item(), config.global_step)
                    writer.add_scalar("reward/collision", reward_collision.item(), config.global_step)
                    writer.add_scalar("reward/off_road", reward_off_road.item(), config.global_step)
                    writer.add_scalar(
                        "reward/lane_distance",
                        reward_lane_distance.item(),
                        config.global_step,
                    )
                    writer.add_scalar("reward/too_fast", reward_too_fast.item(), config.global_step)
                    writer.add_scalar("reward/off_route", reward_off_route.item(), config.global_step)
                    writer.add_scalar("reward/comfort", reward_comfort.item(), config.global_step)
                    writer.add_scalar("reward/ttc", reward_ttc.item(), config.global_step)

                if COMFORT_LOGGING:
                    comfort_lon_acceleration = torch.sum(comfort_lon_acceleration) / num_total_returns_all_processes
                    comfort_lat_acceleration = torch.sum(comfort_lat_acceleration) / num_total_returns_all_processes
                    comfort_jerk_metric = torch.sum(comfort_jerk_metric) / num_total_returns_all_processes
                    comfort_lon_jerk_metric = torch.sum(comfort_lon_jerk_metric) / num_total_returns_all_processes
                    comfort_yaw_accel = torch.sum(comfort_yaw_accel) / num_total_returns_all_processes
                    comfort_yaw_rate = torch.sum(comfort_yaw_rate) / num_total_returns_all_processes

                    writer.add_scalar(
                        "comfort/comfort_lon_acceleration",
                        comfort_lon_acceleration.item(),
                        config.global_step,
                    )
                    writer.add_scalar(
                        "comfort/comfort_lat_acceleration",
                        comfort_lat_acceleration.item(),
                        config.global_step,
                    )
                    writer.add_scalar(
                        "comfort/comfort_jerk_metric",
                        comfort_jerk_metric.item(),
                        config.global_step,
                    )
                    writer.add_scalar(
                        "comfort/comfort_lon_jerk_metric",
                        comfort_lon_jerk_metric.item(),
                        config.global_step,
                    )
                    writer.add_scalar(
                        "comfort/comfort_yaw_accel",
                        comfort_yaw_accel.item(),
                        config.global_step,
                    )
                    writer.add_scalar(
                        "comfort/comfort_yaw_rate",
                        comfort_yaw_rate.item(),
                        config.global_step,
                    )

                if windowed_avg_return >= config.max_training_score:
                    config.max_training_score = windowed_avg_return
                    # Same model could reach multiple high scores
                    if config.best_iteration != update:
                        save(agent, None, config, exp_folder, "model_best.pth", None)
                        config.best_iteration = update

        # Optimizing the policy and value network
        if config.use_lstm:
            assert cfg.num_envs_per_gpu % num_minibatches == 0
            assert not config.use_dd_ppo_preempt
            assert not config.use_world_model_loss

            envsperbatch = cfg.num_envs_per_gpu // num_minibatches
            envinds = np.arange(cfg.num_envs_per_gpu)
            flatinds = np.arange(local_batch_size).reshape(local_bs_per_env, cfg.num_envs_per_gpu)

        b_inds_original = np.arange(num_collected_steps * cfg.num_envs_per_gpu)

        if config.use_dd_ppo_preempt:
            b_inds_original = np.resize(b_inds_original, (local_batch_size,))

        # if config.use_world_model_loss:
        #     b_inds_world_model_original = b_inds_original + b_wm_added_index[b_inds_original]  # TODO

        clipfracs = []

        t3.toc(msg=f"Rank:{rank}, Data pre-processing.")
        t4.tic()

        if cfg.schedule_free:
            optimizer.train()

        for latest_epoch in range(cfg.update_epochs):
            approx_kl_divs = []
            if config.use_lstm:
                np.random.shuffle(envinds)
            else:
                p = np.random.permutation(len(b_inds_original))
                b_inds = b_inds_original[p]
                # if config.use_world_model_loss:
                #     b_inds_world_model = b_inds_world_model_original[p]

            total_steps = local_batch_size
            step_size = local_minibatch_size
            if config.use_lstm:
                total_steps = cfg.num_envs_per_gpu
                step_size = envsperbatch

            for start in range(0, total_steps, step_size):
                if config.use_lstm:
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    lstm_state = (
                        initial_lstm_state[0][:, mbenvinds],
                        initial_lstm_state[1][:, mbenvinds],
                    )
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                else:
                    end = start + local_minibatch_size
                    mb_inds = b_inds[start:end]
                    lstm_state = None

                # if config.use_world_model_loss:
                #     mb_inds_world_model = b_inds_world_model[start:end]
                if config.use_exploration_suggest:
                    b_exploration_suggests_sampled = b_exploration_suggests[mb_inds]
                else:
                    b_exploration_suggests_sampled = None
                b_obs_sampled = {
                    "bev_semantics": b_obs["bev_semantics"][mb_inds],
                    "measurements": b_obs["measurements"][mb_inds],
                    "value_measurements": b_obs["value_measurements"][mb_inds],
                }
                # Don't need action, so we don't unscale
                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    exploration_loss,
                    _,
                    _,
                    distribution,
                    pred_sem,
                    pred_measure,
                    _,
                ) = agent.forward(
                    b_obs_sampled,
                    actions=b_actions[mb_inds],
                    exploration_suggests=b_exploration_suggests_sampled,
                    lstm_state=lstm_state,
                    done=b_dones[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if cfg.clip_vloss:
                    # Value clipping is not implemented with HL_Gauss loss
                    assert config.use_hl_gauss_value_loss is False
                    newvalue = newvalue.view(-1)
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # if config.use_hl_gauss_value_loss:
                    #   target_pdf = rl_u.hl_gaus_pdf(b_returns[mb_inds], config.hl_gauss_std, config.hl_gauss_vmin,
                    #                                 config.hl_gauss_vmax, config.hl_gauss_bucket_size)
                    #   v_loss = F.cross_entropy(newvalue, target_pdf)
                    if config.use_hl_gauss_value_loss:
                        pass
                    else:
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # if config.use_world_model_loss:
                #   b_mask_sampled = b_world_model_mask[mb_inds_world_model]
                #   total_valid_items = torch.sum(b_mask_sampled)
                #   semantic_labels = image_to_class_labels(b_obs['bev_semantics'][mb_inds_world_model])
                #   semantic_loss = F.cross_entropy(pred_sem, semantic_labels, reduction='none')
                #   semantic_loss = torch.mean(semantic_loss, dim=(1, 2)) * b_mask_sampled
                #   semantic_loss = torch.sum(semantic_loss) / total_valid_items
                #   measure_loss = F.l1_loss(pred_measure, b_obs['measurements'][mb_inds_world_model], reduction='none')
                #   measure_loss = torch.mean(measure_loss, dim=1) * b_mask_sampled
                #   measure_loss = torch.sum(measure_loss) / total_valid_items
                #   world_model_loss = 0.5 * semantic_loss + 0.5 * measure_loss

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                if config.use_exploration_suggest:
                    loss = loss + cfg.expl_coef * exploration_loss

                # if config.use_world_model_loss:
                #     loss = loss + config.world_model_loss_weight * world_model_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                old_mu_sampled = b_old_mus[mb_inds]
                old_sigmas_sampled = b_old_sigmas[mb_inds]
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()

                    # We compute approx KL according to roach
                    old_distribution = agent.module.action_dist.proba_distribution(old_mu_sampled, old_sigmas_sampled)
                    kl_div = torch.distributions.kl_divergence(old_distribution.distribution, distribution)
                    approx_kl_divs.append(kl_div.mean())

                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean()]

            approx_kl = torch.mean(torch.stack(approx_kl_divs))
            # Gloo doesn't support AVG, so we implement it via sum / world size
            torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
            approx_kl = approx_kl / world_size
            if cfg.target_kl is not None and config.lr_schedule == "kl":
                if approx_kl > cfg.target_kl:
                    if config.lr_schedule_step is not None:
                        config.kl_early_stop += 1
                        if config.kl_early_stop >= config.lr_schedule_step:
                            config.current_learning_rate *= 0.5
                            config.kl_early_stop = 0

                    break

        if cfg.schedule_free:
            optimizer.eval()

        del b_obs  # Remove large array
        t4.toc(msg=f"Rank:{rank}, Training.")
        t5.tic()

        config.latest_iteration = update
        # Avg value to log over all Environments
        # Sync with 3 envs takes 4 ms.
        # Gloo doesn't support AVG, so we implement it via sum / world size
        torch.distributed.all_reduce(v_loss, op=torch.distributed.ReduceOp.SUM)
        v_loss = v_loss / world_size

        torch.distributed.all_reduce(pg_loss, op=torch.distributed.ReduceOp.SUM)
        pg_loss = pg_loss / world_size

        torch.distributed.all_reduce(entropy_loss, op=torch.distributed.ReduceOp.SUM)
        entropy_loss = entropy_loss / world_size

        if config.use_exploration_suggest:
            torch.distributed.all_reduce(exploration_loss, op=torch.distributed.ReduceOp.SUM)
            exploration_loss = exploration_loss / world_size

        # if config.use_world_model_loss:
        #     torch.distributed.all_reduce(world_model_loss, op=torch.distributed.ReduceOp.SUM)
        #     world_model_loss = world_model_loss / world_size

        torch.distributed.all_reduce(old_approx_kl, op=torch.distributed.ReduceOp.SUM)
        old_approx_kl = old_approx_kl / world_size

        torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
        approx_kl = approx_kl / world_size

        b_values = b_values[b_inds_original]
        torch.distributed.all_reduce(b_values, op=torch.distributed.ReduceOp.SUM)
        b_values = b_values / world_size

        b_returns = b_returns[b_inds_original]
        torch.distributed.all_reduce(b_returns, op=torch.distributed.ReduceOp.SUM)
        b_returns = b_returns / world_size

        b_advantages = b_advantages[b_inds_original]
        torch.distributed.all_reduce(b_advantages, op=torch.distributed.ReduceOp.SUM)
        b_advantages = b_advantages / world_size

        clipfracs = torch.mean(torch.stack(clipfracs))
        torch.distributed.all_reduce(clipfracs, op=torch.distributed.ReduceOp.SUM)
        clipfracs = clipfracs / world_size

        if rank == 0:
            save(
                agent,
                optimizer,
                config,
                exp_folder,
                f"model_latest_{update:09d}.pth",
                f"optimizer_latest_{update:09d}.pth",
            )
            frac = update / num_updates
            if config.current_eval_interval_idx < len(config.eval_intervals):
                if frac >= config.eval_intervals[config.current_eval_interval_idx]:
                    save(
                        agent,
                        None,
                        config,
                        exp_folder,
                        f"model_eval_{update:09d}.pth",
                        None,
                    )
                    config.current_eval_interval_idx += 1

            # Cleanup file from last epoch
            for file in os.listdir(exp_folder):
                if file.startswith("model_latest_") and file.endswith(".pth"):
                    if file != f"model_latest_{update:09d}.pth":
                        old_model_file = os.path.join(exp_folder, file)
                        if os.path.isfile(old_model_file):
                            os.remove(old_model_file)
                if file.startswith("optimizer_latest_") and file.endswith(".pth"):
                    if file != f"optimizer_latest_{update:09d}.pth":
                        old_model_file = os.path.join(exp_folder, file)
                        if os.path.isfile(old_model_file):
                            os.remove(old_model_file)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate",
                optimizer.param_groups[0]["lr"],
                config.global_step,
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), config.global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), config.global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), config.global_step)
            if config.use_exploration_suggest:
                writer.add_scalar("losses/exploration", exploration_loss.item(), config.global_step)
            # if config.use_world_model_loss:
            #     writer.add_scalar("losses/world_model", world_model_loss.item(), config.global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), config.global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), config.global_step)
            writer.add_scalar("losses/clipfrac", clipfracs.item(), config.global_step)
            writer.add_scalar("losses/explained_variance", explained_var, config.global_step)
            writer.add_scalar("losses/latest_epoch", latest_epoch, config.global_step)
            writer.add_scalar("charts/discounted_returns", b_returns.mean().item(), config.global_step)
            writer.add_scalar("charts/advantages", b_advantages.mean().item(), config.global_step)
            # Adjusted so it doesn't count the first epoch which is slower than the rest (converges faster)
            writer.add_scalar(
                "charts/SPS",
                int(local_processed_samples / (time.time() - start_time)),
                config.global_step,
            )
            writer.add_scalar("charts/restart", 0, config.global_step)

            print(f"SPS: {int(local_processed_samples / (time.time() - start_time))}")

        t5.toc(msg=f"Rank:{rank}, Logging")

    env.close()
    if rank == 0:
        writer.close()

        save(
            agent,
            optimizer,
            config,
            exp_folder,
            "model_final.pth",
            "optimizer_final.pth",
        )
        wandb.finish(exit_code=0, quiet=True)
        logger.info("Done training.")
