import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from supersuit.multiagent_wrappers import padding_wrappers

import sumo_rl

if __name__ == "__main__":
    ray.init()

    env_name = "experiment "

    env = sumo_rl.parallel_env(
        net_file=".../experiment.net.xml",
        route_file=".../experiment.rou.xml",
        out_csv_name=".../DQN_result",
        use_gui=False,
        num_seconds=80000,
    )

    env = padding_wrappers.pad_observations_v0(env)
    env = padding_wrappers.pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)

    register_env(env_name, lambda _: env)

    config = (
        DQNConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            grad_clip=None,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "DQN",
        name="DQN",
        stop={"timesteps_total": 1500000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
