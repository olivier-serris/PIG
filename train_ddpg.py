import numpy as np
import gymnasium as gym
import os, sys
from arguments_ddpg import get_args
from algos.ddpg_agent import ddpg_agent

# from goal_env import *
# from goal_env.mujoco import *
import random
import torch
import gymnasium_robotics
import gym_gmazes


def get_env_params(env):
    obs, infos = env.reset()
    # close the environment
    params = {
        "obs": obs["observation"].shape[0],
        "goal": obs["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
    }
    params["max_timesteps"] = env.get_wrapper_attr("_max_episode_steps")
    return params


def launch(args):
    # create the ddpg_agent
    # print(print(gym.envs.all))
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    # set random seeds for reproduce
    # env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    env.reset(seed=args.seed)
    env_params["max_test_timesteps"] = test_env.get_wrapper_attr("_max_episode_steps")
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params, test_env)
    ddpg_trainer.learn()


if __name__ == "__main__":
    # get the params
    args = get_args()
    launch(args)
