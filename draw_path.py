from algos.ddpg_agent import ddpg_agent
import json
import os
import gymnasium as gym
import argparse
from train_ddpg import get_env_params, get_args
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import itertools
import torch
import re


def get_highest(directory):
    pattern = re.compile(r".*_(\d+)\.pt$")
    highest_number = None
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if highest_number is None or number > highest_number:
                highest_number = number

    return highest_number


def look_policy(agent: ddpg_agent, env, visu=None):

    policy = agent.planner_policy
    # if visu:
    #     env.render_mode = "human"

    total_success_rate = []
    eval_options = env.get_wrapper_attr("eval_options")
    n_eval = eval_options["goal_pos"].shape[0]
    options = {k: v[n_eval - 1] for k, v in eval_options.items()}
    for i in range(n_eval):
        # for _ in range(self.args.n_test_rollouts):
        # options = {k: v[i] for k, v in eval_options.items()}
        policy.reset()
        per_success_rate = []
        observation, infos = env.reset(options=options)

        obs = observation["observation"]
        g = observation["desired_goal"]

        for num in range(agent.env_params["max_timesteps"]):
            with torch.no_grad():
                act_obs, act_g = agent._preproc_inputs(obs, g)
                actions = policy(
                    act_obs,
                    act_g,
                    agent.args.plan_budget,
                    ref_loss=agent.goal_loss,
                    jump=agent.args.jump,
                )
            # observation_new, rew, done, info = self.test_env.step(actions)
            observation_new, reward, terminated, truncated, info = env.step(actions)
            if visu:
                env.get_wrapper_attr("render")()

            obs = observation_new["observation"]
            g = observation_new["desired_goal"]
            # per_success_rate.append(info["is_success"])
            per_success_rate.append(info["success"])
            done = terminated or truncated
            if done:
                break
        total_success_rate.append(info["success"])

    total_success_rate = np.array(total_success_rate)
    global_success_rate = np.mean(total_success_rate)


def plot_graph_path(ax: Axes, nodes, path, edges):
    ax.scatter(nodes[:, 0], nodes[:, 1], color="black")

    lc = LineCollection(edges, alpha=0.1, zorder=0)
    ax.add_collection(lc)
    ax.plot(path[:, 0], path[:, 1], color="red")
    ax.scatter(path[-1, 0], path[-1, 1], color="red")


def load_agent_and_env(path):
    hp_path = os.path.join(path, "config.txt")
    with open(hp_path, "r") as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    args.resume = True
    args.loading = True
    args.resume_path = os.path.join(path, "checkpoints")
    args.resume_epoch = get_highest(args.resume_path)
    args.fps = True
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    env_params = get_env_params(env)
    agent = ddpg_agent(args, env, env_params, test_env)
    return agent, test_env


def get_graph(agent: ddpg_agent, draw_env, observation, save_path) -> None:
    obs, goal = observation["observation"], observation["desired_goal"]
    obs, goal = agent._preproc_inputs(obs, goal)
    agent.planner_policy(
        obs,
        goal,
        agent.args.plan_budget,
        ref_loss=agent.goal_loss,
        jump=agent.args.jump,
    )
    ag = observation["achieved_goal"].reshape(1, -1)
    # landmarks = np.concatenate([ag, agent.planner_policy.landmarks.cpu()])
    landmarks = agent.planner_policy.landmarks.cpu()
    dists_pairwise = agent.planner_policy.dists_pairwise
    goal_series = np.concatenate([ag, agent.planner_policy.goal_series])
    edge_idx = itertools.product(
        range(dists_pairwise.shape[0]), range(dists_pairwise.shape[1])
    )
    edge_idx = np.array(
        [
            (i, j)
            for (i, j) in edge_idx
            if agent.planner_policy.dists_pairwise[i, j] > agent.planner_policy.clip_v
        ]
    )
    edges = [[landmarks[i], landmarks[j]] for (i, j) in edge_idx]

    ax = plt.subplot()
    if draw_env:
        draw_env(ax)
    plot_graph_path(
        ax,
        np.array(landmarks),
        np.array(goal_series),
        edges,
    )
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # path = "runs/ddpgNov15_16-05-00_DubinsUMaze-v0"
    # path = "runs/ddpgNov18_16-36-01_DubinsUMaze-v0"
    # path = "runs/ddpgNov18_18-50-25_Dubins3UMaze-v0"
    path = "runs/ddpgNov28_18-45-36_PointMaze_UMaze-v3"

    agent, env = load_agent_and_env(path)
        eval_options = env.get_wrapper_attr("eval_options")
        n_eval = eval_options[next(iter(eval_options))].shape[0]
        try:
            draw_env = env.get_wrapper_attr("plot")
        except:
            draw_env = None
        folder = os.path.join(path, "plots")

        # options = {k: v[n_eval - 1] for k, v in eval_options.items()}
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i in range(n_eval):
            options = {k: v[i] for k, v in eval_options.items()}
            for j in range(5):
                agent.planner_policy.reset()
                observation, infos = env.reset(options=options)
                get_graph(
                    agent, draw_env, observation, os.path.join(folder, f"eval_{i}_{j}")
                )

    # look_policy(agent, env, "human")
