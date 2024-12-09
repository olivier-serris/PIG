#!/usr/bin/env bash
SEED=$1
GROUP=$2
python train_ddpg.py \
--env-name Dubins3UMaze-v0 \
--test Dubins3UMaze-v0 \
--device cuda:0 \
--random-eps 0.2 \
--gamma 0.99 \
--n-epochs 1500 \
--period 3 \
--distance 0.1 \
--fps \
--landmark 400 \
--initial-sample 500 \
--clip-v -38 \
--jump \
--lambda_goal_loss 0.001 \
--jump_temp 10 \
--seed ${SEED} \
--eval-freq 25 \
--n_eval 5 \
--group $GROUP 

# epoch * max_ep_steps = total train timesteps
#1500*200=300_000 steps