#!/usr/bin/env bash
SEED=$1
GROUP=$2
python train_ddpg.py \
--env-name PointMaze_UMaze-eval-v3 \
--test PointMaze_UMaze-eval-v3 \
--device cuda:0 \
--random-eps 0.2 \
--gamma 0.99 \
--n-epochs 667 \
--period 3 \
--distance 0.45 \
--fps \
--landmark 400 \
--initial-sample 500 \
--clip-v -38 \
--jump \
--lambda_goal_loss 0.001 \
--jump_temp 10 \
--seed ${SEED} \
--n_eval 5 \
--group $GROUP 

# epoch * max_ep_steps = total train timesteps
#667*300=200100 steps