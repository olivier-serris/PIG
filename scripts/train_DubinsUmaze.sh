#!/usr/bin/env bash
SEED=$1
GROUP=$2
python train_ddpg.py \
--env-name DubinsUMaze-v0 \
--test DubinsUMaze-v0 \
--device cuda:0 \
--random-eps 0.2 \
--gamma 0.99 \
--n-epochs 2143 \
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
--eval-freq 70 \
--n_eval 5 \
--group $GROUP 

# epoch * max_ep_steps = total train timesteps
#2143*70=150010 steps