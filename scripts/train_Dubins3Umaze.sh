#!/usr/bin/env bash
SEED=$1
python train_ddpg.py \
--env-name Dubins3UMaze-v0 \
--test Dubins3UMaze-v0 \
--random-eps 0.2 \
--device cuda:0 \
--gamma 0.99 \
--n-epochs 1250 \
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
--n_eval 5 \
--mode online \
--group g0