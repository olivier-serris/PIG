#!/usr/bin/env bash
SEED=$1
python train_ddpg.py \
--env-name AntMaze_UMaze-eval-v5 \
--test AntMaze_UMaze-eval-v5 \
--device cuda:0 \
--random-eps 0.2 \
--gamma 0.99 \
--n-epochs 1430 \
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
--group g0