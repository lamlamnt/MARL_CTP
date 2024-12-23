#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 30 --log_directory "initial_node_30_0.8" --time_steps 10000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_30_relabel_0.8" --prop_stoch 0.8
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 30 --log_directory "initial_node_30_0.4" --time_steps 10000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_30_relabel_0.4" --prop_stoch 0.4
