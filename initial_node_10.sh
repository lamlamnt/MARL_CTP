#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "initial_node_10_0" --time_steps 6000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_10_relabel_0" 
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "initial_node_10_0.2" --time_steps 6000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_10_relabel_0.2" 
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "initial_node_10_0.8" --time_steps 6000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_10_relabel_0.8" 

