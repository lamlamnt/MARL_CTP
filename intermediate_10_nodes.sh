#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "pessimistic_normal_0.4" --time_steps 6000000 --learning_rate 0.001 --num_steps_before_update 3600 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.15 --ent_coeff 0.12 --graph_identifier "node_10_relabel_0.4" --prop_stoch 0.4
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "pessimistic_mixed" --time_steps 6000000 --learning_rate 0.001 --num_steps_before_update 3600 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.15 --ent_coeff 0.12 --graph_identifier "node_10_mixed" 
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "pessimistic_deterministic" --time_steps 6000000 --learning_rate 0.001 --num_steps_before_update 3600 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.15 --ent_coeff 0.12 --graph_identifier "node_10_relabel_0.4" --prop_stoch 0.4 --deterministic_inference_policy True
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 10 --log_directory "pessimistic_different_random_seed" --time_steps 6000000 --learning_rate 0.001 --num_steps_before_update 3600 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.15 --ent_coeff 0.12 --graph_identifier "node_10_relabel_0.4" --prop_stoch 0.4 --random_seed_for_training 100 --random_seed_for_inference 101

#do it for 0, 0.2, and 0.8 10 nodes
#30 nodes cnn