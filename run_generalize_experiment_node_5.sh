#!/bin/bash

prop_stoch=(0.4 0.8)

#3mil timesteps (Train on 2 GPUs -> aim for )
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 5 --log_directory "relabel_node_5_1_mil" --time_steps 1000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_5_relabel_0.4" --wandb_mode online --wandb_project_name generalize_node_5
