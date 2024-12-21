#!/bin/bash

prop_stoch = 0.4
clip_eps=(0.2 0.4 0.1)
vf_coeff=(0.1 0.05 0.2)
ent_coeff=(0.05 0.1 0.2)
learning_rate=(0.0001 0.001)
num_update_epochs=(6 8 4)
num_steps_before_update=(2400 3600)
ent_coeff_schedule=("plateau" "sigmoid")

#Try on prop_stoch 0.4 first to eliminate hyperparameters
#Maybe add some FC layers at the end of DenseNet?
CUDA_VISIBLE_DEVICES=1 python main_ppo.py --n_node 5 --log_directory "relabel_node_5_1_mil" --time_steps 1000000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_5_relabel_0.4" --wandb_mode online --wandb_project_name "generalize_node_5"

#Run initial for prop 0, 0.2, and 0.8