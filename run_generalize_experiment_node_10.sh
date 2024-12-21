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
CUDA_VISIBLE_DEVICES=0 python main_ppo.py --n_node 10 --log_directory "hyperparameter_sweep" --time_steps 15000 --learning_rate 0.0001 --num_steps_before_update 2400 --clip_eps 0.2 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "plateau" --vf_coeff 0.1 --graph_identifier "node_10_relabel_0.4" --wandb_mode online --wandb_project_name "test_generalize_node_10" --wandb_sweep True --prop_stoch 0.4 --sweep_run_count 3 --yaml_file "sweep_config_node_10.yaml"


