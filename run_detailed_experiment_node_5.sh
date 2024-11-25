#!/bin/bash

random_seed_for_training=(30 35)
prop_stoch=(0.4 0.7)

#Hyperparameters to tune
learning_rate=(0.001 0.01)
num_update_epochs=(4 6 8)
num_steps_before_update=(600 800)
vf_coeff=(0.05 0.4)
clip_eps=(0.1 0.2 0.4)
gae_lambda=(0.9 0.95)

for clip_eps_value in "${clip_eps[@]}"
do
    for vf_coeff_value in "${vf_coeff[@]}"
    do
        for num_update_epochs_value in "${num_update_epochs[@]}"
        do
            for lr in "${learning_rate[@]}"
            do
                for gae_lambda_value in "${gae_lambda[@]}"
                do
                    for num_steps_before_update_value in "${num_steps_before_update[@]}"
                    do
                        for i in "${!random_seed_for_training[@]}"
                        do
                            seed="${random_seed_for_training[$i]}"
                            stoch="${prop_stoch[$i]}"
                            log_dir="node5_seed_30/lr_${lr}_num_update_${num_update_epochs_value}_num_steps_${num_steps_before_update_value}_vf_${vf_coeff_value}_clip_${clip_eps_value}_gae_${gae_lambda_value}"
                            python main_ppo.py --n_node 5 --time_steps 200000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node5_different_graphs_random_30 --learning_rate "$lr" --num_update_epochs "$num_update_epochs_value" --num_steps_before_update "$num_steps_before_update_value" --vf_coeff "$vf_coeff_value" --clip_eps "$clip_eps_value" --gae_lambda "$gae_lambda_value"
                        done
                    done
                done
            done
        done
    done
done