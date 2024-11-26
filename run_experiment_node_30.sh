#!/bin/bash

random_seed_for_training=(30 30)
prop_stoch=(0.8 0.4)
num_update_epochs=(10)
num_steps_before_update=(2400)
num_minibatches=(8)
ent_coeff=(0.1 0.2 0.3)
clip_eps=(0.1)
learning_rate=(0.01)
vf_coeff=0.05
deter=False
horizon_length_factor=3

for lr in "${learning_rate[@]}"
do
    for num_update_epochs_value in "${num_update_epochs[@]}"
    do
        for num_steps_before_update_value in "${num_steps_before_update[@]}"
        do
            for num_minibatches_value in "${num_minibatches[@]}"
            do
                for ent_coeff_value in "${ent_coeff[@]}"
                do
                    for clip_eps_value in "${clip_eps[@]}"
                    do
                        for i in "${!random_seed_for_training[@]}"
                        do
                            seed="${random_seed_for_training[$i]}"
                            stoch="${prop_stoch[$i]}"
                            log_dir="node30/seed_${seed}_prop_${stoch}_lr_${lr}_update_${num_update_epochs_value}_steps_${num_steps_before_update_value}_minibatch_${num_minibatches_value}_ent_${ent_coeff_value}_clip_${clip_eps_value}"
                            python main_ppo.py --n_node 30 --time_steps 10000000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node30_first_tune --num_update_epochs "$num_update_epochs_value" --num_steps_before_update "$num_steps_before_update_value" --num_minibatches "$num_minibatches_value" --ent_coeff "$ent_coeff_value" --clip_eps "$clip_eps_value" --vf_coeff "$vf_coeff" --deterministic_inference_policy "$deter" --learning_rate "$lr" --horizon_length_factor "$horizon_length_factor"
                        done
                    done
                done
            done
        done
    done
done