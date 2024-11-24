#!/bin/bash

random_seed_for_training=(32 31 30 33 34 35)
prop_stoch=(0.4 0.4 0.4 0.5 0.6 0.7)

#Hyperparameters to tune
deterministic_inference_policy=(True False)
anneal_lr=(True False)
learning_rate=(0.00025 0.001)
num_update_epochs=(4 6 8)
num_steps_before_update=(200 600)
num_minibatches=(4 8)
ent_coeff=(0.01 0.05 0.07)
vf_coeff=(0.05 0.2 0.5)
clip_eps=(0.1 0.2 0.4)
gae_lambda=(0.9 0.95)

for policy in "${deterministic_inference_policy[@]}"
do
    for i in "${!random_seed_for_training[@]}"
    do
        seed="${random_seed_for_training[$i]}"
        stoch="${prop_stoch[$i]}"
        log_dir="node5/random_seed_${seed}_prop_stoch_${stoch}_deterministic_${policy}"
        python main_ppo.py --n_node 5 --time_steps 100000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node5_different_graphs --deterministic_inference_policy "$policy"
    done
done
