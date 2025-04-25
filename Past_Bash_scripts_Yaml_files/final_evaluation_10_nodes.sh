#!/bin/bash

random_seed_for_training=(30 32 34)
prop_stoch=(0.2 0.4)

for prop in "${prop_stoch[@]}"
do
    for seed in "${random_seed_for_training[@]}"
    do
        next_seed=$((seed + 1))
        log_dir="node10_results/prop_${prop}_${seed}"
        graph_identifier_name="node_10_${prop}_training_14k"
        CUDA_VISIBLE_DEVICES=0 python main_ppo.py --n_node 10 --log_directory "$log_dir" --time_steps 6000000 --network_type "Densenet_Same" --learning_rate 0.001 --num_steps_before_update 3600 --clip_eps 0.1366 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.128 --ent_coeff 0.174 --deterministic_inference_policy True --graph_identifier "$graph_identifier_name" --prop_stoch "$prop" --num_stored_graphs 14000 --factor_inference_timesteps 1000 --num_minibatches 1 --optimizer_norm_clip 1.0 --frequency_testing 20 --factor_testing_timesteps 50 --random_seed_for_training "$seed" --random_seed_for_inference "$next_seed" 
    done
done


