#!/bin/bash

random_seed_for_training=(30 70 71 72 73 74 75)
prop_stoch=(0.4 0.8 0.3 0.5 0.7 0.8 0.9)

for run in "${learning_rates[@]}"
do
    log_dir = "node10/random_seed_${run}_prop_stoch_${run}"
    python main_ppo.py --n_node 10 --time_steps 100000 --log_directory logs/ --random_seed_for_training $run --prop_stoch $run --wandb_mode online --wandb_project_name node10_different_graphs
done
