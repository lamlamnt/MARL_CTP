#!/bin/bash

random_seed_for_training=(30 70 71 72 73 74 75)
prop_stoch=(0.4 0.8 0.3 0.5 0.7 0.8 0.9)

for i in "${!random_seed_for_training[@]}"
do
    seed="${random_seed_for_training[$i]}"
    stoch="${prop_stoch[$i]}"
    log_dir="node10/random_seed_${seed}_prop_stoch_${stoch}"
    python main_ppo.py --n_node 10 --time_steps 100000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node10_different_graphs
done
