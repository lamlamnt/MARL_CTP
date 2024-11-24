#!/bin/bash

random_seed_for_training=(32 31 30 33 34 35)
prop_stoch=(0.4 0.4 0.4 0.5 0.6 0.7)

for i in "${!random_seed_for_training[@]}"
do
    seed="${random_seed_for_training[$i]}"
    stoch="${prop_stoch[$i]}"
    log_dir="node5/random_seed_${seed}_prop_stoch_${stoch}"
    python main_ppo.py --n_node 5 --time_steps 100000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node5_different_graphs
done
