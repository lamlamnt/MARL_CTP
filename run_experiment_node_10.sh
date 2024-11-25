#!/bin/bash

random_seed_for_training=(72 73 74 75)
prop_stoch=(0.5 0.7 0.8 0.9)
num_update_epochs=(4 6 8)
num_steps_before_update=(600 1000 1200)
clip_eps=(0.1 0.2 0.4)
vf_coeff=(0.01 0.05)
deterministic_inference_policy=(True False)

for deter in "${deterministic_inference_policy[@]}"
do
    for vf_coeff_value in "${vf_coeff[@]}"
    do
        for clip_eps_value in "${clip_eps[@]}"
        do
            for num_steps_before_update_value in "${num_steps_before_update[@]}"
            do
                for num_update_epochs_value in "${num_update_epochs[@]}"
                do
                    for i in "${!random_seed_for_training[@]}"
                    do
                        seed="${random_seed_for_training[$i]}"
                        stoch="${prop_stoch[$i]}"
                        log_dir="node10/seed_${seed}_prop_${stoch}_update_${num_update_epochs_value}_steps_${num_steps_before_update_value}_vf_${vf_coeff_value}_clip_${clip_eps_value}_deter_${deter}"
                        python main_ppo.py --n_node 10 --time_steps 500000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node10_different_graphs --num_update_epochs "$num_update_epochs_value" --num_steps_before_update "$num_steps_before_update_value" --vf_coeff "$vf_coeff_value" --clip_eps "$clip_eps_value" --deterministic_inference_policy "$deter"
                    done
                done
            done
        done
    done
done