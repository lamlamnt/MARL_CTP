#!/bin/bash

cd ..

random_seed_for_training=(31 33 34 35)
prop_stoch=(0.4 0.5 0.6 0.7)

#Hyperparameters to tune
deterministic_inference_policy=(False)
anneal_lr=(True False)
learning_rate=(0.00025 0.001)
num_update_epochs=(4 6 8)
num_steps_before_update=(600)
num_minibatches=(4)
ent_coeff=(0.05)
vf_coeff=(0.05 0.2 0.4)
clip_eps=(0.1 0.2 0.4)
#gae_lambda=(0.9 0.95)

for clip_eps_value in "${clip_eps[@]}"
do
    for vf_coeff_value in "${vf_coeff[@]}"
    do
        for ent_coeff_value in "${ent_coeff[@]}"
        do
            for num_minibatches_value in "${num_minibatches[@]}"
            do
                for num_steps_before_update_value in "${num_steps_before_update[@]}"
                do
                    for num_update_epochs_value in "${num_update_epochs[@]}"
                    do
                        for lr in "${learning_rate[@]}"
                        do
                            for anneal in "${anneal_lr[@]}"
                            do
                                for policy in "${deterministic_inference_policy[@]}"
                                do
                                    for i in "${!random_seed_for_training[@]}"
                                    do
                                        seed="${random_seed_for_training[$i]}"
                                        stoch="${prop_stoch[$i]}"
                                        log_dir="node5_full/${seed}_prop_${stoch}_deter_${policy}_anneal_${anneal}_lr_${lr}_num_update_${num_update_epochs_value}_num_steps_${num_steps_before_update_value}_minibatches_${num_minibatches_value}_ent_${ent_coeff_value}_vf_${vf_coeff_value}_clip_${clip_eps_value}"
                                        python main_ppo.py --n_node 5 --time_steps 200000 --log_directory "$log_dir" --random_seed_for_training "$seed" --prop_stoch "$stoch" --wandb_mode online --wandb_project_name node5_different_graphs_full --deterministic_inference_policy "$policy" --anneal_lr "$anneal" --learning_rate "$lr" --num_update_epochs "$num_update_epochs_value" --num_steps_before_update "$num_steps_before_update_value" --num_minibatches "$num_minibatches_value" --ent_coeff "$ent_coeff_value" --vf_coeff "$vf_coeff_value" --clip_eps "$clip_eps_value"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done