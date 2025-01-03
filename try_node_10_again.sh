#Same as before but minibatch 1 and more epochs
CUDA_VISIBLE_DEVICES=0 python main_ppo.py --n_node 10 --log_directory "mixed_10_minibatch_1" --time_steps 6000000 --learning_rate 0.001 --num_steps_before_update 3600 --clip_eps 0.2 --num_update_epochs 10 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.15 --ent_coeff 0.12 --graph_identifier "node_10_mixed" --deterministic_inference_policy True --num_stored_graphs 2000 --factor_inference_timesteps 1000 --num_minibatch 1

#Same as before but minibatch 2 and more epochs

#Share parameters and more sampled graphs

#Both together (share parameters, minibatch 1, more epochs, more sampled graphs)