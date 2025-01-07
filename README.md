**Instructions for setting up**
<br>1. Use JaxMARL's Dockerfile (https://github.com/FLAIROx/JaxMARL/blob/main/Dockerfile) to build a Docker image with the addition of the following packages.
<br>2. pip install pytest-print jax-tqdm==0.2.2 networkx pandas openpyxl dm-haiku wandb

**Example**
```
python main_ppo.py --n_node 30 --log_directory "node_30_example" --network_type "Densenet_Same" --time_steps 6000000 --learning_rate 0.001 --num_steps_before_update 3000 --clip_eps 0.14 --num_update_epochs 6 --division_plateau 4 --ent_coeff_schedule "plateau" --vf_coeff 0.13 --ent_coeff 0.17 --graph_identifier "node_30_relabel_0.8" --prop_stoch 0.8 --num_minibatches 2 --horizon_length_factor 1 --densenet_growth_rate 20 --densenet_bn_size 2 --factor_inference_timesteps 500
```

**Note**
Only for single agent for now. Multi-agent will be in a different repo. 



