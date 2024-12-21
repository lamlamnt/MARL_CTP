#!/bin/bash

echo "Generating 5-node graphs ..."
CUDA_VISIBLE_DEVICES="" python main_ppo.py --n_node 5 --log_directory "store" --graph_mode "store" --graph_identifier "node_5_relabel_0.4" --prop_stoch 0.4 --num_stored_graphs 2000
echo "Generating 10-node graphs with prop_stoch 0 ..."
CUDA_VISIBLE_DEVICES="" python main_ppo.py --n_node 10 --log_directory "store" --graph_mode "store" --graph_identifier "node_10_relabel_0" --prop_stoch 0 --num_stored_graphs 2000
echo "Generating 10-node graphs with prop_stoch 0.8 ..."
CUDA_VISIBLE_DEVICES="" python main_ppo.py --n_node 10 --log_directory "store" --graph_mode "store" --graph_identifier "node_10_relabel_0.8" --prop_stoch 0.8 --num_stored_graphs 2000
echo "Generating 10-node graphs with prop_stoch 0.2 ..."
CUDA_VISIBLE_DEVICES="" python main_ppo.py --n_node 10 --log_directory "store" --graph_mode "store" --graph_identifier "node_10_relabel_0.2" --prop_stoch 0.2 --num_stored_graphs 2000


