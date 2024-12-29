#Store 30-node graphs with origin 20 and 10
CUDA_VISIBLE_DEVICES="" python main_ppo.py --n_node 30 --log_directory "store" --graph_mode "store" --graph_identifier "node_30_origin_20" --prop_stoch 0.4 --num_stored_graphs 2000 --origin_node 20
CUDA_VISIBLE_DEVICES="" python main_ppo.py --n_node 30 --log_directory "store" --graph_mode "store" --graph_identifier "node_30_origin_10" --prop_stoch 0.4 --num_stored_graphs 2000 --origin_node 10

