#!/bin/bash

echo "Timesteps: $1"

# Define the configurations for the experiments
learning_rates=("0.001" "0.0001")
batch_sizes=("32" "64" "128")
buffer_sizes=("1000" "2000","3000")
target_net_update_freqs=("10" "40" "100" "200")
optimizer=("Adam" "RMSprop" "Adabelief" "Adamw")
replay_buffer_type=("uniform" "per")
network_size=("[600,300,100,50]" "[400,200,30]")

# Loop over all combinations of configurations
for lr in "${learning_rates[@]}"
do
  for bs in "${batch_sizes[@]}"
  do
    for buffer in "${buffer_sizes[@]}"
    do
      for target_freq in "${target_net_update_freqs[@]}"
      do
        for opt in "${optimizer[@]}"
        do
          for replay in "${replay_buffer_type[@]}"
          do
            for network_size in "${network_size[@]}"
            do
                # Run the experiment (sequentially)
                log_dir="logs/lr_${learning_rate}_bs_${batch_size}_buffer_${buffer_size}_target_freq_${target_net_update_freq}_opt_${optimizer}_replay_${replay_buffer_type}_network_size_${network_size}"
                echo "Running experiment with lr=$lr, batch_size=$bs, buffer_size=$buffer, target_net_update_freq=$target_freq, optimizer=$opt, replay_buffer_type=$replay, network_size=$network_size"
                python main_dqn.py --n_node 10 --grid_size 15 --time_steps $1 --log_directory log_dir --learning_rate $lr --batch_size $bs --buffer_size $buffer --target_net_update_freq $target_freq --optimizer $opt --replay_buffer_type $replay --network_size $network_size
            done
        done
      done
    done
  done
done

echo "All experiments finished!"