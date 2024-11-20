#!/bin/bash

learning_rates=("0.001" "0.00025" "0.0001")
clip_range=("0.1" "0.3")
ent_coef=("0.01" "0.03")
num_steps_before_update=("128" "256" "512")
num_minibatches=("4" "6")
network_activation=("tanh" "relu")
network_type=("CNN","Narrow")

for lr in "${learning_rates[@]}"
do
  for cr in "${clip_range[@]}"
  do
    for ec in "${ent_coef[@]}"
    do
      for nsbu in "${num_steps_before_update[@]}"
      do
        for nmb in "${num_minibatches[@]}"
        do
          for na in "${network_activation[@]}"
          do
            for nt in "${network_type[@]}"
            do
              log_dir="logs/lr_${lr}_cr_${cr}_ec_${ec}_nsbu_${nsbu}_nmb_${nmb}_na_${na}_nt_${nt}"
              echo "Running experiment with lr=$lr, clip_range=$cr, ent_coef=$ec, num_steps_before_update=$nsbu, num_mini_batch=$nmb, network_activation=$na, network_type=$nt"
              python main_ppo.py --n_node 30 --time_steps 2000000 --log_directory $log_dir --learning_rate $lr --clip_range $cr --ent_coef $ec --num_steps_before_update $nsbu --num_minibatches $nmb --network_activation $na --network_type $nt
            done
          done
        done
      done
    done
  done
done


