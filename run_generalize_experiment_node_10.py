#!/bin/bash

clip_eps=(0.2 0.4)
vf_coeff=(0.1 0.05 0.2)
ent_coeff=(0.05 0.1 0.2)
learning_rate=(0.0001 0.001)
num_update_epochs=(4 6 8)
num_steps_before_update=(2400 4800)
ent_coeff_schedule=("plateau" "sigmoid" "linear")
#network - relu with kaiming and tanh with orthogonal
#Run on 2 GPUs
#For 10 mil timesteps
#Leaky relu?
