#!/bin/bash

prop_stoch=(0.4 0.8)
clip_eps=(0.2 0.4 0.1)
vf_coeff=(0.1 0.05 0.2)
ent_coeff=(0.05 0.1 0.2)
learning_rate=(0.0001 0.001)
num_update_epochs=(4 6 8)
num_steps_before_update=(2400 4800 1500)
ent_coeff_schedule=("plateau" "sigmoid" "linear")

#Just try tanh on one set -> should be enough to know if good or not
#Maybe add some FC layers at the end of DenseNet?