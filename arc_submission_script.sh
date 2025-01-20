#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=medium
#SBATCH --job-name=rl-ctp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lam.lam@pmb.ox.ac.uk

#Maybe include cpu_per_task
cd $SCRATCH || exit 1

#Copy MARL folder to $SCRATCH
rsync -av $DATA/MARL_CTP ./

module purge
module load Anaconda3
source activate /data/engs-goals/pemb6454/marl_3

echo "Copied MARL folder and entered conda environment"

#not sure if srun interferes with jax
export XLA_PYTHON_CLIENT_PREALLOCATE=true 
export XLA_PYTHON_CLIENT_MEM_FRACTION=1
cd MARL_CTP
echo "GPU Memory Allocation:"
#This cmd is just for me to see which GPU (how much memory) is allocated to this job. 
#Does not serve the purpose of checking how much GPU memory is used during training
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
srun python main_ppo.py --n_node 30 --log_directory "node_30_prop_0.4_small_densenet" --time_steps 10000000 --network_type "Densenet_Same" --learning_rate 0.001 --num_steps_before_update 6000 --clip_eps 0.1366 --num_update_epochs 6 --division_plateau 3 --ent_coeff_schedule "sigmoid" --vf_coeff 0.128 --ent_coeff 0.174 --deterministic_inference_policy False --graph_identifier "node_30_0.4_training_30k_inference_1k" --prop_stoch 0.4 --num_stored_graphs 30000 --factor_inference_timesteps 500 --num_minibatches 1 --optimizer_norm_clip 1.0 --frequency_testing 20 --factor_testing_timesteps 50 --densenet_bn_size 2 --densenet_growth_rate 20

#Increase to 10mil timesteps
#For 32GB, Probably 6000 with small Densenet OR big DenseNet with 3000
#Use Fortinbras to determine exact memory usage

#Copy the log directory folder back to $DATA
rsync -av --exclude=input --exclude=bin ./Logs/node_30_prop_0.4_small_densenet $DATA/MARL_CTP/ARC