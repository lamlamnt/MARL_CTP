#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1 --constraint="gpu_mem:32GB&[scratch:weka|scratch:gpfs]"
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=test-arc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lam.lam@pmb.ox.ac.uk

#Maybe include cpu_per_task
cd $SCRATCH || exit 1

#Copy MARL folder to $SCRATCH
rsync -av $HOME/pemb6454/MARL_CTP ./

module purge
module load Anaconda3
source activate /data/engs-goals/pemb6454/marl_2

echo "Copied MARL folder and entered conda environment"

#not sure if srun interferes with jax
#export XLA_PYTHON_CLIENT_PREALLOCATE=true 
#export XLA_PYTHON_CLIENT_MEM_FRACTION=1
cd MARL_CTP
echo "Current directory: $(pwd)"
srun python main_ppo.py --n_node 5 --log_directory "test" --time_steps 100000 --network_type "CNN" --learning_rate 0.001 --num_steps_before_update 1000 --clip_eps 0.1366 --num_update_epochs 6 --division_plateau 5 --ent_coeff_schedule "sigmoid" --vf_coeff 0.128 --ent_coeff 0.174 --deterministic_inference_policy True --graph_identifier "node_5_mixed" --prop_stoch 0.4 --num_stored_graphs 2000 --factor_inference_timesteps 1000 --num_minibatches 1 --optimizer_norm_clip 1.0 --frequency_testing 20 --factor_testing_timesteps 50 

#Copy the log directory folder back to $HOME or $DATA
rsync -av --exclude=input --exclude=bin ./Logs/test $HOME/pemb6454/MARL_CTP/ARC