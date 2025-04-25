#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --nodelist=htc-g053
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=super-large
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lam.lam@pmb.ox.ac.uk

#--gres=gpu:1 --constraint='gpu_mem:96GB'
#GPU name: --gres=gpu:h100:1
#Memory constraint: gres=gpu:1 --constraint='gpu_mem:16GB'

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
echo "Allocated Node(s): $SLURM_NODELIST"
echo "Running on: $(hostname)"
echo "GPU Memory Allocation:"
#This cmd is just for me to see which GPU (how much memory) is allocated to this job. 
#Does not serve the purpose of checking how much GPU memory is used during training
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
export WANDB_API_KEY="267ce0358c86cf2f418a382f7be9e01c6464c124"
srun python main_ppo.py --n_node 30 --log_directory "super_large" --time_steps 16000000 --network_type "Densenet_Same" --learning_rate 0.001 --num_steps_before_update 6000 --clip_eps 0.1366 --num_update_epochs 6 --division_plateau 3 --ent_coeff_schedule "sigmoid" --vf_coeff 0.1 --ent_coeff 0.15 --deterministic_inference_policy False --graph_identifier "node_30_0.8_training_30k_inference_1k" --prop_stoch 0.8 --num_stored_graphs 30000 --factor_inference_timesteps 500 --num_minibatches 1 --optimizer_norm_clip 1.0 --frequency_testing 20 --factor_testing_timesteps 50 --densenet_bn_size 4 --densenet_growth_rate 44 --wandb_mode online --wandb_project_name node30_0.8_96gb_sweep

#Increase to 10mil timesteps
#For 32GB, 6000 with small Densenet. 6000 did not offer significant improvements compared to 2000.
#For 16gb, network 4 32 and 2000 steps ok. When using 100k timesteps and cpu 32gb, 3000 steps was ok too.
#80-94GB on H100 (short partition) - specify h100 -> node not available
#Maybe cpu mem also plays a role

#Copy the log directory folder back to $DATA
rsync -av --exclude=input --exclude=bin ./Logs/super_large $DATA/MARL_CTP/ARC