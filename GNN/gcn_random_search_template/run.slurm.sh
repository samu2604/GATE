sbatch --wait << EOF
#!/bin/bash

#SBATCH -J run_gnn
#SBATCH -o "$HOME/GhostFreePro/GNN/slurm_out_%j.job"
#SBATCH -e "$HOME/GhostFreePro/GNN/slurm_err_%j.job"
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1 
#SBATCH -t 00:55:00
#SBATCH -c 4
#SBATCH --mem=15G 
#SBATCH --nice=10000


source $HOME/.bashrc
conda activate pyg-env

echo "Called with base directory $1 and budget $2"

python $HOME/GhostFreePro/GNN/train.py --config=$HOME/GhostFreePro/GNN/config

EOF
