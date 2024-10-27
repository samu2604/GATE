sbatch --wait << EOF
#!/bin/bash

#SBATCH -J train
#SBATCH --output=$HOME/GhostFreePro/GNN/logs/job_output_%j.txt
#SBATCH --error=$HOME/GhostFreePro/GNN/logs/job_error_%j.txt
#SBATCH --time=20:00:00 
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

export PYTHONPATH="$HOME:$PYTHONPATH"
source $HOME/.bashrc
conda activate pyg_gpu 

python $HOME/GhostFreePro/GNN/train.py

EOF