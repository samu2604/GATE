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
#SBATCH --qos=gpu
#SBATCH --nice=10000

source $HOME/.bashrc
export PYTHONPATH="$HOME:$PYTHONPATH"
conda activate pyg_gpu 

python $HOME/GhostFreePro/GNN/positional_encoding.py

EOF