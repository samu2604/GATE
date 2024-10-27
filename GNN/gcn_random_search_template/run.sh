sbatch --wait << EOF
#!/bin/bash

#SBATCH -J run_gcn
#SBATCH -o "${1}/slurm_out_%j.job"
#SBATCH -e "${1}/slurm_err_%j.job"
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1 
#SBATCH -t 00:55:00
#SBATCH -c 4
#SBATCH --mem=15G 
#SBATCH --nice=10000 

source $HOME/.bashrc
conda activate pyg_gpu 

python $HOME/GhostFreePro/GNN/train.py --config ${1}/config

EOF