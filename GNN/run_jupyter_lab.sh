#!/bin/bash

#SBATCH -o slurm_jupyter_%j.txt
#SBATCH -e slurm_error_%j.txt
#SBATCH -J jupyter
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --partition=cpu_p
#SBATCH --nice=10000

source $HOME/.bashrc

echo "Starting jupyter..."

chmod 600 $HOME/slurm_jupyter_$SLURM_JOB_ID.job

conda deactivate
conda activate pytorch_geometric

jupyter-lab --port 8888 --no-browser --ip=0.0.0.0 --notebook-dir=$HOME/GhostFreePro

#if you want to remove error 

rm $HOME/slurm_jupyter_$SLURM_JOB_ID.job
