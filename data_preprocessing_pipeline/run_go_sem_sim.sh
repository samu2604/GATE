#!/bin/bash
for index_1 in {1..100}; do
    index_2=$((index_1 + 1))
    sbatch << EOF
#!/bin/bash
#SBATCH -J r-script
#SBATCH --output=$HOME/GhostFreePro/data_preprocessing_pipeline/r_logs/job_output_%j.txt
#SBATCH --error=$HOME/GhostFreePro/data_preprocessing_pipeline/r_logs/job_error_%j.txt
#SBATCH --time=30:00:00 
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate r-env-cluster 

Rscript $HOME/GhostFreePro/data_preprocessing_pipeline/go_similarity_partial.R $index_1 $index_2

EOF
done
