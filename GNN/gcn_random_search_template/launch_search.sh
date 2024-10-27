source $HOME/.bashrc
conda activate snakemake

##snakemake --snakefile $HOME/GhostFreePro/GCN_experiment/gcn_random_searches/$1/Snakefile > $HOME/GhostFreePro/GCN_experiment/gcn_random_searches/$1/log.out
nohup snakemake --snakefile $HOME/GhostFreePro/GNN/gcn_random_searches/$1/Snakefile --latency-wait 60 -j 50 > $HOME/GhostFreePro/GNN/gcn_random_searches/$1/log.out


