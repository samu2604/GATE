#!/bin/bash

source $HOME/.bashrc
conda activate hyband-snakemake

hyband generate 6 2 --bracket 0 --last-stage 0 --template-dir $HOME/GhostFreePro/GNN/gcn_random_search_template/ --output-dir $HOME/GhostFreePro/GNN/gcn_random_searches/$1
