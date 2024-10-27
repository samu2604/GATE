#!/bin/bash

for filename in $1* 
do
     python /home/samuele/GhostFreePro/GNN/train.py --config=$filename/config
done    
#python /home/samuele/GhostFreePro/GNN/train.py --config=/home/samuele/GhostFreePro/GNN/config & python /home/samuele/GhostFreePro/GNN/train.py --config=/home/samuele/GhostFreePro/GNN/config & 