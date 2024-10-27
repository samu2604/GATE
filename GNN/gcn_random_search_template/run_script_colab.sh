#!/bin/bash

for filename in $1* 
do
     python /content/gdrive/MyDrive/project_folder/GhostFreePro/GNN/train.py --config=$filename/config
done    