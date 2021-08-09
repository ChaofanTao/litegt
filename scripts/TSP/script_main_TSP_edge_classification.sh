#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_tsp
# tmux detach
# pkill python

# bash script_main_TSP_edge_classification.sh

############
# TSP
############

python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Jaccard' --config 'configs/TSP_LiteGT.json' --L 12 --graph_kernel jaccard

python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Jaccard_nodeSampling' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel jaccard

python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Jaccard_nodeSampling_dimReduing' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel jaccard --dim_reduce True

python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Adder' --config 'configs/TSP_LiteGT.json' --L 12 --graph_kernel adder

python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Adder_nodeSampling' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel adder

python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Adder_nodeSampling_dimReduing' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel adder --dim_reduce True

