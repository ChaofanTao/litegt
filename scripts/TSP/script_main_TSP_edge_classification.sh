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
tmux new -s screen_LiteGT -d
tmux send-keys "source activate LiteGT" C-m
tmux send-keys "
python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Jaccard' --config 'configs/TSP_LiteGT.json' --L 12 --graph_kernel jaccard
wait" C-m
tmux send-keys "
python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Jaccard_nodeSampling' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel jaccard
wait" C-m
tmux send-keys "
python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Jaccard_nodeSampling_dimReduing' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel jaccard --dim_reduce True
wait" C-m
tmux send-keys "
python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Adder' --config 'configs/TSP_LiteGT.json' --L 12 --graph_kernel adder
wait" C-m
tmux send-keys "
python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Adder_nodeSampling' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel adder
wait" C-m
tmux send-keys "
python main_TSP_edge_classification.py --dataset TSP --log_path './logs/TSP_Adder_nodeSampling_dimReduing' --config 'configs/TSP_LiteGT.json' --L 12  --jaccard_sparse True --topk_factor 5 --graph_kernel adder --dim_reduce True
wait" C-m
tmux send-keys "tmux kill-session -t screen_LiteGT" C-m
