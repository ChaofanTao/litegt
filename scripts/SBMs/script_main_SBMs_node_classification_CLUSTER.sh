#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# SBM_CLUSTER
############

tmux new -s screen_LiteGT -d
tmux send-keys "source activate LiteGT" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_CLUSTER --log_path './logs/CLUSTER_Jaccard'   --config 'configs/SBMs_LiteGT_CLUSTER' --graph_kernel jaccard 
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_CLUSTER --log_path './logs/CLUSTER_Jaccard_nodeSampling'   --config 'configs/SBMs_LiteGT_CLUSTER.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel jaccard
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_CLUSTER --log_path './logs/CLUSTER_Jaccard_nodeSampling_dimReduing'   --config 'configs/SBMs_LiteGT_CLUSTER.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel jaccard --dim_reduce True
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_CLUSTER --log_path './logs/CLUSTER_Adder'   --config 'configs/SBMs_LiteGT_CLUSTER.json' --graph_kernel adder
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_CLUSTER --log_path './logs/CLUSTER_Adder_nodeSampling'   --config 'configs/SBMs_LiteGT_CLUSTER.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel adder
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_CLUSTER --log_path './logs/CLUSTER_Adder_nodeSampling_dimReduing'   --config 'configs/SBMs_LiteGT_CLUSTER.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel adder --dim_reduce True
wait" C-m
tmux send-keys "tmux kill-session -t screen_LiteGT" C-m









