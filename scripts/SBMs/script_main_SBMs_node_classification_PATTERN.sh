#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# SBM_PATTERN
############

tmux new -s screen_LiteGT -d
tmux send-keys "source activate LiteGT" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2  --log_path './logs/PATTERN_Jaccard'   --config 'configs/SBMs_LiteGT_PATTERN.json' --graph_kernel jaccard
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Jaccard_nodeSampling'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel jaccard
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Jaccard_nodeSampling_dimReduing'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel jaccard --dim_reduce True
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Adder'   --config 'configs/SBMs_LiteGT_PATTERN.json' --graph_kernel adder
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Adder_nodeSampling'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel adder
wait" C-m
tmux send-keys "
python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Adder_nodeSampling_dimReduing'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel adder --dim_reduce True
wait" C-m
tmux send-keys "tmux kill-session -t screen_LiteGT" C-m









