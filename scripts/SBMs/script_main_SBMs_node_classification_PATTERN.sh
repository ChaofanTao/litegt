#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# SBM_PATTERN
############


python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2  --log_path './logs/PATTERN_Jaccard'   --config 'configs/SBMs_LiteGT_PATTERN.json' --graph_kernel jaccard

python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Jaccard_nodeSampling'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel jaccard

python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Jaccard_nodeSampling_dimReduing'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel jaccard --dim_reduce True

python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Adder'   --config 'configs/SBMs_LiteGT_PATTERN.json' --graph_kernel adder

python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Adder_nodeSampling'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel adder

python main_SBMs_node_classification.py --dataset SBM_PATTERN --L 8 --pos_enc_dim 2 --log_path './logs/PATTERN_Adder_nodeSampling_dimReduing'   --config 'configs/SBMs_LiteGT_PATTERN.json'  --topk_factor 5 --jaccard_sparse True --graph_kernel adder --dim_reduce True









