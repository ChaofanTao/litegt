"""
    load LiteGT network for edge classification 
"""
from nets.TSP_edge_classification.LiteGT_net import LiteGTNet

def LiteGT(net_params):
    return LiteGTNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'LiteGT': LiteGT
    }
        
    return models[MODEL_NAME](net_params)