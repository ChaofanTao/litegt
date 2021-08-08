"""
    load LiteGT network for node classification
"""

from nets.SBMs_node_classification.LiteGT_net import LiteGTNet

def LiteGT(net_params):
    return LiteGTNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'LiteGT': LiteGT
    }
        
    return models[MODEL_NAME](net_params)