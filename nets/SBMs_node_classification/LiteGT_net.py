import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
import dgl.function as fn
import random

"""
    LiteGT network for node classification
"""
from layers.LiteGT_layer import LiteGTLayer
from layers.mlp_readout_layer import MLPReadout

# initialize edge attention score uniformly
def initialize_score(dst_field, out_field):
    def func(edges):
        return {out_field: 1.0 / edges.dst[dst_field]} # elment-wise product
    return func

# compute jaccard sparsity metric for node sampling
def jaccard_sparse_metric(out_field):
    def func(nodes):
        return {out_field: torch.max(nodes.mailbox['jaccard'], dim = 1)[0] - torch.mean(nodes.mailbox['jaccard'], dim = 1) - torch.log(nodes.data['in_degree_1h']) }
    return func

class LiteGTNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.num_heads = net_params['n_heads']
        self.graph_kernel = net_params['graph_kernel']
        self.jaccard_sparse = net_params['jaccard_sparse']
        self.dim_reduce = net_params['dim_reduce']
 

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.double_sparse = net_params['double_sparse']
        self.topk_factor = net_params['topk_factor']
        
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # do dimension reducing on the last four layers
        if self.dim_reduce: 
            self.layers = nn.ModuleList([LiteGTLayer(hidden_dim, hidden_dim, num_heads, 
                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.double_sparse, self.topk_factor, self.graph_kernel) for layer_idx in range(n_layers-4)])
            self.layers.append(LiteGTLayer(hidden_dim//2, hidden_dim//2, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel, self.dim_reduce, hidden_dim))
            self.layers.append(LiteGTLayer(hidden_dim//2, hidden_dim//2, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel))
            self.layers.append(LiteGTLayer((hidden_dim*3)//10, (out_dim*3)//10, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel, self.dim_reduce, hidden_dim//2))
            self.layers.append(LiteGTLayer((hidden_dim*3)//10, (out_dim*3)//10, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel))
            self.MLP_layer = MLPReadout((out_dim*3)//10, n_classes)
        else:   
            self.layers = nn.ModuleList([LiteGTLayer(hidden_dim, hidden_dim, num_heads, 
                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.double_sparse, self.topk_factor, self.graph_kernel) for layer_idx in range(n_layers-1)])
            self.layers.append(LiteGTLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.double_sparse, self.topk_factor, self.graph_kernel))
            self.MLP_layer = MLPReadout(out_dim, n_classes)
        if self.graph_kernel == 'jaccard':
            self.score_linear_trans = nn.Linear(num_heads//2, num_heads//2, bias=False)
        else:
            self.score_linear_trans = nn.Linear(num_heads, num_heads, bias=False)
 
    def forward(self, g, h, e, h_lap_pos_enc=None):
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        h = self.in_feat_dropout(h)

        if self.graph_kernel == 'jaccard':
            self.initialize_edge_score(g, self.num_heads//2)
        else:
            self.initialize_edge_score(g, self.num_heads)
 
        if self.jaccard_sparse:
            eids = g.edges()
            g.ndata['in_degree_1h'] = g.in_degrees().float()
            g.send_and_recv(eids, fn.copy_edge('jaccard','jaccard'), jaccard_sparse_metric('jaccard_M'))
            
            # select topk nodes
            k = np.ceil(8 * self.topk_factor * g.num_nodes() / 3000).astype('int').item()
            _, jaccard_top_indices = torch.topk(g.ndata['jaccard_M'].squeeze(), k, dim=0, sorted=False)
            sampled_graph = dgl.in_subgraph(g, jaccard_top_indices)
            sampled_block = dgl.to_block(sampled_graph, jaccard_top_indices)
            i = 0
            for conv in self.layers:
                h = conv(g, h, jaccard_top_indices, sampled_graph, sampled_block)
                del g.ndata['wV'], g.ndata['z']
        else:
            for conv in self.layers:
                h = conv(g, h)
                del g.ndata['wV'], g.ndata['z']
        h_out = self.MLP_layer(h)
        return h_out

    def initialize_edge_score(self, g, num_heads):
        g.ndata['in_degree'] = g.in_degrees().expand(num_heads, g.num_nodes()).transpose(1, 0).unsqueeze(-1).float()
        g.apply_edges(initialize_score('in_degree', 'score'))

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)
        return loss



        
