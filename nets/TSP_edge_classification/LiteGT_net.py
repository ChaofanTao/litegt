import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
import random

"""
    LiteGT network for edge classification 
"""
from layers.LiteGT_edge_layer import LiteGTLayer
from layers.mlp_readout_layer import MLPReadout

# initialize edge attention score uniformly
def initialize_score(dst_field, out_field):
    def func(edges):
        return {out_field: 1.0 / edges.dst[dst_field]}
    return func

# copy edge and node information into message box
def copy_info(info1, info2, m1, m2):
    def func(edges):
        return {m1: edges.data[info1], m2: edges.src[info2]}
    return func

# compute jaccard sparsity metric for node sampling
def jaccard_sparse_metric(out_field):
    def func(nodes):
        return {out_field: torch.max(nodes.mailbox['jaccard'], dim = 1)[0] - torch.mean(nodes.mailbox['jaccard'], dim = 1) - torch.log(nodes.data['in_degree_1h'])}
    return func

class LiteGTNet(nn.Module): # did not find the position which utilizes this class
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        n_classes = net_params['n_classes']
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
 
        self.graph_kernel = net_params['graph_kernel']
        self.jaccard_sparse = net_params['jaccard_sparse']
        self.n_classes = n_classes
        self.double_sparse = net_params['double_sparse']
        self.topk_factor = net_params['topk_factor']
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.dim_reduce = net_params['dim_reduce']
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        if self.graph_kernel == 'jaccard':
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim//2)
        else:
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        
        if self.dim_reduce:
            self.layers = nn.ModuleList([LiteGTLayer(hidden_dim, hidden_dim, num_heads, 
                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.double_sparse, self.topk_factor, self.graph_kernel) for layer_idx in range(n_layers-4)])
            self.layers.append(LiteGTLayer(hidden_dim//2, hidden_dim//2, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel, self.dim_reduce, hidden_dim))
            self.layers.append(LiteGTLayer(hidden_dim//2, hidden_dim//2, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel))
            self.layers.append(LiteGTLayer((hidden_dim*3)//10, (out_dim*3)//10, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel, self.dim_reduce, hidden_dim//2))
            self.layers.append(LiteGTLayer((hidden_dim*3)//10, (out_dim*3)//10, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual, self.double_sparse, self.topk_factor, self.graph_kernel))
            self.MLP_layer = MLPReadout(2*(out_dim*3)//10, n_classes)
        else:
            self.layers = nn.ModuleList([ LiteGTLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                        self.layer_norm, self.batch_norm, self.residual, self.double_sparse, self.topk_factor, self.graph_kernel) for layer_idx in range(n_layers-1)]) 
            self.layers.append(LiteGTLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual, self.double_sparse, self.topk_factor, self.graph_kernel))
            self.MLP_layer = MLPReadout(2*out_dim, n_classes)

    def forward(self, g, h, e, h_lap_pos_enc=None):
        # input embedding
        h = self.embedding_h(h.float())
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())

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
                h, e = conv(g, h, e, jaccard_top_indices, sampled_graph, sampled_block)
                del g.ndata['wV'], g.ndata['z']
        else: 
            # convnets
            for conv in self.layers:
                h, e = conv(g, h, e)
                del g.ndata['wV'], g.ndata['z']
        g.ndata['h'] = h
  
        def _feat_concat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            return {'feat_cat': e}

        def _edge_feat(edges):
            e = self.MLP_layer(edges.data['feat_cat'])
            return {'e': e}

        g.apply_edges(_feat_concat)
        g.apply_edges(_edge_feat)
        return g.edata['e']

    def initialize_edge_score(self, g, num_heads):
        g.ndata['in_degree'] = g.in_degrees().expand(num_heads, g.num_nodes()).transpose(1, 0).unsqueeze(-1).float()
        g.apply_edges(initialize_score('in_degree', 'score_D'))
        g.edata['score_D'] = g.edata['score_D'].repeat(1, 1, self.out_dim//self.num_heads) / (self.out_dim//self.num_heads)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)
        return loss
