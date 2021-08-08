import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
import os

 
"""
    LiteGT Layer with edge features
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func

def exp(field1, field2):
    def func(edges):
        # clamp for softmax numerical stability
        return {field2: torch.exp((edges.data[field1].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

def normalize_new_score(edge_attri, node_attri):
    def func(edges):
        return {edge_attri: edges.data[edge_attri] / edges.dst[node_attri]}

    return func

def src_add_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] - edges.dst[dst_field]).abs().sum(-1, keepdim=True)}
    return func

def src_add_dst_dimD(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] - edges.dst[dst_field]).abs()}
    return func

def initialize_score(dst_field, out_field):
    def func(edges):
        return {out_field: 1.0 / edges.dst[dst_field]} 
    return func

"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, double_sparse, topk_factor, graph_kernel, use_bias):
        super().__init__()
        self.topk_factor = topk_factor
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.double_sparse = double_sparse
        self.graph_kernel = graph_kernel
        
        if self.graph_kernel == 'jaccard':
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * (num_heads//2), bias=True)
                self.K = nn.Linear(in_dim, out_dim * (num_heads//2), bias=True)
                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.proj_e = nn.Linear(in_dim//2, out_dim * (num_heads//2), bias=True)           
            else:
                self.Q = nn.Linear(in_dim, out_dim * (num_heads//2), bias=False)
                self.K = nn.Linear(in_dim, out_dim * (num_heads//2), bias=False)
                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.proj_e = nn.Linear(in_dim//2, out_dim * (num_heads//2), bias=False)
        else:
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)  
                self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)         
            else:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.Q.weight, gain=gain)
        nn.init.xavier_normal_(self.K.weight, gain=gain)
        nn.init.xavier_normal_(self.V.weight, gain=gain)

    # no node sampling on the first branch
    def propagate_with_full_attention(self):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        # softmax
        g.apply_edges(exp('score', 'score'))
        # Send weighted values to target nodes
        eids = g.edges()
        # the second branch with jaccard kernel
        if self.graph_kernel == 'jaccard':
            g.edata['jaccard_expand'] = g.edata['jaccard'].expand(self.num_heads//2, g.num_edges()).transpose(1, 0).unsqueeze(-1)
            g.edata['score_new'] = torch.cat((g.edata['score'], g.edata['jaccard_expand']), dim = 1)
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_new', 'V_h'), fn.sum('V_h', 'wV')) 
            g.send_and_recv(eids, fn.copy_edge('score_new', 'score_new'), fn.sum('score_new', 'z'))
            del g.edata['jaccard_expand']
        # the second branch with adder kernel
        elif self.graph_kernel == 'adder':
            g.apply_edges((src_add_dst_dimD('K_h', 'Q_h', 'score_tmp'))) 
            g.apply_edges(scaling('score_tmp', np.sqrt(self.out_dim)))
            g.apply_edges(imp_exp_attn('score_tmp', 'proj_e'))
            g.edata['e_out'][:, self.num_heads//2:, :] = g.edata['score_tmp'][:, self.num_heads//2:, :]
            g.edata['score_tmp'] = g.edata['score_tmp'].abs().sum(-1, keepdim = True)
            g.edata['score_new'] = torch.cat((g.edata['score'][:, :self.num_heads//2, :], g.edata['score_tmp'][:, self.num_heads//2:, :]), dim = 1)
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_new', 'V_h'), fn.sum('V_h', 'wV')) 
            g.send_and_recv(eids, fn.copy_edge('score_new', 'score_new'), fn.sum('score_new', 'z'))
            del g.edata['score_tmp']
        # the second branch with the same reg.-softmax kernel as the first branch
        else:
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    # no attention is computed on the graph
    def propagate_without_attention(self, g):
        g.edata['score'] = g.edata['score_D'].clone()
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        # softmax
        g.apply_edges(exp('score', 'score'))
        eids = g.edges()
        # the second branch with jaccard kernel
        if self.graph_kernel == 'jaccard':
            g.edata['jaccard_expand'] = g.edata['jaccard'].expand(self.num_heads//2, g.num_edges()).transpose(1, 0).unsqueeze(-1)
            g.edata['score_new'] = torch.cat((g.edata['score'][:, :self.num_heads//2, :], g.edata['jaccard_expand']), dim = 1)
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_new', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score_new', 'score_new'), fn.sum('score_new', 'z'))
            del g.edata['jaccard_expand']
        # the second branch with simply average
        else: 
            g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def block_graph_computation_jaccardtopk(self, g, top_indices, sampled_graph, sampled_block): 
        if sampled_block is None:
            sampled_graph = dgl.in_subgraph(g, top_indices)
            sampled_block = dgl.to_block(sampled_graph, top_indices)
        sg_eids = sampled_block.edges()
        sampled_block.apply_edges(src_dot_dst('K_h', 'Q_h', 'new_score_D'))
        sampled_block.apply_edges(scaling('new_score_D', np.sqrt(self.out_dim)))
        sampled_block.apply_edges(imp_exp_attn('new_score_D', 'proj_e'))
        g.edata['e_out'][sampled_graph.edata[dgl.EID][sampled_block.edata[dgl.EID]], :, :] = sampled_block.edata['new_score_D']
        sampled_block.apply_edges(exp('new_score_D', 'new_score'))
        sampled_block.send_and_recv(sg_eids, fn.copy_edge('new_score', 'new_score'), fn.sum('new_score', 'new_z'))
        sampled_block.apply_edges(normalize_new_score('new_score', 'new_z'))
        g.edata['score_1'][sampled_graph.edata[dgl.EID][sampled_block.edata[dgl.EID]], :, :] = sampled_block.edata['new_score']
    
    def propagate_graph_sparse_attention(self, g, k, jaccard_topk = None, sampled_graph = None, sampled_block = None):
        eids = g.edges()
        # without attention computation on graph
        if k == 0:
            self.propagate_without_attention(g)
        # without node sampling on the first franch
        elif k == g.num_nodes():
            self.propagate_with_full_attention(g)
        # with node sampling on the first franch
        else:
            g.edata['e_out'] = g.edata['proj_e'].clone()
            g.apply_edges(initialize_score('in_degree', 'score_1'))
            # compute the information aggregation on the sampled sub-graph only 
            self.block_graph_computation_jaccardtopk(g, jaccard_topk, sampled_graph, sampled_block)
            # the second branch with jaccard kernel
            if self.graph_kernel == 'jaccard':
                g.edata['jaccard_expand'] = g.edata['jaccard'].expand(self.num_heads//2, g.num_edges()).transpose(1, 0).unsqueeze(-1)
                g.edata['score_1'] = torch.cat((g.edata['score_1'], g.edata['jaccard_expand']), dim = 1)
                g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_1', 'V_h'), fn.sum('V_h', 'wV'))
                g.send_and_recv(eids, fn.copy_edge('score_1', 'score_1'), fn.sum('score_1', 'z'))
            # the second branch with adder kernel
            elif self.graph_kernel == 'adder':
                g.apply_edges((src_add_dst_dimD('K_h', 'Q_h', 'score_tmp'))) 
                g.apply_edges(scaling('score_tmp', np.sqrt(self.out_dim)))
                g.apply_edges(imp_exp_attn('score_tmp', 'proj_e'))
                g.edata['e_out'][:, self.num_heads//2:, :] = g.edata['score_tmp'][:, self.num_heads//2:, :]
                g.edata['score_tmp'] = g.edata['score_tmp'].abs().sum(-1, keepdim = True)
                g.edata['score_new'] = torch.cat((g.edata['score_1'][:, :self.num_heads//2, :], g.edata['score_tmp'][:, self.num_heads//2:, :]), dim = 1)
                g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_new', 'V_h'), fn.sum('V_h', 'wV'))
                g.send_and_recv(eids, fn.copy_edge('score_new', 'score_new'), fn.sum('score_new', 'z')) 
                del g.edata['score_tmp']
            # the second branch with the same reg.-softmax kernel as the first branch
            else: 
                g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_1', 'V_h_tmp'), fn.sum('V_h_tmp', 'wV'))
                g.send_and_recv(eids, fn.copy_edge('score_1', 'score_1'), fn.sum('score_1', 'z'))

    def forward(self, g, h, e, jaccard_topk = None, sampled_graph = None, sampled_block = None):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        g.ndata['Q_h'] = Q_h.view(g.num_nodes(), -1, self.out_dim)
        g.ndata['K_h'] = K_h.view(g.num_nodes(), -1, self.out_dim)
        g.ndata['V_h'] = V_h.view(g.num_nodes(), -1, self.out_dim)
        g.edata['proj_e'] = proj_e.view(g.num_edges(), -1,  self.out_dim)

        if sampled_block is not None:
            sampled_block.srcdata['Q_h'] = Q_h[sampled_block.srcdata['_ID']].view(sampled_block.num_src_nodes(), -1, self.out_dim)
            sampled_block.dstdata['Q_h'] = Q_h[sampled_block.dstdata['_ID']].view(sampled_block.num_dst_nodes(), -1, self.out_dim)
            sampled_block.srcdata['K_h'] = K_h[sampled_block.srcdata['_ID']].view(sampled_block.num_src_nodes(), -1, self.out_dim)
            sampled_block.dstdata['K_h'] = K_h[sampled_block.dstdata['_ID']].view(sampled_block.num_dst_nodes(), -1, self.out_dim)
            sampled_block.edata['proj_e'] = proj_e[sampled_graph.edata[dgl.EID][sampled_block.edata[dgl.EID]]].view(sampled_block.num_edges(), -1, self.out_dim)
        
        if self.double_sparse:
            k = np.ceil(8 * self.topk_factor * g.num_nodes() / 3000).astype('int').item()
        else: 
            k = g.num_nodes()
        self.propagate_graph_sparse_attention(g, k, jaccard_topk, sampled_graph, sampled_block)
    
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        e_out = g.edata['e_out']

        return h_out, e_out
    
 
class LiteGTLayerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, layer_norm=False, batch_norm=True, residual=True, double_sparse=1, topk_factor=5, graph_kernel = '0', dim_reduce = False, ori_hidden_dim = None, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        self.graph_kernel = graph_kernel
        self.dim_reduce = dim_reduce
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, double_sparse, topk_factor, graph_kernel, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        if self.graph_kernel == 'jaccard':
            self.O_e = nn.Linear(out_dim//2, out_dim//2)
        else:
            self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            if self.graph_kernel == 'jaccard':
                self.layer_norm1_e = nn.LayerNorm(out_dim//2)
            else:
                self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            if self.graph_kernel == 'jaccard':
                self.batch_norm1_e = nn.BatchNorm1d(out_dim//2)
            else:
                self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        if self.graph_kernel == 'jaccard':
            self.FFN_e_layer1 = nn.Linear(out_dim//2, out_dim*2)
            self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim//2)
        else:
            self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
            self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            if self.graph_kernel == 'jaccard':
                self.layer_norm2_e = nn.LayerNorm(out_dim//2)
            else:
                self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            if self.graph_kernel == 'jaccard':
                self.batch_norm2_e = nn.BatchNorm1d(out_dim//2)
            else:
                self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        self.reset_parameters()
        if self.dim_reduce:
            self.hidden_pool_h = nn.Linear(ori_hidden_dim, in_dim)
            if self.graph_kernel == 'jaccard':
                self.hidden_pool_e = nn.Linear(ori_hidden_dim//2, in_dim//2)
            else:
                self.hidden_pool_e = nn.Linear(ori_hidden_dim, in_dim)
       

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.O_h.weight, gain=gain)
        nn.init.xavier_normal_(self.O_e.weight, gain=gain)
        nn.init.xavier_normal_(self.FFN_h_layer1.weight, gain=gain)
        nn.init.xavier_normal_(self.FFN_h_layer2.weight, gain=gain)
        nn.init.xavier_normal_(self.FFN_e_layer1.weight, gain=gain)
        nn.init.xavier_normal_(self.FFN_e_layer2.weight, gain=gain)
         
    def forward(self, g, h, e, jaccard_topk = None, sampled_graph = None, sampled_block = None):
        if self.dim_reduce:
            h = self.hidden_pool_h(h)
            e = self.hidden_pool_e(e)
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e, jaccard_topk, sampled_graph, sampled_block)
        
        h_t = h_attn_out.view(-1, self.out_channels)
        if self.graph_kernel == 'jaccard':
            e = e_attn_out.view(-1, self.out_channels//2)
        else:
            e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h_t, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)
#
        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)