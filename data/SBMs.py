
import time
import os
import pickle
import numpy as np

import dgl
import torch

from scipy import sparse as sp
import numpy as np
import networkx as nx

import hashlib

import pdb
import torch.nn.functional as F
import dgl.function as fn


class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 name,
                 split):

        self.split = split
        self.is_test = split.lower() in ['test', 'val'] 
        with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()
    

    def _prepare(self):

        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        for data in self.dataset:

            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(node_features.size(0))
            g.ndata['feat'] = node_features.long()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())

            # adding edge features for Residual Gated ConvNet
            #edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1 # dim same as node feature dim
            g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            self.graph_lists.append(g)
            self.node_labels.append(data.node_label)


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        # data_dir = 'data/SBMs'
        data_dir = 'data/SBMs/'
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in SBMsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

def compute_common_neighbors(common_neighbors, neighbors):
    def func(edges):
        return {common_neighbors: edges.src[neighbors].int() & edges.dst[neighbors].int()} # 这一步极度消耗内存
    return func

def jaccard_similarity(jaccard, neighbor_num, in_edges):
    def func(edges):
        return {jaccard: edges.data[neighbor_num] / (edges.src[in_edges] + edges.dst[in_edges] - edges.data[neighbor_num])}
    return func

def compute_jaccard_similarity(g):
    g.ndata['id'] = F.one_hot(g.nodes(), g.num_nodes()).float()
    g.ndata['in_degree'] = g.in_degrees().float()
    eids = g.edges()
    g.send_and_recv(eids, fn.copy_src('id', 'id'), fn.sum('id', 'neighbors'))

    g.apply_edges(compute_common_neighbors('common_neighbors', 'neighbors'))
    g.edata['neighbor_num'] = torch.sum(g.edata['common_neighbors'], dim = 1)
    g.apply_edges(jaccard_similarity('jaccard', 'neighbor_num', 'in_degree'))

    g.send_and_recv(eids, fn.copy_edge('jaccard', 'jaccard'), fn.sum('jaccard', 'sum_jaccard'))
    g.ndata['sum_jaccard'] += 1e-6 
    # normalization
    g.apply_edges(fn.e_div_v('jaccard', 'sum_jaccard', 'jaccard')) 
    g.edata['jaccard'] += 1e-6 
    del g.ndata['id'], g.ndata['in_degree'], g.ndata['neighbors'], g.ndata['sum_jaccard'], g.edata['common_neighbors'], g.edata['neighbor_num']
    return g
    
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return g
    
class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        # data_dir = 'data/SBMs/'
        data_dir = 'data/SBMs/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
    
    def _compute_jaccard_similarity(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [compute_jaccard_similarity(g) for g in self.train.graph_lists]
        self.val.graph_lists = [compute_jaccard_similarity(g) for g in self.val.graph_lists]
        self.test.graph_lists = [compute_jaccard_similarity(g) for g in self.test.graph_lists]

    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

