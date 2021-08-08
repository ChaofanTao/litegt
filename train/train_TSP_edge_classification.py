"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
from tqdm import tqdm
from train.metrics import binary_f1_score
 
def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    gpu_mem = 0
    pbar = tqdm(enumerate(data_loader))
    for iter, (batch_graphs, batch_labels) in pbar:
        pbar.set_description("batch id", iter)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
        
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        f1 = binary_f1_score(batch_scores, batch_labels)
        epoch_train_f1 += f1
        nb_data += batch_labels.size(0)
        pbar.set_postfix({'loss':loss.item(),'f1':f1})
    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, optimizer
 
def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        
    return epoch_test_loss, epoch_test_f1

