"""
    IMPORTING LIBS
"""
import dgl
import logging
import utils

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict  
 
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.SBMs_node_classification.load_net import gnn_model 
from data.data import LoadData 




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        logging.info('cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
        device = torch.device("cuda")
    else:
        logging.info('cuda not available')
        device = torch.device("cpu")
    return device




"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    logging.info("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    logging.info('MODEL/Total parameters:{} {}'.format(MODEL_NAME, total_param))
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    start0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = dataset.name

    if params['resume']:
        mode = 'a'
    else:
        mode = 'w'

    # load the processed data, which includes lapician position encoding, jaccard similarity
    if DATASET_NAME == 'SBM_PATTERN':
        data_dir = './data/SBMs/processed_PATTERN_data.pth'
    elif DATASET_NAME == 'SBM_CLUSTER':
        data_dir = './data/SBMs/processed_CLUSTER_data.pth'

    SBMdata = torch.load(data_dir) 
    dataset = SBMdata['dataset']  
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, out_dir = dirs
    device = net_params['device']

    # write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', mode) as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    dgl.random.seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(params['seed'])
        torch.cuda.manual_seed(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info("Training Graphs: {}".format(len(trainset)))
    logging.info("Validation Graphs: {}".format(len(valset)))
    logging.info("Test Graphs: {}".format(len(testset)))
    logging.info("Number of Classes: {}".format(net_params['n_classes']))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
     
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    
    epoch_test_accs = [] # added
    epoch_list = []
     
    # import train and evaluate functions
    from train.train_SBMs_node_classification import train_epoch, evaluate_network 

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, num_workers=0) # should be shuffle, but here for inference time check
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, num_workers=0)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, num_workers=0)

    start_epoch = -1 

    # train from saved checkpoint
    if params['resume']:
        ckpt_dir = os.path.join(root_ckpt_dir, "RUN_/epoch_"+params['epoch_id']+".pth")
        checkpoint = torch.load(ckpt_dir) 
        model.load_state_dict(checkpoint['net']) 
        optimizer.load_state_dict(checkpoint['optimizer']) 
        start_epoch = checkpoint['epoch'] 
        scheduler = checkpoint['scheduler']

        epoch_val_accs = checkpoint['epoch_val_accs']
        epoch_test_accs = checkpoint['epoch_test_accs']
        epoch_train_accs = checkpoint['epoch_train_accs']
        epoch_list = checkpoint['epoch_list']
        per_epoch_time = checkpoint['per_epoch_time']

    # at any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(start_epoch+1, params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)
 
                start = time.time()

                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    
                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)        
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                epoch_test_accs.append(epoch_test_acc)
                epoch_list.append(epoch)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)

                per_epoch_time.append(time.time()-start)

                logging.info("""Epoch: {}, lr: {}, train/_loss: {}, val/_loss: {}, train/_acc: {}, val/_acc: {},
                 test/_acc: {}"""\
                 .format(epoch, optimizer.param_groups[0]['lr'], epoch_train_loss, epoch_val_loss, 
                            epoch_train_acc, epoch_val_acc, epoch_test_acc))
                
                scheduler.step(epoch_val_loss)

                # saving checkpoint
                checkpoint = {
                        "net": model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        "epoch": epoch,
                        "scheduler": scheduler,
                        "epoch_val_accs": epoch_val_accs,
                        "epoch_test_accs": epoch_test_accs,
                        "epoch_train_accs": epoch_train_accs,
                        "epoch_list": epoch_list,
                        "per_epoch_time": per_epoch_time}
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(checkpoint, '{}.pth'.format(ckpt_dir + "/epoch_" + str(epoch)))

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    logging.info("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break
                    
                # stop training after params['max_time'] hours
                if time.time()-start0 > params['max_time']*3600:
                    logging.info('-' * 89)
                    logging.info("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early because of KeyboardInterrupt')
    
    
    _, test_acc = evaluate_network(model, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, device, train_loader, epoch)
    logging.info("Test Accuracy: {:.4f}".format(test_acc))
    logging.info("Train Accuracy: {:.4f}".format(train_acc))
    logging.info("Convergence Time (Epochs): {:.4f}".format(epoch))
    logging.info("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-start0))
    logging.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    best_val_id = epoch_val_accs.index(max(epoch_val_accs))
    logging.info("Best Test Accuracy: {:.4f}".format(epoch_test_accs[best_val_id]))
    logging.info("Best Train Accuracy: {:.4f}".format(epoch_train_accs[best_val_id]))
    logging.info("Best Model (Epochs): {:.4f}".format(epoch_list[best_val_id]))
    logging.info("TIME TAKEN: {:.4f}s".format(sum(per_epoch_time[:best_val_id+1])))
    logging.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time[:best_val_id+1])))

    files = glob.glob(ckpt_dir + '/*.pth')
    for file in files:
        epoch_nb = file.split('_')[-1]
        epoch_nb = int(epoch_nb.split('.')[0])
        if epoch_nb == best_val_id or epoch_nb == epoch:
            pass  
        else:
            os.remove(file)

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    
    with open(write_file_name + '.txt', mode) as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_acc, train_acc, epoch, (time.time()-start0)/3600, np.mean(per_epoch_time)))

        




if __name__ == "__main__":
    """
        USER CONTROLS
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', type=int, default=0, help="Please give a value for gpu id")
    parser.add_argument('--model', type=str, default="LiteGT", help="Please give a value for model name")
    parser.add_argument('--dataset', type=str, default="SBM_CLUSTER", help="Please give a value for dataset name")
    parser.add_argument('--out_dir', type=str, default="out/SBMs/", help="Please give a value for out_dir")
    parser.add_argument('--seed', type=int, default=41, help="Please give a value for seed")
    parser.add_argument('--epochs', type=int, default=1000, help="Please give a value for epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Please give a value for batch_size")
    parser.add_argument('--init_lr', type=float, default=0.0005, help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', type=int, default=10, help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', type=int, default=5, help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', type=int, default=12, help="Please give a value for L")
    parser.add_argument('--hidden_dim', type=int, default=80, help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', type=int, default=80, help="Please give a value for out_dim")
    parser.add_argument('--residual', type=bool, default=True, help="Please give a value for residual")
    parser.add_argument('--edge_feat', type=bool, default=True, help="Please give a value for edge_feat")
    parser.add_argument('--readout', type=str, default="mean", help="Please give a value for readout")
    parser.add_argument('--n_heads', type=int, default=8, help="Please give a value for n_heads")
    parser.add_argument('--in_feat_dropout', type=float, default=0.0, help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', type=float, default=0.0, help="Please give a value for dropout")
    parser.add_argument('--layer_norm', type=bool, default=False, help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', type=bool, default=True, help="Please give a value for batch_norm")
    parser.add_argument('--self_loop', type=bool, default=False, help="Please give a value for self_loop")
    parser.add_argument('--max_time', type=int, default=24, help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', type=int, default=10, help="Please give a value for pos_enc_dim")
    parser.add_argument('--lap_pos_enc', type=bool, default=True, help="Please give a value for lap_pos_enc")
    parser.add_argument('--log_path', type=str, default="./logs/result_log", help="Please give a value for log_path")
    parser.add_argument('--resume', type=bool, default=False, help="Please give a value for resume")
    parser.add_argument('--resume_time', type=str, default="0", help="Please give a value for resume_time")
    parser.add_argument('--epoch_id', type=int, default=0, help="Please give a value for epoch_id")
    parser.add_argument('--double_sparse', type=bool, default=False, help="Please give a value for double_sparse")
    parser.add_argument('--topk_factor', type=int, default=5, help="Please give a value for topk_factor")
    parser.add_argument('--graph_kernel', type=str, default="jaccard", help="Please give a value for graph_kernel")
    parser.add_argument('--jaccard_sparse', type=bool, default=False, help="Please give a value for jaccard_sparse")
    parser.add_argument('--dim_reduce', type=bool, default=False, help="Please give a value for dim_reduce")
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    if args.resume is not None:
        mode = 'a'
    else:
        mode = 'w'
    log_dir_id = args.log_path.rfind("/")
    log_dir = args.log_path[:log_dir_id]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(args.log_path+ '.log') 
    utils.setup_logging(log_file_path, mode)
    logging.info('save the log to {}'.format(log_file_path))  
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    net_params['out_dir'] = out_dir
    
    args_dict  = vars(args)
    params.update((k,v) for k,v in args_dict.items() if k in params and v is not None)
    net_params.update((k,v) for k,v in args_dict.items() if k in net_params and v is not None)
    params = edict(params)
    net_params = edict(net_params)
    if net_params['jaccard_sparse']:
        net_params['double_sparse'] = True


    
      
    # SBM
    net_params['in_dim'] = torch.unique(dataset.train[0][0].ndata['feat'],dim=0).size(0) # node_dim (feat is an integer)
    net_params['n_classes'] = torch.unique(dataset.train[0][1],dim=0).size(0)
    

    if params['resume']:
        root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME +  "_" + params['resume_time']
        root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_" + params['resume_time']
        write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_" + params['resume_time']
        write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME +  "_" + params['resume_time']
        dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, out_dir
    else:
        root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME +  "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME +  "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, out_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

    
    
    
    
    
    


























