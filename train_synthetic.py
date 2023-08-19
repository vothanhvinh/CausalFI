#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np
import torch
import torch.nn as nn
import scipy
import random
from scipy.stats import sem
import torchbnn as bnn
from torchbnn.utils import freeze, unfreeze
from model import *
from datasets import SynData50Sources
from evaluation import *

parser = argparse.ArgumentParser()
parser.add_argument('n_sources', type=int)
parser.add_argument('replicate', type=int)
args = parser.parse_args()

source_id_to_run = args.replicate
num_source_to_run = args.n_sources

print('n_sources {}, replicate {}'.format(args.n_sources, args.replicate))


device_id = 0
print('PyTorch version', torch.__version__)
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  torch.cuda.set_device(device_id)
  print('Use ***GPU***')
  print(torch.cuda.get_device_properties(device_id).total_memory/1024/1024/1024,'GB')
else:
  print('Use CPU')
device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")


RND_SEED = 2023
random.seed(RND_SEED)
np.random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
torch.cuda.manual_seed_all(RND_SEED)
torch.backends.cudnn.deterministic=True

# Configuration
training_iter_z = 10000
training_iter_zhat = 10000
training_iter_y = 10000
learning_rate = 1e-3
display_per_iters=100
hidden_size = 10
output_dir = 'save_outputs'

# Load data
dataset = SynData50Sources()
source_size = dataset.source_size
train_size = dataset.train_size
test_size = dataset.test_size
val_size = dataset.val_size
M = dataset.n_sources

test_stats_lst = []
for m in [num_source_to_run]:
  loss_lst = []
  test_stats = []
  for i, (data_train, data_test, data_val) in enumerate(dataset.get_train_test_val(m_sources=m)):
    if i!=source_id_to_run-1:
      continue

    source_ranges_train = [(idx, idx+train_size) for idx in range(0,m*train_size,train_size)]
    source_ranges_test = [(idx, idx+test_size) for idx in range(0,M*test_size,test_size)]
    source_ranges_val = [(idx, idx+val_size) for idx in range(0,m*val_size,val_size)]
    print(source_ranges_train)
    print(source_ranges_test)
    print('======================================================================================')
    print('# Source {}, Replicate: {}'.format(m, i+1))
    print('======================================================================================')

    # Training data
    Wtr, Ytr, Y_cftr, mutr, Xtr = data_train[0][1].reshape(-1)[:m*train_size],\
                              data_train[0][2].reshape(-1)[:m*train_size],\
                              data_train[1][0].reshape(-1)[:m*train_size],\
                              np.concatenate((data_train[1][1],data_train[1][2]),axis=1)[:m*train_size],\
                              data_train[0][0][:m*train_size]
    Ttr = len(Ytr)
    xtr = torch.from_numpy(Xtr.reshape(Ttr,-1)).float().to(device)
    ytr = torch.from_numpy(Ytr.reshape(-1,1)).float().to(device)
    wtr = torch.from_numpy(Wtr.reshape(-1,1)).float().to(device)

    # Testing data
    Wte, Yte, Y_cfte, mute, Xte, Xte_orgi = data_test[0][1].reshape(-1)[:M*test_size],\
                              data_test[0][2].reshape(-1)[:M*test_size],\
                              data_test[1][0].reshape(-1)[:M*test_size],\
                              np.concatenate((data_test[1][1],data_test[1][2]),axis=1)[:M*test_size],\
                              data_test[0][0][:M*test_size],\
                              data_test[1][3][:M*test_size]

    Tte = len(Yte)
    xte = torch.from_numpy(Xte.reshape(Tte,-1)).float().to(device)
    yte = torch.from_numpy(Yte.reshape(-1,1)).float().to(device)
    wte = torch.from_numpy(Wte.reshape(-1,1)).float().to(device)


    # Train
    print('*** P(Z|X,Y,W)')
    model_server_z, model_sources_z = trainZ_FedGrads(train_x=xtr[:,10:],
                                                      train_w=wtr.reshape(-1),
                                                      train_y=ytr.reshape(-1),
                                                      train_z=xtr[:,:10],
                                                      n_sources=m, source_ranges=source_ranges_train,
                                                      hidden_size=hidden_size,
                                                      training_iter=training_iter_z, learning_rate=learning_rate,
                                                      display_per_iters=display_per_iters)
    
#     print('*** P(Zr~|X,Zr)')
#     model_server_zhat, model_sources_zhat = trainZhat_FedGrads(train_x=xtr[:,10:],
#                                                                 train_y=ytr,
#                                                                 train_w=wtr,
#                                                                 model_z=model_sources_z,
#                                                                 dim_z=xtr[:,:10].shape[1],
#                                                                 n_sources=m, source_ranges=source_ranges_train,
#                                                                 training_iter=training_iter_zhat, learning_rate=learning_rate,
#                                                                 display_per_iters=display_per_iters)

    print('*** P(Y|X,Z,W), P(Zr~|X,Zr)')
    model_server_zhaty, model_sources_zhaty = trainY_FedGrads(train_x=xtr[:,10:],
                                                      train_w=wtr.reshape(-1),
                                                      train_y=ytr.reshape(-1),
                                                      model_z=model_sources_z,
                                                      dim_z=xtr[:,:10].shape[1],
                                                      n_sources=m, source_ranges=source_ranges_train,
                                                      hidden_size=hidden_size,
                                                      training_iter=training_iter_y, learning_rate=learning_rate,
                                                      display_per_iters=display_per_iters)
    model_server_zhat = model_server_zhaty.model_zhat
    model_sources_zhat = [model.model_zhat for model in model_sources_zhaty]
    model_server_y = model_server_zhaty.model_y
    model_sources_y = [model.model_y for model in model_sources_zhaty]
    # model_server_y, model_sources_y = trainY_FedParams(train_x=xtr[:,10:],
    #                                                   train_w=wtr.reshape(-1),
    #                                                   train_y=ytr.reshape(-1),
    #                                                   model_z=model_sources_z,
    #                                                   dim_z=10,
    #                                                   n_sources=m, source_ranges=source_ranges_train,
    #                                                   training_iter=100, num_agg=200,
    #                                                   learning_rate=learning_rate,
    #                                                   display_per_iters=display_per_iters)

    
    # Test
    y0pred, y1pred = pred_y0y1(model_server_zhat=model_server_zhat, model_server_y=model_server_y,
                              test_x=xte[:,10:], test_z=xte[:,:10],
                              test_w=wte, test_y=yte, n_sources=m, 
                              source_ranges_test=source_ranges_test, idx_sources_to_test=list(range(M)))

    eval = Evaluation(mute[:,0], mute[:,1])
    y0pred = y0pred.detach().cpu().numpy()
    y1pred = y1pred.detach().cpu().numpy()
    test_stats.append((eval.absolute_err_ate(y0pred,y1pred), eval.pehe(y0pred, y1pred)))

    np.savez('{}/synthetic_test_stats_m{}_replicate{}.npz'.format(output_dir, m,i+1), test_stats=np.asarray(test_stats))
  test_stats = np.asarray(test_stats)

