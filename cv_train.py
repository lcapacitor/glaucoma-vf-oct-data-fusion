#!/usr/bin/env python3
#---------------------------------------------------------------------------------------------
# Description: Train the AE data fusion model with 10 fold cross-validation.
# Author     : Leo Yan Li-Han
# Date       : January, 2025
# version    : 2.0
# License    : MIT License
# Usage     :
#       (1) CV train models with mock data using default parameters:   
#               python cv_train.py
#       (2) CV train models with mock data using specified parameters: 
#               python cv_train.py --argument_1 value_1 --argument_2 value_2
#---------------------------------------------------------------------------------------------
import os
import copy
import joblib
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import *
from model import MLP_Encoder, MLP_Decoder, MLP_AE_model

#==========================================================
# Set random seed
#==========================================================
print('Current device:', TORCH_DEVICE)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
#==========================================================
# Dataset
#==========================================================
class VF_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        data_arr= self.data[index]
        data_ts = torch.from_numpy(data_arr).type(torch.FloatTensor)
        return data_ts

#==========================================================
# Loss functions
#==========================================================
def reconstruction_loss(pred, tar, loss_type, in_type, is_norm):
    if in_type.lower()=='vf+rnfl+age':
        pred_vf, pred_rnfl, pred_age= pred[:, :52], pred[:, 52:-1], pred[:, -1]
        tar_vf, tar_rnfl  , tar_age = tar[:, :52], tar[:, 52:-1], tar[:, -1]
    elif in_type.lower()=='vf+rnfl':
        pred_vf, pred_rnfl, pred_age= pred[:, :52], pred[:, 52:], tar[:, :52]      # pred_age and tar_age are placeholders
        tar_vf, tar_rnfl, tar_age   = tar[:, :52], tar[:, 52:], tar[:, :52]
    elif in_type.lower()=='rnfl+age':
        pred_vf, pred_rnfl, pred_age= tar[:, :256], pred[:, :256], pred[:, -1]     # pred_vf is a placeholder
        tar_vf , tar_rnfl, tar_age  = tar[:, :256], tar[:, :256], tar[:, -1]
    elif in_type.lower()=='rnfl':
        pred_vf, pred_rnfl, pred_age= tar, pred, tar # pred_vf, pred_age, tar_vf, and tar_age are a placeholders
        tar_vf , tar_rnfl , tar_age = tar, tar, tar
    elif in_type.lower()=='vf+age': 
        pred_vf, pred_rnfl, pred_age= pred[:, :52], tar[:, :52], pred[:, -1]   # pred_rnfl is a placeholder
        tar_vf , tar_rnfl, tar_age  = tar[:, :52], tar[:, :52],  tar[:, -1]
    else:
        raise ValueError("Unsupported in_type for reconstruction_loss:", in_type)
    if loss_type.lower()=='mse':
        loss_func = nn.MSELoss()
    elif loss_type.lower()=='mae':
        loss_func = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. 'MSE' or 'MAE' only. ")
    if is_norm:
        if 'vf' in in_type.lower():
            pred_vf = de_normalize(pred_vf, V_MIN, V_MAX)
            tar_vf  = de_normalize(tar_vf, V_MIN, V_MAX)
        if 'age' in in_type.lower():
            pred_age  = de_normalize(pred_age, A_MIN, A_MAX)
            tar_age   = de_normalize(tar_age, A_MIN, A_MAX)  
        if 'rnfl' in in_type.lower():
            pred_rnfl = de_normalize(pred_rnfl, R_MIN, R_MAX)
            tar_rnfl  = de_normalize(tar_rnfl, R_MIN, R_MAX)
    vf_loss  = loss_func(pred_vf, tar_vf) 
    rnfl_loss= loss_func(pred_rnfl, tar_rnfl)
    age_loss = loss_func(pred_age, tar_age)
    rec_loss = torch.mean(vf_loss + rnfl_loss + age_loss)
    return rec_loss

def encoding_loss(pred, tar, loss_type, d_type, is_norm):
    if loss_type.lower()=='mse':
        loss_func = nn.MSELoss()
    elif loss_type.lower()=='mae':
        loss_func = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. 'MSE' or 'MAE' only. ")
    if is_norm:
        if d_type.lower()=='rnfl':
            d_min, d_max = R_MIN, R_MAX
        elif d_type.lower()=='vf':
            d_min, d_max = V_MIN, V_MAX
        else:
            raise ValueError("Unsupported d_type:", d_type)
        pred = de_normalize(pred, d_min, d_max)
        tar  = de_normalize(tar, d_min, d_max)
    return loss_func(pred, tar)

#==========================================================
# Training functions
#==========================================================
def train(train_data, valid_data, params):
    #-------------------------------------------
    # Parameters
    #-------------------------------------------
    cur_fold  = params['cur_fold']
    is_norm   = params['is_norm']
    num_epochs= params['num_epochs']
    batch_size= params['batch_size']
    hid_dim   = params['hidden_dim']
    init_lr   = params['init_lr']
    wt_decay  = params['wt_decay']
    loss_type = params['loss_type']
    in_type   = params['in_type']
    lambda_z  = params['lambda_z']
    disp_gap  = params['disp_gap']
    out_path  = params['out_path']
    enc_n_std = params['enc_noise_std']
    #-------------------------------------------
    if disp_gap>0:
        print('=' * 60)
        print(f"CV-fold-{cur_fold}: train_data: {train_data.shape}, valid_data: {valid_data.shape}")
        print('=' * 60)
    #-------------------------------------------
    if in_type.lower()=='vf+rnfl+age':
        in_dim, out_dim = 52+256+1, 52+256+1
    elif in_type.lower()=='vf+rnfl':
        in_dim, out_dim = 52+256, 52+256
    elif in_type.lower()=='rnfl+age':
        in_dim, out_dim = 256+1, 256+1
    elif in_type.lower()=='rnfl':
        in_dim, out_dim = 256, 256
    elif in_type.lower()=='vf+age':
        in_dim, out_dim = 52+1, 52+1
    else:
        raise ValueError('Unsupported in_type:', in_type)
    z_dim = 52
    print(f'IN dim: {in_dim}, OUT dim:{out_dim}, Encoding dim: {z_dim}')
    #-------------------------------------------
    # Define dataloaders, model, and optimizers
    #-------------------------------------------
    vf_dataset  = {'train': VF_dataset(train_data), 'valid' : VF_dataset(valid_data)}
    dataloaders = {'train': DataLoader(vf_dataset['train'], batch_size=batch_size, shuffle=True,  num_workers=0),
                   'valid': DataLoader(vf_dataset['valid'], batch_size=batch_size, shuffle=False, num_workers=0)}
    encoder = MLP_Encoder(in_dim, hid_dim, z_dim, is_norm).to(TORCH_DEVICE)
    decoder = MLP_Decoder(z_dim, hid_dim, out_dim, is_norm).to(TORCH_DEVICE)
    ae_model= MLP_AE_model(encoder, decoder, enc_n_std).to(TORCH_DEVICE)
    #-------------------------------------------
    b1, b2 = 0.5, 0.999
    optimizer = optim.Adam(ae_model.parameters(), lr=init_lr, betas=(b1, b2), weight_decay=wt_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=20)
    #-------------------------------------------
    # Training
    #-------------------------------------------
    trained_models = None
    train_records  = {phase:[] for phase in ['train', 'valid']}
    epoch_iterator = tqdm(range(num_epochs), leave=False) if disp_gap==0 else range(num_epochs)
    best_loss, counter, early_stop = np.inf, 0, False
    #-------------------------------------------
    for epoch in epoch_iterator:
        if optimizer.param_groups[0]['lr'] < 1e-6:
            counter += 1
        if counter > 5:
            early_stop = True
        if disp_gap>0 and (epoch%disp_gap==0 or early_stop):
            print(f'\nEpoch {epoch}/{num_epochs}, lr: {optimizer.param_groups[0]["lr"]}, weight_decay:{wt_decay}') 
            print('-' * 40)
        #-------------------------------------------
        for phase in ['train', 'valid']:
            running_loss, tmp_rec_loss, tmp_enc_loss = 0., 0., 0.
            for b, data in enumerate(dataloaders[phase]):
                if phase=='valid':
                    ae_model.eval()
                else:
                    ae_model.train()
                #-------------------------------------------
                # Process data
                data    = data.to(TORCH_DEVICE)
                real_vf = data[:,:52]
                if in_type.lower()=='vf+rnfl+age':
                    data_in = data.clone()
                if in_type.lower()=='vf+rnfl':
                    data_in = data[:,:-1]
                if in_type.lower()=='rnfl+age':
                    data_in = data[:,52:]
                if in_type.lower()=='rnfl':
                    data_in = data[:,52:-1]
                if in_type.lower()=='vf+age':
                    data_in = torch.cat( [data[:,:52], data[:,-1].view(-1,1)], dim=1 )
                #-------------------------------------------
                # Forward pass
                data_out, encoding = ae_model(data_in)
                #-------------------------------------------
                # Compute loss
                rec_loss = reconstruction_loss(data_out, data_in, loss_type, in_type, is_norm)
                enc_loss = encoding_loss(encoding, real_vf, loss_type, 'vf', is_norm)
                loss = (1-lambda_z)*rec_loss + lambda_z*enc_loss
                #-------------------------------------------
                # Back propagation
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.cpu().detach().item()
                tmp_rec_loss += rec_loss.cpu().detach().item()
                tmp_enc_loss += enc_loss.cpu().detach().item()
            #-------------------------------------------
            # Compute epoch loss, update containers
            epoch_loss = running_loss/(b+1)
            tmp_epoch_rloss = tmp_rec_loss/(b+1)
            tmp_epoch_eloss = tmp_enc_loss/(b+1)
            train_records[phase].append(epoch_loss)
            if phase=='valid':
                scheduler.step(epoch_loss)
            if phase=='valid' and (epoch_loss < best_loss):
                trained_models = copy.deepcopy(ae_model)
                best_loss = epoch_loss
            if disp_gap>0 and (epoch%disp_gap==0 or early_stop):
                print(f'{phase}: Total loss: {epoch_loss:.4f}, Recon: {tmp_epoch_rloss:.4f}, Enc: {tmp_epoch_eloss:.4f}')
        if early_stop:
            break
    #-------------------------------------------
    # Output
    #-------------------------------------------
    out_loss = os.path.join(out_path, f'losses_cv_fold_{cur_fold}.pkl')
    out_model= os.path.join(out_path, f'models_cv_fold_{cur_fold}.pkl')
    joblib.dump(train_records,  out_loss)
    torch.save(trained_models, out_model)
    return train_records, trained_models


def cv_train():
    #-------------------------------------------
    # Parse argument
    #       num_cv:         (int)   number of CV fold, default: 10
    #       batch_size:     (int)   batch size in training, default: 64
    #       hidden_dim:     (int)   MLP hidden layer dimension, default: 200
    #       init_lr:        (float) initial learning rate, default: 0.001
    #       wt_decay:       (float) weight decay, default: 0.0005
    #       loss_type:      (str)   type of loss function, default: mse
    #       num_epochs:     (int)   number of training epochs, default: 1000
    #       lambda_z:       (float) weight factor to balance reconstruction and encoding lossess, default: 0.6
    #       disp_gap:       (int)   display training results every n epochs, default: 100
    #       is_norm:        (bool)  data normalization, default: True
    #       in_type:        (str)   input data types, default: vf+rnfl+age
    #       enc_noise_std:  (float) the standard deviation of the Gaussian noise (zero mean) added to the encoding
    #-------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cv',       nargs='?', type=int,   default=MODEL_PARAMETERS['num_cv'])
    parser.add_argument('--batch_size',   nargs='?', type=int,   default=MODEL_PARAMETERS['batch_size'])
    parser.add_argument('--hidden_dim',   nargs='?', type=int,   default=MODEL_PARAMETERS['hidden_dim'])
    parser.add_argument('--init_lr',      nargs='?', type=float, default=MODEL_PARAMETERS['init_lr'])
    parser.add_argument('--wt_decay',     nargs='?', type=float, default=MODEL_PARAMETERS['wt_decay'])
    parser.add_argument('--loss_type',    nargs='?', type=str,   default=MODEL_PARAMETERS['loss_type'])
    parser.add_argument('--num_epochs',   nargs='?', type=int,   default=MODEL_PARAMETERS['num_epochs'])
    parser.add_argument('--lambda_z',     nargs='?', type=float, default=MODEL_PARAMETERS['lambda_z'])
    parser.add_argument('--disp_gap',     nargs='?', type=int,   default=MODEL_PARAMETERS['disp_gap'])
    parser.add_argument('--is_norm',      nargs='?', type=bool,  default=MODEL_PARAMETERS['is_norm'])
    parser.add_argument('--in_type',      nargs='?', type=str,   default=MODEL_PARAMETERS['in_type'])
    parser.add_argument('--enc_noise_std',nargs='?', type=float, default=MODEL_PARAMETERS['enc_noise_std'])
    args = parser.parse_args()
    #-------------------------------------------
    params = vars(args)
    PARAM_STR = ''
    for k,v in params.items():
        PARAM_STR += f'_{v}'
    print('\n')
    print('='*100)
    print('Model parameters:', PARAM_STR)
    print('='*100)
    out_path = f'./ae_models/trained_with_mock_data/{PARAM_STR}/'
    params['out_path'] = out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #-------------------------------------------
    # Generate the mock data: (n, 309)
    #-------------------------------------------
    mock_data = generate_mock_cv_data(n=1000, rnd_seed=SEED)
    #-------------------------------------------
    # CV training with mock data
    #-------------------------------------------
    kf = KFold(n_splits=params['num_cv'])
    for cur_fold, (train_idx, valid_idx) in enumerate(kf.split(mock_data)):
        params['cur_fold'] = cur_fold
        train_data = mock_data[train_idx]
        valid_data = mock_data[valid_idx]
        train(train_data, valid_data, params)


#=========================================================
if __name__ == '__main__':
    cv_train()