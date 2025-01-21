#!/usr/bin/env python3
#---------------------------------------------------------------------------------------------
# Description: Visualize the data fusion results
# Author     : Leo(Yan) Li
# Date       : January 22, 2024
# version    : 1.0
# License    : MIT License
# Arguments  :
#       num_cv:     (int)   number of CV fold, default: 10
#       batch_size: (int)   batch size in training, default: 64
#       hidden_dim: (int)   MLP hidden layer dimension, default: 200
#       init_lr:    (float) initial learning rate, default: 0.001
#       wt_decay:   (float) weight decay, default: 0.0005
#       loss_type:  (str)   type of loss function, default: mse
#       num_epochs: (int)   number of training epochs, default: 1000
#       lambda_z:   (float) weight factor to balance reconstruction and encoding lossess, default: 0.6
#       disp_gap:   (int)   display training results every n epochs, default: 100
#       is_norm:    (bool)  data normalization, default: True
#       in_type:    (str)   input data types, default: vf+rnfl+age
# Usage:
#       (1) Evaluate models trained with default parameters (using mock data):   
#               python cv_evaluate.py
#       (2) Evaluate models trained with specified parameters (using mock data): 
#               python cv_evaluate.py --argument_1 value_1 --argument_2 value_2
#---------------------------------------------------------------------------------------------
import os
import torch
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *


def eval_visual_cv_ae(test_data, trained_model, train_record, params):
    #---------------------------------------
    # Evaluation parameters
    #---------------------------------------
    test_encoder= trained_model.Encoder.eval()
    test_decoder= trained_model.Decoder.eval()
    in_type     = params['in_type']
    is_norm     = params['is_norm']
    cur_fold    = params['cur_fold']
    #---------------------------------------
    # Load data
    #---------------------------------------
    test_vf     = test_data[:, :52]
    test_rnfl   = test_data[:, 52:-1]
    test_ages   = test_data[:, -1]
    test_vf_ts  = torch.from_numpy(test_vf).type(torch.FloatTensor).to(TORCH_DEVICE)
    test_rnfl_ts= torch.from_numpy(test_rnfl).type(torch.FloatTensor).to(TORCH_DEVICE)
    test_data_ts= torch.from_numpy(test_data).type(torch.FloatTensor).to(TORCH_DEVICE)
    #---------------------------------------
    # Forward pass based on data selection
    #---------------------------------------
    if in_type.lower()=='vf+rnfl+age':
        fuse_vf   = test_encoder(test_data_ts)
        recon_data= test_decoder(fuse_vf)
        recon_vf  = recon_data[:,:52]
        recon_rnfl= recon_data[:,52:-1]
        recon_age = recon_data[:,-1]
    elif in_type.lower()=='vf+rnfl':
        fuse_vf   = test_encoder(test_data_ts[:,:-1])
        recon_data= test_decoder(fuse_vf)
        recon_vf  = recon_data[:,:52]
        recon_rnfl= recon_data[:,52:]
    elif in_type.lower()=='rnfl+age':
        fuse_vf   = test_encoder(test_data_ts[:,52:])
        recon_data= test_decoder(fuse_vf)
        recon_vf  = fuse_vf
        recon_rnfl= recon_data[:,:256]
        recon_age = recon_data[:,-1]
    elif in_type.lower()=='rnfl':
        fuse_vf   = test_encoder(test_data_ts[:,52:-1])
        recon_data= test_decoder(fuse_vf)
        recon_vf  = fuse_vf
        recon_rnfl= recon_data
    elif in_type.lower()=='vf+age':
        test_data_in = torch.cat([test_data_ts[:,:52], test_data_ts[:,-1].view(-1,1)], dim=1)
        fuse_vf   = test_encoder(test_data_in)
        recon_data= test_decoder(fuse_vf)
        recon_vf  = recon_data[:,:52]
        recon_age = recon_data[:,-1]
        recon_rnfl= test_rnfl_ts
    else:
        raise ValueError("Unsupported data in_type:", in_type)
    recon_vf  = recon_vf.cpu().detach().numpy()
    recon_rnfl= recon_rnfl.cpu().detach().numpy()
    fuse_vf   = fuse_vf.cpu().detach().numpy()
    fuse_vf   = np.clip(fuse_vf, V_MIN, V_MAX)
    #-----------------------------------------------------------
    if is_norm:
        test_vf   = de_normalize(test_vf,   V_MIN, V_MAX)
        test_rnfl = de_normalize(test_rnfl, R_MIN, R_MAX)
        test_ages = de_normalize(test_ages, A_MIN, A_MAX)
        fuse_vf   = de_normalize(fuse_vf,   V_MIN, V_MAX)
        recon_vf  = de_normalize(recon_vf,  V_MIN, V_MAX)
        recon_rnfl= de_normalize(recon_rnfl,R_MIN, R_MAX)
        if 'age' in in_type.lower():
            recon_age = de_normalize(recon_age, A_MIN, A_MAX)
    #-----------------------------------------------------------
    # Evaluation results: 
    #   1. Reconstruction loss: VF reconstruction MAE 
    #   2. Reconstruction loss: RNFL reconstruction MAE
    #   3. Overall reconstruction loss
    #   4. Encoding loss: VF-Encoding MAE
    #-----------------------------------------------------------
    print('-'*50)
    print(f'CV-fold-{cur_fold}:')
    vf_rec_mae  = np.mean(np.abs(recon_vf-test_vf), axis=1)
    rnfl_rec_mae= np.mean(np.abs(recon_rnfl-test_rnfl), axis=1)
    vf_fuse_mae = np.mean(np.abs(fuse_vf-test_vf), axis=1)
    print(f'  1.VF recon      MAE: {print_stats(vf_rec_mae,  r=2)}')
    print(f'  2.RNFL recon    MAE: {print_stats(rnfl_rec_mae,r=2)}')
    print(f'  3.Overall recon MAE: {np.mean([np.mean(vf_rec_mae), np.mean(rnfl_rec_mae)]):.2f}')
    print(f'  4.Encoding loss MAE: {print_stats(vf_fuse_mae, r=2)}')
    #-----------------------------------------------------------
    # Plot learning curve
    #-----------------------------------------------------------
    loss_type = params['loss_type']
    fig, ax = plt.subplots(figsize=(7,5))
    for phase in train_record.keys():
        g_loss = train_record[phase]
        if phase.lower()=='valid':
            phase = 'test'
        ax.plot(np.arange(len(g_loss)), g_loss, label=f'{phase} loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(f'{loss_type.upper()} Loss')
        ax.grid(True, ls='--')
        ax.legend()
        ax.set_title(f'CV-fold-{cur_fold}')
    plt.tight_layout()
    plt.show()


def evaluate_ae_results():
    #-----------------------------------------------------------
    # Parse argument
    #-----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cv',       nargs='?',const=MODEL_PARAMETERS['num_cv'],       type=int,  default=MODEL_PARAMETERS['num_cv'])
    parser.add_argument('--batch_size',   nargs='?',const=MODEL_PARAMETERS['batch_size'],   type=int,  default=MODEL_PARAMETERS['batch_size'])
    parser.add_argument('--hidden_dim',   nargs='?',const=MODEL_PARAMETERS['hidden_dim'],   type=int,  default=MODEL_PARAMETERS['hidden_dim'])
    parser.add_argument('--init_lr',      nargs='?',const=MODEL_PARAMETERS['init_lr'],      type=float,default=MODEL_PARAMETERS['init_lr'])
    parser.add_argument('--wt_decay',     nargs='?',const=MODEL_PARAMETERS['wt_decay'],     type=float,default=MODEL_PARAMETERS['wt_decay'])
    parser.add_argument('--loss_type',    nargs='?',const=MODEL_PARAMETERS['loss_type'],    type=str,  default=MODEL_PARAMETERS['loss_type'])
    parser.add_argument('--num_epochs',   nargs='?',const=MODEL_PARAMETERS['num_epochs'],   type=int,  default=MODEL_PARAMETERS['num_epochs'])
    parser.add_argument('--lambda_z',     nargs='?',const=MODEL_PARAMETERS['lambda_z'],     type=float,default=MODEL_PARAMETERS['lambda_z'])
    parser.add_argument('--disp_gap',     nargs='?',const=MODEL_PARAMETERS['disp_gap'],     type=int,  default=MODEL_PARAMETERS['disp_gap'])
    parser.add_argument('--is_norm',      nargs='?',const=MODEL_PARAMETERS['is_norm'],      type=bool, default=MODEL_PARAMETERS['is_norm'])
    parser.add_argument('--in_type',      nargs='?',const=MODEL_PARAMETERS['in_type'],      type=str,  default=MODEL_PARAMETERS['in_type'])
    parser.add_argument('--enc_noise_std',nargs='?',const=MODEL_PARAMETERS['enc_noise_std'],type=float,default=MODEL_PARAMETERS['enc_noise_std'])
    args = parser.parse_args()
    #-----------------------------------------------------------
    params = vars(args)
    PARAM_STR = ''
    for k,v in params.items():
        PARAM_STR += f'_{v}'
    print('\n')
    print('='*100)
    print('Model parameters:', PARAM_STR)
    print('='*100)
    #-----------------------------------------------------------
    for cur_fold in range(params['num_cv']):
        record_path = f'./ae_models/trained_with_mock_data/{PARAM_STR}/losses_cv_fold_{cur_fold}.pkl'
        model_path  = f'./ae_models/trained_with_mock_data/{PARAM_STR}/models_cv_fold_{cur_fold}.pkl'
        if not os.path.exists(model_path):
            raise ValueError("The given model path does not exists, please check...", model_path)
        params['cur_fold'] = cur_fold
        train_record = joblib.load(record_path)
        trained_model= torch.load(model_path, map_location=TORCH_DEVICE)
        mock_test_data = generate_mock_cv_data(n=100, rnd_seed=SEED)
        eval_visual_cv_ae(mock_test_data, trained_model, train_record, params)


if __name__ == '__main__':
    evaluate_ae_results()
