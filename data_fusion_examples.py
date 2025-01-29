#!/usr/bin/env python3
#---------------------------------------------------------------------------------------------
# Description: Visualize data fusion results with real VF and OCT data for three eyes
# Author     : Leo Yan Li-Han
# Date       : January, 2025
# version    : 2.0
# License    : MIT License
# Usage:
#       (1) Show all three examples:   
#               python visualize_examples.py
#       (2) Show one example with mild VF defect: 
#               ptyhon visualize_examples.py --eye mild
#---------------------------------------------------------------------------------------------
import torch
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from utils import *


def display_example(selected_eye):
    if selected_eye.lower()=='all':
        eye_types = ['mild', 'moderate', 'severe']
    else:
        eye_types = [selected_eye]
    for selected_eye in eye_types:
        #-------------------------------------------
        # Load model
        #-------------------------------------------
        model_path   = f'./ae_models/example_models/model_{selected_eye}.pkl'
        trained_model= torch.load(model_path, map_location=TORCH_DEVICE)
        test_encoder = trained_model['encoder'].eval()
        test_decoder = trained_model['decoder'].eval()
        #-------------------------------------------
        # Load data
        #-------------------------------------------
        test_data   = joblib.load('./data/example_eyes.pkl')[f'{selected_eye}'].reshape(1, -1)
        test_vf     = test_data[0, :52]
        test_rnfl   = test_data[0, 52:-1]
        test_age    = test_data[0, -1]
        test_data_ts= torch.from_numpy(test_data).type(torch.FloatTensor).to(TORCH_DEVICE)
        #-------------------------------------------
        # Data fusion
        #-------------------------------------------
        fused_vf  = test_encoder(test_data_ts).squeeze()
        recon_data= test_decoder(fused_vf).squeeze()
        recon_vf  = recon_data[:52].cpu().detach().numpy()
        recon_rnfl= recon_data[52:-1].cpu().detach().numpy()
        recon_age = recon_data[-1].cpu().detach().item()
        fused_vf  = fused_vf.cpu().detach().numpy()
        #-------------------------------------------
        # Denormalization
        #-------------------------------------------
        test_vf   = de_normalize(test_vf,   V_MIN, V_MAX)
        test_rnfl = de_normalize(test_rnfl, R_MIN, R_MAX)
        test_age  = de_normalize(test_age,  A_MIN, A_MAX)
        fused_vf  = de_normalize(fused_vf,  V_MIN, V_MAX)
        recon_vf  = de_normalize(recon_vf,  V_MIN, V_MAX)
        recon_age = de_normalize(recon_age, A_MIN, A_MAX)
        recon_rnfl= de_normalize(recon_rnfl,R_MIN, R_MAX)
        # Using a 1D Gaussian filter to smooth the reconstructed RNFLT curve for better visualization
        recon_rnfl= gaussian_filter1d(recon_rnfl, sigma=3, mode='reflect')
        #-------------------------------------------
        # Calculate MD and PSD
        #-------------------------------------------
        vf_norm = test_age*VF_NORM_SLOPE + VF_NORM_INTERCEPT
        td_in, td_fuse, td_rec = test_vf-vf_norm, fused_vf-vf_norm, recon_vf-vf_norm
        md_in   = np.average(td_in,  weights=VF_MD_WEIGHTS)
        md_fuse = np.average(td_fuse,weights=VF_MD_WEIGHTS)
        md_rec  = np.average(td_rec, weights=VF_MD_WEIGHTS)
        psd_in, psd_fuse, psd_rec = np.std(td_in), np.std(td_fuse), np.std(td_rec)
        #-------------------------------------------
        # Visualization
        #-------------------------------------------
        fig= plt.figure(figsize=(12, 3))
        gs = gridspec.GridSpec(1, 5, figure=fig)
        ax1, ax2, ax3, ax4 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3:])
        ax1.imshow(vf_gray_image(test_vf), cmap='gray')
        ax2.imshow(vf_gray_image(fused_vf), cmap='gray')
        ax3.imshow(vf_gray_image(recon_vf), cmap='gray')
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax1.set_title(fr'VF Input: MD=${md_in:.1f}$ PSD={psd_in:.1f}'+f'\nAge={test_age:.1f}', fontsize=9)
        ax2.set_title(fr'VF Fused: MD=${md_fuse:.1f}$ PSD={psd_fuse:.1f}', fontsize=9)
        ax3.set_title(fr'VF Recon: MD=${md_rec:.1f}$ PSD={psd_rec:.1f}'+f'\nAge={recon_age:.1f}', fontsize=9)
        ax4.plot(np.arange(360), np.interp(np.arange(360), np.linspace(0,360,256), test_rnfl), label=rf'Input:  mRNFLT={np.mean(test_rnfl):.1f} $\mu$m',alpha=0.9)
        ax4.plot(np.arange(360), np.interp(np.arange(360), np.linspace(0,360,256), recon_rnfl),label=rf'Recon: mRNFLT={np.mean(recon_rnfl):.1f} $\mu$m',alpha=0.9)
        ax4.set_xlabel('ONH degree')
        ax4.set_ylabel(r'RNFLT ($\mu$m)')
        ax4.legend()
        ax4.set_ylim(0, 250)
        ax4.set_xlim(0, 360)
        ax4.set_xticks(np.arange(0, 361, 45))
        ax4.set_xticklabels([rf'{x}$^\circ$' for x in np.arange(0, 361, 45)])
        ax4.grid(True, ls='--')
        plt.tight_layout()
        plt.savefig(f'./figures/example_{selected_eye}.jpeg', dpi=300)
        plt.show()

def visualize_examples():
    #-------------------------------------------
    # Parse argument
    #       eye: (str) Specify one eye to display. 
    #                  The one with mild, moderate, severe VF defects, or all of them.
    #                  Default value 'all'
    #-------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--eye', choices=['mild', 'moderate', 'severe', 'all'], default='all')
    args = parser.parse_args()
    selected_eye = vars(args)['eye']
    display_example(selected_eye)


if __name__ == '__main__':
    visualize_examples()