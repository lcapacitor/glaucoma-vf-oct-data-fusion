#!/usr/bin/env python3
#-------------------------------------------------------
# Description: Define all helper functions
# Author     : Leo Yan Li-Han
# Date       : January 22, 2024
# version    : 1.0
# License    : MIT License
#-------------------------------------------------------
import joblib
import numpy as np
from PIL import Image
from constants import *
#-------------------------------------------------------
# Visualization
#-------------------------------------------------------
def pad52_54(vf52, side, pad):
    v54 = np.zeros(54)
    if side == 0 or side == 'OD':
        v54[BS_INDEX_OD_54] = pad
        v54[:25] = vf52[:25]
        v54[26:34]=vf52[25:33]
        v54[35:]  =vf52[33:]
    if side == 1 or side == 'OS':
        v54[BS_INDEX_OS_54] = pad
        v54[:19] = vf52[:19]
        v54[20:28]=vf52[19:27]
        v54[29:]  =vf52[27:]
    return v54

def vfData_Padding(vf, site, p):
    if len(vf) == 52:
        vf = pad52_54(vf, site, p)
    vfMat = p * np.ones((8, 9))
    if site == 'OD' or site == 0:
        vfMat[0, 3:7] = vf[:4]
        vfMat[1, 2:8] = vf[4:10]
        vfMat[2, 1:9] = vf[10:18]
        vfMat[3, :]   = vf[18:27]
        vfMat[4, :]   = vf[27:36]
        vfMat[5, 1:9] = vf[36:44]
        vfMat[6, 2:8] = vf[44:50]
        vfMat[7, 3:7] = vf[50:]
    if site == 'OS' or site == 1:
        vfMat[0, 2:6] = vf[:4]
        vfMat[1, 1:7] = vf[4:10]
        vfMat[2, :8]  = vf[10:18]
        vfMat[3, :]   = vf[18:27]
        vfMat[4, :]   = vf[27:36]
        vfMat[5, :8]  = vf[36:44]
        vfMat[6, 1:7] = vf[44:50]
        vfMat[7, 2:6] = vf[50:]
    return vfMat

def grayscale_pattern(interp_dB, vf_gray_pattern):
    if interp_dB==interp_dB:
        if interp_dB<=1:
            v = vf_gray_pattern['pattern_v0']
        elif interp_dB<=6:
            v = vf_gray_pattern['pattern_v1']
        elif interp_dB<=11:
            v = vf_gray_pattern['pattern_v6']
        elif interp_dB<=16:
            v = vf_gray_pattern['pattern_v11']
        elif interp_dB<=21:
            v = vf_gray_pattern['pattern_v16']
        elif interp_dB<=26:
            v = vf_gray_pattern['pattern_v21']
        elif interp_dB<=31:
            v = vf_gray_pattern['pattern_v26']
        else:
            v = vf_gray_pattern['pattern_v31']
        im = Image.fromarray(v.to_numpy().astype(bool))
        im = im.resize((24, 24), Image.NEAREST)
        assert im.mode == '1'
        im = im.convert("LA")
        im_arr = np.asarray(im)[:,:,0]
    else:
        im_arr = np.ones((24,24))*255
    return im_arr

def vf_gray_image(vf52):
    vf54 = pad52_54(vf52, 'OD', 0)
    vf_mat_28by28 = np.matmul(MATRIX_784x54, vf54).reshape(28, 28)
    vf_gray_mat = np.zeros((28*24, 28*24))
    for p_row in range(28):
        for p_col in range(28):
            row_s = p_row*24
            row_e = row_s+24
            col_s = p_col*24
            col_e = col_s+24
            interp_dB = vf_mat_28by28[p_row, p_col]
            im_block  = grayscale_pattern(interp_dB, vf_gray_pattern)
            vf_gray_mat[row_s:row_e, col_s:col_e] = im_block
    return vf_gray_mat

def vf_point_image(vf52, ax, dtype):
    vf_mat= vfData_Padding(vf52, 'OD', np.nan)
    if dtype.lower()=='vf':
        eps_x, c_th = 0.2, 15
        d_min, d_max = V_MIN, V_MAX
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. 'vf' only. ")
    eps_y, fs = 0.2, 6
    ax.imshow(vf_mat, cmap='gray', vmin=d_min, vmax=d_max)
    for y in range(8):
        for x in range(9):
            idx = 9 * y + x
            if idx in EMPTY_INDEX_BS:
                continue
            text  = str(int(np.around(vf_mat[y, x], 0)))
            color = 'white' if vf_mat[y, x]<c_th else 'black'
            ax.text(x-eps_x, y+eps_y, text, color=color, fontsize=fs)
    return ax

#-------------------------------------------------------
# Data processing
#-------------------------------------------------------
def generate_mock_cv_data(n, rnd_seed=SEED):
    np.random.seed(rnd_seed)
    return np.random.random(size=(n,309))

def load_example_data():
    return joblib.load(EXAMPLE_EYE_PATH)

def normalize(data, dmin, dmax):
    return (data-dmin)/(dmax-dmin)

def de_normalize(data, dmin, dmax):
    return data*(dmax-dmin)+dmin

def print_stats(data, r=1):
    p25, p75 = np.percentile(data, [25,75])
    avg, std = np.mean(data), np.std(data)
    sem  = std / np.sqrt(len(data))
    ci95_low, ci95_up = avg-1.96*sem, avg+1.96*sem
    text = f'MEAN={round(avg,r)}, SEM={round(sem,r)}, STD={round(std,r)}, 95%CI:{round(ci95_low,r)} to {round(ci95_up,r)}, MEDIAN={round(np.median(data),r)}, IQR: {round(p25,r)} to {round(p75,r)}, n={len(data)}' 
    return text