import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d
from utils import vf_gray_image, vf_point_image, normalize, de_normalize
from constants import *


class Decode_VF_Visualization(object):
    """
    Decode a given VF (array(52) to RNFL thickness profile using the trained Decoder.
    Input: 
        ae_model_or_path_to_model:  Pytorch model or String or None
    """
    def __init__(self, ae_model_or_path_to_model):
        super(Decode_VF_Visualization, self).__init__()
        if type(ae_model_or_path_to_model)==str:
            if not os.path.exists(ae_model_or_path_to_model):
                raise ValueError("Cannot find AE model in:", ae_model_or_path_to_model)
            else:
                self.ae_model = torch.load(ae_model_or_path_to_model, map_location='cpu').eval()
        elif isinstance(ae_model_or_path_to_model, torch.nn.Module):
            self.ae_model = ae_model_or_path_to_model.to('cpu').eval()
        elif ae_model_or_path_to_model is None:
            try:
                model_path = './ae_models/example_models/model_decode_vf.pkl'
                self.ae_model = torch.load(model_path, map_location='cpu').eval()
            except Exception as e:
                raise e
        else:
            raise ValueError(f"Unsupported Decoder type: {type(ae_model_or_path_to_model)}. Expecting: String or Pytorch Model")


    def calculate_clock_hour_mean_rnflt(self, rnflt_profile):
        """
        Calcuate the clock hour mean RNFL thickness.
        Input: 
            rnflt_profile: 1D-array with length <= 360
        Output:
            clock_hour_mean_values: dict, clock-hour mean values
            clock_hour_mean_labels: dict, whether clock-hour mean values are normal (1) or abnormal (0)
            clock_hour_rnfl_index_range: Clock-hour index range for the RNFLT array (for visualizaiton).
        """
        assert len(rnflt_profile)<=360, f"Unsupported RNFL thickenss length: {len(rnflt_profile)}. Should be <= 360"
        if len(rnflt_profile)<360:
            rnflt_profile = np.interp(np.arange(360), np.linspace(0,360,len(rnflt_profile)), rnflt_profile)
        clock_hours = np.arange(1, 13, 1)
        clock_hour_rnfl_index_range = { 
            10:[15,45],11:[45,75],12:[75,105],1:[105,135],2:[135,165],3:[165,195],\
            4:[195,225],5:[225,255],6:[255,285],7:[285,315],8:[315,345],9:[[345, 360],[0, 15]]}
        clock_hour_mean_values = {h:None for h in clock_hours}
        clock_hour_mean_labels = {h:None for h in clock_hours}
        for h in clock_hours:
            if h!=9:
                rnfl_hour_mean = rnflt_profile[clock_hour_rnfl_index_range[h][0]:clock_hour_rnfl_index_range[h][1]].mean()
            else:
                rnfl_hour_mean = np.mean([rnflt_profile[clock_hour_rnfl_index_range[h][0][0]:clock_hour_rnfl_index_range[h][0][1]].mean(), 
                                         rnflt_profile[clock_hour_rnfl_index_range[h][1][0]:clock_hour_rnfl_index_range[h][1][1]].mean()])
            clock_hour_mean_values[h] = rnfl_hour_mean
            clock_hour_mean_labels[h] = 0 if rnfl_hour_mean<RNFL_NORM_LB[h] else 1 
        return clock_hour_mean_values, clock_hour_mean_labels, clock_hour_rnfl_index_range


    def rnfl_clock_hour_pie(self, rnflt_profile, ax=None):
        """
        Plot the clock-hour pie chart of the RNFL thickness profile
        Sectors are marked in red if the clock-hour mean is lower than the lower bound of normal values.
        Otherwise, they are marked in green, similar to the clinical OCT RNFL report. 
        """
        fsize      = 5
        pie_sizes  = np.ones(12)*30
        clock_hours= np.arange(1, 13, 1)
        _, clock_hour_mean_labels, _ = self.calculate_clock_hour_mean_rnflt(rnflt_profile)
        pie_colors = ['tab:red' if clock_hour_mean_labels[c]==0 else 'tab:green' for c in clock_hours]
        if ax is not None:
            ax.pie(pie_sizes, labels=clock_hours, startangle=75, colors=pie_colors, counterclock=False, wedgeprops = {"edgecolor":"dimgray",'alpha':0.6}, textprops={'fontsize': fsize})
            return ax
        else:
            plt.pie(pie_sizes, labels=clock_hours, startangle=75, colors=pie_colors, counterclock=False, wedgeprops = {"edgecolor":"dimgray",'alpha':0.6}, textprops={'fontsize': fsize})
            plt.tight_layout()
            plt.show()


    def decode_from_vf_space(self, base_vf, is_display=True, **kwargs):
        """
            Decode any given VF using the trained decoder
            Intput:
                base_vf: array(52)      
                    # The VF test to decode, if None then use uniform healthy VF with all sensivity of 30 dB.
                is_display: Boolean
                    # To control whether to show the decode results. The result includes the input VF, reconstructed VF,
                    # the reconstructed RNFL thickness profile curve, the reference normal range of RNFL thickness,
                    # and clock-hour regions where below the lower bound of the normal values. 
            Output:
                recon_vf:               Reconstructed VF, array(52)
                recon_rnfl_interp_360:  Reconstructed RNFLT, array(360)
        """
        #-------------------------------------------
        # Base VF
        if base_vf is None:
            base_vf = np.ones(52) * 30
        #-------------------------------------------
        # Reconstruction from given VF
        vf_norm = normalize(base_vf, V_MIN, V_MAX)
        vf_norm_ts = torch.from_numpy(vf_norm).type(torch.FloatTensor).to('cpu')
        recon_data = self.ae_model.Decoder(vf_norm_ts)
        recon_vf   = recon_data[:52].cpu().detach().numpy()
        recon_rnfl = recon_data[52:-1].cpu().detach().numpy()
        #-------------------------------------------
        # Denormalization
        recon_vf  = de_normalize(recon_vf,  V_MIN, V_MAX)
        recon_rnfl= de_normalize(recon_rnfl,R_MIN, R_MAX)
        recon_rnfl_proc = gaussian_filter1d(recon_rnfl, sigma=3, mode='reflect')
        recon_rnfl_interp_360 = np.interp(np.arange(360), np.linspace(0,360,256), recon_rnfl_proc)
        #-------------------------------------------
        # Plot results
        #-------------------------------------------
        fsize, rnfl_ylim = 8, (0, 300)
        clock_hours = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #-------------------------------------------
        # Plot VFs and RNFLT curve
        fig= plt.figure(figsize=(10, 3))
        gs = gridspec.GridSpec(1, 4, figure=fig)
        ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2:])
        ax1 = vf_point_image(base_vf, ax1, 'vf')                # Show as numerical plots
        ax2 = vf_point_image(recon_vf, ax2, 'vf')
        #ax1.imshow(vf_gray_image(base_vf),  cmap='gray')       # Or show as grayscale plots
        #ax2.imshow(vf_gray_image(recon_vf), cmap='gray')
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax1_title = 'VF Input'
        if kwargs:
            if 'VF_pattern' in kwargs.keys():
                ax1_title = f"VF Input: {kwargs['VF_pattern']}"
        ax1.set_title(ax1_title, fontsize=fsize)
        ax2.set_title('VF Recon', fontsize=fsize)
        ax3.plot(np.arange(360), recon_rnfl_interp_360, label=rf'Recon: mRNFLT={np.mean(recon_rnfl_proc):.1f} $\mu$m',alpha=0.9)
        #-------------------------------------------
        # Plot Normal range
        ax3.fill_between(np.arange(0, 361, 30), y1=[RNFL_NORM_LB[c] for c in clock_hours], y2=[RNFL_NORM_UB[c] for c in clock_hours], label='Normal Range', color='tab:green', alpha=0.2)
        #-------------------------------------------
        # Plot Abnomral hours
        _, clock_hour_rnfl_labels, clock_hour_rnfl_index_range = self.calculate_clock_hour_mean_rnflt(recon_rnfl_interp_360)
        is_labeled = False
        for c in clock_hour_rnfl_labels.keys():
            clock_hour_label = clock_hour_rnfl_labels[c]
            clock_hour_range = clock_hour_rnfl_index_range[c]
            if clock_hour_label==0:
                lab_text = '' if is_labeled else '<5-th percentile of Normal'
                is_labeled = True
                c_prev = 12 if c==1 else c-1
                c_next = 1 if c==12 else c+1
                y2_range = [(RNFL_NORM_UB[c_prev]+RNFL_NORM_UB[c])/2, (RNFL_NORM_UB[c]+RNFL_NORM_UB[c_next])/2]
                if c==9:        # The temporal region of the ONH
                    ax3.fill_between(clock_hour_range[0], y1=0, y2=y2_range, color='tab:red', alpha=0.15, label=lab_text)
                    ax3.fill_between(clock_hour_range[1], y1=0, y2=y2_range, color='tab:red', alpha=0.15)
                else:
                    ax3.fill_between(clock_hour_range, y1=0, y2=y2_range, color='tab:red', alpha=0.15, label=lab_text)
        #-------------------------------------------
        ax3.set_xlabel('ONH Clock Hours', fontsize=fsize)
        ax3.set_ylabel(r'RNFLT ($\mu$m)', fontsize=fsize)
        ax3.tick_params(axis='both', which='both', labelsize=fsize)
        ax3.legend(fontsize=fsize)
        ax3.set_ylim(rnfl_ylim)
        ax3.set_xlim(0, 360)
        ax3.set_xticks(np.arange(0, 361, 30))
        ax3.set_xticklabels(clock_hours)
        ax3.grid(True, ls='--')
        #-------------------------------------------
        # Plot ONH RNFLT Piechart
        axins = inset_axes(ax3, width="40%", height="40%", loc="upper left", borderpad=0)
        axins = self.rnfl_clock_hour_pie(recon_rnfl_interp_360, axins)
        plt.tight_layout()
        plt.savefig(f"./figures/VF_Decode_{kwargs['VF_pattern']}.jpeg", dpi=300)
        if is_display:      
            plt.show()
        else:
            plt.close()
        return recon_vf, recon_rnfl_interp_360, ax1, ax2, ax3


#=================================================
if __name__ == '__main__':
    # Intialize the VF_decoder
    vf_decoder = Decode_VF_Visualization(ae_model_or_path_to_model='./ae_models/example_models/model_decode_vf.pkl')

    # Decode and Visualize for VFs with typically glaucoma defects,
    # including Nasal Step, Arcuate, Hemifield, and Tunnel Vision. 
    for key in COMMON_GLAUCOMA_VF_DAMAGE_INDEX.keys():
        pattern_list = COMMON_GLAUCOMA_VF_DAMAGE_INDEX[key]
        for p, pattern in enumerate(pattern_list):
            base_vf = np.ones(52)*30
            base_vf[pattern] = 0
            vf_decoder.decode_from_vf_space(base_vf, VF_pattern=f'{key}_{p+1}')