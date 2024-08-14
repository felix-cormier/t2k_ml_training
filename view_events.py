import numpy as np
import h5py
import time

import torch

import sys

import pandas as pd

import WatChMaL.analysis.utils.math as math


# torch.cuda.set_device(7)


from joblib import dump, load


print(sys.path)
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/dataset/cnn/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/dataset/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/data_utils/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/')
from cnn_dataset import CNNDataset, CNNDatasetDeadPMT, CNNDatasetScale

import watchmal.dataset.data_utils as du

indices = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/train_val_test_nFolds3_fold0.npz' 
h5file = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/combine_combine.hy'
pmt_pos = '/data/thoriba/t2k/imagefile/skdetsim_imagefile.npy'


cnn = CNNDatasetDeadPMT(h5file, pmt_pos, dead_pmts_file='/data/fcormier/Public/dead_pmts.sk4.txt',
                        use_dead_pmt_mask=True, use_hit_mask=True)

index_list_file = '/data/thoriba/t2k/plots/muon_mom_fixed_dead_fully_cut/dead_with_mask/residual_check/residuals_info_idx.csv'
res_info = pd.read_csv(index_list_file)
# data manipulation
res_info['fractional_residual'] = res_info['fractional_residual'].apply(lambda x: float(x.strip('[]')) )
res_info['pred'] = res_info['pred'].apply(lambda x: float(x.strip('[]')) )
res_info['truth'] = res_info['truth'].apply(lambda x: float(x.strip('[]')) )
res_info['towall'] = res_info['towall'].apply(lambda x: float(x))
res_info['event_index'] = res_info['event_index'].apply(lambda x: int(x))

res_info_fitqun = pd.read_csv('/data/thoriba/t2k/plots/muon_mom_fixed_dead_fully_cut/dead_with_mask/fitqun_performance/muon_residuals_fitqun2.csv')
print(res_info_fitqun.columns)
res_info_fitqun.columns = ['event_index', 'true', 'pred', 'frac_residual', 'nhits', 'towall', 'visible_energy']
res_info_fitqun['event_index'] = res_info_fitqun['event_index'].apply(lambda x: int(x))

positions = h5py.File(h5file,mode='r')['positions']


index_list_neg2 = res_info[res_info['fractional_residual'] <= -2]['event_index'].to_numpy()
index_list_neg1 = res_info[res_info['fractional_residual'] <= -1]['event_index'].to_numpy()

index_list_towall200_neg_1 = res_info[(res_info['towall'] >= 0) & (res_info['fractional_residual'] <= -2)]['event_index'].to_numpy()


plot_dest = '/data/thoriba/t2k/plots/muon_mom_fixed_dead_fully_cut/dead_with_mask/residual_check/images_with_FQ_newMOMdef/'


dwalls = []
for i in index_list_towall200_neg_1:
    dwalls.append(float(math.dwall(positions[i])[0]))


# import matplotlib.pyplot as plt
# plt.hist(dwalls, bins=50)
# plt.xlabel('Dwall')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dwall (towall > 200 and residual < -1) Num_events = ' + str(len(dwalls)))
# plt.savefig(plot_dest + 'dwall_hist_2.png')


# exit()
printed = 0
for i in index_list_towall200_neg_1:
    print('-------------------index =', i, '-------------------')
    item  = cnn.__getitem__(i)
    for key in item:
        # if key == 'data':
            # print(item[key][0].shape)
            # chrg = item[key][1].flatten()
            # print('charge shape', chrg.shape)

            # print('non-zero charge count', torch.count_nonzero(chrg))
            # print('total charge', torch.sum(chrg))
            # print('non-zero charges', chrg[chrg > 0])
        # print(f'----------{key}----------')
        print(key, item[key])
    printed += 1

    print('positions', positions[i])
    print('dwall', math.dwall(positions[i]))
    dwall_i = round(float(math.dwall(positions[i])[0]), 1)

    print(round(res_info_fitqun[res_info_fitqun["event_index"] == i]["true"], 2))

    note = ""
    note += f'true_ML={round(res_info[res_info["event_index"] == i]["truth"].to_numpy()[0], 2)}, '
    note += f'pred_ML={round(res_info[res_info["event_index"] == i]["pred"].to_numpy()[0], 2)}, '
    note += f'res_ML={round(res_info[res_info["event_index"] == i]["fractional_residual"].to_numpy()[0], 2)}, '
    note += f'total_chrg={round(res_info[res_info["event_index"] == i]["total_charge"].to_numpy()[0],1)}, '
    note += f'nhits={res_info[res_info["event_index"] == i]["nhits"].to_numpy()[0]}, '
    note += f'towall={round(res_info[res_info["event_index"] == i]["towall"].to_numpy()[0], 1)}, '
    note += f'dwall={dwall_i}\n'

    note += f'true_FQ={round(res_info_fitqun[res_info_fitqun["event_index"] == i]["true"].to_numpy()[0], 2)}, '
    note += f'pred_FQ={round(res_info_fitqun[res_info_fitqun["event_index"] == i]["pred"].to_numpy()[0], 2)}, '
    note += f'res_FQ={round(res_info_fitqun[res_info_fitqun["event_index"] == i]["frac_residual"].to_numpy()[0], 2)}, '
    note += f'nhits_FQ={res_info_fitqun[res_info_fitqun["event_index"] == i]["nhits"].to_numpy()[0]}, '
    note += f'towall_FQ={round(res_info_fitqun[res_info_fitqun["event_index"] == i]["towall"].to_numpy()[0], 1)}, '
    note += f've_FQ={round(res_info_fitqun[res_info_fitqun["event_index"] == i]["visible_energy"].to_numpy()[0], 1)}'
    

    r = np.sqrt(positions[i][0][0]**2 + positions[i][0][1]**2)
    title = f'{i}th event at vertex $(r, \\theta, z) = ({r:.1f}, ?, {positions[i][0][2]:.1f})$'

    y_lables = ['PMT Time', 'PMT Charge', 'Dead PMTs (SK4)', 'Hit PMTs']
    plot_names = ['time_', 'chrg_', 'dead_', 'hits_']
    for a, (y_label, plot_name) in enumerate(zip(y_lables, plot_names)):
        du.save_fig_dead(item['data'][a], None,  None, None, y_label=y_label,
                         counter=i, output_path=f'{plot_dest}{plot_name}', dead_pmt_percent=-1, note=note, title=title)
        # du.generic_histogram(item['data'][a].flatten().numpy(), x_name=y_label, output_name=plot_name+ str(i) + '_hist', output_path=f'{plot_dest}', title=f'{i}-th event {y_label}', doNorm=True, bins =100)
    # if printed >= 5:
    #     break


