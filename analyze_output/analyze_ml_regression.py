import csv
import numpy as np

import sys

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.optimize import curve_fit

from plotting import regression_analysis, regression_analysis_perVar, compute_residuals
from analyze_output.utils.math import get_cherenkov_threshold

import WatChMaL.analysis.utils.fitqun as fq
import WatChMaL.analysis.utils.math as math

import h5py
from runner_util import electron_shower_depth, lq, mom_from_energies, mom_to_range_dicts, range_from_energy

def gaussian(x, a, mean, sigma):
     return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def export_mom_residuals(truth_0, pred_0, rootfiles, event_ids, directions, total_charge, visible_energy, towall, nhits, indices, settings):
     fract_residuals = (truth_0 - pred_0) / truth_0
     large_residuals_index = []
     for i in range(len(fract_residuals)):
          if fract_residuals[i] < - 1:
               large_residuals_index.append(i)
     print("index of large residuals after the cut", large_residuals_index)
     large_residuals_idx_wrt_dataset = indices[large_residuals_index]
     print("index of large residuals before the cut", large_residuals_idx_wrt_dataset)

     file_path = '/data/thoriba/t2k/plots/muon_mom_fixed_dead_fully_cut/dead_with_mask/residual_check/residuals_info_idx2.csv'
     data = []
     for i, idx in enumerate(large_residuals_index):
          data.append({
               'event_index': large_residuals_idx_wrt_dataset[i],
               'rootfile': rootfiles[idx],
               'event_id': event_ids[idx],
               'truth': truth_0[idx],
               'pred': pred_0[idx],
               'fractional_residual': fract_residuals[idx],
               'direction': directions[idx],
               'total_charge': total_charge[idx],
               'visible_energy': visible_energy[idx],
               'towall': towall[idx],
               'nhits': nhits[idx]
          })

     # Write data to CSV file
     with open(file_path, mode='w', newline='') as file:
          writer = csv.DictWriter(file, fieldnames=data[0].keys())
          writer.writeheader()
          writer.writerows(data)


#def analyze_ml_regression(input_path, target, ml_path, output_plot_path, label, fitqun_path=None):
def analyze_ml_regression(settings):

     #First argument is where to save plot
     #Second one where to get data
     files = settings.mlPath
     target = str(settings.target)
     preds = np.load(files+'predicted_'+target+'.npy')
     truth = np.load(files+target+'.npy')
     labels = np.load(files + 'labels.npy')

     # print('loaded truth', truth)



     if settings.doCombination:
          (_, fq_labels, _, fitqun_hash), (mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom) = fq.read_fitqun_file(settings.fitqunPath+'fitqun_combine.hy', regression=True)
          ml_combine_path = settings.inputPath
          hy = h5py.File(ml_combine_path+'combine_combine.hy', "r")
          indices = np.load(files + 'indices.npy')

          # calculate number of hits 
          events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
          nhits = (events_hits_index[indices+1] - events_hits_index[indices]).squeeze()
          total_charge = np.array([part.sum() for part in np.split(hy['hit_charge'], np.cumsum(nhits))[:-1]])
          #total_charge_2 = np.add.reduceat(hy['hit_charge'], np.cumsum(nhits)[:-1])
          rootfiles = np.array(hy['root_files'])[indices].squeeze()
          event_ids = np.array(hy['event_ids'])[indices].squeeze()
          energies = np.array(hy['energies'])[indices].squeeze()
          directions = np.array(hy['directions'])[indices].squeeze()
          positions = np.array(hy['positions'])[indices].squeeze()
          angles = math.angles_from_direction(directions)
          towall = math.towall(positions, angles, tank_axis = 2)
          dwalls = math.dwall(positions)

          nhits_cut = settings.nhitsCut
          towall_cut = settings.towallCut
          dwall_cut = settings.dwallCut

          ranges = range_from_energy(energies, labels)

          #Apply cuts (1 / 3)   
          variables = [event_ids, rootfiles, preds, truth, labels, energies, directions, total_charge, towall, nhits, indices, dwalls, ranges]
          variables = [var[(nhits> nhits_cut)] for var in variables]
          event_ids, rootfiles, preds, truth, labels, energies, directions, total_charge, towall, nhits, indices, dwalls, ranges = variables

          #Apply cuts (2 / 3)  Extract ML events that are also in fitqun 
          ml_hash = fq.get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)
          intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, return_indices=True)
          variables = [preds, truth, labels, energies, directions, total_charge, towall, nhits, indices, dwalls, ranges]
          variables = [var[comm2] for var in variables]
          preds, truth, labels, energies, directions, total_charge, towall, nhits, indices, dwalls, ranges = variables

     cheThr = list(map(get_cherenkov_threshold, labels))
     visible_energy = energies - cheThr


     print(f"PARTICLE LABEL: {settings.particleLabel}")
     
     particle_nhits_towall_dwall_mask = (labels==settings.particleLabel) & (nhits > nhits_cut) & (towall > towall_cut) & (dwalls > dwall_cut)

     print('events that would be excluded by fully contained cut', len(towall) - np.sum((labels==settings.particleLabel) & (towall > 2*ranges)))
     print('events that would be excluded by particle_nhits_towall_dwall_cut', len(towall) -  np.sum(particle_nhits_towall_dwall_mask))

     # Apply cuts (3 / 3) 
     if settings.fullyContainedCut:
          print('fully contained applied')
          particle_nhits_towall_dwall_mask = particle_nhits_towall_dwall_mask & (towall > 2 * ranges)
     variables = [preds, truth, directions, total_charge, visible_energy, indices, towall, nhits, dwalls, ranges]
     variables = [var[particle_nhits_towall_dwall_mask] for var in variables]
     preds, truth, directions, total_charge, visible_energy, indices, towall, nhits, dwalls, ranges = variables

     correction = 1
     if "positions" in target or "directions" in target:
          pred_x = preds[:,0]*correction 
          pred_y = preds[:,1]*correction
          pred_z = preds[:,2]*correction 

          truth_x = truth[:,0]*correction 
          truth_y = truth[:,1]*correction 
          truth_z = truth[:,2]*correction 
          truth_0 = np.stack((truth_x, truth_y, truth_z), axis=1)
          pred_0 = np.stack((pred_x, pred_y, pred_z), axis=1)
     if "energies" in target or "momenta" in target:
          truth_0 = np.ravel(truth)
          pred_0 = np.ravel(preds)

     flag_output_residuals = 0
     if flag_output_residuals and target == 'momenta':
          export_mom_residuals(truth_0, pred_0, rootfiles, event_ids, directions, total_charge, visible_energy, towall, nhits, indices, settings)
    

     vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
     single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
     multi_analysis = {}
     # comment out below to speed up analyze multiple regression
     bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, variable=towall)
     multi_analysis["towall"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=visible_energy)
     multi_analysis["ve"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=total_charge)
     multi_analysis["tot_charge"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     return single_analysis, multi_analysis 


def save_residual_residual_plot(settings, targets=['positions', 'momenta'], axes=['Longitudinal', 'Global'], ml_paths=None, plotsavepath=None):
     """
     Saves a residual-vs-residual scatter plot by looking at 2 residual vectors assocaited to 2 different regression targets.

     Params
     --------------
     settings: 
     targets: list of 2 strings
     axes: list of 2 strings
     ml_paths: list of 2 strings
     plotsavepath: str
     """
     
     #First argument is where to save plot
     #Second one where to get data

     print('combining two regression results from ', ml_paths)

     
     
     res_res_list = []

     for (files, target, v_axis) in zip(ml_paths, targets, axes):

          ml_combine_path = settings.inputPath
          hy = h5py.File(ml_combine_path+'combine_combine.hy', "r")

          preds = np.load(files+'predicted_'+str(target)+'.npy')
          truth = np.load(files+target+'.npy')
          labels = np.load(files + 'labels.npy')
          indices = np.load(files + 'indices.npy')

          # print('truth shape', truth.shape)
          # print('preds shape', preds.shape)
          # print('indices shape', indices.shape)

          # calculate number of hits 
          events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
          nhits = (events_hits_index[indices+1] - events_hits_index[indices]).squeeze()
          total_charge = np.array([part.sum() for part in np.split(hy['hit_charge'], np.cumsum(nhits))[:-1]])
          #total_charge_2 = np.add.reduceat(hy['hit_charge'], np.cumsum(nhits)[:-1])
          rootfiles = np.array(hy['root_files'])[indices].squeeze()
          event_ids = np.array(hy['event_ids'])[indices].squeeze()
          energies = np.array(hy['energies'])[indices].squeeze()
          directions = np.array(hy['directions'])[indices].squeeze()
          positions = np.array(hy['positions'])[indices].squeeze()
          angles = math.angles_from_direction(directions)
          towall = math.towall(positions, angles, tank_axis = 2)

          nhits_cut = 200

          # print('cut nhits', np.sum(nhits> nhits_cut))

          # print('truth shape', truth.shape)
          # print('preds shape', preds.shape)
          # print('indices shape', indices.shape)
          # print('nhits > nhits_cut', (nhits> nhits_cut).shape)

          #Apply cuts
          event_ids = event_ids[(nhits> nhits_cut)]
          rootfiles = rootfiles[(nhits> nhits_cut)]
          preds = preds[(nhits> nhits_cut)]
          truth = truth[(nhits> nhits_cut)]
          labels = labels[(nhits> nhits_cut)]
          energies = energies[(nhits> nhits_cut)]
          directions = directions[(nhits> nhits_cut)]
          total_charge = total_charge[(nhits> nhits_cut)]
          towall = towall[(nhits> nhits_cut)]
          
          # do this before nhits gets changed
          indices = indices[(nhits> nhits_cut)]
          
          nhits = nhits[(nhits> nhits_cut)]

          # print('truth shape', truth.shape)
          # print('preds shape', preds.shape)
          # print('indices shape', indices.shape)
          # print('nhits > nhits_cut', (nhits> nhits_cut).shape)

          



          cheThr = list(map(get_cherenkov_threshold, labels))
          visible_energy = energies - cheThr

          print(f"PARTICLE LABEL: {settings.particleLabel}")
          preds = preds[labels==settings.particleLabel]
          truth = truth[labels==settings.particleLabel]
          directions = directions[labels==settings.particleLabel]
          total_charge = total_charge[labels==settings.particleLabel]
          visible_energy = visible_energy[labels==settings.particleLabel]
          towall = towall[labels==settings.particleLabel]
          nhits = nhits[labels==settings.particleLabel]

          indices = indices[labels==settings.particleLabel]

          # print('cut for truth', labels==settings.particleLabel)
          # print('particle label', settings.particleLabel)
          # print('truth after', truth)
          # print('a', preds[:,0].shape)
          # print('b', truth[:,0].shape)

          correction = 1

          if "positions" in target or "directions" in target:
               pred_x = preds[:,0]*correction 
               pred_y = preds[:,1]*correction
               pred_z = preds[:,2]*correction 

               truth_x = truth[:,0]*correction 
               truth_y = truth[:,1]*correction 
               truth_z = truth[:,2]*correction 
               truth_0 = np.stack((truth_x, truth_y, truth_z), axis=1)
               pred_0 = np.stack((pred_x, pred_y, pred_z), axis=1)
          if "energies" in target or "momenta" in target:
               truth_0 = np.ravel(truth)
               pred_0 = np.ravel(preds)

          print('truth_0', truth_0)
          print('truth_0 shape', truth_0.shape)
          
          # print('energies shape', energies.shape)
          # print('visible energies shape', visible_energy.shape)
          
          
          # residuals along Vertex Axis (v_a)
          residuals = compute_residuals(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, plot_path = settings.outputPlotPath, v_axis=v_axis)
          print('residuals', residuals.shape)
          print('indices', indices.shape)

          print('residuals mean', np.mean(residuals))
          print('residaul std', np.std(residuals))

          sorted_indices = np.argsort(indices)
          sorted_res = residuals[sorted_indices]

          print("first 10 residuals:", sorted_res[:10])
          print("first 10 indices:  ", indices[sorted_indices][:10])

          res_res_list.append(sorted_res)
     
     # fig, ax = plt.subplots()
     # ax.scatter(res_res_list[0], res_res_list[1], s= 0.1)
     # ax.set_xlabel(f'{targets[0]} residuals for {axes[0]} axis')
     # ax.set_ylabel(f'{targets[1]} residuals for {axes[1]} axis')
     # ax.set_title('Residual vs Residual Plot (Corrrelation coeff = ' + str(round(np.corrcoef(res_res_list[0], res_res_list[1])[0,1], 4)) + ')')
     # # ax.set_ybound([-150, 150])
     # fig.savefig(settings.outputPlotPath + f'scatter_{v_axis}_axis_res_res_{targets[0]}_{targets[1]}.png')
     # print(f'Saved residual vs residual plot')

     
     df = pd.DataFrame({
          f'{targets[0]} residuals for {axes[0]} axis': res_res_list[0],
          f'{targets[1]} residuals for {axes[1]} axis': res_res_list[1]
     })

     # joint_plot = sns.jointplot(data=df, x=f'{targets[0]} residuals for {axes[0]} axis', y=f'{targets[1]} residuals for {axes[1]} axis', kind='scatter', 
     #                       marginal_kws=dict(bins=50, fill=True))
     # plt.savefig(settings.outputPlotPath + f'{v_axis}_axis_res_res_{targets[0]}_{targets[1]}.png')
     # plt.clf()

     joint_plot_heat = sns.jointplot(data=df,
                                     x=f'{targets[0]} residuals for {axes[0]} axis', 
                                     y=f'{targets[1]} residuals for {axes[1]} axis', kind='hist', 
                                     marginal_kws=dict(bins=5000, fill=True),
                                     color='blue')
     
     # plt.xlabel()
     plt.ylabel(f'{targets[1]} residuals for {axes[1]} axis')
     plt.ylim([-0.05, .05])
     plt.xlim([-15, 15])
     plt.savefig(settings.outputPlotPath + f'{v_axis}_axis_res_res_{targets[0]}_{targets[1]}_hist_zoom_03.png')

     corr = np.corrcoef(res_res_list[0], res_res_list[1]) #mom_res: momentum residuals, #pos_res: position residuals
     print('corr=', corr)


     return


def save_residual_plot(settings, feature_name='energy', v_axis='Longitudinal'):

     #First argument is where to save plot
     #Second one where to get data
     files = settings.mlPath
     target = str(settings.target)
     preds = np.load(files+'predicted_'+target+'.npy')
     truth = np.load(files+target+'.npy')
     labels = np.load(files + 'labels.npy')

     # print('loaded truth', truth)

     ml_combine_path = settings.inputPath
     hy = h5py.File(ml_combine_path+'combine_combine.hy', "r")
     indices = np.load(files + 'indices.npy')



     # calculate number of hits 
     events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
     nhits = (events_hits_index[indices+1] - events_hits_index[indices]).squeeze()
     total_charge = np.array([part.sum() for part in np.split(hy['hit_charge'], np.cumsum(nhits))[:-1]])
     #total_charge_2 = np.add.reduceat(hy['hit_charge'], np.cumsum(nhits)[:-1])
     rootfiles = np.array(hy['root_files'])[indices].squeeze()
     event_ids = np.array(hy['event_ids'])[indices].squeeze()
     energies = np.array(hy['energies'])[indices].squeeze()
     directions = np.array(hy['directions'])[indices].squeeze()
     positions = np.array(hy['positions'])[indices].squeeze()
     angles = math.angles_from_direction(directions)
     towall = math.towall(positions, angles, tank_axis = 2)

     nhits_cut = 200

     # print('cut nhits', np.sum(nhits> nhits_cut))

     #Apply cuts
     event_ids = event_ids[(nhits> nhits_cut)]
     rootfiles = rootfiles[(nhits> nhits_cut)]
     preds = preds[(nhits> nhits_cut)]
     truth = truth[(nhits> nhits_cut)]
     labels = labels[(nhits> nhits_cut)]
     energies = energies[(nhits> nhits_cut)]
     directions = directions[(nhits> nhits_cut)]
     total_charge = total_charge[(nhits> nhits_cut)]
     towall = towall[(nhits> nhits_cut)]
     nhits = nhits[(nhits> nhits_cut)]



     cheThr = list(map(get_cherenkov_threshold, labels))
     visible_energy = energies - cheThr


     # print('truth', truth)

     print(f"PARTICLE LABEL: {settings.particleLabel}")
     preds = preds[labels==settings.particleLabel]
     truth = truth[labels==settings.particleLabel]
     directions = directions[labels==settings.particleLabel]
     total_charge = total_charge[labels==settings.particleLabel]
     visible_energy = visible_energy[labels==settings.particleLabel]
     towall = towall[labels==settings.particleLabel]
     nhits = nhits[labels==settings.particleLabel]

     # print('cut for truth', labels==settings.particleLabel)
     # print('particle label', settings.particleLabel)
     # print('truth after', truth)
     # print('a', preds[:,0].shape)
     # print('b', truth[:,0].shape)

     correction = 1

     if "positions" in target or "directions" in target:
          pred_x = preds[:,0]*correction 
          pred_y = preds[:,1]*correction
          pred_z = preds[:,2]*correction 

          truth_x = truth[:,0]*correction 
          truth_y = truth[:,1]*correction 
          truth_z = truth[:,2]*correction 
          truth_0 = np.stack((truth_x, truth_y, truth_z), axis=1)
          pred_0 = np.stack((pred_x, pred_y, pred_z), axis=1)
     if "energies" in target or "momenta" in target:
          truth_0 = np.ravel(truth)
          pred_0 = np.ravel(preds)

     # print('truth_0', truth_0)
     # print('truth_0 shape', truth_0.shape)
     # print('energies shape', energies.shape)
     # print('visible energies shape', visible_energy.shape)
     
     # print('energies mask shape', energies[labels==settings.particleLabel].shape)
     
     # residuals along Vertex Axis (v_a)
     v_a_residuals = compute_residuals(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, plot_path = settings.outputPlotPath, v_axis=v_axis)
     
     fig, ax = plt.subplots()
     if feature_name == 'energy':
          feature = energies[labels==settings.particleLabel]
     elif feature_name == 'visible energy':
          feature = visible_energy
     elif feature_name == 'directions':
          feature = directions
     elif feature_name == 'total_charge':
          feature = total_charge
     elif feature_name == 'towall':
          feature = towall
     elif feature_name == 'nhit':
          feature = nhits
     
     
     ax.scatter(feature, v_a_residuals, s= 0.1)
     ax.set_xlabel(feature_name)
     ax.set_ylabel(f'Residual Along {v_axis} Axis')
     corr = '(Correlaion coefficient = ' + str(round(np.corrcoef(feature, v_a_residuals)[0,1], 4)) + ')'
     ax.set_title(f'Residual ({v_axis}) vs {feature_name}')
     # ax.set_ybound([-150, 150])
     fig.savefig(settings.outputPlotPath + f'scatter_{v_axis}_axis_residual_vs_{feature_name}.png')
     print(f'Saved residual plots {v_axis} vs {feature_name}')

     return
     # vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, plot_path = settings.outputPlotPath)
     # single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
     # multi_analysis = {}
     # bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, variable=towall)
     # multi_analysis["towall"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     # bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=visible_energy)
     # multi_analysis["ve"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     # bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=total_charge)
     # multi_analysis["tot_charge"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     return single_analysis , multi_analysis 