import numpy as np

import sys

import matplotlib
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from plotting import regression_analysis, regression_analysis_perVar
from analyze_output.utils.math import get_cherenkov_threshold

import WatChMaL.analysis.utils.fitqun as fq
import WatChMaL.analysis.utils.math as math

from runner_util import range_from_energy

import h5py

def gaussian(x, a, mean, sigma):
     return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

#def analyze_ml_regression(input_path, target, ml_path, output_plot_path, label, fitqun_path=None):
def analyze_ml_regression(settings):

     #First argument is where to save plot
     #Second one where to get data
     files = settings.mlPath
     target = str(settings.target)
     preds = np.load(files+'predicted_'+target+'.npy')
     truth = np.load(files+target+'.npy')
     labels = np.load(files + 'labels.npy')

     ml_combine_path = settings.inputPath
     try:
          hy = h5py.File(ml_combine_path+'multi_combine.hy', "r")
     except FileNotFoundError:
          try:
               hy = h5py.File(ml_combine_path+'digi_combine.hy', "r")
          except FileNotFoundError:
               try:
                    hy = h5py.File(ml_combine_path+'combine_combine.hy', "r")
               except FileNotFoundError:
                    print(f"Could not find input file in {ml_combine_path}")
                    return 0

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
     dwall = math.dwall(positions, tank_axis = 2)
     # calculate number of hits 
     events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
     nhits = (events_hits_index[indices+1] - events_hits_index[indices]).squeeze()

     nhits_cut = 200



     if settings.doCombination:
          (_, fq_labels, _, fitqun_hash), (mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom), fq_truth = fq.read_fitqun_file(settings.fitqunPath+'fitqun_combine.hy', regression=True)
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
          dwall = math.dwall(positions, tank_axis = 2)

          nhits_cut = 200

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
          dwall = dwall[(nhits> nhits_cut)]
          nhits = nhits[(nhits> nhits_cut)]


          ml_hash = fq.get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)
          intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, return_indices=True)
          if not "stopMu" in settings.fitqunPath:
               preds = preds[comm2]
               truth = truth[comm2]
               labels = labels[comm2]
               energies = energies[comm2]
               directions = directions[comm2]
               total_charge = total_charge[comm2]
               towall = towall[comm2]
               dwall = dwall[comm2]
               nhits = nhits[comm2]


     cheThr = list(map(get_cherenkov_threshold, labels))
     visible_energy = energies - cheThr
     ve_cut = 1000
     towall_cut = 150
     ranges = range_from_energy(energies, labels)
     is_fully_contained = towall > ranges
     print(f"Only looking at ML events < {ve_cut} MeV")

     print(f"PARTICLE LABEL: {settings.particleLabel}")
     print(f"ML LABELS: {np.unique(labels,return_counts=True)}")
     temp_visible_energy = np.copy(visible_energy)
     temp_towall = np.copy(towall)

     quality_cuts =  (visible_energy < ve_cut) & (nhits > nhits_cut) & (towall > towall_cut) & (is_fully_contained) 

     preds = preds[(labels==settings.particleLabel) & quality_cuts ]
     truth = truth[(labels==settings.particleLabel) & quality_cuts ]
     directions = directions[(labels==settings.particleLabel) & quality_cuts ]
     total_charge = total_charge[(labels==settings.particleLabel) & quality_cuts ]
     dwall = dwall[(labels==settings.particleLabel) & quality_cuts ]
     towall = towall[(labels==settings.particleLabel) & quality_cuts ]

     visible_energy = visible_energy[(labels==settings.particleLabel) & quality_cuts ]
     nhits = nhits[(labels==settings.particleLabel) & quality_cuts ]

     #print(preds[:,0].shape)
     #print(truth[:,0].shape)

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


     print(f"PREDS: {pred_0}, truth: {truth_0}")
     vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
     single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
     print(f"SINGLE ML ANALYSIS: {single_analysis}")
     multi_analysis = {}
     bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, variable=dwall)
     multi_analysis["dwall"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, variable=towall)
     multi_analysis["towall"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=visible_energy)
     multi_analysis["ve"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
     #bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=total_charge)
     #multi_analysis["tot_charge"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     return single_analysis, multi_analysis 