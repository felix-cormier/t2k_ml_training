import numpy as np

import sys

import matplotlib
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from plotting import regression_analysis, regression_analysis_perVar, regression_analysis_output
from analyze_output.utils.math import get_cherenkov_threshold

import WatChMaL.analysis.utils.fitqun as fq
import WatChMaL.analysis.utils.math as math

from runner_util import range_from_energy

import h5py

def gaussian(x, a, mean, sigma):
     return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

#def analyze_ml_regression(input_path, target, ml_path, output_plot_path, label, fitqun_path=None):
def analyze_ml_regression(settings, dir_f, tw_f):

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
     zenith = np.cos(angles[:,0]) 
     azimuth = angles[:,1]*180/np.pi 
     towall = math.towall(positions, angles, tank_axis = 2)
     dwall = math.dwall(positions, tank_axis = 2)
     # calculate number of hits 
     events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
     nhits = (events_hits_index[indices+1] - events_hits_index[indices]).squeeze()

     nhits_cut = 200



     if settings.doCombination:
          (_, fq_labels, _, fitqun_hash), (mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom), fq_truth, nhits_fq, qtot_fq = fq.read_fitqun_file(settings.fitqunPath+'fitqun_combine.hy', regression=True)
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
          zenith = np.cos(angles[:,0]) 
          azimuth = angles[:,1]*180/np.pi 
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
          angles = angles[(nhits> nhits_cut)]
          positions = positions[(nhits> nhits_cut)]
          total_charge = total_charge[(nhits> nhits_cut)]
          towall = towall[(nhits> nhits_cut)]
          zenith = zenith[(nhits> nhits_cut)]
          azimuth = azimuth[(nhits> nhits_cut)]
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
               angles = angles[comm2]
               positions = positions[comm2]
               total_charge = total_charge[comm2]
               towall = towall[comm2]
               zenith = zenith[comm2]
               azimuth = azimuth[comm2]
               dwall = dwall[comm2]
               nhits = nhits[comm2]


     cheThr = list(map(get_cherenkov_threshold, labels))
     visible_energy = energies - cheThr
     ve_cut = 1000
     min_ve_cut = 30
     towall_cut = 150
     ranges = range_from_energy(energies, labels)
     is_fully_contained = towall > ranges
     print(f"Only looking at ML events < {ve_cut} MeV")

     print(f"PARTICLE LABEL: {settings.particleLabel}")
     print(f"ML LABELS: {np.unique(labels,return_counts=True)}")
     temp_visible_energy = np.copy(visible_energy)
     temp_towall = np.copy(towall)

     quality_cuts =  (visible_energy < ve_cut) & (nhits > nhits_cut) & (towall > towall_cut) & (is_fully_contained) & (visible_energy > min_ve_cut)

     preds = preds[(labels==settings.particleLabel) & quality_cuts ]
     truth = truth[(labels==settings.particleLabel) & quality_cuts ]
     directions = directions[(labels==settings.particleLabel) & quality_cuts ]
     angles = angles[(labels==settings.particleLabel) & quality_cuts ]
     positions = positions[(labels==settings.particleLabel) & quality_cuts ]
     total_charge = total_charge[(labels==settings.particleLabel) & quality_cuts ]
     dwall = dwall[(labels==settings.particleLabel) & quality_cuts ]
     towall = towall[(labels==settings.particleLabel) & quality_cuts ]
     azimuth = azimuth[(labels==settings.particleLabel) & quality_cuts ]
     zenith = zenith[(labels==settings.particleLabel) & quality_cuts ]

     visible_energy = visible_energy[(labels==settings.particleLabel) & quality_cuts ]
     nhits = nhits[(labels==settings.particleLabel) & quality_cuts ]

     #For stopMu matching
     if "stopMu" in settings.fitqunPath:
          int_dw, comm1_dw, comm2_dw = np.intersect1d(directions[:,0], dir_f[:,0], return_indices=True)
          int_tw, comm1_tw, comm2_tw = np.intersect1d(directions[:,1], dir_f[:,1], return_indices=True)
          int_comm, comm1_comm, comm2_comm= np.intersect1d(comm1_dw, comm1_tw, return_indices=True)
          print(f"INTERSECT OF FQ AND ML: len of ML ({len(directions[:,0])}), len of FQ ({len(dir_f[:,0])}), dw: {len(int_dw)}, tw: {len(int_tw)}, int comm: {len(int_comm)}")
          #preds = preds[comm1_dw[comm1_comm]]
          #truth = truth[comm1_dw[comm1_comm]]
          #directions = directions[comm1_dw[comm1_comm]]
          #total_charge = total_charge[comm1_dw[comm1_comm]]
          #dwall = dwall[comm1_dw[comm1_comm]]
          #towall = towall[comm1_dw[comm1_comm]]
          #visible_energy = visible_energy[comm1_dw[comm1_comm]]
          #nhits = nhits[comm1_dw[comm1_comm]]

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


     vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath, analysis_var_names=['towall', 've', 'z'], analysis_vars=[towall, visible_energy, positions[:,2]])
     single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
     multi_analysis = {}
     output_analysis = {}
     if settings.doVarPlots:
          if not settings.getfiTQunTruth:
               bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, variable=dwall, bins_min=0, bins_max=1600, bins_num=32)
               multi_analysis["dwall"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
          bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir = directions, target=target, extra_string="ML_"+settings.plotName, save_plots=False, variable=towall, bins_min=150, bins_max=1600, bins_num=30)
          multi_analysis["towall"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
          bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=visible_energy, bins_min=0, bins_max=1000, bins_num=20)
          multi_analysis["ve"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
          bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=positions[:,2], bins_min=-1810, bins_max=1810, bins_num=36)
          multi_analysis["z"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
          bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=zenith, bins_min=-1, bins_max=1, bins_num=20)
          multi_analysis["zenith"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
          bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=azimuth, bins_min=-180, bins_max=180, bins_num=36)
          multi_analysis["azimuth"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
          #bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=truth_0, pred=pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, variable=total_charge)
          #multi_analysis["tot_charge"] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     if settings.doOutputPlots:
          if "positions" in target:
               #X
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0[:,0], pred_0[:,0], dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-1650, bins_max=1650, bins_num=36)
               output_analysis["X"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #Y
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0[:,1], pred_0[:,1], dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-1650, bins_max=1650, bins_num=36)
               output_analysis["Y"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #Z
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0[:,2], pred_0[:,2], dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=1650, bins_max=1820, bins_num=34)
               output_analysis["Z"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #dwall
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(math.towall(truth_0, angles, tank_axis = 2), math.towall(pred_0, angles, tank_axis = 2), dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=0, bins_max=1600, bins_num=32)
               output_analysis["dwall"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #towall
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(math.dwall(truth_0, tank_axis=2), math.dwall(pred_0, tank_axis=2), dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=0, bins_max=2000, bins_num=40)
               output_analysis["towall"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
          if "momenta" in target:
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0, pred_0, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=30, bins_max=1000, bins_num=50)
               output_analysis["momenta"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
          if "directions" in target:
               #X
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0[:,0], pred_0[:,0], dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-1., bins_max=1., bins_num=40)
               output_analysis["X"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #Y
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0[:,1], pred_0[:,1], dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-1., bins_max=1., bins_num=40)
               output_analysis["Y"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #Z
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(truth_0[:,2], pred_0[:,2], dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-1., bins_max=1., bins_num=40)
               output_analysis["Z"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               true_angles = math.angles_from_direction(truth_0)
               true_zenith = np.cos(true_angles[:,0]) 
               true_azimuth = true_angles[:,1]*180/np.pi 
               pred_angles = math.angles_from_direction(pred_0)
               pred_zenith = np.cos(pred_angles[:,0]) 
               pred_azimuth = pred_angles[:,1]*180/np.pi 
               #azimuth
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(true_azimuth, pred_azimuth, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-180, bins_max=180, bins_num=40)
               output_analysis["azimuth"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]
               #polar
               bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict = regression_analysis_output(true_zenith, pred_zenith, dir=directions, target=target,extra_string="ML_"+settings.plotName, save_plots=False, bins_min=-1., bins_max=1., bins_num=40)
               output_analysis["polar"] = [bin_dict, pred_dict, pred_error_dict, truth_dict, truth_error_dict]

     return single_analysis, multi_analysis, output_analysis 