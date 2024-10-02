import numpy as np

import os

import h5py

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


from scipy.optimize import curve_fit
from scipy.stats.mstats import mquantiles_cimj

import WatChMaL.analysis.utils.fitqun as fq

from tqdm import tqdm

import WatChMaL.analysis.utils.math as math

from plotting import regression_analysis, regression_analysis_perVar
from analyze_output.utils.math import get_cherenkov_threshold

from matplotlib.ticker import NullFormatter

from runner_util import range_from_energy


def check_fitqun_correlations(fq_pos, fq_dir, fq_mom, true_pos, true_dir, true_mom, label):

    
     total_magnitude_pred, longitudinal_component_pred, _ = math.decompose_along_direction(fq_pos[:,0:3], true_dir)
     total_magnitude_true, longitudinal_component_true, _ = math.decompose_along_direction(true_pos[:,0:3], true_dir)
     print(f"FQ mom: {fq_mom}")
     print(f"TRUE mom: {true_mom}")
     pos_res = longitudinal_component_true - longitudinal_component_pred
     mom_res = (true_mom - fq_mom)/true_mom
     print(f"POS RES: {pos_res}")
     print(f"MOM RES: {mom_res}")

     nullfmt = NullFormatter()         # no labels

     # definitions for the axes
     left, width = 0.1, 0.65
     bottom, height = 0.1, 0.65
     bottom_h = left_h = left + width + 0.02

     rect_scatter = [left, bottom, width, height]
     rect_histx = [left, bottom_h, width, 0.2]
     rect_histy = [left_h, bottom, 0.2, height]

     # start with a rectangular Figure
     plt.figure(1, figsize=(8, 8))

     axScatter = plt.axes(rect_scatter)
     axHistx = plt.axes(rect_histx)
     axHisty = plt.axes(rect_histy)

     # no labels
     axHistx.xaxis.set_major_formatter(nullfmt)
     axHisty.yaxis.set_major_formatter(nullfmt)


     axScatter.hist2d(mom_res,pos_res, bins=(100,100), range=((-0.08,0.08),(-30,30)))
     #plt.hist2d(mom_res,pos_res, bins=(100,100), range=((-0.5,0.5),(-200,200)))
     axScatter.set_xlabel("fiTQun Momentum Residual")
     axScatter.set_ylabel("fiTQun Longitudinal Position Residual")

     axHistx.hist(mom_res, bins=100, range=(-0.08,0.08))
     axHisty.hist(pos_res, bins=100, range=(-30,30), orientation='horizontal')


     plt.savefig("outputs/fq_true_correlations_trueDir_electrons.png")

     corr = np.corrcoef(mom_res, pos_res)

     print(f"Correlation: {corr}")


def analyze_fitqun_regression(settings):
     '''
     Plot fiTQun specific regression results using un_normalize(), regression_analysis(), and read_fitqun_file().
     Args:
         hy_path (str, optional): directory where fitqun_combine.hy and combine_combine.hy files are located. 
         true_path (str, optional): directory where true_positions.npy file is located.
     Returns:
         None
     '''
     # get values out of fitqun file, where mu_1rpos and e_1rpos are the positions of muons and electrons respectively
     if settings.getfiTQunTruth:
         (_, labels, _, fitqun_hash), (mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom), fq_truth, nhits = fq.read_fitqun_file(settings.fitqunPath+'/fitqun_combine.hy', regression=True, fq_truth=settings.getfiTQunTruth)
     else:
         (_, labels, _, fitqun_hash), (mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom), fq_truth = fq.read_fitqun_file(settings.fitqunPath+'/fitqun_combine.hy', regression=True, fq_truth=settings.getfiTQunTruth)

     # read in the indices file
     idx = np.array(sorted(np.load(settings.mlPath + "/indices.npy")))

     # read in the main HDF5 file that has the rootfiles and event_ids
     nhits_cut=200
     try:
          hy = h5py.File(settings.inputPath+'/multi_combine.hy', "r")
     except FileNotFoundError:
          try:
               hy = h5py.File(settings.inputPath+'/digi_combine.hy', "r")
          except FileNotFoundError:
               try:
                    hy = h5py.File(settings.inputPath+'/combine_combine.hy', "r")
               except FileNotFoundError:
                    print(f"Could not find input file in {settings.inputPath}")
                    return 0

     if settings.getfiTQunTruth:
        print("Get fiTQun truth")
        positions = np.array(fq_truth[0]).squeeze()
        directions = np.array(fq_truth[1]).squeeze()
        momenta = np.array(fq_truth[2]).squeeze()
        labels = labels.squeeze() 
        labels[labels==5] = 0
        labels[labels==6] = 0
        energies = math.energy_from_momentum(momenta, labels)
        print(f"INITIAL MOMENTA: {momenta}")
        print(f"INITIAL ENERGIES: {energies}")
        print(f"ENERGIES: {np.amin(energies)}, {np.amax(energies)}")
        #TEMPORARY
     else:
        positions = np.array(hy['positions'])[idx].squeeze()
        directions = np.array(hy['directions'])[idx].squeeze()
        energies = np.array(hy['energies'])[idx].squeeze()
        labels = np.array(hy['labels'])[idx].squeeze()
        print(f"OG Labels: {np.unique(labels, return_counts=True)}")
        momenta = np.ones(energies.shape[0])
        momenta[labels == 1] = np.sqrt(np.multiply(energies[labels==1], energies[labels==1]) - np.multiply(momenta[labels==1]*0.5,momenta[labels==1]*0.5))
        #momenta = np.sqrt(np.multiply(energies, energies) - np.multiply(momenta*0.5,momenta*0.5))
        momenta[labels == 0] = np.sqrt(np.multiply(energies[labels==0], energies[labels==0]) - np.multiply(momenta[labels==0]*105.7,momenta[labels==0]*105.7))
        momenta[labels == 2] = np.sqrt(np.multiply(energies[labels==2], energies[labels==2]) - np.multiply(momenta[labels==2]*139.584,momenta[labels==2]*139.584))
        # calculate number of hits 
        events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
        nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()
        rootfiles = np.array(hy['root_files'])[idx].squeeze()
        event_ids = np.array(hy['event_ids'])[idx].squeeze()
        event_ids = event_ids
        rootfiles = rootfiles
        ml_hash = fq.get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)
        intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, return_indices=True)
        energies = energies


     angles = math.angles_from_direction(directions)
     towall = math.towall(positions, angles, tank_axis = 2)
     dwall = math.dwall(positions, tank_axis = 2)

     ranges = range_from_energy(energies, labels)
     is_fully_contained = towall >  ranges





     #Apply cuts
     #positions = positions[(nhits> nhits_cut)]
     #directions = directions[(nhits> nhits_cut)]
     #labels = labels[(nhits> nhits_cut)]
     #momenta = momenta[(nhits> nhits_cut)]
     #towall = towall[(nhits> nhits_cut)]
     #nhits = nhits[(nhits> nhits_cut)]

     towall_nan = np.argwhere(~np.isnan(towall))
     #towall_nan = np.ones(len(towall),dtype=bool)
     print(f"TOWALL: {np.amin(towall[towall_nan])}, {np.amax(towall[towall_nan])}, {np.isnan(towall[towall_nan])}, {towall}")



     # load in the true positions 
     # Not necessary in current config
     '''
     true_target = np.load(npy_path + target+".npy")
     print('true_'+target+'.shape =', true_target.shape)

     # unnormalize the true_positions
     if 'positions' in target:
        tp = []
        for t in true_target:
            tp.append(un_normalize(t))
        true_target = np.array(tp)

     print('true_'+target+' =', true_target)
     print('mu_1rpos =', mu_1rpos)
     print('e_1rpos =', e_1rpos)
     '''

     # use intersect1d to find the intersection of fitqun_hash and ml_hash, specifically, intersect is a 
     # sorted 1D array of common and unique elements, comm1 is the indices of the first occurrences of 
     # the common values in fitqun_hash, comm2 is the indices of the first occurrences of the common values 
     # in ml_hash. So we use comm1 to index fitqun_hash and comm2 to index ml_hash.
     if not "stopMu" in settings.fitqunPath:
        print(f"UNIQUE IN COMM2: {np.unique(comm2,return_counts=True)}")
        print(f"UNIQUE IN COMM1: {np.unique(comm1,return_counts=True)}")
        fitqun_labels = labels[comm2]
        fitqun_mu_1rpos = mu_1rpos[comm1].squeeze() 
        fitqun_e_1rpos = e_1rpos[comm1].squeeze() 
        fitqun_pi_1rpos = pi_1rpos[comm1].squeeze() 
        fitqun_mu_1rdir = mu_1rdir[comm1].squeeze() 
        fitqun_e_1rdir = e_1rdir[comm1].squeeze() 
        fitqun_pi_1rdir = pi_1rdir[comm1].squeeze() 
        fitqun_mu_1rmom = mu_1rmom[comm1].squeeze() 
        fitqun_e_1rmom = e_1rmom[comm1].squeeze() 
        fitqun_pi_1rmom = pi_1rmom[comm1].squeeze() 
        positions = positions[comm2]
        directions = directions[comm2]
        momenta = momenta[comm2]
        energies = energies[comm2]
        towall = towall[comm2]
        dwall = dwall[comm2]
        nhits = nhits[comm2]
        is_fully_contained = is_fully_contained[comm2]
     else:
        fitqun_labels = labels[towall_nan].squeeze()
        print(f"Fitqun Labels: {fitqun_labels}")
        fitqun_mu_1rpos = mu_1rpos[towall_nan].squeeze() 
        fitqun_e_1rpos = e_1rpos[towall_nan].squeeze() 
        fitqun_pi_1rpos = pi_1rpos[towall_nan].squeeze() 
        fitqun_mu_1rdir = mu_1rdir[towall_nan].squeeze() 
        fitqun_e_1rdir = e_1rdir[towall_nan].squeeze() 
        fitqun_pi_1rdir = pi_1rdir[towall_nan].squeeze() 
        fitqun_mu_1rmom = mu_1rmom[towall_nan].squeeze() 
        fitqun_e_1rmom = e_1rmom[towall_nan].squeeze() 
        fitqun_pi_1rmom = pi_1rmom[towall_nan].squeeze() 
        positions = positions[towall_nan].squeeze()
        directions = directions[towall_nan].squeeze()
        momenta = momenta[towall_nan].squeeze()
        energies = energies[towall_nan].squeeze()
        angles = angles[towall_nan].squeeze()
        towall = towall[towall_nan].squeeze()
        dwall = dwall[towall_nan].squeeze()
        nhits = nhits[towall_nan].squeeze()
        is_fully_contained = is_fully_contained[towall_nan].squeeze()
     cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
     visible_energy = energies - cheThr
     ve_cut = 1000
     towall_cut = 150

     #positions = positions[(visible_energy< ve_cut)]
     #directions = directions[(visible_energy< ve_cut)]
     #fitqun_labels = fitqun_labels[(visible_energy< ve_cut)]
     #momenta = momenta[(visible_energy< ve_cut)]
     #towall = towall[(visible_energy< ve_cut)]
     #nhits = nhits[(visible_energy< ve_cut)]



     #print(f"FQ LABELS: {np.unique(fitqun_labels, return_counts=True)}")
     #check_fitqun_correlations(fitqun_e_1rpos[fitqun_labels==1], fitqun_e_1rdir[fitqun_labels==1], fitqun_e_1rmom[fitqun_labels==1], positions[fitqun_labels==1], directions[fitqun_labels==1], momenta[fitqun_labels==1], 1)
     old_visible_energy = visible_energy.copy()
     old_nhits = nhits.copy()
     old_towall = towall.copy()

     print(np.unique(old_nhits > nhits_cut, return_counts=True))
     print(np.unique(old_visible_energy < ve_cut, return_counts=True))
     print(np.unique(towall > towall_cut, return_counts=True))
     print(np.unique(is_fully_contained,return_counts=True))

     quality_cuts = (old_visible_energy < ve_cut) & (old_nhits > nhits_cut) & (towall > towall_cut) & (is_fully_contained)
     print(np.unique(quality_cuts,return_counts=True))

     if "positions" in settings.target:
         truth = positions[quality_cuts]
         fitqun_mu = fitqun_mu_1rpos[quality_cuts]
         fitqun_e = fitqun_e_1rpos[quality_cuts]
         fitqun_pi = fitqun_pi_1rpos[quality_cuts]
     elif "directions" in settings.target:
        truth = directions[quality_cuts]
        fitqun_mu = fitqun_mu_1rdir[quality_cuts]
        fitqun_e = fitqun_e_1rdir[quality_cuts]
        fitqun_pi = fitqun_pi_1rdir[quality_cuts]
     elif "momentum" in settings.target or "momenta" in settings.target:
        truth = momenta[quality_cuts]
        fitqun_mu = fitqun_mu_1rmom[quality_cuts]
        fitqun_e = fitqun_e_1rmom[quality_cuts]
        fitqun_pi = fitqun_pi_1rmom[quality_cuts]

     fitqun_labels = fitqun_labels[quality_cuts]
     print(np.unique(fitqun_labels,return_counts=True))


     visible_energy = visible_energy[quality_cuts]
     nhits = nhits[quality_cuts]
     towall = towall[quality_cuts]
     dwall = dwall[quality_cuts]

     true_0, pred_0, ve_0, tw_0, dw_0, nhits_0, dir_0 = [], [], [], [], [], [], []
     true_1, pred_1, ve_1, tw_1, dw_1, nhits_1, dir_1 = [], [], [], [], [], [], []
     true_2, pred_2, ve_2, tw_2, dw_2, nhits_2, dir_2 = [], [], [], [], [], [], []
     for i in range(len(fitqun_labels)):
         # LABEL 0 - muons  
         if fitqun_labels[i] == 0:
             true_0.append(truth[i])
             pred_0.append(fitqun_mu[i])
             ve_0.append(visible_energy[i])
             tw_0.append(towall[i])
             dw_0.append(dwall[i])
             nhits_0.append(nhits[i])
             dir_0.append(directions[i])

         elif fitqun_labels[i] == 2:
             true_2.append(truth[i])
             pred_2.append(fitqun_pi[i])
             ve_2.append(visible_energy[i])
             tw_2.append(towall[i])
             dw_2.append(dwall[i])
             nhits_2.append(nhits[i])
             dir_2.append(directions[i])

         # LABEL 1 - electrons  
         else:
             true_1.append(truth[i])
             pred_1.append(fitqun_e[i])
             ve_1.append(visible_energy[i])
             tw_1.append(towall[i])
             dw_1.append(dwall[i])
             nhits_1.append(nhits[i])
             dir_1.append(directions[i])

     # convert lists to arrayss
     true_0 = np.array(true_0)
     true_1 = np.array(true_1)
     true_2 = np.array(true_2)
     pred_0 = np.array(pred_0)
     pred_1 = np.array(pred_1)
     pred_2 = np.array(pred_2)
     tw_0 = np.array(tw_0)
     tw_1 = np.array(tw_1)
     tw_2 = np.array(tw_2)
     dw_0 = np.array(dw_0)
     dw_1 = np.array(dw_1)
     dw_2 = np.array(dw_2)
     dir_0 = np.array(dir_0)
     dir_1 = np.array(dir_1)
     dir_2 = np.array(dir_2)
     print(f"FITQUN TRUTH: {true_0}")
     print(f"FITQUN PRED: {pred_0}")
     print(f"FITQUN VE: {np.array(ve_0)}")

     #print(true_0.shape)
     single_analysis = []
     multi_analysis = {}
     if settings.particleLabel==0:
        print('######## fiTQun MUON EVENTS ########')
        print(f"FITQUN TRUE: {true_0}")
        print(f"FITQUN PRED: {pred_0}")

        vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target=settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
        single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=dw_0)
        multi_analysis['dwall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=tw_0)
        multi_analysis['towall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=ve_0)
        multi_analysis['ve'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     if settings.particleLabel==1:
        print('######## fiTQun ELECTRON EVENTS ########')
        #regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, target = target, extra_string="fitqun_Electrons", save_plots=False, variable=ve_1)
        vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target=settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
        single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=dw_1)
        multi_analysis['dwall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=tw_1)
        multi_analysis['towall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=ve_1)
        multi_analysis['ve'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     if settings.particleLabel==2:
        print('######## fiTQun PION EVENTS ########')
        #regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, target = target, extra_string="fitqun_Electrons", save_plots=False, variable=ve_1)
        vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=true_2, pred=pred_2, dir=dir_2, target=settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
        single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_2, pred=pred_2, dir=dir_2, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=dw_2)
        multi_analysis['dwall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_2, pred=pred_2, dir=dir_2, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=tw_2)
        multi_analysis['towall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_2, pred=pred_2, dir=dir_2, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=ve_2)
        multi_analysis['ve'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     return single_analysis, multi_analysis
