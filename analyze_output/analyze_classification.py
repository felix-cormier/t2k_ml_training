import numpy as np

from analyze_output.utils.plotting import softmax_plots
from analyze_output.utils.math import get_cherenkov_threshold

import os

import h5py

from WatChMaL.analysis.classification import WatChMaLClassification
from WatChMaL.analysis.classification import plot_efficiency_profile, plot_rocs
from WatChMaL.analysis.utils.plotting import plot_legend
from WatChMaL.analysis.utils.binning import get_binning
from WatChMaL.analysis.utils.fitqun import read_fitqun_file, make_fitqunlike_discr, get_rootfile_eventid_hash, plot_fitqun_comparison
import WatChMaL.analysis.utils.math as math

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


from scipy.optimize import curve_fit
from scipy.stats.mstats import mquantiles_cimj

import WatChMaL.analysis.utils.fitqun as fq

from tqdm import tqdm

from math import log10, floor, ceil

def analyze_classification(settings):


    # retrieve test indices
    #idx = np.array(sorted(np.load(str(settings.mlPath) + "/indices.npy")))
    idx = np.array(np.load(str(settings.mlPath) + "/indices.npy"))
    #idx = np.unique(idx)
    print(f"Saved Indices {idx}")
    softmax = np.array(np.load(str(settings.mlPath) + "/softmax.npy"))
    labels_test = np.array(np.load(str(settings.mlPath) + "/labels.npy"))
    #positions_array = np.array(np.load(str(newest_directory) + "/pred_positions.npy"))
    #true_positions_array = np.array(np.load(str(newest_directory) + "/true_positions.npy"))

    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(settings.inputPath+"/digi_combine.hy", "r")
    print(hy["labels"].shape)
    angles = np.array(hy['angles'])[idx].squeeze() 
    labels = np.array(hy['labels'])[idx].squeeze() 
    veto = np.array(hy['veto'])[idx].squeeze()
    print(f"OG labels: {labels_test} ")
    energies = np.array(hy['energies'])[idx].squeeze()
    print(f"new labels: {labels}")
    positions = np.array(hy['positions'])[idx].squeeze()
    #positions=true_positions_array.squeeze()
    directions = math.direction_from_angles(angles)
    rootfiles = np.array(hy['root_files'])[idx].squeeze()
    event_ids = np.array(hy['event_ids'])[idx].squeeze()
    #positions_ml = positions_array.squeeze()

    # calculate number of hits 
    events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
    nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()
    total_charge = np.array([part.sum() for part in np.split(hy['hit_charge'], np.cumsum(nhits))[:-1]])

    # calculate additional parameters 
    towall = math.towall(positions, angles, tank_axis = 2)
    dwall = math.dwall(positions, tank_axis = 2)
    momentum = math.momentum_from_energy(energies, labels)
    ml_cheThr = list(map(get_cherenkov_threshold, labels))
    ml_visible_energy = energies - ml_cheThr


    #softmax_sig = [softmax[label] for label in settings.signalLabels]
    #softmax_bkg = [softmax[label] for label in settings.bkgLabels]

    softmax_e = softmax[labels_test==1]
    softmax_mu = softmax[(labels_test==0) & (ml_visible_energy < 1200)]
    softmax_pi = softmax[(labels_test==2) & (ml_visible_energy < 1200)]

    if settings.signalLabels==1:
        pass
        #softmax_plots([softmax_e[:,1], softmax_e[:,0]+softmax_e[:,2]], ['e-score', 'mu+pi-score'], extra_label='Electrons only', file_path=settings.outputPlotPath)
        #softmax_plots([softmax_mu[:,1], softmax_mu[:,0]+softmax_mu[:,2]], ['e-score', 'mu+pi-score'], extra_label='Muons only', file_path=settings.outputPlotPath)
        #softmax_plots([softmax_pi[:,1], softmax_pi[:,0]+softmax_pi[:,2]], ['e-score', 'mu+pi-score'], extra_label='Pi+ only', file_path=settings.outputPlotPath)

    if settings.signalLabels==0:
        softmax_plots([softmax_mu[:,0], softmax_mu[:,1]], ['mu-score', 'e-score'], extra_label='Muon only', file_path=settings.outputPlotPath, bins=50)
        softmax_plots([softmax_mu[:,0], softmax_mu[:,2]], ['mu-score', 'pi-score'], extra_label='Muon only', file_path=settings.outputPlotPath, bins=50)
        softmax_plots([softmax_mu[:,0]/softmax_mu[:,2]], ['mu-score over pi-score'], extra_label='Muon only', file_path=settings.outputPlotPath, bins=101, range=[0,100])

    #softmax_plots([np.log(np.divide(softmax_pi[:,0], softmax_pi[:,2])), np.log(np.divide(softmax_mu[:,0], softmax_mu[:,2]))], ['Pi+ only','Muons only'], extra_label='ln mu over pi', range=[-100,100], file_path=settings.outputPlotPath)


    #Save ids and rootfiles to compare to fitqun, after applying cuts
    ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)


    softmax_sum = np.sum(softmax,axis=1)
    print(f"SOFTMAX SUM: {np.amin(softmax_sum)}")




    do_fitqun=False
    if os.path.isfile(str(settings.fitqunPath) + "/fitqun_combine.hy") and settings.doFiTQun:
        print("Running fiTQun")
        do_fitqun=True
        fitqun_discr, fitqun_pi_discr, fitqun_labels, fitqun_mom, fitqun_hash = read_fitqun_file(str(settings.fitqunPath) + "/fitqun_combine.hy", plotting=False, regression=False)
        print(f'len idx: {len(idx)}, len fitqun: {len(fitqun_discr)}')
        fitqun_idx = np.array(range(len(fitqun_discr)))
        fitqun_hash = np.array(fitqun_hash)[fitqun_idx].squeeze()
        fitqun_discr = fitqun_discr[fitqun_idx].squeeze() 
        if int(settings.bkgLabels)==2:
            fitqun_pi_discr = fitqun_pi_discr[fitqun_idx].squeeze() 
        fitqun_labels = fitqun_labels[fitqun_idx].squeeze() 
        fitqun_mom = fitqun_mom[fitqun_idx].squeeze() 
        fitqun_energy = math.energy_from_momentum(fitqun_mom, fitqun_labels)
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_energy - fitqun_cheThr

        #Get the ids that are in both ML and fitqun samples
        intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, assume_unique=True, return_indices=True)
        print(f'intersect: {intersect.shape}, comm1: {comm1.shape}, comm2: {comm2.shape}')
        print(len(comm1))
        print(len(comm2))

        fitqun_matched_energies = energies[comm2]
        fitqun_total_charge = total_charge[comm2]
        fitqun_dwall = dwall[comm2]
        fitqun_az = (angles[:,1]*180/np.pi)[comm2]
        fitqun_polar = np.cos(angles[:,0])[comm2] 
        fitqun_towall = towall[comm2]
        fitqun_discr = fitqun_discr[comm1]
        fitqun_pi_discr = fitqun_pi_discr[comm1]
        fitqun_labels = fitqun_labels[comm1]
        fitqun_idx = fitqun_idx[comm2]
        fitqun_mom = momentum[comm2]
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_matched_energies - fitqun_cheThr

        temp = np.abs(fitqun_labels[fitqun_towall > 100]-fitqun_discr[fitqun_towall > 100])

        print(f"fitqun e- avg towall > 100): {1-np.sum(temp[fitqun_labels[fitqun_towall > 100]==1])/len(temp[fitqun_labels[fitqun_towall > 100]==1])}")
        print(f"fitqun mu- avg (towall > 100): {1-np.sum(temp[fitqun_labels[fitqun_towall > 100]==0])/len(temp[fitqun_labels[fitqun_towall > 100]==0])}")

    elif os.path.isfile(str(settings.fitqunPath) + "/fitqun_combine.hy") and settings.doFiTQun:
        print("Running fiTQun")
        fitqun_discr, fitqun_pi_discr, fitqun_labels, fitqun_mom, fitqun_hash, fitqun_truth = read_fitqun_file(str(settings.fitqunPath) + "/fitqun_combine.hy", plotting=False, regression=False, fq_truth=True)
        print(f'len idx: {len(idx)}, len fitqun: {len(fitqun_discr)}')
        fitqun_idx = np.array(range(len(fitqun_discr)))
        fitqun_hash = np.array(fitqun_hash)[fitqun_idx].squeeze()
        fitqun_discr = fitqun_discr[fitqun_idx].squeeze() 
        if 2 in settings.bkgLabels:
            fitqun_pi_discr = fitqun_pi_discr[fitqun_idx].squeeze() 
        #Geant 4 particle codes, 6 (5) are muons (anti-muons)
        fitqun_labels = fitqun_labels[fitqun_idx].squeeze() 
        fitqun_labels[fitqun_labels==5] = 0
        fitqun_labels[fitqun_labels==6] = 0

        fitqun_mom = fitqun_mom[fitqun_idx].squeeze() 
        fitqun_energy = math.energy_from_momentum(fitqun_mom, fitqun_labels)
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_energy - fitqun_cheThr

        fq_truth_pos = np.array(fitqun_truth[0]).squeeze()
        fq_truth_dir = np.array(fitqun_truth[1]).squeeze()
        fq_truth_mom = np.array(fitqun_truth[2]).squeeze()
        fq_truth_energy = math.energy_from_momentum(fq_truth_mom, fitqun_labels)
        fitqun_truth_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_truth_visible_energy = fq_truth_energy - fitqun_truth_cheThr

        fq_truth_angles = math.angles_from_direction(np.array(fq_truth_dir).squeeze(), zenith_axis=2)
        fq_truth_towall = math.towall(np.array(fq_truth_pos).squeeze(), np.array(fq_truth_angles).squeeze(), tank_axis = 2)

        #fitqun_hy_electrons = (fitqun_labels == 1)
        #fitqun_hy_muons = (fitqun_labels == 0)
        if int(settings.signalLabels) == 0:
            fitqun_hy_electrons = (fitqun_labels == 0)
            fitqun_hy_muons = (fitqun_labels == 2)
        elif int(settings.signalLabels) == 1:
            fitqun_hy_electrons = (fitqun_labels == 1)
            fitqun_hy_muons = (fitqun_labels == 0)
        fitqun_basic_cuts = ((fitqun_hy_electrons | fitqun_hy_muons) & (fq_truth_towall > 100) & (fitqun_truth_visible_energy < 1200) )
            
        fitqun_mom_binning = get_binning(fq_truth_mom, 11, minimum=100, maximum=1000)
        fitqun_ve_binning = get_binning(fitqun_truth_visible_energy, 11, minimum=100, maximum=1200)
        fitqun_towall_binning = get_binning(fq_truth_towall, 30, minimum=0, maximum=3000)

        stride1 = settings.mlPath
        fitqun_pi_run_result = [WatChMaLClassification(stride1, 'fiTQun', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        (fitqun_pi_run_result[0]).selection = fitqun_basic_cuts
        fitqun_pi_run_result[0].cut = fitqun_pi_discr.astype(bool)


        
    # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
    nhit_cut = nhits > 200 #25
    towall_cut = towall > 150
    ve_cut = ml_visible_energy < 1000
    # veto_cut = (veto == 0)
    if int(settings.signalLabels) == 0:
        hy_electrons = (labels == 0)
        hy_muons = (labels == 2)
    elif int(settings.signalLabels) == 1:
        hy_electrons = (labels == 1)
        hy_muons = (labels == 0)
    #hy_electrons = (labels == 1)
    #hy_muons = (labels == 0)
    print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
    basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut & ve_cut)

    # set class labels and decrease values within labels to match either 0 or 1 
    if type(settings.signalLabels) == list:
        e_label = [settings.signalLabels]
    else:
        e_label = [settings.signalLabels]
    if type(settings.bkgLabels) == list:
        mu_label = settings.bkgLabels
    else:
        mu_label = settings.bkgLabels
    #labels = [x - 1 for x in labels]

    # get the bin indices and edges for parameters
    polar_binning = get_binning(np.cos(angles[:,0]), 10, -1, 1)
    az_binning = get_binning(angles[:,1]*180/np.pi, 10, -180, 180)
    mom_binning = get_binning(momentum, 11, minimum=0., maximum=1000)
    total_charge_binning = get_binning(total_charge, 50, minimum=0, maximum=10000)
    visible_energy_binning = get_binning(ml_visible_energy, 11, minimum=0., maximum=1000)
    dwall_binning = get_binning(dwall, 15, minimum=0, maximum=1600)
    towall_binning = get_binning(towall, 57, minimum=150, maximum=3000)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    stride1 = settings.mlPath
    run_result = [WatChMaLClassification(stride1, 'ResNet', labels, idx, basic_cuts, color="blue", linestyle='-')]
    #print(f"UNIQUE IN LABLES: {np.unique(fitqun_labels, return_counts=True)}")

    
    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    #Fitqun
    if do_fitqun and not settings.getfiTQunTruth:
        fitqun_hy_electrons = (fitqun_labels == int(settings.signalLabels))
        if int(settings.signalLabels) == 0:
            fitqun_hy_muons = (fitqun_labels == 2)
        elif int(settings.signalLabels) == 1:
            fitqun_hy_muons = (fitqun_labels == 0)
        fitqun_basic_cuts = ((fitqun_hy_electrons | fitqun_hy_muons) & (fitqun_towall > 100) & (fitqun_visible_energy < 1200) )
        fitqun_mom_binning = get_binning(fitqun_mom, 11, minimum=0., maximum=1000)
        fitqun_ve_binning = get_binning(fitqun_visible_energy, 11, minimum=0., maximum=1000)
        fitqun_tc_binning = get_binning(fitqun_total_charge, 50, minimum=0, maximum=10000)
        fitqun_towall_binning = get_binning(fitqun_towall, 30, minimum=0, maximum=3000)
        fitqun_az_binning = get_binning(fitqun_az, 10, minimum=-180, maximum=180)
        fitqun_polar_binning = get_binning(fitqun_polar, 10, minimum=-1, maximum=1)
        fitqun_run_result = [WatChMaLClassification(stride1, 'test', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        (fitqun_run_result[0]).selection = fitqun_basic_cuts
        fitqun_run_result[0].cut = fitqun_discr.astype(bool)

        fitqun_pi_run_result = [WatChMaLClassification(stride1, 'ML', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        (fitqun_pi_run_result[0]).selection = fitqun_basic_cuts
        fitqun_pi_run_result[0].cut = fitqun_pi_discr.astype(bool)

    if True:
        #For electron/muon
        #ALSO have to change e/mu definitions for fitqun and ML
        if do_fitqun:
            if int(settings.signalLabels) == 0:
                cut_pi_discr = fitqun_pi_discr[fitqun_basic_cuts]
                fitqun_mu_eff = np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 0]))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==0])
                fitqun_bkg_rej = 1/(1-np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 2]-1))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==2]))
            elif int(settings.signalLabels) == 1:
                cut_pi_discr = fitqun_discr[fitqun_basic_cuts]
                fitqun_mu_eff = np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 1]))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==1])
                fitqun_bkg_rej = 1/(1-np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 0]-1))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==0]))
        else:
            fitqun_pi_eff = 0
            fitqun_bkg_rej = 0
        #For mu/pi+
        #cut_pi_discr = fitqun_pi_discr[fitqun_basic_cuts]
        #fitqun_muon_eff = np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] ==0])/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==2])
        #fitqun_bkg_rej = np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 2]-1))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==2])
        if settings.doFiTQun:
            print(f"fiTQun signal efficiency: {fitqun_mu_eff}, fiTQun bkg rejection: {fitqun_bkg_rej}")
        
        if int(settings.signalLabels) == 0:
            print(f"DISCRIMINATOR: {run_result[0].discriminator(0,2)}")
            discriminator_muons = run_result[0].discriminator(0,2)[basic_cuts][labels[basic_cuts]==0]
            discriminator_muons_ascend = np.sort(discriminator_muons)
            print(discriminator_muons_ascend)
            print(f"# muons: {discriminator_muons.shape[0]}, number above discriminator{discriminator_muons[discriminator_muons > 0.36085752719162684].shape[0]}")
            discriminator_muons_index = int(np.ceil((1-fitqun_mu_eff)*discriminator_muons_ascend.shape[0]))
            print(discriminator_muons_index)
            print(f"muon index i {discriminator_muons_index} out of {discriminator_muons_ascend.shape[0]}, discriminator: {discriminator_muons_ascend[discriminator_muons_index]}")
            fig_roc, ax_roc = plot_rocs(run_result, e_label, mu_label, selection=basic_cuts, x_label="Muon Tagging Efficiency", y_label="Pion Rejection",
                    legend='best', mode='rejection', fitqun=(fitqun_mu_eff, fitqun_bkg_rej), label='ML', x_lim=(0.85,1.0), y_lim=(0.5,300))
            fig_roc.savefig(settings.outputPlotPath + '/ml_mu_pi_roc.png', format='png')
        elif int(settings.signalLabels) == 1:
            fig_roc, ax_roc = plot_rocs(run_result, e_label, mu_label, selection=basic_cuts, x_label="Electron Tagging Efficiency", y_label="Muon Rejection",
                    legend='best', mode='rejection', fitqun=(fitqun_mu_eff, fitqun_bkg_rej), label='ML', x_lim=(0.9,1.0), y_lim=(1.,5000))
            fig_roc.savefig(settings.outputPlotPath + '/ml_e_mu_roc.png', format='png')

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.961
    muon_efficiency = 1 - muon_rejection
    print(e_label)
    for r in run_result:
        r.cut_with_constant_binned_efficiency(e_label, mu_label, 0.98, binning = visible_energy_binning, select_labels = e_label)

    label='eMuPi Training'


    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, polar_ax_e = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_az_fig, az_ax_e = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_mom_fig, mom_ax_e = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_ve_fig, ve_ax_e = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=e_label, x_label="True Visible Energy [MeV]", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_tc_fig, tc_ax_e = plot_efficiency_profile(run_result, total_charge_binning, select_labels=e_label, x_label="Total PMT Charge", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_towall_fig, towall_ax_e = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    if do_fitqun or settings.getfiTQunTruth:
        e_mom_fig_fitqun, mom_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_mom_binning, select_labels=e_label, x_label="fiTQun e Momentum [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_ve_fig_fitqun, ve_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_ve_binning, select_labels=e_label, x_label="fiTQun Visible energy [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        #e_tc_fig_fitqun, tc_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_tc_binning, select_labels=e_label, x_label="fiTQun Total Charge", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_towall_fig_fitqun, towall_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_towall_binning, select_labels=e_label, x_label="Truth toWall [cm]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        #e_az_fig_fitqun, az_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_az_binning, select_labels=e_label, x_label="Truth Azimuth [deg]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax_mu = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_az_fig, az_ax_mu = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_mom_fig, mom_ax_mu = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_ve_fig, ve_ax_mu = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=mu_label, x_label="True Visible Energy [MeV]", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_tc_fig, tc_ax_mu = plot_efficiency_profile(run_result, total_charge_binning, select_labels=mu_label, x_label="Total Charge", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_towall_fig, towall_ax_mu = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    if do_fitqun or settings.getfiTQunTruth:
        mu_mom_fig_fitqun, mom_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_mom_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_ve_fig_fitqun, ve_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_ve_binning, select_labels=mu_label, x_label="fiTQun Visible Energy [MeV]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        #mu_tc_fig_fitqun, tc_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_tc_binning, select_labels=mu_label, x_label="fiTQun Total Charge", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_towall_fig_fitqun, towall_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_towall_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        #mu_az_fig_fitqun, az_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_az_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)

    # save plots of effiency as a function of specific parameters
    e_polar_fig.savefig(settings.outputPlotPath + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(settings.outputPlotPath + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(settings.outputPlotPath + 'e_momentum_efficiency.png', format='png')
    e_ve_fig.savefig(settings.outputPlotPath + 'e_ve_efficiency.png', format='png')
    e_tc_fig.savefig(settings.outputPlotPath + 'e_tc_efficiency.png', format='png')
    if do_fitqun or settings.getfiTQunTruth:
        e_mom_fig_fitqun.savefig(settings.outputPlotPath + 'fitqun_e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(settings.outputPlotPath + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(settings.outputPlotPath + 'e_towall_efficiency.png', format='png')

    mu_polar_fig.savefig(settings.outputPlotPath + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(settings.outputPlotPath + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(settings.outputPlotPath + 'mu_momentum_efficiency.png', format='png')
    mu_ve_fig.savefig(settings.outputPlotPath + 'mu_ve_efficiency.png', format='png')
    if do_fitqun or settings.getfiTQunTruth:
        mu_mom_fig_fitqun.savefig(settings.outputPlotPath + 'fitqun_mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(settings.outputPlotPath + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(settings.outputPlotPath + 'mu_towall_efficiency.png', format='png')

    if do_fitqun or settings.getfiTQunTruth:
        plot_fitqun_comparison(settings.outputPlotPath, mom_ax_e, mom_ax_fitqun_e, mom_ax_mu, mom_ax_fitqun_mu, 'mom_combine', 'Truth Momentum [MeV]')
        plot_fitqun_comparison(settings.outputPlotPath, towall_ax_e, towall_ax_fitqun_e, towall_ax_mu, towall_ax_fitqun_mu, 'towall_combine', 'Towall [cm]')
        plot_fitqun_comparison(settings.outputPlotPath, ve_ax_e, ve_ax_fitqun_e, ve_ax_mu, ve_ax_fitqun_mu, 've_combine', 'Truth Visible Energy [MeV]')
        #plot_fitqun_comparison(settings.outputPlotPath, tc_ax_e, tc_ax_fitqun_e, tc_ax_mu, tc_ax_fitqun_mu, 'tc_combine', 'Total Charge')
        #plot_fitqun_comparison(plot_output, az_ax_e, az_ax_fitqun_e, az_ax_mu, az_ax_fitqun_mu, 'az_combine', 'Truth Azimuth [deg]')



    # remove comment for ROC curves of single run 
    return run_result[0]