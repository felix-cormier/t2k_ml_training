import numpy as np

import h5py

from scipy.stats import chisquare
from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile
from analysis.utils.plotting import plot_legend
from analysis.utils.binning import get_binning
import analysis.utils.math as math
import analysis.classification

def get_cherenkov_threshold(label):
    threshold_dict = {0: 160., 1:0.8, 2: 0.}
    return threshold_dict[label]

def efficiency_plots(inputPath, arch_name, newest_directory, plot_output, label=None):

    # retrieve test indices
    idx = np.array(sorted(np.load(str(newest_directory)+'/indices.npy')))

    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(inputPath, "r")
    #print(list(hy.keys()))
    angles = np.array(hy['angles'])[idx].squeeze() 
    labels = np.array(hy['labels'])[idx].squeeze() 
    veto = np.array(hy['veto'])[idx].squeeze()
    energies = np.array(hy['energies'])[idx].squeeze()
    positions = np.array(hy['positions'])[idx].squeeze()
    directions = math.direction_from_angles(angles)
    cheThr = list(map(get_cherenkov_threshold, np.ravel(labels)))
    visible_energy = energies - cheThr
    
    print('np.unique', np.unique(labels))
    # calculate number of hits 
    events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
    nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()

    # calculate additional parameters 
    towall = math.towall(positions, angles, tank_axis = 2)
    dwall = math.dwall(positions, tank_axis = 2)
    momentum = math.momentum_from_energy(energies, labels)

    # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
    nhit_cut = nhits > 0 #25
    # veto_cut = (veto == 0)
    hy_electrons = (labels == 1)
    hy_muons = (labels == 0)
    basic_cuts = ((hy_electrons | hy_muons) & nhit_cut)
    #print('basic cuts = ', basic_cuts)
    # set class labels and decrease values within labels to match either 0 or 1 
    e_label = [1]
    mu_label = [0]
    #labels = [x - 1 for x in labels]

    # get the bin indices and edges for parameters
    polar_binning = get_binning(np.cos(angles[:,0]), 10, -1, 1)
    az_binning = get_binning(angles[:,1]*180/np.pi, 10, -180, 180)
    mom_binning = get_binning(momentum, 10)
    dwall_binning = get_binning(dwall, 10)
    towall_binning = get_binning(towall, 10)
    visible_energy_binning = get_binning(visible_energy, 10)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    stride1 = newest_directory
    run_result = [WatChMaLClassification(stride1, 'test', labels, idx, basic_cuts, color="blue", linestyle='-')]

    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    #muon_rejection = 0.99876
    muon_rejection = 0.9925
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_fixed_efficiency(e_label, mu_label, muon_efficiency, select_labels = mu_label, selection = basic_cuts)
    

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, e_polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label, y_lim = (99, 100))
    e_az_fig, e_az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label, y_lim = (99, 100))
    e_mom_fig, mom_ax = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, mu_polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Muon Miss-PID [%]", errors=True, x_errors=False, label=label, y_lim = (0, 2))
    mu_az_fig, mu_az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Muon Miss-PID [%]", errors=True, x_errors=False, label=label, y_lim = (0, 2))
    mu_mom_fig, mom_ax = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    
    Visible_energy_mu_fig, visible_energy_mu_ax = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=mu_label, x_label="Visible energy (MeV)", y_label="Muon Miss-PID [%]", errors=True, x_errors=False, label=label, y_lim = (0, 4))
    Visible_energy_elec_fig, visible_energy_elec_ax = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=e_label, x_label="Visible energy (MeV)", y_label="Electron Miss-PID [%]", errors=True, x_errors=False, label=label, y_lim = (0, 20))

    # save plots of effiency as a function of specific parameters
    e_polar_fig.savefig(plot_output + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(plot_output + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(plot_output + 'e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(plot_output + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(plot_output + 'e_towall_efficiency.png', format='png')

    mu_polar_fig.savefig(plot_output + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(plot_output + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(plot_output + 'mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(plot_output + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(plot_output + 'mu_towall_efficiency.png', format='png')
    
    Visible_energy_mu_fig.savefig(plot_output + 'Visible_energy_mu_efficiency.png', format = 'png')
    Visible_energy_elec_fig.savefig(plot_output + 'Visible_energy_elec_efficiency.png', format = 'png')
    
    lines_mu = mu_az_ax.lines[0]
    lines_elec = e_az_ax.lines[0]

    x_data_mu = lines_mu.get_xdata()
    y_data_mu = lines_mu.get_ydata()

    x_data_elec = lines_elec.get_xdata()
    y_data_elec = lines_elec.get_ydata()

    p_mu = np.polyfit(x_data_mu, y_data_mu, 1)
    p_e = np.polyfit(x_data_elec, y_data_elec, 1)

    chi_squared_mu = np.sum((np.polyval(p_mu, x_data_mu) - y_data_mu) ** 2)
    chi_squared_e = np.sum((np.polyval(p_e, x_data_elec) - y_data_elec) ** 2)
    
    print('chi mu reduced', chi_squared_mu/9)
    print('chi e reduced', chi_squared_e/9)

    #print('y e az ax = ', chisquare(y_data_elec, m_e, ddof = 9))
    #print('y mu az ax = ', chisquare(y_data_mu, m_mu, ddof = 9))
    # remove comment for ROC curves of single run 
    return run_result[0]
