import numpy as np

import h5py

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

import scipy

import WatChMaL.analysis.utils.fitqun as fq
import WatChMaL.analysis.utils.math as math


from scipy.optimize import linear_sum_assignment

from tqdm import tqdm


def get_ml_file(file, target, isMC=False):
    preds = np.load(file+'predicted_'+target+'.npy')
    if isMC:
        truth = np.load(file+target+'.npy')
        return preds, truth
    else:
        return preds

def get_secondaries_file(filepath):
    temp = h5py.File(filepath)
    return temp


def apply_cuts(mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom, qtot, nhits, total_charge_cut=[0,150000], nhits_cut=200):

    keep = ((qtot > total_charge_cut[0]) & (qtot < total_charge_cut[1]) & (nhits > nhits_cut))
    print(np.unique(keep, return_counts=True))

    mu_1rpos = mu_1rpos[keep]
    e_1rpos = e_1rpos[keep]
    pi_1rpos = pi_1rpos[keep]

    mu_1rdir = mu_1rdir[keep]
    e_1rdir = e_1rdir[keep]
    pi_1rdir = pi_1rdir[keep]

    mu_1rmom = mu_1rmom[keep]
    e_1rmom = e_1rmom[keep]
    pi_1rmom = pi_1rmom[keep]

    return mu_1rpos, e_1rpos, pi_1rpos, mu_1rdir, e_1rdir, pi_1rdir, mu_1rmom, e_1rmom, pi_1rmom 

def plot_with_ratio(data, mc, label_data, label_mc, bins, range, savename, xlabel, truth=None, zoom=False):
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ns_data_noDensity, bins_data_noDensity = np.histogram(data,
                      bins=bins,
                      range=range
                      )
    ns_data, bins_data = np.histogram(data,
                      bins=bins,
                      range=range,
                      density=True
                      )
    ns_mc_noDensity, bins_mc_noDensity = np.histogram(mc,
                      bins=bins,
                      range=range
                      )


    num_data_events = np.sum(ns_data_noDensity)
    num_mc_events = np.sum(ns_mc_noDensity)

    ns, bins, patches = ax1.hist(mc, density=True,
                      histtype='stepfilled',
                      bins=bins,
                      range=range,
                      alpha=0.8,
                      label=label_mc
                      )

    if truth is not None:
        ns_truth, bins_truth, patches = ax1.hist(truth, density=True,
                        histtype='step',
                        bins=bins,
                        range=range,
                        alpha=0.7,
                        color='orange',
                        label='MC Truth'
                        )

    sigma_data = np.sqrt(ns_data_noDensity)
    sigma_mc = np.sqrt(ns_mc_noDensity)

    num_data_events_density = np.sum(ns_data) 
    num_mc_events_density = np.sum(ns) 

    sigma_data = sigma_data*(num_data_events_density/num_data_events)
    sigma_mc = sigma_mc*(num_mc_events_density/num_mc_events)

    ax1.errorbar(x=(bins[:-1] + bins[1:]) / 2, y=ns_data, yerr=sigma_data, fmt='o', capsize=2, label=label_data, color='black')
    ax1.legend()
    ax1.set_xlim(range[0], range[1])


    sigma_ratio = (ns_data/ns)* np.sqrt( np.square((sigma_data/ns_data)) + np.square(sigma_mc/ns)  )


    ax2.errorbar((bins[:-1] + bins[1:]) / 2,     # this is what makes it comparable
            ns_data / ns, # maybe check for div-by-zero!
            yerr=sigma_ratio,
            fmt='.',
            color='black')
    plt.xlim(range[0], range[1])
    ax2.plot([range[0], range[1]], [1,1], linestyle='dotted')
    if zoom:
        ax2.set_ylim(0.8,1.2)

    ax1.set_ylabel('Arb. Units')
    ax2.set_ylabel('Ratio (Data/MC)')
    ax2.set_xlabel(xlabel)
    plt.savefig(savename)
    plt.clf()

def apply_quality_cuts(cut, pred):
    return pred[cut]

def do_2d_endcap_plot(settings, x, y, z, savename, min_z=1780, max_z=1840, num_bins=74):
    x_bins = np.linspace(-1300, 1300, num_bins)
    y_bins = np.linspace(-1300, 1300, num_bins)

    H, xedges, yedges = np.histogram2d(x, y, bins = [x_bins, y_bins], weights = z)
    H_counts, xedges, yedges = np.histogram2d(x, y, bins = [x_bins, y_bins]) 
    print(f"Counts in {savename}: {np.mean(H_counts[H_counts > 0])}")
    H = H/H_counts
    H_x = np.nanmean(H, axis=0)
    H_y = np.nanmean(H, axis=1)
    print(np.array(~np.isnan(H)).shape)
    H_x_std = np.nanstd(H, axis=0)/np.sqrt(np.count_nonzero(~np.isnan(H),axis=0))
    H_y_std = np.nanstd(H, axis=1)/np.sqrt(np.count_nonzero(~np.isnan(H),axis=1))


    plt.errorbar(np.squeeze((xedges[:-1] + xedges[1:]) / 2), np.squeeze(H_x), H_x_std, fmt='', label='X projection endcap Z')
    plt.xlabel("X [cm]")
    plt.ylabel("Proj. Avg. Reco Z [cm]")
    plt.savefig(settings.outputPlotPath+'/Xproj'+savename)
    plt.clf()
    plt.close()

    plt.errorbar((xedges[:-1] + xedges[1:]) / 2, H_y, H_x_std, fmt='', label='Y projection endcap Z')
    plt.xlabel("Y [cm]")
    plt.ylabel("Proj. Avg. Reco Z [cm]")
    plt.savefig(settings.outputPlotPath+'/Yproj'+savename)
    plt.clf()
    plt.close()

    plt.imshow(H.T, origin='lower',  cmap='jet',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=min_z, vmax=max_z)
    cbar = plt.colorbar()
    cbar.set_label('Average Reco Z [cm]', rotation=90)
    plt.xlabel("Reco X [cm]")
    plt.ylabel("Reco Y [cm]")
    plt.savefig(settings.outputPlotPath+'/'+savename)
    plt.clf()
    plt.close()

def laurel_plots(settings, x,y,z, mom, savename, z_min, z_max):
    x = x[(z>z_min) & (z<z_max)]
    mom = mom[(z>z_min) & (z<z_max)]

    plt.hist2d(np.ravel(x),np.ravel(mom), bins=[20,20], cmap='jet')
    plt.xlabel('X [cm]')
    plt.ylabel('Mom [MeV]')
    plt.colorbar()
    plt.savefig(settings.outputPlotPath+savename)
    plt.clf()
    plt.close()

def get_pmt_positions():
    positions_file = "/home/fcormier/t2k/ml/t2k_ml_training/data/geofile_skdetsim.npz"
    positions = np.load(positions_file)['position']
    return positions

def return_preds_number(preds_rootfiles):

    #Just to extract one number , didn't work
    number = [x.decode('UTF-8') for x in preds_rootfiles]
    #print(f"nubmer 0: {number[1]}")
    #number = np.char.split(number,'/')
    #number = [x[-1] for x in number]
    #number = [x[1] for x in number]
    #print(f"number: {number[1]}")
    number = [str(int(''.join(filter(str.isdigit, x)))) for x in number]

    return number

def return_subevents_number(subevents_rootfiles):

    number = [x.decode('UTF-8') for x in subevents_rootfiles]
    number = np.char.split(number,'/')
    number = [x[-1] for x in number]
    number = np.char.split(number,'_')
    number = [x[-1] for x in number]
    number = np.char.split(number,'.')
    number = [x[0] for x in number]

    return number

def combine_hash(rootfile_id, event_id, rootfile_string=True):
    event_id = np.char.mod('%d', event_id)
    if not rootfile_string:
        rootfile_id = np.char.mod('%d', rootfile_id)
    combined_hash = np.char.add(rootfile_id, event_id)
    return combined_hash

def  get_inference_subevent_correspondence(subevents, rootfiles_preds, eventids_preds):

    rootfiles_subevents = subevents['root_files']
    eventids_subevents = subevents['event_ids']


    rootfiles_preds = [item.tobytes() for item in rootfiles_preds ]

    #Looks like data/stopmu_mc_1390_for_atmnu_test_064382.N.root, want N
    #print(f"rootfiles preds {np.unique(rootfiles_preds, return_counts=True)}")
    #print(f"rootfiles subevents {np.unique(rootfiles_subevents, return_counts=True)}")
    if 'atmnu' in str(rootfiles_preds[0]): 
        preds_rootfile_number = return_preds_number(rootfiles_preds)
    else:
        preds_rootfile_number = return_subevents_number(rootfiles_preds)
    #Looks like /scratch/fcormier/sk/stopmu/root/mc/stopmuMC_N.root, want N
    #print(f"rootfiles subevents {np.array(rootfiles_subevents)[1]}")
    if 'atmnu' in str(rootfiles_subevents[0]): 
        preds_subevents_number = return_preds_number(rootfiles_subevents)
    else:
        preds_subevents_number = return_subevents_number(rootfiles_subevents)
    #The two Ns will match

    subevents_hash = combine_hash(preds_subevents_number, eventids_subevents)
    #print(f"subevents rootfiles uniques: {np.unique(preds_subevents_number,return_counts=True)}, subevents eventids uniques: {np.unique(eventids_subevents,return_counts=True)[0].shape},")
    #print(f"preds rootfiles uniques: {np.unique(preds_rootfile_number,return_counts=True)}, preds eventids uniques: {np.unique(eventids_preds,return_counts=True)[0].shape} \n {np.unique(eventids_preds,return_counts=True)[1].shape},")
    preds_hash = combine_hash(preds_rootfile_number, eventids_preds)
    #print(f"subevents hash: {subevents_hash[0]}")
    #print(f"preds hash: {preds_hash[0]}")

    #Comm1 maps subevents_hash to some common order
    #Comm2 maps preds_hash to some commond order
    intersect, comm1, comm2 = np.intersect1d(subevents_hash, preds_hash, return_indices=True)

    return intersect, comm1, comm2, subevents_hash, preds_hash

def get_fitqun_correspondence(subevents, fq_se, isData=False, sub_map=None):

    if not isData:
        #print(f"se nhit: { np.array(subevents['nhit'])}, fq nhit: {np.array(fq_se['nhit'])}")
       #print(f"se nhita: { np.array(subevents['nhita'])}, fq nhita: {np.array(fq_se['nhita'])}")
        if sub_map is not None:
            subevents_hash = combine_hash(np.array(subevents['nhit'])[sub_map],np.array(subevents['nhita'])[sub_map], rootfile_string=False)
            subevents_hash = combine_hash(subevents_hash,np.array(subevents['nev'][sub_map]))
        else:
            subevents_hash = combine_hash(np.squeeze(np.array(subevents['position'], dtype=np.int32))[:,0],np.squeeze(np.array(subevents['position'], dtype=np.int32))[:,1], rootfile_string=False)
            subevents_hash = combine_hash(subevents_hash,np.squeeze(np.array(subevents['position'], dtype=np.int32))[:,2])

        fq_se_hash = combine_hash(np.squeeze(np.array(fq_se['position'], dtype=np.int32))[:,0],np.squeeze(np.array(fq_se['position'], dtype=np.int32))[:,1],rootfile_string=False)
        fq_se_hash = combine_hash(fq_se_hash,np.squeeze(np.array(fq_se['position'], dtype=np.int32))[:,2])
        intersect, comm1, comm2 = np.intersect1d(subevents_hash, fq_se_hash, return_indices=True)
        return intersect, comm1, comm2, subevents_hash, fq_se_hash
    else:
        if sub_map is not None:
            subevents_hash = combine_hash(np.array(subevents['nev'])[sub_map],np.array(subevents['nrun'])[sub_map], rootfile_string=False)
            subevents_hash = combine_hash(subevents_hash,np.array(subevents['nsub'])[sub_map])
        else:
            subevents_hash = combine_hash(np.array(subevents['nev']),np.array(subevents['nrun']), rootfile_string=False)
            subevents_hash = combine_hash(subevents_hash,np.array(subevents['nsub']))
        fq_se_hash = combine_hash(np.array(fq_se['nev']),np.array(fq_se['nrun']), rootfile_string=False)
        fq_se_hash = combine_hash(fq_se_hash,np.array(fq_se['nsub']))
        intersect, comm1, comm2 = np.intersect1d(subevents_hash, fq_se_hash, return_indices=True)
        return intersect, comm1, comm2, subevents_hash, fq_se_hash

def get_directMC_correspondence(mc_pos, fq_se, isData=False, sub_map=None):

    mc_pos_hash = combine_hash(np.squeeze(np.array(mc_pos, dtype=np.int32))[:,0],np.squeeze(np.array(mc_pos, dtype=np.int32))[:,1], rootfile_string=False)
    mc_pos_hash = combine_hash(mc_pos_hash,np.squeeze(np.array(mc_pos, dtype=np.int32))[:,2])

    fq_se_hash = combine_hash(np.squeeze(np.array(fq_se['position'], dtype=np.int32))[:,0],np.squeeze(np.array(fq_se['position'], dtype=np.int32))[:,1],rootfile_string=False)
    fq_se_hash = combine_hash(fq_se_hash,np.squeeze(np.array(fq_se['position'], dtype=np.int32))[:,2])
    intersect, comm1, comm2 = np.intersect1d(mc_pos_hash, fq_se_hash, return_indices=True)
    return intersect, comm1, comm2, mc_pos_hash, fq_se_hash

#Hack for buggy subevents hits index
def modify_array(arr):
    # Create a copy of the array to avoid modifying the original
    modified_arr = arr.copy()
    print(f"modified array shape: {modified_arr.shape}")

    # Iterate through the array
    for i in range(1, len(modified_arr)):
        if i==0:
            continue
        if (modified_arr[i] == 107 and (modified_arr[i-1] > modified_arr[i])) or (modified_arr[i] == 0 and (modified_arr[i-1] > modified_arr[i])):
            # Add the previous element to all subsequent elements
            print(f"first: {modified_arr[i]}")
            modified_arr[i:] += modified_arr[i - 1]
            print(f"second: {modified_arr[i]}")

    return modified_arr


def align_datasets(dataset1, dataset2, common_values1, common_values2):
    """
    Align two datasets based on a common column (variable), even if the rows are unordered and one dataset is multi-dimensional.

    Parameters:
        dataset1 (numpy.ndarray): First dataset.
        dataset2 (numpy.ndarray): Second dataset (can be multi-dimensional).
        common_col_idx1 (int): Index of the common column in the first dataset.
        common_col_idx2 (int): Index of the common column in the second dataset.

    Returns:
        aligned_dataset1 (numpy.ndarray): Rows from dataset1 aligned with dataset2.
        aligned_dataset2 (numpy.ndarray): Rows from dataset2 aligned with dataset1.
    """
    if (dataset1.shape[0] != common_values1.shape[0]):
        print(f"Datasets 1 {dataset1.shape[0]} and common values {common_values1.shape[0]} not same length")
        return -1, -1 
    if (dataset2.shape[0] != common_values2.shape[0]):
        print(f"Datasets 2 {dataset2.shape[0]} and common values {common_values2.shape[0]} not same length")
        return -1, -1 

    # Find the intersection of the common variable values
    common_values = np.intersect1d(common_values1, common_values2)

    # Create dictionaries for quick lookup
    dataset1_dict = {value: row for row, value in zip(dataset1, common_values1)}
    dataset2_dict = {value: row for row, value in zip(dataset2, common_values2)}

    # Align rows by matching the common values
    aligned_dataset1 = np.array([dataset1_dict[value] for value in common_values])
    aligned_dataset2 = np.array([dataset2_dict[value] for value in common_values])

    return aligned_dataset1, aligned_dataset2

def find_min_distance_matches(array1, array2):
    """
    Find matches between two arrays of 3D positions with the least distance.

    Parameters:
        array1 (numpy.ndarray): First array of 3D positions with shape (N, 3).
        array2 (numpy.ndarray): Second array of 3D positions with shape (M, 3).

    Returns:
        matches (list of tuples): List of matched indices [(i, j)], where i is an index in array1 and j is an index in array2.
        distances (list): List of distances corresponding to the matches.
    """
    # Calculate pairwise distances between all points in array1 and array2
    distance_matrix = np.linalg.norm(array1[:, np.newaxis, :] - array2[np.newaxis, :, :], axis=2)

    # Solve the assignment problem (minimize total distance)
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Extract the matches and their distances
    matches = list(zip(row_indices, col_indices))
    distances = distance_matrix[row_indices, col_indices]

    return matches, distances

def transform_array(arr, total_len):
    """
    Converts an array so that each element is the original element except the 0th, minus one.
    Then appends the total length minus one at the end.
    
    Parameters:
        arr (numpy.ndarray): Input 1D NumPy array.

    Returns:
        numpy.ndarray: Transformed array.
    """
    transformed = arr[1:] - 1  # Subtract 1 from all elements except the first
    transformed = np.append(transformed, total_len - 1)  # Append length-1 at the end
    return transformed

def transform_array_mc(arr):
    """
    Converts an array so that each element is the original element except the 0th, minus one.
    Then appends the total length minus one at the end.
    
    Parameters:
        arr (numpy.ndarray): Input 1D NumPy array.

    Returns:
        numpy.ndarray: Transformed array.
    """
    transformed = arr[0:] - 1  # Subtract 1 from all elements
    return transformed




def analyze_ml_regression_dataMC(settings, total_charge_cut, nhits_cut):
    file_mc = settings.MLMCPath
    file_data = settings.MLDataPath

    file_subevent_mc = settings.MLMCsubeventInfo
    file_subevent_data = settings.MLDatasubeventInfo

    subevents_mc = get_secondaries_file(file_subevent_mc)
    subevents_data = get_secondaries_file(file_subevent_data)

    fq_se_mc = get_secondaries_file(settings.FQMCSecondariesPath)
    fq_se_data = get_secondaries_file(settings.FQDataSecondariesPath)



    if settings.doRegression:
        target = str(settings.target)
        preds_mc, truth = get_ml_file(file_mc, target, isMC=True)
        print(f"File MC: {file_mc}")
        preds_data = get_ml_file(file_data, target)
        print(f"File data: {file_data}")
        data_rootfiles = np.load(file_data+"/root_files.npy") 
        data_eventids = np.load(file_data+"/event_ids.npy") 
        mc_rootfiles = np.load(file_mc+"/root_files.npy") 
        mc_eventids = np.load(file_mc+"/event_ids.npy") 
    elif settings.doClassification:
        preds_mc = np.load(file_mc+"/softmax.npy")
        preds_data = np.load(file_data+"/softmax.npy")

    if settings.doRegression and "momenta" in target:
        bins = 28
        range=[100,1500]
        plot_with_ratio(np.ravel(preds_data), np.ravel(preds_mc), 'Data Reco ' + target, 'MC Reco ' + target, bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_momentum.png", 'Reco Momentum [MeV]', truth=truth)
        plot_with_ratio(np.ravel(preds_data), np.ravel(preds_mc), 'Data Reco ' + target, 'MC Reco ' + target, bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_momentum_zoom.png", 'Reco Momentum [MeV]', truth=truth, zoom=True)

    elif settings.doRegression and "positions" in target:

        mc_positions = np.load(file_mc+"/positions.npy") 


        print(f"mc subevent size: {np.unique(subevents_mc['fqnse'], return_counts=True)}, subevents index: {subevents_mc['subevents_hit_index']}")
        print(f"data subevent size: {np.unique(subevents_data['fqnse'], return_counts=True)}, subevents index: {subevents_data['subevents_hit_index']}")

        intersect_subpreds_mc, sub_map_mc, preds_map_mc, subevents_hash_mc, preds_hash_mc = get_inference_subevent_correspondence(subevents_mc, mc_rootfiles, mc_eventids)
        intersect_subfq_mc, sub_map_fq_mc, fq_map_sub_mc, subevents_fq_hash_mc, fq_hash_mc = get_fitqun_correspondence(subevents_mc, fq_se_mc)

        intersect_predsfq_mc, preds_map_fq_mc, fq_map_preds_mc, predsFQ_hash_mc, fqPreds_hash_mc = get_directMC_correspondence(mc_positions, fq_se_mc)


        intersect_subpreds_data, sub_map_data, preds_map_data, subevents_hash_data, preds_hash_data = get_inference_subevent_correspondence(subevents_data, data_rootfiles, data_eventids)
        intersect_fqsub_data, sub_map_fq_data, fq_map_sub_data, subevents_fq_hash_data, fq_hash_data = get_fitqun_correspondence(subevents_data, fq_se_data, isData=True)



        
        #Want mapping between fq and preds
        #print(f"SUB MAP FQ: {sub_map_fq_data}, {sub_map_fq_data.shape}, shuffle: {np.random.shuffle(sub_map_fq_mc)}, {np.array(np.random.shuffle(sub_map_fq_mc)).shape}")
        #mc_conv = preds_map_mc
        #data_conv = preds_map_data

        #print(f"Subevents hash mc: {subevents_hash_mc.shape}, se fq hash: {subevents_fq_hash_mc.shape}")
        
        #subevents_to_preds_mc = subevents_hash_mc[sub_map_mc]
        #subevents_to_fq_mc = subevents_hash_mc[sub_map_fq_mc]
        #subevents_to_preds_data = subevents_hash_data[sub_map_data]
        #subevents_to_fq_data = subevents_hash_data[sub_map_fq_data]
        #mc_map_intersect, mc_map_sub, mc_map_fq = np.intersect1d(subevents_hash_mc, subevents_fq_hash_mc, return_indices=True)
        #data_map_intersect, data_map_sub, data_map_fq = np.intersect1d(subevents_hash_data, subevents_hash_mc[], return_indices=True)

        #preds_fqOrder_mc = (preds_mc[mc_conv])[mc_map_sub]
        #preds_fqOrder_mc = preds_mc[np.isin(preds_hash_mc, intersect_subpreds_mc)]
        #preds_fqOrder_mc = preds_fqOrder_mc[np.isin(preds_hash_mc, subevents_hash_mc)]
        #preds_fqOrder_data = preds_data[np.isin(preds_hash_data, intersect_subpreds_data)]
        #preds_fqOrder_data = preds_fqOrder_data[np.isin(preds_hash_data, subevents_hash_data)]
        #preds_fqOrder_data = (preds_data[data_conv])[data_map_sub]

        #print(f"fq by subevents mc: {fq_se_mc['fqnse']}, data: {fq_se_data['fqnse']}")
        #print(f"fq by subevents mc: {fq_se_mc['fqnse']}, data: {fq_se_data['fqnse']}")
        #print(f"fq by subevents mc: {fq_se_mc['fqipeak']}, data: {fq_se_data['fqipeak']}")

        #print(f"subevents hits index data: {np.unique(np.array(fq_se_data['subevents_hit_index']), return_counts=True)}")
        #print(f"subevents hits index mc: {np.unique(np.array(fq_se_mc['subevents_hit_index']), return_counts=True)}")

        #fq_fqOrder_nse_mc = np.array(fq_se_mc['fqnse'])[np.isin(fq_hash_mc, intersect_subfq_mc)]
        #fq_fqOrder_fqipeak_mc = np.array(fq_se_mc['fqipeak'])[np.isin(fq_hash_mc, intersect_subfq_mc)]
        #fq_fqOrder_seHitIndex_mc = np.array(fq_se_mc['subevents_hit_index'])[np.isin(fq_hash_mc, intersect_subfq_mc)]


        #fq_fqOrder_fqipeak_mc = fq_fqOrder_fqipeak_mc[sub_map_mc]
        #fq_fqOrder_nse_mc = fq_fqOrder_nse_mc[sub_map_mc]
        #fq_fqOrder_fqipeak_mc = fq_fqOrder_fqipeak_mc[sub_map_mc]
        #fq_fqOrder_seHitIndex_mc = fq_fqOrder_seHitIndex_mc[sub_map_mc]
        #fq_fqOrder_mc = fq_fqOrder_mc[sub_map_mc]
        #fq_fqOrder_nse_data = np.array(fq_se_data['fqnse'])[np.isin(fq_hash_data, intersect_fqsub_data)]
        #fq_fqOrder_fqipeak_data = np.array(fq_se_data['fqipeak'])[np.isin(fq_hash_data, intersect_fqsub_data)]
        #fq_fqOrder_seHitIndex_data = np.array(fq_se_data['subevents_hit_index'])[np.isin(fq_hash_data, intersect_fqsub_data)]


        se_hits_index_mod_data = transform_array(np.array(fq_se_data['subevents_hit_index'])+1, np.array(fq_se_data['fq1rpos']).shape[0])#modify_array(np.array(fq_se_data['subevents_hit_index']))
        se_hits_index_mod_mc = np.array(fq_se_mc['subevents_hit_index'])#+1, np.array(fq_se_mc['fq1rpos']).shape[0]) #transform_array(np.array(fq_se_mc['subevents_hit_index']))#modify_array(np.array(fq_se_mc['subevents_hit_index']))
        fq_fqOrder_fq1rpos_data = (np.array(fq_se_data['fq1rpos'])[se_hits_index_mod_data])
        fq_fqOrder_fq1rpos_mc = (np.array(fq_se_mc['fq1rpos'])[se_hits_index_mod_mc])
        fq_fqOrder_fqipeak_data = (np.array(fq_se_data['fqipeak'])[se_hits_index_mod_data])
        fq_fqOrder_fqipeak_mc = (np.array(fq_se_mc['fqipeak'])[se_hits_index_mod_mc])
        fq_fqOrder_pos_mc = np.array(fq_se_mc['position']) 
        fq_fqOrder_nse_mc = np.array(fq_se_mc['fqnse']) 
        fq_fqOrder_nse_data = np.array(fq_se_data['fqnse']) 



        #Align data
        preds_fqOrder_data, fq_fqOrder_fq1rpos_data = align_datasets(preds_data, fq_fqOrder_fq1rpos_data[fq_map_sub_data], preds_hash_data, subevents_hash_data[sub_map_fq_data])
        _, se_hits_index_mod_data = align_datasets(preds_data, se_hits_index_mod_data[fq_map_sub_data], preds_hash_data, subevents_hash_data[sub_map_fq_data])
        _, fq_fqOrder_fqipeak_data = align_datasets(preds_data, fq_fqOrder_fqipeak_data[fq_map_sub_data], preds_hash_data, subevents_hash_data[sub_map_fq_data])
        _, fq_fqOrder_nse_data = align_datasets(preds_data, fq_fqOrder_nse_data[fq_map_sub_data], preds_hash_data, subevents_hash_data[sub_map_fq_data])
        #Align MC
        #preds_fqOrder_mc, fq_fqOrder_fq1rpos_mc = align_datasets(preds_mc, fq_fqOrder_fq1rpos_mc[fq_map_sub_mc], preds_hash_mc, subevents_hash_mc[sub_map_fq_mc])
        #pos_fqOrder_mc, fq_fqOrder_pos_mc = align_datasets(mc_positions, fq_fqOrder_pos_mc[fq_map_sub_mc], preds_hash_mc, subevents_hash_mc[sub_map_fq_mc])
        #hash_fqOrder_mc, sub_fqOrder_hash_mc = align_datasets(preds_hash_mc, subevents_hash_mc[sub_map_fq_mc], preds_hash_mc, subevents_hash_mc[sub_map_fq_mc])

        preds_fqOrder_mc = preds_mc[preds_map_fq_mc]
        fq_fqOrder_fq1rpos_mc = fq_fqOrder_fq1rpos_mc[fq_map_preds_mc]
        fq_fqOrder_fqipeak_mc = fq_fqOrder_fqipeak_mc[fq_map_preds_mc]
        se_hits_index_mod_mc = se_hits_index_mod_mc[fq_map_preds_mc]

        pos_fqOrder_mc =mc_positions[preds_map_fq_mc]
        fq_fqOrder_pos_mc = fq_fqOrder_pos_mc[fq_map_preds_mc]
        fq_fqOrder_nse_mc = fq_fqOrder_nse_mc[fq_map_preds_mc]

        #print(f"ML true pos: {pos_fqOrder_mc.shape}, FQ true pos: {fq_fqOrder_pos_mc.shape}, ML unequal: {pos_fqOrder_mc[np.where((np.squeeze(fq_fqOrder_pos_mc) != np.squeeze(pos_fqOrder_mc)).all(axis=1))[0]]}, FQ unequal: {fq_fqOrder_pos_mc[np.where((np.squeeze(fq_fqOrder_pos_mc) != np.squeeze(pos_fqOrder_mc)).all(axis=1))[0]]}")

        #print(f"MC ml hash: {hash_fqOrder_mc}, FQ hash: {sub_fqOrder_hash_mc}")



        debug=False
        if debug:
            matches_data, distances_data = find_min_distance_matches(preds_fqOrder_data, fq_fqOrder_fq1rpos_data[0:10,1,:])
            matches_mc, distances_mc = find_min_distance_matches(np.squeeze(preds_fqOrder_mc[0:10,:]), np.squeeze(fq_fqOrder_fq1rpos_mc[:,1,:]))

            for match, distance in zip(matches_data,distances_data):
                print(f"Data match: {match}, distance: {distance}, pred: {preds_fqOrder_data[match[0]]}, fq: {fq_fqOrder_fq1rpos_data[match[1], 1,:]}")
            for match, distance in zip(matches_mc,distances_mc):
                print(f"MC match: {match}, distance: {distance}, pred: {preds_fqOrder_mc[match[0]]}, fq: {fq_fqOrder_fq1rpos_mc[match[1],1,:]}")


            print(f"preds fqOrder MC : {preds_fqOrder_mc[0:10]}")
            print(f"fq fqOrder MC : {fq_fqOrder_fq1rpos_mc[0:10]}")

        fq_fqOrder_fq1rpos_data = fq_fqOrder_fq1rpos_data[:,1,:]
        fq_fqOrder_fq1rpos_mc = fq_fqOrder_fq1rpos_mc[:,1,:]


        #Difference in data between ML and fitqun
        data_predsFQ_diff = np.median(np.abs(np.abs(fq_fqOrder_fq1rpos_data[:,0]) - np.abs(preds_fqOrder_data[:,0])))
        mc_predsFQ_diff = np.median(np.abs(np.abs(fq_fqOrder_fq1rpos_mc[:,0]) - np.abs(preds_fqOrder_mc[:,0])))

        print(f"Difference between Data ML and fiTQun x position: {data_predsFQ_diff}")
        print(f"Difference between mc ML and fiTQun x position: {mc_predsFQ_diff}")

        if data_predsFQ_diff > 50 or mc_predsFQ_diff > 50:
            print(f"WARNING, Data ({data_predsFQ_diff}) or fiTQun ({mc_predsFQ_diff}) is > 50, Exiting...")
            return 0
        
        #Cuts like fiTQun data/MC paper

        #Cut 1, exactly 1 decay electron
        cut_1decayE_mc = fq_fqOrder_nse_mc == 2  
        cut_1decayE_data = fq_fqOrder_nse_data == 2  

        print(f"cut 1 decay E, MC : {np.unique(cut_1decayE_mc, return_counts=True)}")
        print(f"cut 1 decay E, data : {np.unique(cut_1decayE_data, return_counts=True)}")


        #Cut 2, decay electron is not in gate

        #Find all events that have more than 1 sub-event, convert to int
        gt_1se_mc = fq_fqOrder_nse_mc ==2
        gt_1se_data = fq_fqOrder_nse_data ==2
        print(f"gt 1se mc: {gt_1se_mc}, gt 1se data: {gt_1se_data}")
        gt_1se_mc = gt_1se_mc.astype(int)
        gt_1se_data = gt_1se_data.astype(int)

        #Add one and times gt_1se to get the index after all events with 2 sub events
        gt_1se_hitIndex_mc = se_hits_index_mod_mc+1 
        gt_1se_hitIndex_mc = gt_1se_hitIndex_mc*gt_1se_mc
        gt_1se_hitIndex_mc = gt_1se_hitIndex_mc[gt_1se_hitIndex_mc > 0]
        gt_1se_hitIndex_data = se_hits_index_mod_data+1 
        gt_1se_hitIndex_data = gt_1se_hitIndex_data*gt_1se_data
        gt_1se_hitIndex_data = gt_1se_hitIndex_data[gt_1se_hitIndex_data > 0]

        cut_2inGate_mc = fq_fqOrder_fqipeak_mc[gt_1se_hitIndex_mc] == 0
        cut_2inGate_data = fq_fqOrder_fqipeak_data[gt_1se_hitIndex_data] == 0

        print(f"cut 2 in-gate decay E, MC : {np.unique(cut_2inGate_mc, return_counts=True)}")
        print(f"cut 2 in-gate decay E, data : {np.unique(cut_2inGate_data, return_counts=True)}")

        

        #preds_mc = preds_fqOrder_mc[fq_fqOrder_mc]

        preds_mc = preds_fqOrder_mc[cut_1decayE_mc]
        preds_mc = preds_mc[cut_2inGate_mc]
        preds_data = preds_fqOrder_data[cut_1decayE_data]
        preds_data = preds_data[cut_2inGate_data]


        #side-entering
        r_bool_cut_mc_se = (np.sqrt(np.square(preds_mc[:,0])+np.square(preds_mc[:,1])) > 1600) & (preds_mc[:,2] < 1750)
        r_bool_cut_data_se = (np.sqrt(np.square(preds_data[:,0])+np.square(preds_data[:,1])) > 1600) & (preds_data[:,2] < 1750)

        #top-entering
        r_bool_cut_mc_te = (np.sqrt(np.square(preds_mc[:,0])+np.square(preds_mc[:,1])) < 1600) & (preds_mc[:,2] > 1750)
        r_bool_cut_data_te = (np.sqrt(np.square(preds_data[:,0])+np.square(preds_data[:,1])) < 1600) & (preds_data[:,2] > 1750)

        print(f"ML pre-cut: {len(preds_mc)}")
        preds_mc_se = apply_quality_cuts(r_bool_cut_mc_se, preds_mc)
        preds_mc_te = apply_quality_cuts(r_bool_cut_mc_te, preds_mc)
        print(f"ML post-cut: {len(preds_mc)}")
        preds_data_se = apply_quality_cuts(r_bool_cut_data_se, preds_data)
        preds_data_te = apply_quality_cuts(r_bool_cut_data_te, preds_data)
        #truth_se = apply_quality_cuts(r_bool_cut_mc_se, truth)
        #truth_te = apply_quality_cuts(r_bool_cut_mc_te, truth)

        do_2d_endcap_plot(settings, np.squeeze(preds_data[:,0]), np.squeeze(preds_data[:,1]), np.squeeze(preds_data[:,2]), "stoppingMuons_ML_Data_position_endcap2Dmap.png")
        do_2d_endcap_plot(settings, np.squeeze(preds_mc[:,0]), np.squeeze(preds_mc[:,1]), np.squeeze(preds_mc[:,2]), "stoppingMuons_ML_mc_position_endcap2Dmap.png")

        save_events=False
        if save_events:
            high_z_data = (preds_data[:,2] > 1850) & (np.sqrt(np.square(preds_data[:,0])+np.square(preds_data[:,1])) < 1200) 
            low_z_data = (preds_data[:,2] < 1760) & (np.sqrt(np.square(preds_data[:,0])+np.square(preds_data[:,1])) < 1200) 
            print(f"number of high z data: {(preds_data[:,2])[high_z_data].shape}, low z data: {(preds_data[:,2])[low_z_data].shape}")
            print(f"high z rootfiles: {np.array([x.tobytes()  for x in data_rootfiles])[high_z_data]}, eventids: {data_eventids[high_z_data]}")
            print(f"low z rootfiles: {np.array([x.tobytes()  for x in data_rootfiles])[low_z_data]}, eventids: {data_eventids[low_z_data]}")

            high_z_mc = (preds_mc[:,2] > 1830) & (np.sqrt(np.square(preds_mc[:,0])+np.square(preds_mc[:,1])) < 1200) 
            low_z_mc = (preds_mc[:,2] < 1760) & (np.sqrt(np.square(preds_mc[:,0])+np.square(preds_mc[:,1])) < 1200) 
            print(f"number of high z mc: {(preds_mc[:,2])[high_z_mc].shape}, low z mc: {(preds_mc[:,2])[low_z_mc].shape}")
            print(f"MC high z rootfiles: {np.array([x.tobytes()  for x in mc_rootfiles])[high_z_mc]}, eventids: {mc_eventids[high_z_mc]}")
            print(f"MC low z rootfiles: {np.array([x.tobytes()  for x in mc_rootfiles])[low_z_mc]}, eventids: {mc_eventids[low_z_mc]}")

            np.savez('data/mc_rootfiles_eventids', rootfiles_high_z=np.array([x.tobytes()  for x in mc_rootfiles])[high_z_mc], 
                                                                            rootfiles_lowz=np.array([x.tobytes()  for x in mc_rootfiles])[low_z_mc],
                                                                            eventids_highz=mc_eventids[high_z_mc], eventids_lowz=mc_eventids[low_z_mc])
            np.savez('data/data_rootfiles_eventids', rootfiles_high_z=np.array([x.tobytes()  for x in data_rootfiles])[high_z_data], 
                                                                            rootfiles_lowz=np.array([x.tobytes()  for x in data_rootfiles])[low_z_data],
                                                                            eventids_highz=data_eventids[high_z_data], eventids_lowz=data_eventids[low_z_data])


        bins = 32 
        range=[1660,1750]
        #plot_with_ratio(np.ravel(np.sqrt(np.square(preds_data[:,0])+np.square(preds_data[:,1]))), np.ravel(np.sqrt(np.square(preds_mc[:,0])+np.square(preds_mc[:,1]))), 'Data Reco R [cm]', 'MC Reco R [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_R.png", 'Reco R Position [cm]', truth=np.sqrt(np.square(truth[:,0])+np.square(truth[:,1])))
        plot_with_ratio(np.ravel(np.sqrt(np.square(preds_data_se[:,0])+np.square(preds_data_se[:,1]))), np.ravel(np.sqrt(np.square(preds_mc_se[:,0])+np.square(preds_mc_se[:,1]))), 'data_se Reco R [cm]', 'mc_se Reco R [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_sideEntering_mc_data_position_R.png", 'Reco R Position [cm]' )
        bins = 32 
        range=[0,1600]
        plot_with_ratio(np.ravel(np.sqrt(np.square(preds_data_te[:,0])+np.square(preds_data_te[:,1]))), np.ravel(np.sqrt(np.square(preds_mc_te[:,0])+np.square(preds_mc_te[:,1]))), 'data_te Reco R [cm]', 'mc_te Reco R [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_topEntering_mc_data_position_R.png", 'Reco R Position [cm]' )
        bins = 40
        range=[-1690,2000]
        plot_with_ratio(np.ravel(preds_data_se[:,2]), np.ravel(preds_mc_se[:,2]), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_sideEntering_mc_Data_position_Z.png", 'Reco Z Position [cm]')
        bins = 40
        range=[1780,1880]
        plot_with_ratio(np.ravel(preds_data_te[:,2]), np.ravel(preds_mc_te[:,2]), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_mc_teData_position_Z_zoom.png", 'Reco Z Position [cm]')

        return 0

        mom_mc, mom_truth = get_ml_file(settings.aux_MLMCMom, 'momenta', isMC=True)
        mom_data = get_ml_file(settings.aux_MLDataMom, 'momenta')
        mom_data = apply_quality_cuts(r_bool_cut_data_se, mom_data)
        mom_mc = apply_quality_cuts(r_bool_cut_mc_se, mom_mc)

        z_min = [1780,1790,1800,1810,1820,1830]
        z_max = [1790,1800,1810,1820,1830,1840]
        for temp_z_min, temp_z_max in zip(z_min, z_max):
            laurel_plots(settings, preds_data[:,0], preds_data[:,1], preds_data[:,2], mom_data, "/laurel_plot_data_ML_x_mom_"+str(temp_z_min)+"_"+str(temp_z_max)+".png", temp_z_min, temp_z_max)



        print(f"x size: {np.squeeze(preds_data[:,0]).shape}, mom size: {np.ravel(mom_data).shape}")
        do_2d_endcap_plot(settings, np.squeeze(preds_data[:,0]), np.squeeze(preds_data[:,1]), np.ravel(mom_data), "stoppingMuons_ML_Data_positionMom_endcap2Dmap.png",200,1500)
        do_2d_endcap_plot(settings, np.squeeze(preds_mc[:,0]), np.squeeze(preds_mc[:,1]), np.ravel(mom_mc), "stoppingMuons_ML_mc_positionMom_endcap2Dmap.png",200,1500)

        
        
        
        dir_mc, dir_truth = get_ml_file(settings.aux_MLMCDir, 'directions', isMC=True)
        dir_data = get_ml_file(settings.aux_MLDataDir, 'directions')

        dir_mc = apply_quality_cuts(r_bool_cut_mc, dir_mc)
        dir_data = apply_quality_cuts(r_bool_cut_data, dir_data)
        dir_truth = apply_quality_cuts(r_bool_cut_mc, dir_truth)

        mom_mc, mom_truth = get_ml_file(settings.aux_MLMCMom, 'momenta', isMC=True)
        mom_data = get_ml_file(settings.aux_MLDataMom, 'momenta')

        mom_mc = apply_quality_cuts(r_bool_cut_mc, mom_mc)
        mom_data = apply_quality_cuts(r_bool_cut_data, mom_data)
        mom_truth = apply_quality_cuts(r_bool_cut_mc, mom_truth)
        
        pmt_positions = get_pmt_positions()
        top_endcap_pmt_positions = pmt_positions[pmt_positions[:,2]== 1810]
        min_distance = np.amin(scipy.spatial.distance.cdist(preds_data[:,0:2],top_endcap_pmt_positions[:,0:2],'euclidean'),axis=1)
        print(f"MIN DISTANCE: {min_distance.shape}")

        close_by_pmt_preds_data = preds_data[(min_distance < 10) & (dir_data[:,2] < -0.8)]
        far_from_pmt_preds_data = preds_data[(min_distance > 40) & (dir_data[:,2] < -0.8)]
        plot_with_ratio(np.ravel(close_by_pmt_preds_data[:,2]), np.ravel(far_from_pmt_preds_data[:,2]), 'X,Y < 10cm from PMT', 'X,Y > 40cm from PMT', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_PMTdistance_Z.png", 'Reco Z Position [cm]')
        do_2d_endcap_plot(settings, np.squeeze(close_by_pmt_preds_data[:,0]), np.squeeze(close_by_pmt_preds_data[:,1]), np.squeeze(close_by_pmt_preds_data[:,2]), "stoppingMuons_closePMTs_ML_Data_position_endcap2Dmap.png", num_bins=37*2)
        do_2d_endcap_plot(settings, np.squeeze(far_from_pmt_preds_data[:,0]), np.squeeze(far_from_pmt_preds_data[:,1]), np.squeeze(far_from_pmt_preds_data[:,2]), "stoppingMuons_farPMTs_ML_Data_position_endcap2Dmap.png", num_bins=37*2)
        do_2d_endcap_plot(settings, np.squeeze(top_endcap_pmt_positions[:,0]), np.squeeze(top_endcap_pmt_positions[:,1]), np.squeeze(top_endcap_pmt_positions[:,2]), "stoppingMuons_pmtPositions_ML_Data_position_endcap2Dmap.png", num_bins=37*2)


        dir_scan_min = [-1.00, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.25]
        dir_scan_max = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.25, 0.0]

        mom_scan_min = [0, 300, 400, 500, 800, 1100]
        mom_scan_max = [300, 400, 500, 800, 1100, 1500]

        temp_preds_data = preds_data[(dir_data[:,2] > -1.0) & (dir_data[:,2] < -0.8)]
        temp_preds_mc = preds_mc[(dir_mc[:,2] > -1.0) & (dir_mc[:,2] < -0.8)]
        do_2d_endcap_plot(settings, np.squeeze(temp_preds_data[:,0]), np.squeeze(temp_preds_data[:,1]), np.squeeze(temp_preds_data[:,2]), "stoppingMuons_lowDir_ML_Data_position_endcap2Dmap.png", num_bins=37*2)
        do_2d_endcap_plot(settings, np.squeeze(temp_preds_mc[:,0]), np.squeeze(temp_preds_mc[:,1]), np.squeeze(temp_preds_mc[:,2]), "stoppingMuons_lowDir_ML_mc_position_endcap2Dmap.png", num_bins=37*2)

        temp_preds_data = preds_data[(dir_data[:,2] > -0.5) & (dir_data[:,2] < 0.)]
        temp_preds_mc = preds_mc[(dir_mc[:,2] > -0.5) & (dir_mc[:,2] < 0.)]
        do_2d_endcap_plot(settings, np.squeeze(temp_preds_data[:,0]), np.squeeze(temp_preds_data[:,1]), np.squeeze(temp_preds_data[:,2]), "stoppingMuons_highDir_ML_Data_position_endcap2Dmap.png", num_bins=50)
        do_2d_endcap_plot(settings, np.squeeze(temp_preds_mc[:,0]), np.squeeze(temp_preds_mc[:,1]), np.squeeze(temp_preds_mc[:,2]), "stoppingMuons_highDir_ML_mc_position_endcap2Dmap.png", num_bins=50)

        for temp_dir_min, temp_dir_max in zip (dir_scan_min, dir_scan_max):
            temp_preds_data = preds_data[:,2][(dir_data[:,2] > temp_dir_min) & (dir_data[:,2] < temp_dir_max)]
            temp_preds_mc = preds_mc[:,2][(dir_mc[:,2] > temp_dir_min) & (dir_mc[:,2] < temp_dir_max)]
            plot_with_ratio(np.ravel(temp_preds_data), np.ravel(temp_preds_mc), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_Z_"+str(temp_dir_min)+"dir"+str(temp_dir_max)+".png", 'Reco Z Position [cm] ('+str(temp_dir_min)+'>dir<'+str(temp_dir_max)+')')

        for temp_mom_min, temp_mom_max in zip (mom_scan_min, mom_scan_max):
            temp_preds_data = preds_data[:,2][(np.ravel(mom_data) > temp_mom_min) & (np.ravel(mom_data) < temp_mom_max)]
            temp_preds_mc = preds_mc[:,2][(np.ravel(mom_mc) > temp_mom_min) & (np.ravel(mom_mc) < temp_mom_max)]
            plot_with_ratio(np.ravel(temp_preds_data), np.ravel(temp_preds_mc), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_Z_"+str(temp_mom_min)+"mom"+str(temp_mom_max)+".png", 'Reco Z Position [cm] ('+str(temp_mom_min)+'>mom<'+str(temp_mom_max)+')')

        
        angles_mc = math.angles_from_direction(dir_mc)
        towall_mc = math.towall(preds_mc, angles_mc, tank_axis = 2)
        dwall_mc = math.dwall(preds_mc, tank_axis = 2)
        angles_data = math.angles_from_direction(dir_data)
        towall_data = math.towall(preds_data, angles_data, tank_axis = 2)
        dwall_data = math.dwall(preds_data, tank_axis = 2)
        angles_truth = math.angles_from_direction(dir_truth)
        towall_truth = math.towall(truth, angles_truth, tank_axis = 2)
        dwall_truth = math.dwall(truth, tank_axis = 2)

        bins = 40
        range=[0,2000]

        plot_with_ratio(np.ravel(towall_data), np.ravel(towall_mc), 'Data ML towall [cm]', 'MC ML towall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_towall.png", 'Reco ML towall [cm]', truth=towall_truth)
        plot_with_ratio(np.ravel(towall_data), np.ravel(towall_mc), 'Data ML towall [cm]', 'MC ML towall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_towall_zoom.png", 'Reco ML towall [cm]', zoom=True)

        bins = 40
        range=[0,40]

        plot_with_ratio(np.ravel(dwall_data), np.ravel(dwall_mc), 'Data ML dwall [cm]', 'MC ML dwall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_dwall.png", 'Reco ML dwall [cm]')
        plot_with_ratio(np.ravel(dwall_data), np.ravel(dwall_mc), 'Data ML dwall [cm]', 'MC ML dwall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_position_dwall_zoom.png", 'Reco ML dwall [cm]', zoom=True)


    elif settings.doRegression and "directions" in settings.target:
        bins = 50 
        range=[-1.,1.]

        angles_mc = math.angles_from_direction(preds_mc)
        zenith_mc = np.cos(angles_mc[:,0]) 
        azimuth_mc = angles_mc[:,1]*180/np.pi 
        print(f"Azimuth mc min: {np.amin(azimuth_mc)}, max: {np.amax(azimuth_mc)}")

        angles_truth = math.angles_from_direction(truth)
        zenith_truth = np.cos(angles_truth[:,0]) 
        azimuth_truth = angles_truth[:,1]*180/np.pi 

        angles_data = math.angles_from_direction(preds_data)
        zenith_data = np.cos(angles_data[:,0]) 
        azimuth_data = angles_data[:,1]*180/np.pi 

        plot_with_ratio(np.ravel(preds_data[:,2]), np.ravel(preds_mc[:,2]), 'Data Reco Z Dir', 'MC Reco Z Dir', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_Z.png", 'Reco Z Dir', truth=truth[:,2])
        plot_with_ratio(np.ravel(preds_data[:,2]), np.ravel(preds_mc[:,2]), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_Z_zoom.png", 'Reco Z Dir', zoom=True)

        plot_with_ratio(np.ravel(preds_data[:,0]), np.ravel(preds_mc[:,0]), 'Data Reco X Dir', 'MC Reco X Dir', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_X.png", 'Reco X Dir', truth=truth[:,0])
        plot_with_ratio(np.ravel(preds_data[:,0]), np.ravel(preds_mc[:,0]), 'Data Reco X [cm]', 'MC Reco X [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_X_zoom.png", 'Reco X Dir', zoom=True)

        plot_with_ratio(np.ravel(preds_data[:,1]), np.ravel(preds_mc[:,1]), 'Data Reco Y Dir', 'MC Reco Y Dir', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_Y.png", 'Reco Y Dir', truth=truth[:,1])
        plot_with_ratio(np.ravel(preds_data[:,1]), np.ravel(preds_mc[:,1]), 'Data Reco Y [cm]', 'MC Reco Y [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_Y_zoom.png", 'Reco Y Dir', zoom=True)

        bins = 50 
        range=[-1.,0]

        plot_with_ratio(np.ravel(zenith_data), np.ravel(zenith_mc), 'Data Reco Zenith', 'MC Reco Zenith', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_zenith.png", 'Reco ML Zenith', truth=zenith_truth)
        plot_with_ratio(np.ravel(zenith_data), np.ravel(zenith_mc), 'Data Reco Zenith', 'MC Reco Zenith', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_zenith.png", 'Reco ML Zenith', zoom=True)

        bins = 50 
        range=[-180,180]

        plot_with_ratio(np.ravel(azimuth_data), np.ravel(azimuth_mc), 'Data Reco azimuth', 'MC Reco azimuth', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_azimuth.png", 'Reco ML azimuth', truth=azimuth_truth)
        plot_with_ratio(np.ravel(azimuth_data), np.ravel(azimuth_mc), 'Data Reco azimuth', 'MC Reco azimuth', bins, range, settings.outputPlotPath+"/stoppingMuons_ML_MCData_direction_azimuth.png", 'Reco ML azimuth', zoom=True)

    
    elif settings.doClassification:
        bins = 50 
        range=[0.,1.]
        plot_with_ratio(preds_data[:,0], preds_mc[:,0], 'Muon Softmax - Data', 'Muon Softmax - MC', bins, range, settings.outputPlotPath+"/stoppingMuons_MCData_classification_muons.png", 'Muon Softmax')
        plot_with_ratio(preds_data[:,1], preds_mc[:,1], 'Electron Softmax - Data', 'Electron Softmax - MC', bins, range, settings.outputPlotPath+"/stoppingMuons_MCData_classification_electrons.png", 'Electron Softmax')
        plot_with_ratio(preds_data[:,2], preds_mc[:,2], 'PiPlus Softmax - Data', 'PiPlus Softmax - MC', bins, range, settings.outputPlotPath+"/stoppingMuons_MCData_classification_piPlus.png", 'PiPlus Softmax')

def analyze_fitqun_regression_dataMC(settings, total_charge_cut, nhits_cut):
    file_mc = settings.FQMCPath
    file_data = settings.FQDataPath

    (_, labels_mc, _, fitqun_hash_mc), (mu_1rpos_mc, e_1rpos_mc, pi_1rpos_mc, mu_1rdir_mc, e_1rdir_mc, pi_1rdir_mc, mu_1rmom_mc, e_1rmom_mc, pi_1rmom_mc), fq_truth_mc, nhits_mc, qtot_mc= fq.read_fitqun_file(file_mc+'/fitqun_combine.hy', regression=True)
    (_, labels_data, _, fitqun_hash_data), (mu_1rpos_data, e_1rpos_data, pi_1rpos_data, mu_1rdir_data, e_1rdir_data, pi_1rdir_data, mu_1rmom_data, e_1rmom_data, pi_1rmom_data), fq_truth_data, nhits_data, qtot_data= fq.read_fitqun_file(file_data+'/fitqun_combine.hy', regression=True)

    print(f"Length pre-cut: {mu_1rpos_mc.shape}")
    mu_1rpos_mc, e_1rpos_mc, pi_1rpos_mc, mu_1rdir_mc, e_1rdir_mc, pi_1rdir_mc, mu_1rmom_mc, e_1rmom_mc, pi_1rmom_mc, = apply_cuts(mu_1rpos_mc, e_1rpos_mc, pi_1rpos_mc, mu_1rdir_mc, e_1rdir_mc, pi_1rdir_mc, mu_1rmom_mc, e_1rmom_mc, pi_1rmom_mc, qtot_mc, nhits_mc, total_charge_cut=total_charge_cut, nhits_cut=nhits_cut)
    print(f"Length post-cut: {mu_1rpos_mc.shape}")
    mu_1rpos_data, e_1rpos_data, pi_1rpos_data, mu_1rdir_data, e_1rdir_data, pi_1rdir_data, mu_1rmom_data, e_1rmom_data, pi_1rmom_data = apply_cuts(mu_1rpos_data, e_1rpos_data, pi_1rpos_data, mu_1rdir_data, e_1rdir_data, pi_1rdir_data, mu_1rmom_data, e_1rmom_data, pi_1rmom_data, qtot_data, nhits_data, total_charge_cut=total_charge_cut, nhits_cut=nhits_cut)

    if settings.doRegression and "momenta" in settings.target:
        bins = 28
        range=[100,1500]
        plot_with_ratio(np.ravel(mu_1rmom_data), np.ravel(mu_1rmom_mc), 'Data fiTQun momentum', 'MC fiTQun momentum', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_momentum.png", 'Reco Momentum [MeV]')
        plot_with_ratio(np.ravel(mu_1rmom_data), np.ravel(mu_1rmom_mc), 'Data fiTQun momentum', 'MC fiTQun momentum', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_momentum_zoom.png", 'Reco Momentum [MeV]', zoom=True)

    elif settings.doRegression and "positions" in settings.target:


        mu_1rpos_mc = np.squeeze(mu_1rpos_mc)
        mu_1rpos_data = np.squeeze(mu_1rpos_data)

        r_bool_cut_mc_se = (np.sqrt(np.square(mu_1rpos_mc[:,0])+np.square(mu_1rpos_mc[:,1])) > 1600) & (mu_1rpos_mc[:,2] < 1750)
        r_bool_cut_data_se = (np.sqrt(np.square(mu_1rpos_data[:,0])+np.square(mu_1rpos_data[:,1])) > 1600) & (mu_1rpos_data[:,2] < 1750)

        r_bool_cut_mc_te = (np.sqrt(np.square(mu_1rpos_mc[:,0])+np.square(mu_1rpos_mc[:,1])) < 1600) & (mu_1rpos_mc[:,2] > 1750)
        r_bool_cut_data_te = (np.sqrt(np.square(mu_1rpos_data[:,0])+np.square(mu_1rpos_data[:,1])) < 1600) & (mu_1rpos_data[:,2] > 1750)


        #r_bool_cut_mc = (np.ravel(np.sqrt(np.square(mu_1rpos_mc[:,0])+np.square(mu_1rpos_mc[:,1]))) < 1300) & (mu_1rpos_mc[:,2] > 1700)
        #r_bool_cut_data = (np.ravel(np.sqrt(np.square(mu_1rpos_data[:,0])+np.square(mu_1rpos_data[:,1]))) < 1300)  & (mu_1rpos_data[:,2] > 1700)

        print(f"ML pre-cut: {len(mu_1rpos_mc)}")
        mu_1rpos_mc_se = apply_quality_cuts(r_bool_cut_mc_se, mu_1rpos_mc)
        mu_1rdir_mc_se = apply_quality_cuts(r_bool_cut_mc_se, mu_1rdir_mc)
        mu_1rpos_mc_te = apply_quality_cuts(r_bool_cut_mc_te, mu_1rpos_mc)
        mu_1rdir_mc_te = apply_quality_cuts(r_bool_cut_mc_te, mu_1rdir_mc)
        print(f"ML post-cut: {len(mu_1rpos_mc)}")
        mu_1rpos_data_se = apply_quality_cuts(r_bool_cut_data_se, mu_1rpos_data)
        mu_1rdir_data_se = apply_quality_cuts(r_bool_cut_data_se, mu_1rdir_data)
        mu_1rpos_data_te = apply_quality_cuts(r_bool_cut_data_te, mu_1rpos_data)
        mu_1rdir_data_te = apply_quality_cuts(r_bool_cut_data_te, mu_1rdir_data)

        do_2d_endcap_plot(settings, np.squeeze(mu_1rpos_data_se[:,0]), np.squeeze(mu_1rpos_data_se[:,1]), np.squeeze(mu_1rpos_data_se[:,2]), "stoppingMuons_fiTQun_sideEntering_data_position_endcap2Dmap.png")
        do_2d_endcap_plot(settings, np.squeeze(mu_1rpos_data_te[:,0]), np.squeeze(mu_1rpos_data_te[:,1]), np.squeeze(mu_1rpos_data_te[:,2]), "stoppingMuons_fiTQun_topEntering_data_position_endcap2Dmap.png")
        do_2d_endcap_plot(settings, np.squeeze(mu_1rpos_mc_se[:,0]), np.squeeze(mu_1rpos_mc_se[:,1]), np.squeeze(mu_1rpos_mc_se[:,2]), "stoppingMuons_fiTQun_sideEntering_mc_position_endcap2Dmap.png")
        do_2d_endcap_plot(settings, np.squeeze(mu_1rpos_mc_te[:,0]), np.squeeze(mu_1rpos_mc_te[:,1]), np.squeeze(mu_1rpos_mc_te[:,2]), "stoppingMuons_fiTQun_topEntering_mc_position_endcap2Dmap.png")


        bins = 32 
        range=[1660,1720]
        plot_with_ratio(np.ravel(np.sqrt(np.square(mu_1rpos_data_se[:,0])+np.square(mu_1rpos_data_se[:,1]))), np.ravel(np.sqrt(np.square(mu_1rpos_mc_se[:,0])+np.square(mu_1rpos_mc_se[:,1]))), 'Data Reco R [cm]', 'MC Reco R [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_sideEntering_mc_Data_position_R.png", 'Reco R Position [cm]')
        bins = 32 
        range=[0,1600]
        plot_with_ratio(np.ravel(np.sqrt(np.square(mu_1rpos_data_te[:,0])+np.square(mu_1rpos_data_te[:,1]))), np.ravel(np.sqrt(np.square(mu_1rpos_mc_te[:,0])+np.square(mu_1rpos_mc_te[:,1]))), 'Data Reco R [cm]', 'MC Reco R [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_topEntering_mc_Data_position_R.png", 'Reco R Position [cm]')
        bins = 40
        range=[-1690,2000]
        plot_with_ratio(np.ravel(mu_1rpos_data_se[:,2]), np.ravel(mu_1rpos_mc_se[:,2]), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_sideEntering_mc_Data_position_Z.png", 'Reco Z Position [cm]')
        bins = 40
        range=[1750,1850]
        plot_with_ratio(np.ravel(mu_1rpos_data_te[:,2]), np.ravel(mu_1rpos_mc_te[:,2]), 'Data Reco Z [cm]', 'MC Reco Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_topEntering_mc_Data_position_Z.png", 'Reco Z Position [cm]')

        angles_mc = math.angles_from_direction(mu_1rdir_mc)
        towall_mc = math.towall(mu_1rpos_mc, angles_mc, tank_axis = 2)
        dwall_mc = math.dwall(mu_1rpos_mc, tank_axis = 2)
        angles_data = math.angles_from_direction(mu_1rdir_data)
        towall_data = math.towall(mu_1rpos_data, angles_data, tank_axis = 2)
        dwall_data = math.dwall(mu_1rpos_data, tank_axis = 2)

        bins = 40
        range=[0,2000]

        plot_with_ratio(np.ravel(towall_data), np.ravel(towall_mc), 'Data fiTQun towall [cm]', 'MC fiTQun towall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_position_towall.png", 'Reco fiTQun towall [cm]')
        plot_with_ratio(np.ravel(towall_data), np.ravel(towall_mc), 'Data fiTQun towall [cm]', 'MC fiTQun towall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_position_towall_zoom.png", 'Reco fiTQun towall [cm]', zoom=True)

        bins = 40
        range=[0,40]

        plot_with_ratio(np.ravel(dwall_data), np.ravel(dwall_mc), 'Data fiTQun dwall [cm]', 'MC fiTQun dwall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_position_dwall.png", 'Reco fiTQun dwall [cm]')
        plot_with_ratio(np.ravel(dwall_data), np.ravel(dwall_mc), 'Data fiTQun dwall [cm]', 'MC fiTQun dwall [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_position_dwall_zoom.png", 'Reco fiTQun dwall [cm]', zoom=True)

    elif settings.doRegression and "directions" in settings.target:
        bins = 50 
        range=[-1.,1.]
        mu_1rdir_mc = np.squeeze(mu_1rdir_mc)
        mu_1rdir_data = np.squeeze(mu_1rdir_data)
        plot_with_ratio(np.ravel(mu_1rdir_data[:,2]), np.ravel(mu_1rdir_mc[:,2]), 'Data fiTQun Z Dir', 'MC fiTQun Z Dir', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_Z.png", 'fiTQun Z Dir')
        plot_with_ratio(np.ravel(mu_1rdir_data[:,2]), np.ravel(mu_1rdir_mc[:,2]), 'Data fiTQun Z [cm]', 'MC fiTQun Z [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_Z_zoom.png", 'fiTQun Z Dir', zoom=True)

        plot_with_ratio(np.ravel(mu_1rdir_data[:,0]), np.ravel(mu_1rdir_mc[:,0]), 'Data fiTQun X Dir', 'MC fiTQun X Dir', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_X.png", 'fiTQun X Dir')
        plot_with_ratio(np.ravel(mu_1rdir_data[:,0]), np.ravel(mu_1rdir_mc[:,0]), 'Data fiTQun X [cm]', 'MC fiTQun X [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_X_zoom.png", 'fiTQun X Dir', zoom=True)

        plot_with_ratio(np.ravel(mu_1rdir_data[:,1]), np.ravel(mu_1rdir_mc[:,1]), 'Data fiTQun Y Dir', 'MC fiTQun Y Dir', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_Y.png", 'fiTQun Y Dir')
        plot_with_ratio(np.ravel(mu_1rdir_data[:,1]), np.ravel(mu_1rdir_mc[:,1]), 'Data fiTQun Y [cm]', 'MC fiTQun Y [cm]', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_Y_zoom.png", 'fiTQun Y Dir', zoom=True)

        angles_mc = math.angles_from_direction(mu_1rdir_mc)
        zenith_mc = np.cos(angles_mc[:,0]) 
        azimuth_mc = angles_mc[:,1]*180/np.pi 

        angles_data = math.angles_from_direction(mu_1rdir_data)
        zenith_data = np.cos(angles_data[:,0]) 
        azimuth_data = angles_data[:,1]*180/np.pi 

        bins = 50 
        range=[-1.,0.]

        plot_with_ratio(np.ravel(zenith_data), np.ravel(zenith_mc), 'Data fiTQun Zenith', 'MC fiTQun Zenith', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_zenith.png", 'Reco fiTQun Zenith')
        plot_with_ratio(np.ravel(zenith_data), np.ravel(zenith_mc), 'Data fiTQun Zenith', 'MC fiTQun Zenith', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_zenith.png", 'Reco fiTQun Zenith', zoom=True)

        bins = 50
        range=[-180,180]

        plot_with_ratio(np.ravel(azimuth_data), np.ravel(azimuth_mc), 'Data fiTQun azimuth', 'MC fiTQun azimuth', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_azimuth.png", 'Reco fiTQun azimuth')
        plot_with_ratio(np.ravel(azimuth_data), np.ravel(azimuth_mc), 'Data fiTQun azimuth', 'MC fiTQun azimuth', bins, range, settings.outputPlotPath+"/stoppingMuons_fiTQun_MCData_direction_azimuth.png", 'Reco fiTQun azimuth', zoom=True)
    
    elif settings.doClassification:
        bins = 50 
        range=[0.,1.]
        plot_with_ratio(preds_data[:,0], preds_mc[:,0], 'Muon Softmax - Data', 'Muon Softmax - MC', bins, range, settings.outputPlotPath+"/stoppingMuons_MCData_classification_muons.png", 'Muon Softmax')
        plot_with_ratio(preds_data[:,1], preds_mc[:,1], 'Electron Softmax - Data', 'Electron Softmax - MC', bins, range, settings.outputPlotPath+"/stoppingMuons_MCData_classification_electrons.png", 'Electron Softmax')
        plot_with_ratio(preds_data[:,2], preds_mc[:,2], 'PiPlus Softmax - Data', 'PiPlus Softmax - MC', bins, range, settings.outputPlotPath+"/stoppingMuons_MCData_classification_piPlus.png", 'PiPlus Softmax')