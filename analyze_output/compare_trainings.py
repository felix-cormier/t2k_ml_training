import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

import math


def calculate_dwall(position):
    """Generates wall and towall variables for input position and direction

    Args:
        position (_type_): three-coordinate position
        direction (_type_): three-coordinate direction

    Returns:
        float, float: wall, towall variables
    """

    #Calculate wall variables
    min_vertical_wall = 1810-np.abs(position[:,2])
    min_horizontal_wall = 1690 - np.sqrt(position[:,0]*position[:,0] + position[:,1]*position[:,1])
    wall = np.minimum(min_vertical_wall,min_horizontal_wall)

    return wall

def get_pred_truth_values(path, variable):
    pred = np.load(path+"/predicted_"+variable+".npy")
    true = np.load(path+"/"+variable+".npy")
    indices = np.load(path+"indices.npy") 
    labels = np.load(path+"labels.npy") 

    return pred, true, indices, labels

def do_histogram(x, x_label, y_label, label, dir_name, savename, bins=None, range=None, clear=True, norm=False, y_log=False):
    if bins is not None and range is not None:
        plt.hist(x, label=label, bins=bins, range=range, alpha=0.5, density=norm)
    else:
        plt.hist(x, label=label, density=norm)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if y_log:
        plt.yscale('log')
    #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=3)    
    plt.legend()
    plt.savefig("/data/fcormier/t2k/ml/training/plots/" + dir_name + "/" + savename + ".png")
    if clear:
        plt.clf()

path_new = "/data/fcormier/training/resnet_aug22_electrons_position_gt200Hits_fullyContained_timeO400S1000_2GeV_4M_1/23082024-100932/"
path_old = "/data/fcormier/training/resnet_sep5_electrons_positions_gt200Hits_fullyContained_janData_timeO400S1000_inference_2/06092024-083139/SK4DeadPMTs/"
variable  = "positions"
label = 1
plot_directory = "new_old_e_pos_compare_4/"
do_res_cuts=True
do_coord_cuts=True

pred_new, true_new, indices_new, labels_new = get_pred_truth_values(path_new, variable)
pred_old, true_old, indices_old, labels_old = get_pred_truth_values(path_old, variable)


new_sort = np.argsort(indices_new)
old_sort = np.argsort(indices_old)

pred_new = pred_new[new_sort]
true_new = true_new[new_sort]
indices_new = indices_new[new_sort]
labels_new = labels_new[new_sort]

pred_old = pred_old[old_sort]
true_old = true_old[old_sort]
indices_old = indices_old[old_sort]
labels_old = labels_old[old_sort]


print(f"len new: {len(indices_new)}, len old: {len(indices_old)} ")
if len(indices_old) == len(indices_new)+1:
    print("old had an two copies of first event, removing duplicate")
    pred_old = pred_old[1:]
    true_old = true_old[1:]
    indices_old = indices_old[1:]
    labels_old = labels_old[1:]

print(np.unique(labels_old, return_counts=True))
print(np.unique(labels_new, return_counts=True))

pred_new = pred_new[labels_new==label]
true_new = true_new[labels_new==label]
indices_new = indices_new[labels_new==label]
labels_new = labels_new[labels_new==label]

pred_old = pred_old[labels_old==label]
true_old = true_old[labels_old==label]
indices_old = indices_old[labels_old==label]
labels_old = labels_old[labels_old==label]


print("Started dwall")

dwall_old = calculate_dwall(true_old)
dwall_new = calculate_dwall(true_new)

pred_dwall_old = calculate_dwall(pred_old)
pred_dwall_new = calculate_dwall(pred_new)

print("Finished dwall")


if np.array_equal(indices_new,indices_old):
    print("SAME INDICES!")
    do_histogram(pred_old[:,0], "Pred X_old" , "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "X_pred_old", bins=100, range=[-1700,1700],clear=False)
    do_histogram(pred_new[:,0], "Pred X_new" , "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "X_pred_new", bins=100, range=[-1700,1700])
    do_histogram(dwall_old, "dwall [mm]" , "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "dwall_old", bins=100, range=[0,800],clear=False)
    do_histogram(dwall_new, "dwall[mm]" , "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "dwall_new", bins=100, range=[0,800])
    do_histogram(pred_old[:,1], "Pred Y_old" , "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "Y_pred_old", bins=100, range=[-1700,1700], clear=False)
    do_histogram(pred_new[:,1], "Pred Y_new" , "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "Y_pred_new", bins=100, range=[-1700,1700])
    do_histogram(pred_old[:,2], "Pred Z_old" , "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "Z_pred_old", bins=100, range=[-1900,1900], clear=False)
    do_histogram(pred_new[:,2], "Pred Z_new" , "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "Z_pred_new", bins=100, range=[-1900,1900])
    pred_diff = pred_old - pred_new
    do_histogram(pred_diff[:,0], "Pred X_old - Pred X_new", "Num. Events", "Electron Position", dir_name = plot_directory, savename = "X_pred_diff", bins=100, range=[-100,100])
    do_histogram(pred_diff[:,1], "Pred Y_old - Pred Y_new", "Num. Events", "Electron Position", dir_name = plot_directory, savename = "Y_pred_diff", bins=100, range=[-100,100])
    do_histogram(pred_diff[:,2], "Pred Z_old - Pred Z_new", "Num. Events", "Electron Position", dir_name = plot_directory, savename = "Z_pred_diff", bins=100, range=[-100,100])
    res_old = pred_old - true_old
    res_global_old = np.sqrt(np.add(np.add(np.square(np.subtract(pred_old[:,0], true_old[:,0])), np.square(np.subtract(pred_old[:,1], true_old[:,1]))), np.square(np.subtract(pred_old[:,2], true_old[:,2]))))
    res_new = pred_new - true_new
    res_global_new = np.sqrt(np.add(np.add(np.square(np.subtract(pred_new[:,0], true_new[:,0])), np.square(np.subtract(pred_new[:,1], true_new[:,1]))), np.square(np.subtract(pred_new[:,2], true_new[:,2]))))
    res_diff = np.abs(res_old) - np.abs(res_new)
    do_histogram(res_diff[:,0], "Residual X_old - Residual X_new", "Num. Events", "Electron Position", dir_name = plot_directory, savename = "X_res_diff", bins=100, range=[-100,100])
    do_histogram(res_diff[:,1], "Residual Y_old - Residual Y_new", "Num. Events", "Electron Position", dir_name = plot_directory, savename = "Y_res_diff", bins=100, range=[-100,100])
    do_histogram(res_diff[:,2], "Residual Z_old - Residual Z_new", "Num. Events", "Electron Position", dir_name = plot_directory, savename = "Z_res_diff", bins=100, range=[-100,100])


    do_histogram(res_old[:,0], "Residual X_old", "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "X_res_old", bins=100, range=[-100,100],clear=False, y_log=True)
    do_histogram(res_new[:,0], "Residual X_new", "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "X_res_new", bins=100, range=[-100,100], y_log=True)
    do_histogram(res_old[:,1], "Residual Y_old", "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "Y_res_old", bins=100, range=[-100,100], clear=False, y_log=True)
    do_histogram(res_new[:,1], "Residual Y_new", "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "Y_res_new", bins=100, range=[-100,100], y_log=True)
    do_histogram(res_old[:,2], "Residual Z_old", "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "Z_res_old", bins=100, range=[-100,100],clear=False, y_log=True)
    do_histogram(res_new[:,2], "Residual Z_new", "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "Z_res_new", bins=100, range=[-100,100], y_log=True)

    do_histogram(res_global_old, "Residual Global [mm]", "Num. Events", "El. Pos. Old Data", dir_name = plot_directory, savename = "Global_res_old", bins=100, range=[0,200],clear=False, y_log=True)
    do_histogram(res_global_new, "Residual Global [mm]", "Num. Events", "El. Pos. New Data", dir_name = plot_directory, savename = "Global_res_new", bins=100, range=[0,200], y_log=True)

    if do_res_cuts:

        res_cuts_min = [0,2,5,10,15,20,50]
        res_cuts_max = [2,5,10,15,20,50,500]

        for cut_min, cut_max in zip(res_cuts_min, res_cuts_max):
            do_histogram((pred_old[(np.abs(res_old[:,0]) > cut_min) & (np.abs(res_old[:,0]) < cut_max)])[:,0], "Pred X_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "X_pred_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700],clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_new[:,0]) > cut_min) & (np.abs(res_new[:,0]) < cut_max)])[:,0], "Pred X_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "X_pred_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((pred_old[(np.abs(res_old[:,0]) > cut_min) & (np.abs(res_old[:,0]) < cut_max)])[:,0], "Pred X_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "X_pred_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700],clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_new[:,0]) > cut_min) & (np.abs(res_new[:,0]) < cut_max)])[:,0], "Pred X_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "X_pred_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((pred_old[(np.abs(res_old[:,1]) > cut_min) & (np.abs(res_old[:,1]) < cut_max)])[:,1], "Pred Y_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Y_pred_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_new[:,1]) > cut_min) & (np.abs(res_new[:,1]) < cut_max)])[:,1], "Pred Y_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Y_pred_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((pred_old[(np.abs(res_old[:,2]) > cut_min) & (np.abs(res_old[:,2]) < cut_max)])[:,2], "Pred Z_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Z_pred_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_new[:,2]) > cut_min) & (np.abs(res_new[:,2]) < cut_max)])[:,2], "Pred Z_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Z_pred_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], norm=True)

            do_histogram((true_old[(np.abs(res_old[:,0]) > cut_min) & (np.abs(res_old[:,0]) < cut_max)])[:,0], "true X_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "X_true_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700],clear=False, norm=True)
            do_histogram((true_new[(np.abs(res_new[:,0]) > cut_min) & (np.abs(res_new[:,0]) < cut_max)])[:,0], "true X_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "X_true_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((true_old[(np.abs(res_old[:,1]) > cut_min) & (np.abs(res_old[:,1]) < cut_max)])[:,1], "true Y_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Y_true_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], clear=False, norm=True)
            do_histogram((true_new[(np.abs(res_new[:,1]) > cut_min) & (np.abs(res_new[:,1]) < cut_max)])[:,1], "true Y_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Y_true_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((true_old[(np.abs(res_old[:,2]) > cut_min) & (np.abs(res_old[:,2]) < cut_max)])[:,2], "true Z_old" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Z_true_old_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], clear=False, norm=True)
            do_histogram((true_new[(np.abs(res_new[:,2]) > cut_min) & (np.abs(res_new[:,2]) < cut_max)])[:,2], "true Z_new" , "Arb.", "El Pos., Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Z_true_new_resDiffCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], norm=True)

            do_histogram((pred_old[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,0], "Pred X_old" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "X_pred_old_GlobalResCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700],clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,0], "Pred X_new" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "X_pred_new_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((pred_old[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,1], "Pred Y_old" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Y_pred_old_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,1], "Pred Y_new" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Y_pred_new_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((pred_old[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,2], "Pred Z_old" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Z_pred_old_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], clear=False, norm=True)
            do_histogram((pred_new[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,2], "Pred Z_new" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Z_pred_new_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], norm=True)

            do_histogram((true_old[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,0], "true X_old" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "X_true_old_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700],clear=False, norm=True)
            do_histogram((true_new[(np.abs(res_global_new) > cut_min) & (np.abs(res_global_new) < cut_max)])[:,0], "true X_new" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "X_true_new_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((true_old[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,1], "true Y_old" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Y_true_old_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], clear=False, norm=True)
            do_histogram((true_new[(np.abs(res_global_new) > cut_min) & (np.abs(res_global_new) < cut_max)])[:,1], "true Y_new" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Y_true_new_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[-1700,1700], norm=True)
            do_histogram((true_old[(np.abs(res_global_old) > cut_min) & (np.abs(res_global_old) < cut_max)])[:,2], "true Z_old" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, Old Data", dir_name = plot_directory, savename = "Z_true_old_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], clear=False, norm=True)
            do_histogram((true_new[(np.abs(res_global_new) > cut_min) & (np.abs(res_global_new) < cut_max)])[:,2], "true Z_new" , "Arb.", "El Pos., Global Res > "+str(cut_min)+", < " + str(cut_max) +"cm res diff, New Data", dir_name = plot_directory, savename = "Z_true_new_GlobalresCutMin"+str(cut_min)+"Max"+str(cut_max), bins=100, range=[-1900,1900], norm=True)
        
    if do_coord_cuts:
        coord_cuts_min = [0,20,50,100,200,400,600,800,1000,1700]
        coord_cuts_max = [20,50,100,200,400,600,800,1000,1700,2000]

        for cut_min, cut_max in zip(coord_cuts_min, coord_cuts_max):
            do_histogram((res_global_old[(np.abs(pred_old[:,0]) > cut_min) & (np.abs(pred_old[:,0]) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., X > "+str(cut_min)+", < " + str(cut_max) +"cm, Old Data", dir_name = plot_directory, savename = "Res_old_XCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],clear=False, norm=True)
            do_histogram((res_global_new[(np.abs(pred_new[:,0]) > cut_min) & (np.abs(pred_new[:,0]) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., X > "+str(cut_min)+", < " + str(cut_max) +"cm, New Data", dir_name = plot_directory, savename = "Res_new_XCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],norm=True)

            do_histogram((res_global_old[(np.abs(pred_old[:,1]) > cut_min) & (np.abs(pred_old[:,1]) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., Y > "+str(cut_min)+", < " + str(cut_max) +"cm, Old Data", dir_name = plot_directory, savename = "Res_old_YCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],clear=False, norm=True)
            do_histogram((res_global_new[(np.abs(pred_new[:,1]) > cut_min) & (np.abs(pred_new[:,1]) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., Y > "+str(cut_min)+", < " + str(cut_max) +"cm, New Data", dir_name = plot_directory, savename = "Res_new_YCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],norm=True)

            do_histogram((res_global_old[(np.abs(pred_old[:,2]) > cut_min) & (np.abs(pred_old[:,2]) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., Z > "+str(cut_min)+", < " + str(cut_max) +"cm, Old Data", dir_name = plot_directory, savename = "Res_old_ZCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],clear=False, norm=True)
            do_histogram((res_global_new[(np.abs(pred_new[:,2]) > cut_min) & (np.abs(pred_new[:,2]) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., Z > "+str(cut_min)+", < " + str(cut_max) +"cm, New Data", dir_name = plot_directory, savename = "Res_new_ZCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],norm=True)

            do_histogram((res_global_old[(np.abs(dwall_old) > cut_min) & (np.abs(dwall_old) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., dwall > "+str(cut_min)+", < " + str(cut_max) +"cm, Old Data", dir_name = plot_directory, savename = "Res_old_dwallCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],clear=False, norm=True)
            do_histogram((res_global_new[(np.abs(dwall_new) > cut_min) & (np.abs(dwall_new) < cut_max)]), "Resolution [mm]" , "Arb.", "El Pos., dwall > "+str(cut_min)+", < " + str(cut_max) +"cm, New Data", dir_name = plot_directory, savename = "Res_new_dwallCutMin"+str(cut_min)+"Max"+str(cut_max), bins=50, range=[0,100],norm=True)


else:
    print("DIFFERENT INDICES!")


