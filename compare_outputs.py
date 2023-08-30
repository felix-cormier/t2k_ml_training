import glob
import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve
from plotting import efficiency_plots
from sklearn import metrics
from torchmetrics import AUROC, ROC
import h5py

filename = "/fast_scratch/ipress/emu/jul31_emu_decayEVar_2/combine_combine.hy"
h5fw = h5py.File(filename, 'r')

def convert_label(label):
    if label == 2:
        return 'Gamma'
    if label == 1:
        return 'Electron'
    if label == 0:
        return 'Muon'
    else:
        return label

def get_cherenkov_threshold(label):
    threshold_dict = {0: 160., 1:0.8, 2: 0.}
    return threshold_dict[label]

class dealWithOutputs():
    def __init__(self, directory) -> None:
        self.directory = directory
        self.list_of_directories = []
        self.list_of_indices_file = []
        self.list_of_input_variables = []
        self.list_of_output_stats = []
        self.list_of_input_files = []
    
    def add_output(self, directory, indices_file, inputFile, input_variables, output_stats):
        self.list_of_directories.append(directory)
        self.list_of_indices_file.append(indices_file)
        self.list_of_input_variables.append(input_variables)
        self.list_of_output_stats.append(output_stats)
        self.list_of_input_files.append(inputFile)

    def find_unique(self):
        set_of_same = []
        #Find which dictionaries are the same
        for i, item_1 in enumerate(self.list_of_input_variables):
            for j, item_2 in enumerate(self.list_of_input_variables):
                #if item_1 == item_2 and i!=j:
                #    set_of_same.append([i,j])
                set_of_same.append([i,j])
        #            print('set_of_same = ', set_of_same)
    
        out = []
        #Combine all same dictionaries into unique sets
        while len(set_of_same)>0:
            first, *rest = set_of_same
            first = set(first)
            print('first1 = ', first)
            lf = -1
            while len(first)>lf:
                lf = len(first)
                
                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r)))>0:
                        first |= set(r)
                        print('first2 = ', first)
                    else:
                        rest2.append(r)     
                rest = rest2
                print('rest = ', rest)
            out.append(first)
            set_of_same = rest

        self.unique_sets = out
        print('self.unique_sets', self.unique_sets)
        

    def calc_stats(self):
        self.final_stats_dict = []
        self.final_variable_dict = []
        for set in self.unique_sets:
            temp_stats = {}
            for index in list(set):
                for stat in self.list_of_output_stats[index]:
                    if stat in temp_stats:
                        temp_stats[stat].append(float(self.list_of_output_stats[index][stat]))
                    else:
                        temp_stats[stat] = [float(self.list_of_output_stats[index][stat])]
            for stat in temp_stats:
                temp_stats[stat] = [np.mean(temp_stats[stat]), np.std(temp_stats[stat])]
            self.final_stats_dict.append(temp_stats)
            self.final_variable_dict.append(self.list_of_input_variables[list(set)[0]])
        #print('self.unique = ', self.unique_sets)
        print('self.final_variable_dict = ', self.final_variable_dict)
        print('self.final_stats_dict = ', self.final_stats_dict)

        #Assumes all dictionaries have same variables
        '''
        for key in self.final_variable_dict[0]:
            learning_rates = [d[key] for d in self.final_variable_dict]
            resultset = [key_2 for key_2, value in self.final_variable_dict[0].items() if key not in key_2]
            for i, items in self.final_variable_dict:
                label = ''
                for var in resultset:
                    pass
                    

            print(learning_rates)
            print(resultset)
        '''
        
    def set_output_directory(self, path):
        """Makes an output file as given in the arguments
        """
        if not(os.path.exists(path) and os.path.isdir(path)):
            try:
                os.makedirs(path)
            except FileExistsError as error:
                print("Directory " + str(path) +" already exists")

    def convert_variables_to_label(self,variable_dict):
        output_string = ''
        for i, key in enumerate(variable_dict):
            if i > 0:
                output_string = output_string + '\n'
            output_string = output_string + key +': ' + variable_dict[key]
        return output_string

    def make_plots(self):
        print(1)
        plot_folder = self.directory + '/plots/'
        self.set_output_directory(plot_folder)
        print(2)
        for i, dir in enumerate(self.list_of_directories):
            print(dir)
            self.set_output_directory(plot_folder+dir.replace(self.directory,''))
            plot_output = plot_folder+dir.replace(self.directory,'')+"/"
            label = self.convert_variables_to_label(self.list_of_input_variables[i])
            run = efficiency_plots(self.list_of_input_files[i], '', dir, plot_output, label=label)
            fig, ax1, ax2 = run.plot_training_progression(y_loss_lim=[0.,2.], doAccuracy=True, label=label)
            fig.tight_layout(pad=2.0) 
            fig.savefig(plot_output + 'log_test.png', format='png')

    def roc(self):
        plot_folder = (self.directory + '/plots/')
        self.set_output_directory(plot_folder)
        for i, dir in enumerate(self.list_of_directories):
            #set output directories
            self.set_output_directory(plot_folder+dir.replace(self.directory,''))
            plot_output = plot_folder+dir.replace(self.directory,'')+"/"
            
            #load softmax and labels and create roc_curve's and calculate the AUC
            softmaxes = np.load(self.list_of_directories[i]+'/softmax.npy')
            labels = np.load(self.list_of_directories[i]+'/labels.npy')
            fpr, tpr, _ = metrics.roc_curve(torch.tensor(labels),torch.tensor(softmaxes[:,1]))
            auroc = AUROC(task="binary")
            auc = auroc (torch.tensor(softmaxes[:,1]),torch.tensor(labels))
            auc = (round(float(auc), 5))
            
            #plot roc curve
            plt.plot(fpr*100, tpr*100, label = 'ROC curve')
            plt.xlabel('False Positive Rate (%)')
            plt.ylabel('True Positive Rate (%)')
            plt.grid()
            plt.legend(title = f'AUC: {auc}')
            plt.title('ROC Curve')
            plt.savefig(plot_output + 'roc.png', format='png')
            plt.close()
            
            #plot high efficiency roc curve
            plt.plot(tpr*100, (100/fpr), label = 'ROC curve')
            plt.xlabel('True Positive Rate (%)')
            plt.ylabel('1/False Positive Rate (1/%)')
            plt.grid()
            plt.xlim(60, 110)
            plt.yscale('log')
            plt.legend(title = f'AUC: {auc}')
            plt.title('High efficiency ROC Curve')
            plt.savefig(plot_output + 'roc_High_Eff.png', format='png')
            plt.close()


    def roc_overlapper(self):
        plot_folder = (self.directory + '/plots/')
        self.set_output_directory(plot_folder)
        for i, dir in enumerate(self.list_of_directories):
            #set output directories
            self.set_output_directory(plot_folder+dir.replace(self.directory,''))
            plot_output = plot_folder+dir.replace(self.directory,'')+"/"
            name = self.list_of_directories[i].split('/')[-1]
            
            #load softmax and labels and create roc_curve's and calculate the AUC
            softmaxes = np.load(self.list_of_directories[i]+'/softmax.npy')
            labels = np.load(self.list_of_directories[i]+'/labels.npy')
            fpr, tpr, _ = metrics.roc_curve(torch.tensor(labels),torch.tensor(softmaxes[:,1]))
            auroc = AUROC(task="binary")
            auc = auroc (torch.tensor(softmaxes[:,1]),torch.tensor(labels))
            auc = (round(float(auc), 5))
            
            #plot roc curve
            plt.plot(fpr*100, tpr*100, label = f'ROC curve {name} \nAUC = {auc}')
            plt.xlabel('False Positive Rate (%)')
            plt.ylabel('True Positive Rate (%)')
            plt.grid()
            plt.legend()
            plt.title('ROC Curve')
        plt.savefig(plot_folder + 'roc_combined.png', format='png')
        plt.close()    


    def he_roc_overlapper(self):
        plot_folder = (self.directory + '/plots/')
        self.set_output_directory(plot_folder)
        for i, dir in enumerate(self.list_of_directories):
            #set output directories
            self.set_output_directory(plot_folder+dir.replace(self.directory,''))
            plot_output = plot_folder+dir.replace(self.directory,'')+"/"
            name = self.list_of_directories[i].split('/')[-1]
            
            #load softmax and labels and create roc_curve's and calculate the AUC
            softmaxes = np.load(self.list_of_directories[i]+'/softmax.npy')
            labels = np.load(self.list_of_directories[i]+'/labels.npy')
            fpr, tpr, _ = metrics.roc_curve(torch.tensor(labels),torch.tensor(softmaxes[:,1]))
            auroc = AUROC(task="binary")
            auc = auroc (torch.tensor(softmaxes[:,1]),torch.tensor(labels))
            auc = (round(float(auc), 5))
            
            #plot high efficiency roc curve
            plt.plot(tpr*100, (100/fpr), label = f'ROC curve {name} \nAUC = {auc}')
            plt.xlabel('True Positive Rate (%)')
            plt.ylabel('1/False Positive Rate (1/%)')
            plt.grid()
            plt.xlim(60, 110)
            plt.yscale('log')
            plt.legend()
            plt.title('High efficiency ROC Curve')
        plt.savefig(plot_folder + 'roc_High_Eff_combined.png', format='png')
        plt.close()

def compare_outputs(folder):
    dirs =  glob.glob(folder+'/*')
    outputs = dealWithOutputs(folder)
    for dir in dirs:
        #Checks if training is done by checking if training_stats.xml is output
        try:
            tree = ET.parse(dir+'/training_stats.xml')
        except FileNotFoundError:
            continue
        root = tree.getroot()
        input_variables = {}
        output_stats = {}
        indices_file = {}
        input_file = ''
        for child in root:
            if 'Variables' in child.tag:
                for child_2 in child:
                    if 'indicesFile' in child_2.tag:
                        indices_file[child_2.tag] = child_2.attrib['var']
                    else:
                        input_variables[child_2.tag] = child_2.attrib['var']
            if 'Stats' in child.tag:
                for child_2 in child:
                    output_stats[child_2.tag] = child_2.attrib['var']
            if 'Files' in child.tag:
                for child_2 in child:
                    if 'inputPath' in child_2.tag:
                        inputFile = child_2.attrib['var']
        outputs.add_output(dir, indices_file, inputFile, input_variables, output_stats)
    #outputs.he_roc_overlapper()
    #outputs.roc_overlapper()
    #outputs.roc()
    outputs.find_unique()
    outputs.calc_stats()
    outputs.make_plots()
    #print('outputs.find_unique() = ', outputs.find_unique())
    #print('outputs.calc_stats() = ', outputs.calc_stats())
    #print('outputs.make_plots() = ', outputs.make_plots())
