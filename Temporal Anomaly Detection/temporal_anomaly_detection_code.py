'''
Anomaly detection of of eigenvalue time-series data.
'''

#Import Necessary Packages
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from numpy import load

### Transferring data: CONTROL Data ###
control_names = ["DsRed2-HeLa_10_26_1.npz","DsRed2-HeLa_10_26_2.npz","DsRed2-HeLa_10_26_3.npz","DsRed2-HeLa_10_26_4.npz",
"DsRed2-HeLa_10_26_5.npz","DsRed2-HeLa_10_26_6.npz","DsRed2-HeLa_10_26_7.npz","DsRed2-HeLa_11_21_3.npz",
"DsRed2-HeLa_11_21_4.npz","DsRed2-HeLa_11_21_5.npz","DsRed2-HeLa_11_21_6.npz","DsRed2-HeLa_11_21_7.npz",
"DsRed2-HeLa_11_21_9.npz","DsRed2-HeLa_11_21_10.npz","DsRed2-HeLa_11_21_11.npz","DsRed2-HeLa_11_21_12.npz",
"HeLa-DsRed2_10_13_1.npz","HeLa-DsRed2_10_13_2.npz","HeLa-DsRed2_10_13_3.npz","HeLa-DsRed2_10_13_4.npz",
"HeLa-DsRed2_10_13_5.npz","HeLa-DsRed2_10_13001_1.npz","HeLa-DsRed2_10_13001_2.npz","HeLa-DsRed2_10_13001_4.npz",
"HeLa-DsRed2_10_13001_5.npz","HeLa-DsRed2_10_13001_6.npz","HeLa-DsRed2_10_13001_7.npz",
"HeLa-DsRed2_10_13001_8.npz","HeLa-DsRed2_10_13001_9.npz"]

control_list = [0] * len(control_names)

for i in range(len(control_names)):
   data = load(control_names[i])
   lst_ultimate = data.files
   for item in lst_ultimate:
       if item == "eigen_vals":
           data_file = data[item]
   control_list[i] = data_file



### Transferring Data: LLO Data ###
from matplotlib.pyplot import figure

llo_names = ["DsRed2-HeLa_2_21_LLO_1.npz","DsRed2-HeLa_2_21_LLO_2.npz","DsRed2-HeLa_2_21_LLO_3.npz",
"DsRed2-HeLa_3_1_LLO_1.npz","DsRed2-HeLa_3_1_LLO_2.npz","DsRed2-HeLa_3_8_LLO002_1.npz","DsRed2-HeLa_3_8_LLO002_2.npz",
"DsRed2-HeLa_3_9_LLO_1.npz","DsRed2-HeLa_3_15_LLO1part1_1.npz","DsRed2-HeLa_3_15_LLO1part1_2.npz",
"DsRed2-HeLa_3_15_LLO1part1_3.npz","DsRed2-HeLa_3_15_LLO1part1_5.npz","DsRed2-HeLa_3_15_LLO2part1_1.npz","DsRed2-HeLa_3_15_LLO2part1_2.npz",
"DsRed2-HeLa_3_15_LLO2part1_3.npz","DsRed2-HeLa_3_23_LLO_1_1.npz","DsRed2-HeLa_3_23_LLO_1_2.npz",
"DsRed2-HeLa_3_23_LLO_1_3.npz","DsRed2-HeLa_3_23_LLO_1_4.npz","DsRed2-HeLa_3_23_LLO_1_5.npz","DsRed2-HeLa_3_31_LLO_1_2.npz",
"DsRed2-HeLa_3_31_LLO_1_3.npz","DsRed2-HeLa_3_31_LLO_1_4.npz","DsRed2-HeLa_3_31_LLO_1_5.npz","DsRed2-HeLa_3_31_LLO_2_1.npz",
"DsRed2-HeLa_3_31_LLO_2_2.npz","DsRed2-HeLa_3_31_LLO_2_3.npz","DsRed2-HeLa_3_31_LLO_2_4.npz","DsRed2-HeLa_3_31_LLO_2_5.npz",
"DsRed2-HeLa_3_31_LLO_3_1.npz","DsRed2-HeLa_3_31_LLO_3_2.npz","DsRed2-HeLa_3_31_LLO_3_3.npz","DsRed2-HeLa_3_31_LLO_3_4.npz",
"DsRed2-HeLa_3_31_LLO_3_5.npz","DsRed2-HeLa_3_31_LLO_3_6.npz","DsRed2-HeLa_4_5_LLO1_1.npz","DsRed2-HeLa_4_5_LLO1_2.npz",
"DsRed2-HeLa_4_5_LLO2_1.npz","DsRed2-HeLa_4_5_LLO2_2.npz","DsRed2-HeLa_11_2_LLO_1.npz","DsRed2-HeLa_11_2_LLO_2.npz",
"DsRed2-HeLa_11_2_LLO_4.npz","DsRed2-HeLa_11_2_LLO_6.npz","DsRed2-HeLa_11_2_LLO_8.npz","DsRed2-HeLa_11_2_LLO_9.npz",
"DsRed2-HeLa_11_2_LLO_10.npz","DsRed2-HeLa_11_2_LLO_11.npz","DsRed2-HeLa_11_3_LLO_1.npz","DsRed2-HeLa_11_3_LLO_2.npz",
"DsRed2-HeLa_11_3_LLO_3.npz","DsRed2-HeLa_11_3_LLO_4.npz","DsRed2-HeLa_11_3_LLO_5.npz","DsRed2-HeLa_11_3_LLO_6.npz",
"DsRed2-HeLa_11_3_LLO_7.npz","DsRed2-HeLa_11_3_LLO_8.npz","DsRed2-HeLa_11_3_LLO_10.npz","DsRed2-HeLa_11_3_LLO_11.npz",
"DsRed2-HeLa_11_22_LLO_30nm_1.npz","DsRed2-HeLa_11_22_LLO_30nm_2.npz","DsRed2-HeLa_11_22_LLO_30nm_3.npz",
"DsRed2-HeLa_11_22_LLO_30nm_4.npz","DsRed2-HeLa_11_22_LLO_30nm_5.npz","DsRed2-HeLa_11_22_LLO_30nm_6.npz",
"DsRed2-HeLa_11_22_LLO_30nm_7.npz","DsRed2-HeLa_11_22_LLO_30nm_10.npz","DsRed2-HeLa_11_22_LLO_30nm_11.npz"]


llo_list = [0] * 66

for i in range(66):
   data = load(llo_names[i])
   lst_ultimate = data.files
   for item in lst_ultimate:
       if item == "eigen_vals":
           data_file = data[item]
   llo_list[i] = data_file


### Transferring Data: MDIVI Data ###
mdivi_names = ["Mdivi1_7_14_1.npz","Mdivi1_7_14_2.npz","Mdivi1_7_14_2_1.npz","Mdivi1_7_14_2_2.npz",
"Mdivi1_7_14_2_3.npz","Mdivi1_7_14_3.npz","Mdivi1_7_14_4.npz","Mdivi1_7_18_1.npz","Mdivi1_7_18_2.npz","Mdivi1_7_18_2_1.npz","Mdivi1_7_18_2_2.npz","Mdivi1_7_18_2_3.npz",
"Mdivi1_7_21_1.npz","Mdivi1_7_21_2.npz","Mdivi1_7_21_2_1.npz","Mdivi1_7_21_2_2.npz","Mdivi1_7_26_1.npz","Mdivi1_7_26_2.npz","Mdivi1_7_26_2_1.npz","Mdivi1_7_26_2_2.npz","Mdivi1_7_26_2_3.npz","Mdivi1_7_26_3_1.npz","Mdivi1_7_26_3_2.npz","Mdivi1_7_26_3_3.npz","Mdivi1_7_28_2.npz","Mdivi1_7_28_2_1.npz","Mdivi1_7_28_2_2.npz","Mdivi1_7_28_2_3.npz","Mdivi1_7_28_2_4.npz","Mdivi1_7_28_3_1.npz","Mdivi1_7_28_3_2.npz"]

mdivi_list = [0] * len(mdivi_names)

for i in range(31):
   data = load(mdivi_names[i])
   lst_ultimate = data.files
   for item in lst_ultimate:
       if item == "eigen_vals":
           data_file = data[item]
   mdivi_list[i] = data_file

### Temporal Anomaly Detection ###

#Lists containing the numpy arrays
llo_list # 66 arrays
mdivi_list # 31 arrays
control_list # 29 arrays

arrays_from_each_experimental_group = [llo_list,mdivi_list,control_list] #List that contains each list of numpy arrays
names_of_arrays = [llo_names,mdivi_names,control_names] #List that contains each list of the file names of the arrays

# Defining plot() function.

def plot(eigen_values, z_scores, title):
    '''
      Plots eigenvalue time-series data, and a
      corresponding z-score curve. My plot() differs from the original in that it excludes the save_fig & outdir_path
       parameters. Excluding these parameters made working with this function within my Python environment much easier.
      '''
    sns.set()
    fig = plt.figure()
    fig.suptitle(title)

    ax = fig.add_subplot(211)
    ax.plot(eigen_values)
    ax.set_ylabel('Magnitude')

    ax = fig.add_subplot(212)
    ax.plot(z_scores)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Signal')
    plt.show()

#Defining temporal_anomaly_detection() function.
def temporal_anomaly_detection(arrays_from_each_experimental_group, k,
                   window, threshold):
    '''
       Generates a figure comprised of a time-series plot
       of the eigenvalue vectors, and an outlier detection
       signals plot.
       My temporal_anomaly_detection() differs from the original in that it
       excludes the vid_name & outdir_path parameters. Excluding these parameters made working with this function
       within my Python environment much easier.

       Parameters
       ----------
       arrays_from_each_experimental_group (replacing eigen_vals parameter): list
           List of 3 sub-lists containing NumPy arrays (NXM)
           There are 3 sub-lists within the list.
           Each sub-list is a list of Numpy arrays
           from each experimental group (control,LLO,MDIVI).
           The first sub-list contains the LLO NumPy arrays.
           The second sub-list contains the MDIVI NumPy arrays.
           The third sub-list contains the control NumPy arrays.
           In each sub-list are matrices comprised of eigenvalue vectors.
           In the temporal anomaly detection function, this object is
           appropriately subsetted so that a plot is generated from
           each NumPy array.
           N represents the number of frames in the
           corresponding video, and M is the number of
           mixture components.

       k: int
           The number of leading eigenvalues to display.
       window: int
           The size of the window to be used for anomaly
           detection.
       threshold: float
           Value used to determine whether a signal value
           is anomalous.
    '''
    for g in range(len(arrays_from_each_experimental_group)):
        experimental_group = arrays_from_each_experimental_group[g] #Extracts the sub-list corresponding to each group
        for a in range(len(experimental_group)):
            eigen_vals_array = experimental_group[a] #Extracts the NumPy array from the sub-list
            eigen_vals_avgs = [np.mean(x) for x in eigen_vals_array] #Finds the average eigenvalue per timepoint
            moving_avgs = np.empty(shape=(eigen_vals_array.shape[0],), dtype= float) * 0
            moving_stds = np.empty(shape=(eigen_vals_array.shape[0],), dtype= float) * 0
            z_scores = np.empty(shape=(eigen_vals_array.shape[0],), dtype= float) * 0
            signals = np.empty(shape=(eigen_vals_array.shape[0],), dtype= float) * 0
            for i in range(window, moving_avgs.shape[0]):
                moving_avgs[i] = np.mean(eigen_vals_avgs[i - window:i])
                moving_stds[i] = np.std(eigen_vals_avgs[i - window:i])
                z_scores[i] = (eigen_vals_avgs[i] - moving_avgs[i]) / moving_stds[i]
                plot_title = names_of_arrays[g][a] + ' Signals Plot'
                for i, score in enumerate(z_scores):
                    if score > threshold:
                      signals[i] = 1
                    elif score < threshold * -1:
                        signals[i] = -1
                    else:
                     signals[i] = 0
            plot(eigen_vals_array[:, :k], signals, title = plot_title)

temporal_anomaly_detection(arrays_from_each_experimental_group,k=10,
                   window= 20, threshold= 2)
