'''
Calculating and Plotting Explained Variance from Eigenvalue NumPy Arrays
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

## Explained Variance ##

#Lists containing the numpy arrays
llo_list # 66 arrays
mdivi_list # 31 arrays
control_list # 29 arrays

def explained_variance(experimental_group, file_names):
    '''
           Generates a figure comprised of a time-series plot
              of 10 leading eigenvalue vectors, and an explained variance
              plot created from those 10 vectors.

              Parameters
              ----------
              experimental_group: list
               A list that contains the NumPy arrays from the experimental
               group (control, LLO, MDIVI)

              file_names: list
               A list consisting of strings. The strings are the file
               names of the NumPy arrays from each experimental group.
               Ex: "Mdivi1_7_28_2_2.npz" is one the strings in the list
        '''

    for q in range(len(experimental_group)):
       leading_eigenvalues = experimental_group[q][:, :10] #Extract the qth array out of the list of arrays and analyze only the first 10 leading eigenvalues (columns)
       explained_variance_array = (np.empty(shape=(leading_eigenvalues.shape[0], leading_eigenvalues.shape[1]), dtype=float)) * 0 #Create an empty array with the same dimensions as this qth array. This where the explained variance is stored.
       for w in range(leading_eigenvalues.shape[0]):
           for z in range(leading_eigenvalues.shape[1]):
               explained_variance_array[w, z] = ((leading_eigenvalues[w, z]) / np.sum(leading_eigenvalues[w,:]))* 100 #Fill each entry of this array with the explained variance calcualtion
       sns.set()
       fig = plt.figure()
       fig.suptitle("Explained Variance Per Timepoint for " + file_names[q])

       ax = fig.add_subplot(211)
       ax.plot(leading_eigenvalues[:, :])
       ax.set_ylabel('Magnitude') #Plotting the 10 leading eigenvalues over time

       ax = fig.add_subplot(212)
       ax.plot(explained_variance_array[:, :]) #Plotting the explained variance
       ax.set_xlabel('Frame')
       ax.set_ylabel('Explained Variance')

       plt.show()




explained_variance(experimental_group= control_list, file_names= control_names)
explained_variance(experimental_group= mdivi_list, file_names= mdivi_names)
explained_variance(experimental_group= llo_list, file_names= llo_names)