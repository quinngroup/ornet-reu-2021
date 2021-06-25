import numpy as np
import matplotlib.pyplot as plt
from numpy import load

### Calculating and Plotting Explained Variance for LLO Eigenvalue Data ###

## Importing the LLO Eigenvalue Arrays

# I quickly converted the names of the numpy files into strings via find and replace on a word doc and put them in the
# list 'llo_names.' There are 66 arrays in total.

llo_names = ["DsRed2-HeLa_2_21_LLO_1.npz","DsRed2-HeLa_2_21_LLO_2.npz","DsRed2-HeLa_2_21_LLO_3.npz","DsRed2-HeLa_3_1_LLO_1.npz","DsRed2-HeLa_3_1_LLO_2.npz",
"DsRed2-HeLa_3_8_LLO002_1.npz","DsRed2-HeLa_3_8_LLO002_2.npz","DsRed2-HeLa_3_9_LLO_1.npz","DsRed2-HeLa_3_15_LLO1part1_1.npz","DsRed2-HeLa_3_15_LLO1part1_2.npz",
"DsRed2-HeLa_3_15_LLO1part1_3.npz","DsRed2-HeLa_3_15_LLO1part1_5.npz","DsRed2-HeLa_3_15_LLO2part1_1.npz","DsRed2-HeLa_3_15_LLO2part1_2.npz",
"DsRed2-HeLa_3_15_LLO2part1_3.npz","DsRed2-HeLa_3_23_LLO_1_1.npz","DsRed2-HeLa_3_23_LLO_1_2.npz","DsRed2-HeLa_3_23_LLO_1_3.npz","DsRed2-HeLa_3_23_LLO_1_4.npz",
"DsRed2-HeLa_3_23_LLO_1_5.npz","DsRed2-HeLa_3_31_LLO_1_2.npz","DsRed2-HeLa_3_31_LLO_1_3.npz","DsRed2-HeLa_3_31_LLO_1_4.npz","DsRed2-HeLa_3_31_LLO_1_5.npz",
"DsRed2-HeLa_3_31_LLO_2_1.npz","DsRed2-HeLa_3_31_LLO_2_2.npz","DsRed2-HeLa_3_31_LLO_2_3.npz","DsRed2-HeLa_3_31_LLO_2_4.npz","DsRed2-HeLa_3_31_LLO_2_5.npz",
"DsRed2-HeLa_3_31_LLO_3_1.npz","DsRed2-HeLa_3_31_LLO_3_2.npz","DsRed2-HeLa_3_31_LLO_3_3.npz","DsRed2-HeLa_3_31_LLO_3_4.npz","DsRed2-HeLa_3_31_LLO_3_5.npz",
"DsRed2-HeLa_3_31_LLO_3_6.npz","DsRed2-HeLa_4_5_LLO1_1.npz","DsRed2-HeLa_4_5_LLO1_2.npz","DsRed2-HeLa_4_5_LLO2_1.npz","DsRed2-HeLa_4_5_LLO2_2.npz",
"DsRed2-HeLa_11_2_LLO_1.npz","DsRed2-HeLa_11_2_LLO_2.npz","DsRed2-HeLa_11_2_LLO_4.npz","DsRed2-HeLa_11_2_LLO_6.npz","DsRed2-HeLa_11_2_LLO_9.npz",
"DsRed2-HeLa_11_2_LLO_10.npz","DsRed2-HeLa_11_2_LLO_11.npz","DsRed2-HeLa_11_3_LLO_1.npz","DsRed2-HeLa_11_3_LLO_2.npz","DsRed2-HeLa_11_3_LLO_3.npz",
"DsRed2-HeLa_11_3_LLO_4.npz","DsRed2-HeLa_11_3_LLO_5.npz","DsRed2-HeLa_11_3_LLO_6.npz","DsRed2-HeLa_11_3_LLO_7.npz","DsRed2-HeLa_11_3_LLO_8.npz",
"DsRed2-HeLa_11_3_LLO_10.npz","DsRed2-HeLa_11_3_LLO_11.npz","DsRed2-HeLa_11_22_LLO_30nm_1.npz","DsRed2-HeLa_11_22_LLO_30nm_2.npz",
"DsRed2-HeLa_11_22_LLO_30nm_3.npz","DsRed2-HeLa_11_22_LLO_30nm_4.npz","DsRed2-HeLa_11_22_LLO_30nm_5.npz","DsRed2-HeLa_11_22_LLO_30nm_6.npz",
"DsRed2-HeLa_11_22_LLO_30nm_7.npz","DsRed2-HeLa_11_22_LLO_30nm_10.npz","DsRed2-HeLa_11_22_LLO_30nm_11.npz"]

# Within 'llo_list,' I will store the 66 eigenvalue arrays.
llo_list = [0] * 66

#Importing arrays
for i in range(66):
    data = load(llo_names[i])     #By calling load() on each string in llo_names, I am loading the actual numpy array itself
    lst_ultimate = data.files     #data.files consists of two "keys", "eigen_vals" & "eigen_vectors." I wish to extract the former
    for item in lst_ultimate:
        if item == "eigen_vals":
            data_file = data[item]     #Extract the array from the data
    llo_list[i] = data_file            #Insert that array into llo_list


## Calculating Explained Variance for each LLO array

#Here I use the same process I used for the control data. Please see ornet-reu-2021/Control/Explained Variance for Control

variance_array_for_each_eigenvalue_llo = [0] * len(llo_list)

for q in range(66):
    variance_array_llo = (np.empty(shape=(llo_list[q].shape[0],llo_list[q].shape[1]), dtype= float)) * 0
    for w in range(llo_list[q].shape[0]):
        for z in range(llo_list[q].shape[1]):
            variance_array_llo[w,z] = ((llo_list[q][w,z])) / np.sum(llo_list[q][w,])
    variance_array_for_each_eigenvalue_llo[q] = variance_array_llo


## Creating the Explained Variance Plots for LLO
#Once more I use the same plotting process I used for the control data. Please see ornet-reu-2021/Control/Explained Variance for Control

pltlist_llo = [0] * len(llo_list)

for z in range(66):
    pltlist_llo[z] = plt.plot(range(variance_array_for_each_eigenvalue_llo[z].shape[0]), variance_array_for_each_eigenvalue_llo[z][:,:])
    plt.xlabel("Time Point")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance Per Timepoint for " + llo_names[z])
    plt.savefig(fname= str(llo_names[z] + ".jpg"))
    plt.show()