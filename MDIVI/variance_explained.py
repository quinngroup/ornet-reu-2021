import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from matplotlib.pyplot import figure

### Calculating and Plotting Explained Variance for MDIVI Eigenvalue Data ###

## Importing the MDIVI Eigenvalue Arrays

# I quickly converted the names of the numpy files into strings via find and replace on a word doc and put them in the
# list 'mdivi_names.' There are 31 arrays in total.

mdivi_names = ["Mdivi1_7_14_1.npz", "Mdivi1_7_14_2.npz", "Mdivi1_7_14_2_1.npz", "Mdivi1_7_14_2_2.npz",
               "Mdivi1_7_14_2_3.npz", "Mdivi1_7_14_3.npz", "Mdivi1_7_14_4.npz", "Mdivi1_7_18_1.npz",
               "Mdivi1_7_18_2.npz", "Mdivi1_7_18_2_1.npz", "Mdivi1_7_18_2_2.npz", "Mdivi1_7_18_2_3.npz",
               "Mdivi1_7_21_1.npz", "Mdivi1_7_21_2.npz", "Mdivi1_7_21_2_1.npz", "Mdivi1_7_21_2_2.npz",
               "Mdivi1_7_26_1.npz", "Mdivi1_7_26_2.npz", "Mdivi1_7_26_2_1.npz", "Mdivi1_7_26_2_2.npz",
               "Mdivi1_7_26_2_3.npz", "Mdivi1_7_26_3_1.npz", "Mdivi1_7_26_3_2.npz", "Mdivi1_7_26_3_3.npz",
               "Mdivi1_7_28_2.npz", "Mdivi1_7_28_2_1.npz", "Mdivi1_7_28_2_2.npz", "Mdivi1_7_28_2_3.npz",
               "Mdivi1_7_28_2_4.npz", "Mdivi1_7_28_3_1.npz", "Mdivi1_7_28_3_2.npz"]

# Within 'mdivi_list,' I will store the 31 eigenvalue arrays.

mdivi_list = [0] * len(mdivi_names)

# Importing arrays: Please see this importing process in ornet-reu-2021/Control/variance_explained.py

for i in range(31):
    data = load(mdivi_names[i])
    lst_ultimate = data.files
    for item in lst_ultimate:
        if item == "eigen_vals":
            data_file = data[item]
    mdivi_list[i] = data_file

## Calculating Explained Variance for each MDIVI array

# Here I use the same process I used for the control data. Please see ornet-reu-2021/Control/variance_explained.py

variance_array_for_each_eigenvalue_mdivi = [0] * len(mdivi_list)

for q in range(31):
    variance_array_mdivi = (np.empty(shape=(mdivi_list[q].shape[0], mdivi_list[q].shape[1]), dtype=float)) * 0
    for w in range(mdivi_list[q].shape[0]):
        for z in range(mdivi_list[q].shape[1]):
            variance_array_mdivi[w, z] = ((mdivi_list[q][w, z])) / np.sum(mdivi_list[q][w,])
    variance_array_for_each_eigenvalue_mdivi[q] = variance_array_mdivi

## Creating the Explained Variance Plots for MDIVI
# Once more I use the same plotting process I used for the control data. Please see ornet-reu-2021/Control/variance_explained.py

pltlist_mdivi = [0] * len(mdivi_list)

figure(figsize=(100, 600), dpi=80)

for z in range(31):
    pltlist_mdivi[z] = plt.plot(range(variance_array_for_each_eigenvalue_mdivi[z].shape[0]),
                                variance_array_for_each_eigenvalue_mdivi[z][:, :])
    plt.xlabel("Time Point")
    plt.ylabel("Explained Variance")
    plt.title("MDIVI Explained Variance Per Timepoint for " + mdivi_names[z])
    plt.savefig(fname=str(mdivi_names[z] + ".jpg"))  # Automatically saves plot as file in working directory
    plt.show()