import numpy as np
import matplotlib.pyplot as plt
from numpy import load

### Calculating and Plotting Explained Variance for Control Eigenvalue Data ###

## Importing the control Eigenvalue Arrays

# I quickly converted the names of the numpy files into strings via find and replace on a word doc and put them in the
# list 'control_names.' There are 29 arrays in total.

control_names = ["DsRed2-HeLa_10_26_1.npz","DsRed2-HeLa_10_26_2.npz","DsRed2-HeLa_10_26_3.npz","DsRed2-HeLa_10_26_4.npz",
"DsRed2-HeLa_10_26_5.npz","DsRed2-HeLa_10_26_6.npz","DsRed2-HeLa_10_26_7.npz","DsRed2-HeLa_11_21_3.npz",
"DsRed2-HeLa_11_21_4.npz","DsRed2-HeLa_11_21_5.npz","DsRed2-HeLa_11_21_6.npz","DsRed2-HeLa_11_21_7.npz",
"DsRed2-HeLa_11_21_9.npz","DsRed2-HeLa_11_21_10.npz","DsRed2-HeLa_11_21_11.npz","DsRed2-HeLa_11_21_12.npz",
"HeLa-DsRed2_10_13_1.npz","HeLa-DsRed2_10_13_2.npz","HeLa-DsRed2_10_13_3.npz","HeLa-DsRed2_10_13_4.npz",
"HeLa-DsRed2_10_13_5.npz","HeLa-DsRed2_10_13001_1.npz","HeLa-DsRed2_10_13001_2.npz","HeLa-DsRed2_10_13001_4.npz",
"HeLa-DsRed2_10_13001_5.npz","HeLa-DsRed2_10_13001_6.npz","HeLa-DsRed2_10_13001_7.npz",
"HeLa-DsRed2_10_13001_8.npz","HeLa-DsRed2_10_13001_9.npz"]

# Within 'control_list,' I will store the 29 eigenvalue arrays.
control_list = [0] * 29

#Importing arrays
for i in range(29):
   data = load(control_names[i])     #By calling load() on each string in control_names, I am loading the actual numpy array itself
   lst_ultimate = data.files     #data.files consists of two "keys", "eigen_vals" & "eigen_vectors." I wish to extract the former
   for item in lst_ultimate:
       if item == "eigen_vals":
           data_file = data[item]     #Extract the array from the data
   control_list[i] = data_file        #Insert that array into control_list


## Calculating Explained Variance for each Control array


variance_array_for_each_eigenvalue_control = [0] * len(control_list)

for q in range(66):
   variance_array_control = (np.empty(shape=(control_list[q].shape[0],control_list[q].shape[1]), dtype= float)) * 0
   for w in range(control_list[q].shape[0]):
       for z in range(control_list[q].shape[1]):
           variance_array_control[w,z] = ((control_list[q][w,z])) / np.sum(control_list[q][w,])
   variance_array_for_each_eigenvalue_control[q] = variance_array_control


## Creating the Explained Variance Plots for Control ##

pltlist_control = [0] * len(control_list)


for z in range(66):
   pltlist_control[z] = plt.plot(range(variance_array_for_each_eigenvalue_control[z].shape[0]), variance_array_for_each_eigenvalue_control[z][:,:])
   plt.xlabel("Time Point")
   plt.ylabel("Explained Variance")
   plt.title("Explained Variance Per Timepoint for " + control_names[z])
   plt.savefig(fname= str(control_names[z] + ".jpg")) # Automatically saves plot as file in working directory
   plt.show()
