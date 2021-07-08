'''
Calculating and Plotting Explained Variance from Eigenvalue NumPy Arrays
'''

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from tqdm import tqdm
from numpy import load

def explained_variance(eigen_vals, file_names):
    '''
               Generates a figure comprised of a time-series plot
                  of 10 leading eigenvalue vectors, and an explained variance
                  plot created from those 10 vectors.
                  Parameters
                  ----------
                  eigen_vals: NumPy Array
                   A NumPy arrays from the experimental
                   group (control, LLO, MDIVI)
                  file_names: string
                   The file names of the NumPy arrays from each experimental group.
                   Ex: "Mdivi1_7_28_2_2.npz"
            '''
    ten_leading_eigenvalues = eigen_vals[:, :10]  #Analyze only the first 10 leading eigenvalues (columns) of the NumPy array
    explained_variance_array = (np.empty(shape=(ten_leading_eigenvalues.shape[0], ten_leading_eigenvalues.shape[1]),
                                         dtype=float)) * 0 #Create an empty array with the same dimensions as the selected array
    for i in range(ten_leading_eigenvalues.shape[0]): #i corresponds to the row of the selected array
        for j in range(ten_leading_eigenvalues.shape[1]): #j corresponds to the column of the selected array
            sum_of_eigenvalue_values_at_row_i = np.sum(ten_leading_eigenvalues[i, :])
            explained_variance_array[i, j] = ((ten_leading_eigenvalues[i, j]) / sum_of_eigenvalue_values_at_row_i) * 100
    sns.set()
    fig = plt.figure()
    fig.suptitle("Explained Variance Per Timepoint for " + file_names)

    ax = fig.add_subplot(211)
    ax.plot(ten_leading_eigenvalues[:, :])
    ax.set_ylabel('Magnitude')  # Plotting the 10 leading eigenvalues over time

    ax = fig.add_subplot(212)
    ax.plot(explained_variance_array[:, 0], label="λ1")
    ax.plot(explained_variance_array[:, 1], label="λ2")
    ax.plot(explained_variance_array[:, 2], label="λ3")
    ax.plot(explained_variance_array[:, 3], label="λ4")
    ax.plot(explained_variance_array[:, 4], label="λ5")
    ax.plot(explained_variance_array[:, 5], label="λ6")
    ax.plot(explained_variance_array[:, 6], label="λ7")
    ax.plot(explained_variance_array[:, 7], label="λ8")
    ax.plot(explained_variance_array[:, 8], label="λ9")
    ax.plot(explained_variance_array[:, 9], label="λ10")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Explained Variance')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    ax.legend(bbox_to_anchor=(1, 1.30), loc="upper left")
    plt.savefig(fname=str(file_names + ".jpg"), dpi = 100)
    plt.show()

def parse_command_line_interface():
    '''
        Parses the command line arguments.
        Returns
        -------
        parsed_args: dict
            Key value pairs of arguments.
        '''
    parser = argparse.ArgumentParser(description = "This script calculates the explained variance of each of the 10 leading eigenvalues over each frame per NumPy Array of Eigenvalues")
    parser.add_argument("-i","--eigen_directory", type = str, required = True, help = "This is the path to the file directory with the NumPy arrays")
    args = vars(parser.parse_args())
    return(args)

parsed_args = parse_command_line_interface() #parsed_args consists of the key value pairs of the argument

path_to_directory = parsed_args["eigen_directory"] #Key to your directory

your_directory = os.listdir(path_to_directory) #Your actual working directory

print(your_directory) #To make sure that you are in the correct directory where the npz files are

for item in your_directory:
    if ".npz" in item:
        data = np.load(item)
        path = data["eigen_vals"]
        file_name = item.split(".")[0]
        print(file_name)
        explained_variance(eigen_vals= path, file_names= file_name)





