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


def parse_command_line_interface():
    '''
        Parses the command line arguments.
        Returns
        -------
        parsed_args: dict
            Key value pairs of arguments.
        '''
    parser = argparse.ArgumentParser(description = "Anomaly detection of of eigenvalue time-series data.")
    parser.add_argument("-i","--eigen_directory", type = str, required = True, help = "This is the path to the file directory with the NumPy arrays")
    args = vars(parser.parse_args())
    return(args)

parsed_args = parse_command_line_interface() #parsed_args consists of the key value pairs of the argument

path_to_directory = parsed_args["eigen_directory"] #Key to your directory

your_directory = os.listdir(path_to_directory) #Your actual working directory

print(your_directory)

def plot(eigen_values, z_scores, title):
    '''
      Plots eigenvalue time-series data, and a
      corresponding z-score curve. My plot() differs from the original in that it excludes the outdir_path
       parameter. Excluding this parameter made working with this function within my Python environment much easier.
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
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(fname=str(title + ".jpg"), dpi=100)
    plt.show()

def temporal_anomaly_detection(eigen_vals, window, k, threshold,file_name):
    eigen_vals_avgs = [np.mean(x) for x in eigen_vals]
    moving_avgs = np.empty(shape=(eigen_vals.shape[0],), dtype= float) * 0
    moving_stds = np.empty(shape=(eigen_vals.shape[0],), dtype= float) * 0
    z_scores = np.empty(shape=(eigen_vals.shape[0],), dtype= float) * 0
    signals = np.empty(shape=(eigen_vals.shape[0],), dtype= float) * 0
    for i in range(window, moving_avgs.shape[0]):
        moving_avgs[i] = np.mean(eigen_vals_avgs[i - window:i])
        moving_stds[i] = np.std(eigen_vals_avgs[i - window:i])
        z_scores[i] = (eigen_vals_avgs[i] - moving_avgs[i]) / moving_stds[i]
        plot_title = file_name + ' Signals Plot ' + 'Window ~ ' + str(window) + ' Threshold ~ ' + str(threshold)
        for i, score in enumerate(z_scores):
            if score > threshold:
                signals[i] = 1
            elif score < threshold * -1:
                signals[i] = -1
            else:
                signals[i] = 0
    plot(eigen_vals[:, :k], signals, title=plot_title)

for item in your_directory:
    if ".npz" in item:
        data = np.load(item)
        path = data["eigen_vals"]
        file_name = item.split(".")[0]
        print(file_name)
        temporal_anomaly_detection(eigen_vals= path, window = 20, k = 10, threshold= 2.0, file_name= file_name)







