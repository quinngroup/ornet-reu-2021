'''
Comparing TAD Simple Average to TAD Weighted Average. Comparing the average proportion of anomalous frames per experimental group
among both TAD versions.
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
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec


def parse_command_line_interface():
    '''
        Parses the command line arguments.
        Returns
        -------
        parsed_args: dict
            Key value pairs of arguments.
        '''
    parser = argparse.ArgumentParser(description = "This script calculates and compares the average proportion of anomalous frames per experimental group among both TAD versions.")
    parser.add_argument("-i","--eigen_directory", type = str, required = True, help = "This is the path to the file directory with the NumPy arrays")
    args = vars(parser.parse_args())
    return(args)

parsed_args = parse_command_line_interface() #parsed_args consists of the key value pairs of the argument

path_to_directory = parsed_args["eigen_directory"] #Key to your directory

your_directory = os.listdir(path_to_directory) #Your actual working directory

print(your_directory) #To make sure that you are in the correct directory where the npz files are


llo_list = []
mdivi_list = []
control_list = []

for item in your_directory:
    if ".npz" and "LLO" in item:
        data = np.load(item)
        eigen_vals = data["eigen_vals"]
        llo_list.append(eigen_vals)
    elif ".npz" and "Mdivi" in item:
        data = np.load(item)
        eigen_vals = data["eigen_vals"]
        mdivi_list.append(eigen_vals)
    elif ".npz" in item:
        data = np.load(item)
        eigen_vals = data["eigen_vals"]
        control_list.append(eigen_vals)

eigenvals_list = [llo_list,mdivi_list,control_list]

def average_proportion_of_anomalous_frames(eigenvals_list, tad_version, window, threshold):
    '''
        Prints three lists. Each entry in each list is the proportion of anomalous frames from the eigenvalue arrays
        from each experimental groups/mitochondrial conditions.
        Additionally prints the average proportion and standard deviation for each list and specifies which TAD version
        and what window and threshold values were used.

        Parameters
        ----------
        eigenvals_list: list
            A list of three sublists. Each sublist contains
            the eigenvalue arrays from each experimental
            group/mitochondrial condition.
        tad_version: string
            Determines which TAD version is used to print
            the proportions. The inputs are "Weighted Average"
            or "Simple Average."
        window: int
            The size of the window to be used for anomaly
            detection.
        threshold: float
            Value used to determine whether a signal value
            is anomalous.
    '''
    experimental_groups = len(eigenvals_list)
    llo_props = [0] * len(llo_list)
    mdivi_props = [0] * len(mdivi_list)
    control_props = [0] * len(control_list)
    proportions_list = [llo_props,mdivi_props,control_props]

    if tad_version == "Weighted Average":
        for f in range(experimental_groups):
            mitochondrial_condition = eigenvals_list[f]
            for j in range(len(mitochondrial_condition)):
                eigen_vals = mitochondrial_condition[j]
                number_of_rows = eigen_vals.shape[0]
                number_of_columns = eigen_vals.shape[1]
                weights_array = np.empty(shape=(number_of_rows, number_of_columns)) * 0
                eigen_vals_weighted_avgs = np.empty(shape=(number_of_rows))
                for row in range(number_of_rows):
                    for column in range(number_of_columns):
                        array_entry = eigen_vals[row, column]
                        sum_of_row = np.sum(eigen_vals[row,])
                        weights_array[row, column] = array_entry * (array_entry / sum_of_row)
                        eigen_vals_weighted_avgs[row] = np.sum(weights_array[row,])
                moving_avgs = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                moving_stds = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                z_scores = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                signals = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                for i in range(window, moving_avgs.shape[0]):
                    moving_avgs[i] = np.mean(eigen_vals_weighted_avgs[i - window:i])
                    moving_stds[i] = np.std(eigen_vals_weighted_avgs[i - window:i])
                    z_scores[i] = (eigen_vals_weighted_avgs[i] - moving_avgs[i]) / moving_stds[i]
                    for i, score in enumerate(z_scores):
                        if score > threshold:
                            signals[i] = 1
                        elif score < threshold * -1:
                            signals[i] = -1
                        else:
                            signals[i] = 0
                        prop_of_anomaly_detections = np.sum(abs(signals)) / mitochondrial_condition[j].shape[0]
                        proportions_list[f][j] = prop_of_anomaly_detections

        for x in proportions_list:
            if len(x) == len(llo_list):
                print("LLO Proportions: " + str(x))
            elif len(x) == len(mdivi_list):
                print("MDIVI Proportions: " + str(x))
            elif len(x) == len(control_list):
                print("Control Proportions: " + str(x))

        for x in proportions_list:
            if len(x) == len(llo_list):
                print("The Mean Prop. of Anomalous Frames for the LLO Arrays using TAD Weighted Average, Window: " +
                      str(window) + " & Threshold: " + str(threshold) + " is " + str(np.mean(x)))
                print("The Standard Deviation is: " + str(np.std(x)))
            elif len(x) == len(mdivi_list):
                print("The Mean Prop. of Anomalous Frames for the MDIVI Arrays using TAD Weighted Average, Window: " +
                      str(window) + " & Threshold: " + str(threshold) + " is " + str(np.mean(x)))
                print("The Standard Deviation is: " + str(np.std(x)))
            elif len(x) == len(control_list):
                print("The Mean Prop. of Anomalous Frames for the Control Arrays using TAD Weighted Average, Window: " +
                      str(window) + " & Threshold: " + str(threshold) + " is " + str(np.mean(x)))
                print("The Standard Deviation is: " + str(np.std(x)))

    elif tad_version == "Simple Average":
        for f in range(experimental_groups):
            mitochondrial_condition = eigenvals_list[f]
            for j in range(len(mitochondrial_condition)):
                eigen_vals = mitochondrial_condition[j]
                eigen_vals_avgs = [np.mean(x) for x in eigen_vals]
                moving_avgs = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                moving_stds = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                z_scores = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                signals = np.empty(shape=(eigen_vals.shape[0],), dtype=float) * 0
                for i in range(window, moving_avgs.shape[0]):
                    moving_avgs[i] = np.mean(eigen_vals_avgs[i - window:i])
                    moving_stds[i] = np.std(eigen_vals_avgs[i - window:i])
                    z_scores[i] = (eigen_vals_avgs[i] - moving_avgs[i]) / moving_stds[i]
                    for i, score in enumerate(z_scores):
                        if score > threshold:
                            signals[i] = 1
                        elif score < threshold * -1:
                            signals[i] = -1
                        else:
                            signals[i] = 0
                        prop_of_anomaly_detections = np.sum(abs(signals)) / mitochondrial_condition[j].shape[0]
                        proportions_list[f][j] = prop_of_anomaly_detections

        for x in proportions_list:
            if len(x) == len(llo_list):
                print("LLO Proportions: " + str(x))
            elif len(x) == len(mdivi_list):
                print("MDIVI Proportions: " + str(x))
            elif len(x) == len(control_list):
                print("Control Proportions: " + str(x))

        for x in proportions_list:
            if len(x) == len(llo_list):
                print("The Mean Prop. of Anomalous Frames for the LLO Arrays using TAD Simple Average, Window: " +
                      str(window) + " & Threshold: " + str(threshold) + " is " + str(np.mean(x)))
                print("The Standard Deviation is: " + str(np.std(x)))
            elif len(x) == len(mdivi_list):
                print("The Mean Prop. of Anomalous Frames for the MDIVI Arrays using TAD Simple Average, Window: " +
                      str(window) + " & Threshold: " + str(threshold) + " is " + str(np.mean(x)))
                print("The Standard Deviation is: " + str(np.std(x)))
            elif len(x) == len(control_list):
                print("The Mean Prop. of Anomalous Frames for the Control Arrays using TAD Simple Average, Window: " +
                      str(window) + " & Threshold: " + str(threshold) + " is " + str(np.mean(x)))
                print("The Standard Deviation is: " + str(np.std(x)))







average_proportion_of_anomalous_frames(eigenvals_list = eigenvals_list, tad_version= "Simple Average",window = 20, threshold = 2.0)
average_proportion_of_anomalous_frames(eigenvals_list = eigenvals_list, tad_version= "Weighted Average",window = 20, threshold = 2.0)


