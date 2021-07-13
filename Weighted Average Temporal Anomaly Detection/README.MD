# WEIGHTED Average Temporal Anomaly Detection

This folder contains the script for temporal anomaly detection using the **weighted average eigenvalue per frame.** 

For each row/frame, in each eigenvals NumPy array, each entry is multiplied by a weight (entry/divided by the sum of the row entries) and then each product in that row is summed to
calculate the weighted average per frame. 

An image that clearly illustrates this process will be uploaded soon. 
