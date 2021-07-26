# Temporal Anomaly Detection Utilizing Weighted Average Eigenvalue Per Frame

Within the "TAD Weighted Average Per Frame" folder, I have included my code for the temporal anomaly detection using the **weighted average eigenvalue per frame.** 

For each row/frame, in each eigenvals NumPy array, each entry is multiplied by a weight (entry/divided by the sum of the row entries) and then each product in that row is summed to
calculate the weighted average per frame. 

An image that clearly illustrates this process will be uploaded soon.

My code file is titled, "temporal_anomaly_detection_weighted_average.py" This is the code I use to create the temporal anomaly detection plots for the eigenvalue NumPy arrays in each experimental group.
