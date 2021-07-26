# Average Proportion of Anomalous Frames for Each Video in Each Mitochondrial Condition

In creating the new temporal anomaly detection code utilizing the weighted average eigenvalue per frame, I sought to compare this new model to the original temporal anomaly 
detection code that utilizes the simple average eigenvalue per frame. 

For each TAD code (simple average, weighted average), I calculated the average proportion of frames declared anomalous from each NumPy array for each experimental group among different
window sizes and thresholds. 

Within this folder are three bar plots. Each plot was generated from three thresholds: threshold = 2.0, threshold = 2.5, and threshold = 3.0. 
The x-axis is the window size and the y-axis is the average proportion of frames declared anomalous for each experimental group. The left side of the plot is data generated 
from the simple average TAD code. The right side of the plot is data generated from the weighted average TAD code. 
