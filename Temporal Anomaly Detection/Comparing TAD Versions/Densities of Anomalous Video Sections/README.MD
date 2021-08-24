# Densities of Anomalous Video Sections in Each Experimental Group
### One of the Two Methods Comparing TAD Simple Average to TAD Weighted Average

Among each data group, compute **the densities of anomalous frames per NumPy array in each experimental group.** Compare these densities at each window size and threshold. 


*How is this data gathered?* For each NumPy array in each experimental group on which the TAD was applied, locate the anomalous frames and what their "positions" are (frame 3, frame 5, frame 80, etc) and divide the position of each anomalous frame by the number of rows in the NumPy array (indicative of number of frames), and display the densities of those anomalous frames for each group.
