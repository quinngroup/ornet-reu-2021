# Explained Variance Python Code

Within the "Explained Variance" repository, I have uploaded my Python code that consists of a funcion that calculates and plots the explained variance from each eigenvalue per time-point/frame
in each Eigenvalue NumPy array from the OrNet Eigenspectrum data. 

Each Eigenvalue NumPy array's rows correspond to time-points/frames and columns correspond to eigenvalues. Only the first 10 leading eigenvalues are being considered in the explained
variance calculations. 

Explained variance is calculated by taking the sum of the eigenvalues in each row and dividing each eigenvalue by that sum. Then, the explained variance from each eigenvalue per time-point/frame is plotted. Below is a crude image illustrating the how the calculation is carried out. 

![](Images/Screenshot%20(172).png)

