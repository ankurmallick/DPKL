Data: 

We use the MNIST dataset which is already present in Tensorflow. 

Code:

The code has been tested with Python 3.6.9, Numpy 1.18.1, and Tensorflow 2.0.0.

1. The code for each model is contained in a file named after it. For eg: 'DPKL.py' contains the code for the DPKL model (similarly for BLR, BNN, and DBLR).

2. Running the file performs 10 trials of classification with n=100,200,...,1000 labeled examples for each trial.

3. Results (Accuracy and negative log likelihood) in the corresponding file in the 'Results' directory. For eg: Results for DPKL are saved in the following files:
   i) acc_dpkl.npy: Matrix of accuracy over 10 trials
  ii) negll_dpkl.npy: Matrix of test negative log likelihood over 10 trials
In each file the ith row of the matrix correspond to the ith value of 'n' (number of labeled examples) in 100,200,...,1000; and the jth column corresponds to the jth trial.
