Data: 

We use the same datasets from the UCI repository as the authors of 'Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive Variance' (https://arxiv.org/abs/1805.10407). They have made the data available at the following link:

https://www.dropbox.com/sh/o5nruvv6g4x8bzn/AAAnyHRcQpW1BAUxntQJ_4kVa?dl=0

The downloaded directory contains sub-folders named after the corresponding dataset. Each sub-folder contains the following npy files:
1) X.npy: Stores the 'n', D-dimensional data points in an nxD matrix
2) y.npy: Stores the 'n' targets in an n dimensional vector

Code:

The code has been tested with Python 3.6.9, Numpy 1.18.1, and Tensorflow 2.0.0.

1. The code for running experiments corresponding to each model is contained in a Jupyter notebook named after it as:
    i) run_expts_ARD.ipynb: GP (with Automatic Relevance Determination)
   ii) run_expts_DKL.ipynb: DKL
  iii) run_expts_DPKL.ipynb: DPKL

2. The path to the dataset(s) can be modified within the notebook (under Experiment Hyperparameters) while the folder-names for the datasets can be entered in a list.

2. Model Hyperparameters are stored in the file params.json which can be edited to change the hyper parameters.

3. Results (RMSE and negative log likelihood) are saved in 'Results' and 'Results_SS' directories (for supervised and semi-supervised settings respectively). For eg: Results for DPKL are saved in the following files:
   i) RMSE_mean_DPKL.npy: Matrix of RMSE means over 10 trials
  ii) RMSE_std_DPKL.npy: Matrix of RMSE std deviations over 10 trials
 iii) Negll_mean_DPKL.npy: Matrix of test negative log likelihood means over 10 trials
  iv) Negll_std_DPKL.npy: Matrix of test negative log likelihood std deviations over 10 trials
For every file, each column of the matrix will correspond to results for a distinct dataset while each row corresponds to a distinct value of 'n' (number of labeled examples) in [50, 100, 200, 300, 400, 500].