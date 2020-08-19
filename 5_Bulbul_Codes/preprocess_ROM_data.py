#***********************************************;
# Preprocessing the ROM data                    ;
# Authors: Bulbul Ahmmed and MKM                ;
# Created on: 5th June 2019                     ;
# Date modified: 10th June 2019                 ;
#***********************************************;
import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib   # This library saves model 
# Starting processing time
tic = time.clock()
#**************Importing data where all output are normalized by MKM**********************;
ROM_A_data       = np.genfromtxt('ROMs_Input_A.txt', skip_header = 1, delimiter = ',', \
                           usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13)) 
# Transforming data
ROM_A_data[:,2]  = np.log10(ROM_A_data[:,2])                 # Log transformation of alpha_L/alpha_T
ROM_A_data[:,4]  = np.log10(ROM_A_data[:,4])                 # Log transformation of perturbation
ROM_A_data[:,5]  = np.where(ROM_A_data[:,5]==0, 1e-8, ROM_A_data[:,5]) # Replacing "0" to 1e-8
ROM_A_data[:,5]  = np.log10(ROM_A_data[:,5])                 # Log transformation of diffusivity
#
input_params     = np.asarray(ROM_A_data[:,0:6], order = 'C') # Input parameters
avg_conc         = np.asarray(ROM_A_data[:,9], order = 'C')   # avg. of conc
avg_sq_conc      = np.asarray(ROM_A_data[:,10], order = 'C')  # avg. sq. of conc
var_conc         = np.asarray(ROM_A_data[:,11], order = 'C')  # variance of conc
#
X_train, X_test, y_train, y_test = train_test_split(input_params, avg_conc, \
test_size = 0.97, random_state = 42) #Avg. Conc
print('Inputs: Training and test data size = ', X_train.shape, X_test.shape)
print('Outputs: Training and test data size = ', y_train.shape, y_test.shape)
# Preprocessing data by scaling (zero mean and unit variance)
scaler           = preprocessing.StandardScaler().fit(X_train)  # This is creating an object for scaling. Also, this can be utilized on test data
scaler.mean_
X_train   = scaler.transform(X_train)                          # Scaling for training data
X_test    = scaler.transform(X_test)                           # Scaling for test data
#****************************************************************************;
'''Saving and loading scaler '''
# This will save only sklearn object rather than whole data file. 
# Also, due to version issue pickling sometime perform poorly.
# joblib.dump(scaler, "standardscaler.save")
# We can load the scaler using following command
# scaler   = joblib.load("standardscaler.save")
#*****************************************************************************;
model     = svm.SVR(kernel = 'rbf', C = 1e4, gamma = 0.1, verbose = True)
model_fit = model.fit(X_train, y_train)
joblib.dump(model_fit, 'svr_A_avg_train_01.pkl') #1% training data and 99% test data
#
model_score_train = model_fit.score(X_train, y_train)
print('Training score = ', model_score_train)
#
model_score_test = model_fit.score(X_test, y_test)
print('Testing score = ', model_score_test)
#****************************************************************************************;
toc = time.clock()
print('Time elapsed in seconds = ', toc - tic)













