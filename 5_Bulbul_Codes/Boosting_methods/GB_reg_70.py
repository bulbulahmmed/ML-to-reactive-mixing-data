#*********************************************************;
# This code is for AdaBoost Reg., Gradient Boosting Reg., ; 
# and Dec. Tree Reg. with AdaBoost, aka multiclass-       ;
# AdaBoost Regression                                     ;
# Authors: Bulbul Ahmmed and MKM                          ;
# Created on: 19th June 2019                              ;
# Date modified: 19th June 2019                           ;
#*********************************************************;
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib   # This library saves model 
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Starting processing time
tic = time.clock()
#**************Importing data where all output are normalized by MKM**********************;
ROM_data       = np.genfromtxt('/scratch/fe/bulbul/MLEF/ml_mixing/5_Bulbul_Codes/ROMs_Input_A.txt', \
skip_header = 1, delimiter = ',', usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13))
# Transforming data
ROM_data[:,2]  = np.log10(ROM_data[:,2])                 # Log transformation of alpha_L/alpha_T
ROM_data[:,4]  = np.log10(ROM_data[:,4])                 # Log transformation of perturbation
ROM_data[:,5]  = np.where(ROM_data[:,5]==0, 1e-8, ROM_data[:,5]) # Replacing "0" to 1e-8
ROM_data[:,5]  = np.log10(ROM_data[:,5])                 # Log transformation of diffusivity
#
input_params     = np.asarray(ROM_data[:,0:6], order = 'C') # Input parameters
avg_conc         = np.asarray(ROM_data[:,9], order = 'C')   # avg. of conc
avg_sq_conc      = np.asarray(ROM_data[:,10], order = 'C')  # avg. sq. of conc
var_conc         = np.asarray(ROM_data[:,11], order = 'C')  # variance of conc
#
test_percent = 0.3
X_train, X_test, y_train, y_test = train_test_split(input_params, avg_conc, \
test_size = test_percent, random_state = 42) #Avg. Conc
#print('Inputs: Training and test data size = ', X_train.shape, X_test.shape)
#print('Outputs: Training and test data size = ', y_train.shape, y_test.shape)
# Preprocessing
scaler           = preprocessing.StandardScaler().fit(X_train)  # This is creating an object for scaling. Also, this can be utilized on test data
# scaler.mean_
X_train   = scaler.transform(X_train)                          # Scaling for training data
X_test    = scaler.transform(X_test)                           # Scaling for test data
#****************************************************************************;
'''Saving and loading scaler '''
#joblib.dump(scaler, "Dec_bagging_standardscaler.save")
# scaler   = joblib.load("standardscaler.save")
#Gradient Boosting params search
GB_param_grid = {'max_features': [5],
                 "min_samples_split": [2], 
                 "min_samples_leaf": [2],
                 "max_depth": [None],
                 "learning_rate": [0.1, 0.25, 0.5, 0.75, 1],
                 'n_estimators': [100, 500, 1000],
                 "subsample": [0.5, 0.7, 0.8],
                 "loss": ['ls'],
                 "n_iter_no_change": [10, 20]}
#dectree_param_grid
''' I have the best model params for dec_tree, 
so I will use that model instead of creating a new one '''
# Gradient Boosting  model
GB_model       = GradientBoostingRegressor(learning_rate=0.1, loss='ls', max_features=5, min_samples_leaf=2, min_samples_split=2, n_estimators=100,n_iter_no_change=20, subsample=0.5, verbose=1, random_state = 42)
#GB_grid_search = GridSearchCV(GB_model, param_grid=GB_param_grid, cv=5, n_jobs=32, iid=False)
GB_model_fit   = GB_model.fit(X_train, y_train)
#
pkl1 = 'GB_scale_ROM_A70.pkl'
joblib.dump(GB_model_fit, pkl1) #1% training data and 99% test data
#print('Best estimator for GB: ', GB_model_fit.best_params_)
#print('Best test score for GB: ', GB_model_fit.best_score_)
#
GB_training_score  = GB_model_fit.score(X_train, y_train)
GB_testing_score   = GB_model_fit.score(X_test, y_test)
#==================================================================================;
# Function: Writing output file onto a file                                        ;
#==================================================================================;
def writing_output(filename, test_percent, training_score, testing_score, time_taken, ML_model):
    with open(filename, 'w') as outf:
        outf.write('Test data is: %f\r\n' % test_percent)
        outf.write('Training score is: %f\r\n' % training_score)
        outf.write('Testing score is: %f\r\n' % testing_score)
        outf.write('Time taken by this process: %f\r\n' % time_taken)
        outf.write(str(ML_model))
#***************************;
# Writing output to a file  ;
#***************************;
toc = time.clock()
time_taken = toc - tic
filename1    = 'GB_output_70.txt'
'''Running the function '''
writing_output(filename1, test_percent, GB_training_score, GB_testing_score, time_taken, GB_model)
#writing_output(filename2, test_percent, dec_tree_training_score, dec_tree_testing_score, time_taken, dec_tree_model)

