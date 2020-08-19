#***********************************************;
# Building RandomForest Classification and      ;
# Regression                                    ;
# Authors: Bulbul Ahmmed and MKM                ;
# Created on: 10th June 2019                    ;
# Date modified: 10th June 2019                 ;
#***********************************************;
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib   # This library saves model 
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
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
X_train, X_test, y_train, y_test = train_test_split(input_params, avg_conc, \
test_size = 0.3, random_state = 42) #Avg. Conc
print('Inputs: Training and test data size = ', X_train.shape, X_test.shape)
print('Outputs: Training and test data size = ', y_train.shape, y_test.shape)
# Preprocessing
scaler           = preprocessing.StandardScaler().fit(X_train)  # This is creating an object for scaling. Also, this can be utilized on test data
# scaler.mean_
X_train   = scaler.transform(X_train)                          # Scaling for training data
X_test    = scaler.transform(X_test)                           # Scaling for test data
#****************************************************************************;
'''Saving and loading scaler '''
joblib.dump(scaler, "Dec_bagging_standardscaler.save")
# scaler   = joblib.load("standardscaler.save")
bag_param_grid = {"bootstrap": [True, False],
                  "bootstrap_features": [True, False],
                  'max_features': [3, 4, 5],
                  "n_estimators": [100, 250, 500, 1000]}
dectree_param_grid = {"max_depth": [2, 3, None],
                      "min_samples_split": [2, 3, 4],
                      'max_features': [3, 4, 5]}
# Creating models
bagging_model    = BaggingRegressor(bootstrap=True, bootstrap_features=False, max_features=5, n_estimators=100,verbose=5, n_jobs=32, random_state=42)
bagging_model_fit  = bagging_model.fit(X_train, y_train)
# Decision trees
dec_tree_model  = DecisionTreeRegressor(max_depth=None, max_features=5, min_samples_split=2, random_state=42)
dec_tree_model_fit    = dec_tree_model.fit(X_train, y_train)
# Bagging grid search
#bagging_grid_search = GridSearchCV(bagging_model, param_grid=bag_param_grid, cv=5, n_jobs=32, iid=False)
#bagging_model_fit   = bagging_grid_search.fit(X_train, y_train)
# Decision tree grid search
#dec_tree_grid_search = GridSearchCV(dec_tree_model, param_grid=dectree_param_grid, cv=5, n_jobs=32, iid=False)
#dec_tree_model_fit   = dec_tree_grid_search.fit(X_train, y_train)
#
joblib.dump(bagging_model_fit, 'bagging_scale_ROM_A70.pkl') #1% training data and 99% test data
joblib.dump(dec_tree_model_fit, 'Dec_tree_scale_ROM_A70.pkl') #1% training data and 99% test data
#print('Best estimator for bagging: ', bagging_model_fit.best_params_)
#print('Best test score for bagging: ', bagging_model_fit.best_score_)
#print('Best estimator for dec tree: ', dec_tree_model_fit.best_params_)
#print('Best test score for dec tree: ', dec_tree_model_fit.best_score_)
#
bagging_training_score  = bagging_model_fit.score(X_train, y_train)
bagging_testing_score   = bagging_model_fit.score(X_test, y_test)
dec_tree_training_score  = dec_tree_model_fit.score(X_train, y_train)
dec_tree_testing_score   = dec_tree_model_fit.score(X_test, y_test)
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
test_percent = 0.3
filename1    = 'bagging_output_70.txt'
filename2    = 'dec_tree_output_70.txt'
'''Running the function '''
writing_output(filename1, test_percent, bagging_training_score, bagging_testing_score, time_taken, bagging_model)
writing_output(filename2, test_percent, dec_tree_training_score, dec_tree_testing_score, time_taken, dec_tree_model)

