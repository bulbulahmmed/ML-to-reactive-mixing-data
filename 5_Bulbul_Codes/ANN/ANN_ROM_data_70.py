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
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import joblib
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
input_params     = np.asarray(ROM_data[:,0:6], order = 'C')  # Input parameters
avg_conc         = np.asarray(ROM_data[:,9],   order = 'C')  # avg. of conc
avg_sq_conc      = np.asarray(ROM_data[:,10],  order = 'C')  # avg. sq. of conc
var_conc         = np.asarray(ROM_data[:,11],  order = 'C')  # variance of conc
var_for_clf      = np.asarray(ROM_data[:,12],  order = 'C')  # For classification
#
test_size = 0.7
X_train, X_test, y_train, y_test = train_test_split(input_params, avg_conc, \
test_size = test_size, random_state = 42) #Avg. Conc
print('Inputs: Training and test data size = ', X_train.shape, X_test.shape)
print('Outputs: Training and test data size = ', y_train.shape, y_test.shape)
# Preprocessing
scaler           = preprocessing.StandardScaler().fit(X_train)  # This is creating an object for scaling. Also, this can be utilized on test data
# scaler.mean_
X_train   = scaler.transform(X_train)                          # Scaling for training data
X_test    = scaler.transform(X_test)                           # Scaling for test data
#****************************************************************************;
'''Saving and loading scaler '''
joblib.dump(scaler, "ANN_standardscaler_70.save")
# scaler   = joblib.load("standardscaler.save")
param_grid = {"hidden_layer_sizes": [5, 25, 50, 75, 100, 200],
              "activation": ['relu', 'tanh', 'logistic'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              "alpha": [0.0001, 0.001, 0.01, 0.1],
              "learning_rate": ['adaptive'],
              "max_iter": [5000]}
# run grid search
model        = MLPRegressor(activation='relu', alpha=0.0001, hidden_layer_sizes=200, learning_rate='adaptive', solver='adam', early_stopping=True, random_state=42)
model_fit    = model.fit(X_train, y_train)
#Best_estimator = {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': 200, 'learning_rate': 'adaptive', 'max_iter': 5000 (this is true for sgd solver) , 'solver': 'adam'}
#model_fit   = model.fit(X_train, y_train)
#model       = MLPClassifiers(bootstrap=False, n_estimators=500, max_depth=None, max_features=4, min_samples_split=2, n_jobs=32, verbose=5, random_state=42)
#model_fit   = model.fit(X_train, y_train)
#grid_search = GridSearchCV(model, param_grid=param_grid, cv=10, n_jobs=32, iid=False)
#model_fit   = grid_search.fit(X_train, y_train)
#joblib.dump(model_fit, 'ANN_ROM_A_1.pkl') #1% training data and 99% test data
#print('Best estimator: ', model_fit.best_params_)
#print('Best test score: ', model_fit.best_score_)
#
training_score  = model_fit.score(X_train, y_train)
testing_score   = model_fit.score(X_test, y_test)
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
ML_model = model_fit
test_percent = test_size
filename    = 'ANN_output_70.txt'
'''Running the function '''
writing_output(filename, test_percent, training_score, testing_score, time_taken, ML_model)





