"""
Author: Bulbul and MKM at LANL 06/13/2019
Modified on:
Confusion matrix plot for ROM_data after applying
RandomForest classification
"""

import time
import pickle
#import cPickle
import numpy as np
#import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
#=====================================;
# Start processing time               ;
#=====================================;
tic  = time.clock()
#=================================================================== ===;
# Importing data, transforming them normalizing them, spliting into     ;
# test and training                                                     ;
#=======================================================================;
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

#***********************************************************;
# Function to run multiple models for different test sizes  ;
#***********************************************************;
''' avg_conc if Reg or var_for_clf if Classification  '''
def run_multiple_model(input_params, avg_conc, test_size, model, filename):
    tic = time.clock()
    X_train, X_test, y_train, y_test = train_test_split(input_params, avg_conc, \
    test_size = test_size, random_state = 42) #Avg. Conc
    # Fiiting model
    model_fit   = model.fit(X_train, y_train)
    training_score  = model_fit.score(X_train, y_train)
    testing_score   = model_fit.score(X_test, y_test)
    print(training_score, testing_score)
    # Writing output to a file  ;
    toc = time.clock()
    time_taken = toc - tic
    ML_model = model_fit
    writing_output(filename, test_size, training_score, testing_score, time_taken, ML_model)
#**************Importing data where all output are normalized by MKM**********************;
test_sizes = [0.99, 0.3, 0.1]
n_estimators, max_features, n_jobs = [1000, 4, 32]
model      = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs, random_state=42)
#
for test_size in test_sizes:
    filename    = 'RF_clf_output_' + str(test_size) + '.txt'
    run_multiple_model(input_params, var_for_clf, test_size, model, filename)

