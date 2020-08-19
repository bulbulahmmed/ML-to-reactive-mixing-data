#**********************************************************************;
#  Species-C: SVR-ROMs construction using various kernels              ;
#  INPUT FEATURES: time, tau, alpha_T, k_f, v_0, diffusivity           ;
#          tau_list     = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]     ;
#          alpha_T_list = [0.0001, 0.001, 0.01, 0.1, 1.0]              ;
#          k_f_list     = [1, 2, 3, 4, 5]                              ;
#          v_0_list     = [0.0001, 0.001, 0.01, 0.1, 1.0]              ;
#          diff_list    = [0.0, 0.1, 0.01, 0.001]                      ;
#          total no of realizations = 2315                             ;
#  OUTPUT: ROMs for mixing Quantities of Interests                     ;
#          (norm_avg_conc_C, norm_avg_sq_conc_C, norm_var_conc_C)      ;
#  AUTHORS: Maruti Mudunuru, Satish Karra                              ;
#  DATE MODIFIED: August-24-2017                                       ;
#**********************************************************************;

import time
import pickle
import cPickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

#========================;
# Start processing time  ;
#========================;
tic = time.clock()

#===============================================================;
#  Inputs and outputs for species-C (and make it C-contiguous)  ;
#===============================================================;
realiz_id = np.genfromtxt('ROMs_Input_C.txt', skip_header = 1, \
                           delimiter = ',', usecols = (0))
realiz_id = np.unique(realiz_id).astype(int)
num_realizations, = realiz_id.shape
ids_list = range(0,num_realizations)
#
ROM_data      = np.genfromtxt('ROMs_Input_C.txt', skip_header = 1, \
                           delimiter = ',', usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13))
ROM_data[:,0] = (1.0/1000.0) * ROM_data[:,0] #Time/Time_max
ROM_data[:,1] = (1.0/0.0005) * ROM_data[:,1] #Tau/Tau_max
ROM_data[:,2] = np.abs((1/np.min(np.log10(ROM_data[:,2]))) * np.log10(ROM_data[:,2])) #(a_L/a_T)/(a_L/a_T)_max
ROM_data[:,3] = (1.0/5.0) * ROM_data[:,3] #k_f/k_f_max
ROM_data[:,4] = np.abs((1/np.min(np.log10(ROM_data[:,4]))) * np.log10(ROM_data[:,4])) #v_0/v_0_max
ROM_data[:,5] = (1.0/0.1) * ROM_data[:,5] #Diffusivity/Diffusivity_max
#
input_params     = np.asarray(ROM_data[:,0:6], order = 'C') #Input parameters
norm_avg_conc    = np.asarray(ROM_data[:,9], order = 'C') #Normalized avg. of conc
norm_avg_sq_conc = np.asarray(ROM_data[:,10], order = 'C') #Normalized avg. sq. of conc
norm_var_conc    = np.asarray(ROM_data[:,11], order = 'C') #Normalized variance of conc
#
print 'Flags of input parameters array = ', input_params.flags
print 'Flags of output normalized avg. array = ', norm_avg_conc.flags
print 'Flags of output normalized avg. sq. array = ', norm_avg_sq_conc.flags
print 'Flags of output normalized variance array = ', norm_var_conc.flags
#
print 'Scaled tau_list = ', list(sorted(set(ROM_data[:,1])))
print 'Scaled alpha_LT_list = ', list(sorted(set(ROM_data[:,2])))
print 'Scaled k_f_list = ', list(sorted(set(ROM_data[:,3])))
print 'Scaled v_0_list = ', list(sorted(set(ROM_data[:,4])))
print 'Scaled diff_list = ', list(sorted(set(ROM_data[:,5])))

#=====================================================================================;
#  Avg.Conc: CONSTRUCT and TEST SVR-ROMs and their performance metrics for species-C  ;
#=====================================================================================;
X_train, X_test, y_train, y_test = train_test_split(input_params, norm_avg_conc, \
test_size = 0.99, random_state = 42) #Avg. Conc
#
print 'Inputs: Training and test data size = ', X_train.shape, X_test.shape
print 'Output: Training and test data size = ', y_train.shape, y_test.shape
print 'Training input flags = ', X_train.flags
print 'Testing input flags =', X_test.flags
print 'Training output flags = ', y_train.flags
print 'Training output flags = ', y_test.flags
#
model = svm.SVR(kernel = 'rbf', C = 1e4, gamma = 0.1, verbose = True)
model_fit = model.fit(X_train, y_train)
joblib.dump(model_fit, 'svr_C_avg_train_01.pkl') #1% training data and 99% test data
#
model_score_train = model_fit.score(X_train, y_train)
print 'Training score = ', model_score_train
#
model_score_test = model_fit.score(X_test, y_test)
print 'Testing score = ', model_score_test

#========================================================================================;
#  Avg.Sq.Conc: CONSTRUCT and TEST SVR-ROMs and their performance metrics for species-C  ;
#========================================================================================;
X_train, X_test, y_train, y_test = train_test_split(input_params, norm_avg_sq_conc, \
test_size = 0.99, random_state = 42) #Avg. sq. conc
#
print 'Inputs: Training and test data size = ', X_train.shape, X_test.shape
print 'Output: Training and test data size = ', y_train.shape, y_test.shape
print 'Training input flags = ', X_train.flags
print 'Testing input flags =', X_test.flags
print 'Training output flags = ', y_train.flags
print 'Training output flags = ', y_test.flags
#
model = svm.SVR(kernel = 'rbf', C = 1e4, gamma = 0.1, verbose = True)
model_fit = model.fit(X_train, y_train)
joblib.dump(model_fit, 'svr_C_avgsq_train_01.pkl') #1% training data and 99% test data
#
model_score_train = model_fit.score(X_train, y_train)
print 'Training score = ', model_score_train
#
model_score_test = model_fit.score(X_test, y_test)
print 'Testing score = ', model_score_test

#================================================================================;
#  DoM: CONSTRUCT and TEST SVR-ROMs and their performance metrics for species-C  ;
#================================================================================;
X_train, X_test, y_train, y_test = train_test_split(input_params, norm_var_conc, \
test_size = 0.99, random_state = 42) #Degree of mixing
#
print 'Inputs: Training and test data size = ', X_train.shape, X_test.shape
print 'Output: Training and test data size = ', y_train.shape, y_test.shape
print 'Training input flags = ', X_train.flags
print 'Testing input flags =', X_test.flags
print 'Training output flags = ', y_train.flags
print 'Training output flags = ', y_test.flags
#
model = svm.SVR(kernel = 'rbf', C = 1e4, gamma = 0.1, verbose = True)
model_fit = model.fit(X_train, y_train)
joblib.dump(model_fit, 'svr_C_DoM_train_01.pkl') #1% training data and 99% test data
#
model_score_train = model_fit.score(X_train, y_train)
print 'Training score = ', model_score_train
#
model_score_test = model_fit.score(X_test, y_test)
print 'Testing score = ', model_score_test

#======================;
# End processing time  ;
#======================;
toc = time.clock()
print 'Time elapsed in seconds = ', toc - tic
