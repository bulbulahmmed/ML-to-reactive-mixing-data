#**********************************************************************;
#  Species-A: SVM-ROMs construction using multiclass classification    ;
#  INPUT FEATURES: time, tau, alpha_T, k_f, v_0, diffusivity           ;
#  OUTPUT CLASSES: Label-1 (Strong: Var_conc = 0 to 0.25)              ;
#                  Label-2 (Moderate: Var_conc = 0.25 to 0.5)          ;
#                  Label-3 (Weak: Var_conc = 0.5 to 0.75)              ;
#                  Label-4 (Confined/Ultra-Weak: Var_conc = 0.75 to 1) ;
#  AUTHORS: Maruti Mudunuru, Satish Karra                              ;
#  DATE MODIFIED: April-18-2017                                        ;
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
#  Inputs and outputs for species-A (and make it C-contiguous)  ;
#===============================================================;
ROM_A_data = np.genfromtxt('ROMs_Input_A.txt', skip_header = 1, delimiter = ',', \
                           usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13))
ROM_A_data[:,0] = 0.001 * ROM_A_data[:,0]
ROM_A_data[:,2] = 10000 * ROM_A_data[:,2]
#
input_params_A = np.asarray(ROM_A_data[:,0:6], order = 'C')
class_var_conc_A = np.asarray(ROM_A_data[:,12], order = 'C')
class_var_conc_A = class_var_conc_A.astype(int)
#
print input_params_A.flags
print class_var_conc_A.flags

#==========================================================================;
# CONSTRUCT and TEST SVM-ROMs and their performance metrics for species-A  ;
#==========================================================================;
input_A_train, input_A_test, class_A_train, class_A_test = \
train_test_split(input_params_A, class_var_conc_A, test_size = 0.99, random_state = 42)
#
print input_A_train.shape, input_A_test.shape
print class_A_train.shape, class_A_test.shape
print input_A_train.flags
print input_A_test.flags
print class_A_train.flags
print class_A_test.flags
#
model_A = svm.SVC(probability = True, random_state = 0, decision_function_shape = 'ovr', verbose = True)
model_A_fit = model_A.fit(input_A_train, class_A_train)
joblib.dump(model_A_fit, 'svm_model_A_test01.pkl') #1% training data and 99% test data
#
model_A_score_train = model_A_fit.score(input_A_train, class_A_train)
print 'Training score = ', model_A_score_train
#
model_A_score_test = model_A_fit.score(input_A_test, class_A_test)
print 'Testing score = ', model_A_score_test
#
#model_A_CV_score = cross_val_score(model_A, input_params_A, class_var_conc_A, scoring = 'accuracy')
#print model_A_CV_score

#======================;
# End processing time  ;
#======================;
toc = time.clock()
print 'Time elapsed in seconds = ', toc - tic
