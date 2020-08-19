#   Classifying ROM data and plotting with Random Forest               ;
#  INPUT FEATURES: time, tau, alpha_T, k_f, v_0, diffusivity           ;
#          tau_list     = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]     ;
#          alpha_T_list = [0.0001, 0.001, 0.01, 0.1, 1.0]              ;
#          k_f_list     = [1, 2, 3, 4, 5]                              ;
#          v_0_list     = [0.0001, 0.001, 0.01, 0.1, 1.0]              ;
#          diff_list    = [0.0, 0.1, 0.01, 0.001]                      ;
#          total no of realizations = 2315                             ;
#  OUTPUT: class probability plots (ground truth vs prob)              ;
#  AUTHORS: Bulbul Ahmmed, Maruti Mudunuru                             ;
#  DATE MODIFIED: June-12-2019                                       ;
#**********************************************************************;
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
import joblib
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import cm

#=====================================;
# Start processing time               ;
#=====================================;
tic  = time.clock()
#=================================================================== ===;
# Importing data, transforming them normalizing them, spliting into     ; 
# test and training                                                     ;
#=======================================================================;
ROM_data       = np.genfromtxt('/scratch/fe/bulbul/MLEF/ml_mixing/5_Bulbul_Codes/ROMs_Input_A.txt', \
skip_header = 1, delimiter = ',', usecols = (0,1,2,3,4,5,6,13))
#
# Transforming data
ROM_data[:,2]  = np.log10(ROM_data[:,2])                 # Log transformation of alpha_L/alpha_T
ROM_data[:,4]  = np.log10(ROM_data[:,4])                 # Log transformation of perturbation
ROM_data[:,5]  = np.where(ROM_data[:,5]==0, 1e-8, ROM_data[:,5]) # Replacing "0" to 1e-8
ROM_data[:,5]  = np.log10(ROM_data[:,5])                 # Log transformation of diffusivity
# Making data
X              = np.asarray(ROM_data[:,1:7], order = 'C') # Input parameters
y              = np.asarray(ROM_data[:,7], order = 'C')  # Classification data
# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, random_state=42)

#==================================================================================;
# Create model, fit, and get scores                                                ;
#==================================================================================;
n_estimators, max_features, n_jobs = [1000, 4, 32]
RFC_model     = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs, random_state=42)
RFC_model_fit = RFC_model.fit(X_train, y_train)
joblib.dump(RFC_model_fit, 'RFC_model_A_train01.pkl') # 1% training data
# print('Training score is:', RFC_model_fit.score(X_train, y_train))
# print('Testing score is:', RFC_model_fit.score(X_test, y_test)) 
training_score  = RFC_model_fit.score(X_train, y_train)
testing_score   = RFC_model_fit.score(X_test, y_test)
#==================================================================================;
# Finding class probabilities also predicting class                                ;
#=================================================================================;
#X_train_pred = RFC_model.predict(X_train) # predicting class
X_train_probabs = RFC_model.predict_proba(X_train) # Predicting class probabilities
''' '''
#==================================================================================;
# Function: get ready the data to plot probabs for true and prediction             ;
#==================================================================================;
def get_probabs_data(idx, ROM_data, classifier_model, index):
    time_step  = ROM_data[idx*1000:idx*1000+1000,1]/1000
    X_input    = ROM_data[idx*1000:idx*1000+1000, 1:7]   # Only for 17th realization
    y_true     = np.abs(ROM_data[idx*1000:idx*1000+1000, index])
    X_probs    = classifier_model.predict_proba(X_input)
    pred_class1 = X_probs[:,0]
    pred_class2 = X_probs[:,1]
    pred_class3 = X_probs[:,2]
    pred_class4 = X_probs[:,3]
    '''Extracting y_class '''
    y_true_class1  = np.where(y_true > 1, 0, y_true)
    y_true_class2  = np.where(((y_true == 1) | (y_true > 2)), 0, y_true)
    y_true_class3  = np.where(((y_true < 3) | (y_true == 4)), 0, y_true)
    y_true_class4  = np.where(y_true < 4, 0, y_true)
    true_class1    = np.where(y_true_class1 > 0, 1, y_true_class1)
    true_class2    = np.where(y_true_class2 > 0, 1, y_true_class2)
    true_class3    = np.where(y_true_class3 > 0, 1, y_true_class3)
    true_class4    = np.where(y_true_class4 > 0, 1, y_true_class4)
    return time_step, true_class1, true_class2, true_class3, true_class4, \
           pred_class1, pred_class2, pred_class3, pred_class4
#==================================================;
#  Function: Plot Data for different realizations  ;
#==================================================;
def plot_probs(str_label_x, str_label_y, prd_label_y, str_fig_name, \
    time_step, true_class1, true_class2, true_class3, true_class4, \
    pred_class1, pred_class2, pred_class3, pred_class4, cmap=plt.cm.get_cmap("winter")):
    
    legend_properties = {'weight':'bold'}
    plt.rc('text', usetex = True)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 12)
    #
    fig, axs = plt.subplots(4, 2, figsize=(7, 8), tight_layout=True)
    axs[0,0].plot(time_step, true_class1)
    axs[0,0].set_ylabel(str_label_y, fontsize=12)
#    axs[0,0].set_title('True class 1') 
    axs[0,1].plot(time_step, pred_class1)
    axs[0,1].set_ylabel(prd_label_y, fontsize=12)
#    axs[0,1].set_title('Prediction class 1')       
    axs[1,0].plot(time_step, true_class2)
#    axs[1,0].set_title('True class 2')
    axs[1,0].set_ylabel(str_label_y, fontsize=12)
    axs[1,1].plot(time_step, pred_class2)
    axs[1,1].set_ylabel(prd_label_y, fontsize=12)
#    axs[1,1].set_title('Prediction class 2')
    axs[2,0].plot(time_step, true_class3)
#    axs[2,0].set_title('True class 3')
    axs[2,0].set_ylabel(str_label_y, fontsize=12)
    axs[2,1].plot(time_step, pred_class3)
    axs[2,1].set_ylabel(prd_label_y, fontsize=12)
#    axs[2,1].set_title('Prediction class 3')
    axs[3,0].plot(time_step, true_class4)
#    axs[3,0].set_title('True class 4')
    axs[3,0].set_xlabel(str_label_x, fontsize=12)
    axs[3,0].set_ylabel(str_label_y, fontsize=12)
    im = axs[3,1].plot(time_step, pred_class4)
    axs[3,1].set_ylabel(prd_label_y, fontsize=12)
#    axs[3,1].set_title('Prediction class 4')
    axs[3,1].set_xlabel(str_label_x)    
    #plt.colorbar(im[3])
    fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.eps', format = 'eps', dpi = 1000)
    plt.show()
'''Need to create same plot as like as ground truth '''
#==================================================================================;
# Function: Writing output file onto a file                                        ;
#==================================================================================;
def writing_output(filename, ML_model, training_score, testing_score):
    with open(filename, 'w') as outf:
        outf.write('Training score is: %f\r\n' % training_score)
        outf.write('Testing score is: %f\r\n' % testing_score)
        outf.write(str(ML_model))
#*********************************************************************************;
''' Executing the writing output function  '''             
#*********************************************************************************;
filename = 'RFC_model_details01.txt'
writing_output(filename, RFC_model, training_score, testing_score)

#******************************************************************************;
#  Plot-1: Species-C: Data and avg of data (Avg of concentration, normalized)  ;
#******************************************************************************;
idx            = 17
#ids_list       = range(0,num_realizations)
str_label_x    = 'Time step'
str_label_y    = 'True class'
prd_label_y    = 'Predicted class probability'
str_plot_title = 'Species-A: Avg. of Concentration'
str_fig_name   = 'A_prob_plot01'
index          = 7 #norm_avg_conc_C
classifier_model = RFC_model
cmap           = plt.cm.get_cmap("winter")
#
time_step, true_class1, true_class2, true_class3, true_class4,\
pred_class1, pred_class2, pred_class3, pred_class4 = get_probabs_data(idx, ROM_data, classifier_model, index)
#
#plot_probs(str_label_x, str_label_y, prd_label_y, str_fig_name, \
#time_step, true_class1, true_class2, true_class3, true_class4, \
#pred_class1, pred_class2, pred_class3, pred_class4)
