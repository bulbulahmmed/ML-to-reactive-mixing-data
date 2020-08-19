#**********************************************************************;
#  SVR-ROMs (Species-C): Prediction of ROMs and R2-scores              ;
#  INPUT FEATURES: time, tau, alpha_T, k_f, v_0, diffusivity           ;
#          tau_list     = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]     ;
#          alpha_T_list = [0.0001, 0.001, 0.01, 0.1, 1.0]              ;
#          k_f_list     = [1, 2, 3, 4, 5]                              ;
#          v_0_list     = [0.0001, 0.001, 0.01, 0.1, 1.0]              ;
#          diff_list    = [0.0, 0.1, 0.01, 0.001]                      ;
#          total no of realizations = 2315                             ;
#  OUTPUT: Validating and predicting ROMs                              ;
#  AUTHORS: Maruti Mudunuru, Satish Karra                              ;
#  DATE MODIFIED: August-24-2017                                       ;
#**********************************************************************;

import time
import pickle
import cPickle
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score 

#=========================;
#  Start processing time  ;
#=========================;
tic = time.clock()

#======================================;
#  Function: r2_score of the SVR-ROMs  ;
#======================================;
def get_r2score(realiz_id, ids_list, X, y, model_fit):
    
    #-------------------------------------;
    #  r2_score for various realizations  ;
    #-------------------------------------;
    r2score_list = np.array([]) 
    #
    for i in ids_list:
        #print i
        input_data  = X[i*1000:i*1000+1000,:]
        y_true      = y[i*1000:i*1000+1000]
        y_predict   = model_fit.predict(input_data)
        y_predict   = np.clip(y_predict, 0.0, 1.0)
        #
        model_r2score = r2_score(y_true, y_predict)
        print 'i, realiz-id, r2-score = ', i, realiz_id[i], model_r2score
        r2score_list = np.array(r2score_list, model_r2score)
    
    return r2score_list

#========================================================================;
#  Function: Plot SVR-ROMs predictions vs Data for certain realizations  ;
#========================================================================;
def plot_SVR_vs_data(ids_list, str_label_x, str_label_y, str_plot_title, \
                     str_fig_name, X, y, model_fit, marker, color):
    
    #------------------------------------------------------------;
    #  Iterate over prediction realizations (selected randomly)  ;
    #------------------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    #
    for i in ids_list:
        #print i
        input_data  = X[i*1000:i*1000+1000,:]
        y_true      = y[i*1000:i*1000+1000]
        y_predict   = model_fit.predict(input_data)
        y_predict   = np.clip(y_predict, 0.0, 1.0)
        #
        plt.rc('text', usetex = True)
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Lucida Grande']
        plt.rc('legend', fontsize = 14)
        ax = fig.add_subplot(111)
        ax.set_xlabel(str_label_x, fontsize = 24, fontweight = 'bold')
        ax.set_ylabel(str_label_y, fontsize = 24, fontweight = 'bold')
        #plt.title(str_plot_title, fontsize = 24, fontweight = 'bold')
        plt.grid(True)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
        col  = color.next()
        mark = marker.next()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot(input_data[range(0,1000,20),0], y_true[range(0,1000,20)], \
                linestyle = 'None', color = col, marker = mark, markersize = 3)
        ax.plot(input_data[:,0], y_predict, linestyle = 'solid', linewidth = 2.5, color = col)
        ax.fill_between(input_data[:,0], y_predict - 0.1, y_predict + 0.1, facecolor = col, \
                        alpha = 0.075)
        #ax.legend(loc = 'upper right', shadow = False, borderpad = 1.025, prop = legend_properties)
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #ax.legend(loc = 'center left', shadow = False, borderpad = 1.025, prop = legend_properties, \
        #          bbox_to_anchor=(1.0,0.5))
    #
    fig.tight_layout()
    fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.eps', format = 'eps', dpi = 1000)
    #plt.show()

#===============================================================;
#  Inputs and outputs for species-C (and make it C-contiguous)  ;
#===============================================================;
realiz_id = np.genfromtxt('ROMs_Input_C.txt', skip_header = 1, \
                           delimiter = ',', usecols = (0))
realiz_id = np.unique(realiz_id).astype(int)
num_realizations, = realiz_id.shape
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

#*************************************************************;
#  Plot-1: Species-C: Data vs prediction (Avg. sq. of. conc)  ;
#*************************************************************;
marker         = itertools.cycle(('o','v','8','s','p','*','h','+','x','^','D'))
color          = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'k', 'y'))
ids_list       = [17, 315, 947, 1595, 1870, 2219] #Goes from 0 to 2315, ids_list = range(0, num_realizations)
#ids_list      = [1870]
str_label_x    = 'Time'
str_label_y    = 'Avg. of Conc.'
str_plot_title = 'Species-C: Avg. of Conc.'
str_fig_name   = 'C_avg_predict'
#
model_fit = joblib.load('svr_C_avg_train_01.pkl') #1% training data and 99% test data 
print 'SVR-Model-C = ', model_fit
#
plot_SVR_vs_data(ids_list, str_label_x, str_label_y, str_plot_title, \
                 str_fig_name, input_params, norm_avg_conc, \
                 model_fit, marker, color) #Avg. of. conc plots
#
r2score_list = get_r2score(realiz_id, ids_list, input_params, \
norm_avg_conc, model_fit) #r2-score of avg. of. conc

#*************************************************************;
#  Plot-2: Species-C: Data vs prediction (Avg. Sq. of. Conc)  ;
#*************************************************************;
ids_list       = [17, 315, 947, 1595, 1870, 2219] #Goes from 0 to 2315, ids_list = range(0, num_realizations)
#ids_list      = [1870]
str_label_x    = 'Time'
str_label_y    = 'Avg. Sq. of Conc.'
str_plot_title = 'Species-C: Avg. Sq. of Conc.'
str_fig_name   = 'C_avgsq_predict'
#
model_fit = joblib.load('svr_C_avgsq_train_01.pkl') #1% training data and 99% test data 
print 'SVR-Model-C = ', model_fit
#
plot_SVR_vs_data(ids_list, str_label_x, str_label_y, str_plot_title, \
                 str_fig_name, input_params, norm_avg_sq_conc, \
                 model_fit, marker, color) #Avg. Sq. of Conc. plots
#
r2score_list = get_r2score(realiz_id, ids_list, input_params, \
norm_avg_sq_conc, model_fit) #r2-score of Avg. Sq. of. Conc.

#************************************************************;
#  Plot-3: Species-C: Data vs prediction (Degree of Mixing)  ;
#************************************************************;
ids_list       = [17, 315, 1400, 1595, 2150, 2219] #Goes from 0 to 2315, ids_list = range(0, num_realizations)
#ids_list      = [1870]
str_label_x    = 'Time'
str_label_y    = 'Degree of mixing'
str_plot_title = 'Species-C: Degree of mixing'
str_fig_name   = 'C_DoM_predict'
#
model_fit = joblib.load('svr_C_DoM_train_01.pkl') #1% training data and 99% test data 
print 'SVR-Model-C = ', model_fit
#
plot_SVR_vs_data(ids_list, str_label_x, str_label_y, str_plot_title, \
                 str_fig_name, input_params, norm_var_conc, \
                 model_fit, marker, color) #Degree of mixing plots
#
r2score_list = get_r2score(realiz_id, ids_list, input_params, \
norm_var_conc, model_fit) #r2-score of Degree of mixing

#======================;
# End processing time  ;
#======================;
toc = time.clock()
print 'Time elapsed in seconds = ', toc - tic
