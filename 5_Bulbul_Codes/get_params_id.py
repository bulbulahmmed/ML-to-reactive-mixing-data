#*************************************************;
# This code will get realization ID for different ;
# parameters                                      ;
# Authors: Bulbul Ahmmed and M.K. Mudunuru        ;
# Date modified: 5th June 2019                    ;
#*************************************************;

# Import libraries
import numpy as np
import pandas as pd

data = np.genfromtxt('ROMs_Params.txt', skip_header = 1, delimiter = ',')

all_params  = data[:,1:]
tau         = data[:,1]
alpha_ratio = data[:,2]
k_f_list    = data[:,3]
v_0_list    = data[:,4]
diff_list   = data[:,5]

unique_tau  = np.unique(tau)
unique_alpha= np.unique(alpha_ratio)
#unique_k_f  = np.unique(k_f_list)
unique_v_0  = np.unique(v_0_list)
unique_diff = np.unique(diff_list)

#print(unique_tau)
#print(unique_alpha)
#print(unique_k_f)
#print(unique_v_0)
#print(unique_diff)
tau_sort   = 0.0001
alpha_sort = 2.0
k_f_sort   = 1.0
v_0_sort   = 0.001
diff_sort  = 0.01
k_f = 5
#for kval in unique_k_f:
#    print(kval)
#    real_id = []
#    data_k_f   = data[(data[:,1] == 0.0001) & (data[:,2] == 4.0) \
#         & (data[:,3] == 1.) & (data[:,4] == 1.e-3) & (data[:,5] == 0.01)]
#    real_id.append(int(data_k_f[0,0]))
#print(real_id)
realization_id  = []
for val1 in unique_alpha:
#   realization_id = []
    data_k_f   = data[(data[:,2] == val1) &  (data[:,3] == k_f) & (data[:,4] == 1.0)]
    print(val1, k_f, int(data_k_f[0,0]))
    realization_id.append(data_k_f)
#print(realization_id)

