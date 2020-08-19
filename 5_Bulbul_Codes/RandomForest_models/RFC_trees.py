"""
Author: Bulbul and MKM at LANL 06/13/2019
Modified on: 
Confusion matrix plot for ROM_data after applying 
RandomForest classification
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
#=====================================;
# Start processing time               ;
#=====================================;
tic  = time.clock()
#=================================================================== ===;
# Importing data, transforming them normalizing them, spliting into     ;
# test and training                                                     ;
#=======================================================================;
ROM_data       = np.genfromtxt('/scratch/fe/bulbul/MLEF/ml_mixing/5_Bulbul_Codes/ROMs_Input_C.txt', \
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

y_pred   = RFC_model.fit(X_train, y_train).predict(X_test)
# Extract single tree
estimator = RFC_model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='RFC_tree.dot',
                rounded = True, proportion = False,
                precision = 2, filled = True)

#*******************************************;
# Plot Decision tree                        ;
#*******************************************;

#plt.savefig(figname)
#plt.show()
