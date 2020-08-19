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
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#=====================================;
# Start processing time               ;
#=====================================;
tic  = time.clock()
#=================================================================== ===;
# Importing data, transforming them normalizing them, spliting into     ;
# test and training                                                     ;
#=======================================================================;
ROM_data       = np.genfromtxt('/scratch/fe/bulbul/MLEF/ml_mixing/5_Bulbul_Codes/ROMs_Input_B.txt', \
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
#RFC_model_fit = RFC_model.fit(X_train, y_train)
#joblib.dump(RFC_model_fit, 'RFC_model_A_train01.pkl') # 1% training data
# print('Training score is:', RFC_model_fit.score(X_train, y_train))
# print('Testing score is:', RFC_model_fit.score(X_test, y_test))
#training_score  = RFC_model_fit.score(X_train, y_train)
#testing_score   = RFC_model_fit.score(X_test, y_test)
#==================================================================================;
# Finding class probabilities also predicting class                                ;
#=================================================================================;
#X_train_pred = RFC_model.predict(X_train) # predicting class
#X_train_probabs = RFC_model.predict_proba(X_train) # Predicting class probabilities
y_pred   = RFC_model.fit(X_train, y_train).predict(X_test)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    ''''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    '''
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
    classes = class_names
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
#    plt.savefig(figname)
    return ax

#*******************************************;
# Plot confusion matrix                     ;
#*******************************************;

np.set_printoptions(precision=2)
class_names = ['Ultra weak', 'Weak', 'Moderate', 'Well']
#class_names = y_train
# Plot non-normalized confusion matrixi
figname  = 'confusion_matrix_C.png'
ax = plot_confusion_matrix(y_test, y_pred, classes=class_names)

# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
plt.savefig(figname)
#plt.show()
