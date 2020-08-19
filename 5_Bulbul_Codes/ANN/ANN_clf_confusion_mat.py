"""
Author: Bulbul and MKM at LANL 08/13/2019
Modified on:
Confusion matrix plot for ROM_data after applying
ANN classification
"""
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
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
test_size = 0.99
X_train, X_test, y_train, y_test = train_test_split(input_params, var_for_clf, \
test_size = test_size, random_state = 42)            # Degree of mixing
# Preprocessing
scaler           = preprocessing.StandardScaler().fit(X_train)  # This is creating an object for scaling. Also, this can be utilized on test data
# scaler.mean_
X_train   = scaler.transform(X_train)                          # Scaling for training data
X_test    = scaler.transform(X_test)                           # Scaling for test data
#****************************************************************************;
model        = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=200, learning_rate='adaptive', solver='adam', early_stopping=True, random_state=42, max_iter=5000)
y_pred        = model.fit(X_train, y_train).predict(X_test)
#==============================================;
# This function plots confusion matrix for     ;
# classification algorithm                     ;
#==============================================;
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
           #ylabel='True label',
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
figname  = 'ANN_confusion_matrix_C.pdf'
figname2 = 'ANN_confusion_matrix_C.eps'
ax = plot_confusion_matrix(y_test, y_pred, classes=class_names)

# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
plt.savefig(figname)
plt.savefig(figname2)

