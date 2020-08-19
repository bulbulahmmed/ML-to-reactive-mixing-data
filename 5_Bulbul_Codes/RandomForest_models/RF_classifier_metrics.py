#***********************************************;
# Building RandomForest Classification and      ;
# Authors: Bulbul Ahmmed and MKM                ;
# Created on: 10th June 2019                    ;
# Date modified: 10th June 2019                 ;
#***********************************************;
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# Starting processing time
tic = time.clock()
#**************Importing data where all output are normalized by MKM**********************;
ROM_data       = np.genfromtxt('/scratch/fe/bulbul/MLEF/ml_mixing/5_Bulbul_Codes/ROMs_Input_A.txt', \
skip_header = 1, delimiter = ',', usecols = (1,2,3,4,5,6,13))
#
# Transforming data
ROM_data[:,2]  = np.log10(ROM_data[:,2])                 # Log transformation of alpha_L/alpha_T
ROM_data[:,4]  = np.log10(ROM_data[:,4])                 # Log transformation of perturbation
ROM_data[:,5]  = np.where(ROM_data[:,5]==0, 1e-8, ROM_data[:,5]) # Replacing "0" to 1e-8
ROM_data[:,5]  = np.log10(ROM_data[:,5])                 # Log transformation of diffusivity
#
X              = np.asarray(ROM_data[:,0:6], order = 'C') # Input parameters
y              = np.asarray(ROM_data[:,6], order = 'C')  # Classification data
# Creating object for diff models
model1   = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
model2   = RandomForestClassifier(n_estimators=1000, random_state=42)
model3   = GaussianNB()
# Voting classifier
emodels  = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('gnb', model3)],
                        voting='soft',
                        weights=[1, 1, 5])
# Predict class probabilities for all classifiers
probabs = [c.fit(X, y).predict_proba(X) for c in (model1, model2, model3, emodels)]
# Get class probabs for the first sample in the dataset
class1_1 = [pr[0, 0] for pr in probabs]
class2_1 = [pr[0, 1] for pr in probabs]
'''Plotting bars for classifiers '''
N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width
fig, ax = plt.subplots()
# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')
# bars for VotingClassifier
p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')
# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegression\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForestClassifier\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
plt.tight_layout()
plt.show()





