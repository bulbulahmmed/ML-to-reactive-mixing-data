#***********************************************;
# Building RandomForest Classification and      ;
# Regression                                    ;
# Authors: Bulbul Ahmmed and MKM                ;
# Created on: 10th June 2019                    ;
# Date modified: 10th June 2019                 ;
#***********************************************;
import time
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
#===================;
# Inputs for plot   ;
#===================;
tic = time.clock()
X                 = np.genfromtxt('/scratch/fe/bulbul/MLEF_Gitlab/5_Bulbul_Codes/Plot/Inputs_for_plots/ROMs_C_non_scaled_input.txt', delimiter = ',')
y                 = np.genfromtxt('/scratch/fe/bulbul/MLEF_Gitlab/5_Bulbul_Codes/Plot/Inputs_for_plots/True_data.txt', skip_header=1, delimiter = ',')
i                 = 0
input_data        = X[i*1000:i*1000+1000,:]
y_true            = y[i*1000:i*1000+1000, 6]
# Model definition
model       = RandomForestRegressor(bootstrap=False, n_estimators=500, max_depth=None, max_features=4, min_samples_split=2, n_jobs=32, verbose=0, random_state=42)
model_fit   = model.fit(input_data, y_true)
y_predict   = model_fit.predict(input_data)
toc = time.clock()
time_taken = toc  - tic
test_score  = model_fit.score(input_data, y_true)
#
print('Time taken for single realization:', time_taken)
print('Test score is:', test_score)

