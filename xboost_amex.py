import pandas as pd
import numpy as np
import csv
import warnings

warnings.filterwarnings('ignore',category=DeprecationWarning)
import xgboost as xgb
warnings.filterwarnings("ignore", category=DeprecationWarning)
from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore',category=DeprecationWarning)

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams


rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('modTrain.csv')
target = 'mvar45'
IDcol = 'mvar0'

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    print target
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 4
        # xgb_param['updater'] = 'grow_gpu'
        # xgb_param['tree_method'] = 'gpu_hist'

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='merror', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    # print cvresult
    print cvresult.shape[0]
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['mvar45'],eval_metric='merror')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['mvar45'].values, dtrain_predictions)
    # print "AUC Score (Train): %f" % metrics.confusion_matrix(dtrain['mvar45'], dtrain_predprob)
                    
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    return alg


predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 # num_class=4,
 # silent=False,
 # tree_method='gpu_hist',
 # updater='grow_gpu',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

# alg=modelfit(xgb1,train,predictors)
# 	
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')





# # TUNING

# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=77, max_depth=5,	
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27,silent=False), 
#  param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(train[predictors],train[target])
# gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# param_test2 = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[2,3,4]
# }

# gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=77, max_depth=5,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27,silent=False), 
#  param_grid = param_test2, scoring='accuracy',n_jobs=4,iid=False, cv=5)
# gsearch2.fit(train[predictors],train[target])
# gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

# param_test3 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }

# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=77, max_depth=5,
#  min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27,silent=False), 
#  param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)

# gsearch3.fit(train[predictors],train[target])

# gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=112, max_depth=5,
#  min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27,silent=False), 
#  param_grid = param_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)

# gsearch4.fit(train[predictors],train[target])
# gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# param_test6 = {
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }
# gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=112, max_depth=5,
#  min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.95,
#  objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27,silent=False), 
#  param_grid = param_test6, scoring='accuracy',n_jobs=4,iid=False, cv=5)
# gsearch6.fit(train[predictors],train[target])
# gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

#############################################################################################
# For leaderboard submission

model=XGBClassifier(
 learning_rate =0.09,
 max_depth=5,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.975,
 objective= 'multi:softmax',
 # reg_aplha=1,
 # num_class=4,
 # silent=False,
 # tree_method='gpu_hist',
 # updater='grow_gpu',
 nthread=4,
 # n_estimators=112,
 n_estimators=10000,
 scale_pos_weight=1,
 seed=27)

# model.fit(train[predictors],train[target])
model=modelfit(model,train,predictors)


# dtrain_predictions = model.predict(train[predictors])
# dtrain_predprob = model.predict_proba(train[predictors])[:,1]
# #Print model report:
# print "\nModel Report"
# print "Accuracy : %.4g" % metrics.accuracy_score(train['mvar45'].values, dtrain_predictions)


val = pd.read_csv('modLeader.csv')
predictors_val = [x for x in val.columns if x not in [IDcol]]

y=model.predict(val[predictors_val])
IDcolVal=val[IDcol]

subList=[]
for i in range(0,IDcolVal.size):
	if y[i]==3:
		continue
	elif y[i]==0:
		subList.append([int(IDcolVal[i]),'Supp'])
	elif y[i]==1:
		subList.append([int(IDcolVal[i]),'Elite'])
	elif y[i]==2:
		subList.append([int(IDcolVal[i]),'Credit'])

with open("tensor_IITMadras_3.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(subList)


#####################################################3

# >>> gsearch1.grid_scores_
# >>> gsearch1.grid_scores_
# [mean: 0.77908, std: 0.00049, params: {'max_depth': 3, 'min_child_weight': 1}, mean: 0.77893, std: 0.00069, params: {'max_depth': 3, 'min_child_weight': 3}, mean: 0.77890, std: 0.00051, params: {'max_depth': 3, 'min_child_weight': 5}, mean: 0.77905, std: 0.00093, params: {'max_depth': 5, 'min_child_weight': 1}, mean: 0.77915, std: 0.00078, params: {'max_depth': 5, 'min_child_weight': 3}, mean: 0.77855, std: 0.00065, params: {'max_depth': 5, 'min_child_weight': 5}, mean: 0.77895, std: 0.00114, params: {'max_depth': 7, 'min_child_weight': 1}, mean: 0.77825, std: 0.00146, params: {'max_depth': 7, 'min_child_weight': 3}, mean: 0.77838, std: 0.00115, params: {'max_depth': 7, 'min_child_weight': 5}, mean: 0.77817, std: 0.00123, params: {'max_depth': 9, 'min_child_weight': 1}, mean: 0.77873, std: 0.00128, params: {'max_depth': 9, 'min_child_weight': 3}, mean: 0.77870, std: 0.00106, params: {'max_depth': 9, 'min_child_weight': 5}]
# >>> gsearch1.best_score_
# 0.7791500215841123
# >>> modelfit(xgb1, train, predictors)
# mvar45

# ([mean: 0.77885, std: 0.00080, params: {'subsample': 0.6, 'colsample_bytree': 0.6}, mean: 0.77878, std: 0.00049, params: {'subsample': 0.7, 'colsample_bytree': 0.6}, mean: 0.77852, std: 0.00062, params: {'subsample': 0.8, 'colsample_bytree': 0.6}, mean: 0.77945, std: 0.00068, params: {'subsample': 0.9, 'colsample_bytree': 0.6}, mean: 0.77908, std: 0.00036, params: {'subsample': 0.6, 'colsample_bytree': 0.7}, mean: 0.77887, std: 0.00082, params: {'subsample': 0.7, 'colsample_bytree': 0.7}, mean: 0.77870, std: 0.00068, params: {'subsample': 0.8, 'colsample_bytree': 0.7}, mean: 0.77925, std: 0.00071, params: {'subsample': 0.9, 'colsample_bytree': 0.7}, mean: 0.77918, std: 0.00043, params: {'subsample': 0.6, 'colsample_bytree': 0.8}, mean: 0.77905, std: 0.00119, params: {'subsample': 0.7, 'colsample_bytree': 0.8}, mean: 0.77915, std: 0.00078, params: {'subsample': 0.8, 'colsample_bytree': 0.8}, mean: 0.77893, std: 0.00081, params: {'subsample': 0.9, 'colsample_bytree': 0.8}, mean: 0.77928, std: 0.00092, params: {'subsample': 0.6, 'colsample_bytree': 0.9}, mean: 0.77937, std: 0.00097, params: {'subsample': 0.7, 'colsample_bytree': 0.9}, mean: 0.77922, std: 0.00099, params: {'subsample': 0.8, 'colsample_bytree': 0.9}, mean: 0.77918, std: 0.00123, params: {'subsample': 0.9, 'colsample_bytree': 0.9}], {'subsample': 0.9, 'colsample_bytree': 0.6}, 0.7794500153458296)
# Model Report
# Accuracy : 0.7863
# {'subsample': 0.9, 'colsample_bytree': 0.9}, 0.779525009105205)
# ,tree_method= 'gpu_hist',		

# max_depth=5
# min_child_weight=3
# gamma=0
# 'subsample': 0.9
# 'colsample_bytree': 0.9