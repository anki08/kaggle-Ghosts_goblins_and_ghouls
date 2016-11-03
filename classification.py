# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:21:45 2016

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')
print(train.head())
print(train.info())
#x=train.groupby('type')['color'].agg('count')
#print(x)
y_train=train['type']
x_train=train.drop(['color','id','type'],axis=1)
le = LabelEncoder().fit(y_train)
y_train = le.transform(y_train)
le = LabelEncoder().fit(train['color'])
y = le.transform(train['color'])

sns_plot=sns.pointplot(x=y_train,y=y)  
plt.show()
fig = sns_plot.get_figure()
fig.savefig('output.png') 
#sns_plot.savefig("output.png")
#colour is not significant in predicting
#so we drop colour
id=test['id']

print(x_train.head())
print(x_train.describe())
params = {'C':[1,5,10,0.1,0.01],'gamma':[0.001,0.01,0.05,0.5,1]}
log_reg = SVC()
#params={'min_samples_leaf':[40]}
clf = GridSearchCV(log_reg ,params, refit='True', n_jobs=1, cv=5)


clf.fit(x_train, y_train)
print(test.head())
x_test=test.drop(['id','color'],axis=1)
y_test = clf.predict(x_test)
print(clf.score(x_train,y_train))
#print(y_test[:])
y_test2=le.inverse_transform(y_test)
#print((clf.score(x_train,y_train)))
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))
#print(y_test2)
submission = pd.DataFrame( { 
                  "type": y_test2
                   },index=id)
#submission.loc[submission["Y"]==1 , "Y"]=0
#submission.loc[submission["Y"]==-1 , "Y"]=0
submission.to_csv('submission3.csv')


