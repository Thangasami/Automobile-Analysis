# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 08:48:12 2021

@author: Thangasami
"""
import os
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

cwd = os.getcwd()
print(cwd)

data = pd.read_csv(r'C:\Users\sthan\Python\Machine Learning\Automobile - Decision Tree\Automobile_data.csv')
data.replace('?', np.nan, inplace = True)

string_col = data.select_dtypes(exclude = np.number).columns.tolist()

num_cols = ['normalized-losses', 'bore',  'stroke', 'peak-rpm',  'price' ]


#convert into numeric data type

for i in num_cols:
    data[i] = pd.to_numeric(data[i], errors = 'raise')


#categorical conversion
for i in data:
    if is_string_dtype(data[i]):
        data[i] = data[i].astype('category').cat.as_unordered()
        
#Cat code conversion
for i in data:
    if(str(data[i].dtype) ==  'category'):
        data[i] = data[i].cat.codes
        
#imputation
data.fillna(data.median(), inplace = True )

#Modeling
X = data.drop('symboling', axis = 1)
y = data['symboling']

#Train test Split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 100)

#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pre = dt.predict(x_val)
a = accuracy_score(y_pre, y_val)
print(a)
print(dt.score(x_val, y_val))

lr = LogisticRegression(max_iter = 10101000                                                                                     )
lr.fit(x_train, y_train)
print(lr.score(x_val, y_val))

#Hyper-Parameter

NoofEstimator = [5, 10, 15, 20]
MinSampleLeaf = [1,3,5,7]
MaxFeature = np.arange(0.1, 1.1, 0.1)
best_score = []

for i in NoofEstimator:
    for j in MinSampleLeaf:
        for k in MaxFeature:
            result = [i, j, k]
            rfc = RandomForestClassifier(n_estimators = i,
                                         min_samples_leaf= j,
                                         max_features = k,
                                         random_state = 100)
            rfc.fit(x_train, y_train)
            result.append(rfc.score(x_train, y_train))
            result.append(rfc.score(x_val, y_val))
            if len(best_score) == 0:
                best_score = result
            elif best_score[4] < result[4]:
                best_score = result
                print(best_score)

print('The final best result is:', best_score)


#Grid Search
rf = RandomForestClassifier(random_state = 100)
rf_grid = GridSearchCV(estimator = rf, param_grid = dict(n_estimators = NoofEstimator,
                                         min_samples_leaf= MinSampleLeaf,
                                         max_features = MaxFeature) )
rf_grid.fit(x_train, y_train)
print(rf_grid.best_estimator_)
print(rf_grid.score(x_val, y_val))


#Grid Search
rf = RandomForestClassifier(random_state = 100)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = dict(n_estimators = NoofEstimator,
                                         min_samples_leaf= MinSampleLeaf,
                                         max_features = MaxFeature) )
rf_random.fit(x_train, y_train)
print(rf_random.best_estimator_)
print(rf_random.score(x_val, y_val))

#Checking the oob Score
rf_o = rf = RandomForestClassifier(oob_score = True)
rf_o.fit(x_train, y_train)
rf_o.oob_score_

#important Features
imp_feature = rf_grid.best_estimator_.feature_importances_
feature_list = list(X.columns)
feature_importance = sorted(zip(imp_feature, feature_list), reverse = True)
df = pd.DataFrame(feature_importance, columns = {'importance', 'feature'})

# Set the style
plt.style.use('bmh')
# list of x locations for plotting
x_values = list(range(len(feature_importance)))
importance= list(df['importance'])
feature= list(df['feature'])
# Make a bar chart
plt.figure(figsize=(15,10))
plt.bar(x_values, importance, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')

#AdaBoost Classifier

rf = RandomForestClassifier(max_features=0.4, n_estimators=20, min_samples_leaf= 1)

ab = AdaBoostClassifier(base_estimator = rf, n_estimators = 50)
ab.fit(x_train, y_train)
print('Adboost Score', ab.score(x_val, y_val))

#XGBoost Classifier
xgb = XGBClassifier(base_estimator = rf, n_estimators = 50)
xgb.fit(x_train, y_train)
print('Adboost Score', xgb.score(x_val, y_val))