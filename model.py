#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:40:23 2022

@author: farishanna
"""

#to manipulate data as a dataframe
import pandas as pd
# To make any needed calculations
import numpy as np

#to visualise data and results
import matplotlib.pyplot as plt
import seaborn as sns

# For managing imbalanced data
from imblearn.over_sampling import SMOTE
#to split data into training and testing, cross-validation, and finding the best parameters for most accurate model
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
#to select best features
from sklearn.feature_selection import SelectKBest, chi2

# To use XGBClassifer algorithm
from xgboost import XGBClassifier
# Creating pipeline
from sklearn.pipeline import Pipeline

#to calculate accuracy score and plot confusion matrix
from sklearn.metrics import accuracy_score, plot_confusion_matrix


df = pd.read_csv('Bank_Personal_Loan_Modelling.csv') # Reading the data into a pandas dataframe
print(df.head()) # Displays first 5 rows
print(df.info()) # Checking for null values and datatypes
print(df.describe()) # Getting data statistics

#  -- from https://stackoverflow.com/questions/29077188/absolute-value-for-column-in-python
df['Experience'] = df['Experience'].abs() # Turning negative values postive

# Checking for any outliers using a boxplot
for column in ['Income', 'CCAvg', 'Mortgage']: 
    sns.boxplot(x=df[column]) 
    plt.show()

# Removing outliers
before = len(df)
df = df.loc[df['Income'] < 220]
df = df.loc[df['CCAvg'] < 10]
print('\nOutliers removed:', before-len(df), '\n')

# Checking histograms of Age and Experience    Obtained from --https://matplotlib.org/3.1.1/gallery/statistics/histogram_multihist.html
fig, ax = plt.subplots() 
# Hist representing Age and Experience
ax.hist(df["Age"], bins=15, alpha=0.5, color="red", label="Age") 
ax.hist(df["Experience"], bins=15, alpha=0.5, color="blue", label="Experience") 
# Adding labels, subtitles and legend
ax.set_xlabel("Years")
ax.set_ylabel("Count")
fig.suptitle("Age and Experience")
ax.legend();

# Checking for imbalances
for column in ['Family', 'Education']:
    pivot = pd.pivot_table(df, values='ID', index=[column], aggfunc='count')
    pivot.plot(kind='bar', legend='')
    plt.show()

# Checking for the counts of the binary columns -- used from: https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/
# To store counts of true and false
true_count = []
false_count = []
# Appends lists 
for column in df.columns[9:]:
    true_count.append(len(df.loc[df[column] == 1]))
    false_count.append(len(df.loc[df[column] == 0]))
# Create appropriate x axis
X_axis = np.arange(len(df.columns[9:]))
fig = plt.figure(figsize=(10, 5))
# Creating bar charts with true or false labels
ab_bar_list = [
               plt.bar(X_axis+0.2, true_count, align='edge', width= 0.4, label = 'True'),
               plt.bar(X_axis-0.2, false_count, align='edge', width= 0.4, label = 'False')
               ]
# X axis names  
plt.xticks(X_axis+0.2, df.columns[9:])
# Title and legend
plt.title("Binary Values Counts")
plt.legend()
plt.show()    

# Heatmap  --style used by: https://www.kaggle.com/yamanizm/personal-loan-eda-ml-98-iamdatamonkey
plt.figure(figsize=(15,5))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.2f')
plt.show()


# Splitting feature variables with the target variable
features = df.drop(columns=['Personal Loan']).values
target = df['Personal Loan']

# Checking for imbalanced data on the target
print(target.value_counts())

# Balancing data using SMOTE
smote = SMOTE()
features, target = smote.fit_resample(features, target)

# K fold splitting training and testing data
kf = KFold(n_splits=10, shuffle=True)
for train, test in kf.split(features):
    features_train, features_test, target_train, target_test = features[train], features[test], target[train], target[test]

# Prints scores of a specified model 
def model_scores(model, grid=False):      
    model.fit(features_train, target_train) # Fit model
    target_pred = model.predict(features_test) # Predict test features
    
    accuracy = round(accuracy_score(target_test, target_pred, normalize = True)*100, 2) # Accuracy
    cv_scores = cross_val_score(model, features, target, cv=kf, n_jobs=-1) # CV Score
    cv_mean = round(np.mean(cv_scores)*100, 2) # Mean of CV Scores
    print('Test Results:', f'{accuracy}%') # Print accuracy
    print('CV Results Mean:', f'{cv_mean}%')  # Print CV Results
    
    # Print best params if gridsearch is used
    if(grid==True):
        print('Best Parameters: ', model.best_params_)
        
    # Boxplot of cv scores and confusion matrix 
    sns.boxplot(x=cv_scores)
    plt.show()  
    plot_confusion_matrix(model, features_test, target_test)
    plt.show()

# Printing scores for XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
print('\nXGBoost Classifier Scores:')
model_scores(model)

# Pipeline --guidance by: https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/
pipe = Pipeline([
('selection', SelectKBest(chi2)),
('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])
# Printing pipeline results
print('\nPipeline Model Scores:')
model_scores(pipe)

# Reading parameters in the config file
file = "parameters.config"
parameters = eval(open(file).read())
# Optimizing pipeline using GridSearchCV
grid = GridSearchCV(pipe, parameters, cv=kf)
# Printing gridsearch results
print('\nGridSearchCV Model Scores:')
model_scores(grid, grid=True)
