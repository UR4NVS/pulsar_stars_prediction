# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:05:01 2019

@author: alban

Dataset: Pavan Raj on kaggle.com
"""
import csv
from os.path import join, dirname
import numpy as np
import mglearn
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd 
from math import *
from sklearn.model_selection._search import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
iris_dataset = load_iris()

pulsar_dataset = pd.read_csv(
        "C:\\Users\\alban\\Documents\\Pÿthon_ML\\Datasets\\data\\pulsar_stars\\pulsar_stars.csv")
pulsar_visualisation = pd.DataFrame(pulsar_dataset)
# .data = pulsar_dataset.values ( .values = transforme les données pandas en NumPy [pour scikit-learn])
# .target = le nom de la classe target dans ce cas : target_class + .values à la fin pour valeurs NumPy
target_names = np.array(['other star', 'pulsar star'])
# feature_names, .columns = donne le nom des colonenes, list(x) converti x en liste
feature_names = list(pulsar_dataset.columns) # ou .keys()

X_train, X_test, y_train, y_test = train_test_split(pulsar_dataset.values, 
                                                    pulsar_dataset.target_class.values,
                                                    stratify=pulsar_dataset.target_class.values, 
                                                    random_state=42)
"""# pipeline pour KMeans et LinearSVC ensemble
pipeline_1 = make_pipeline(KMeans(), LinearSVC())
# print(sorted(pipeline_1.get_params().keys()))

param_grid_1 = {'linearsvc__C' :[0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10, 20, 50, 100, 1000], 
                'linearsvc__penalty': ['l1', 'l2'], 
                'kmeans__n_clusters': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
grid_1 = GridSearchCV(pipeline_1, param_grid=param_grid_1, cv=5, n_jobs=-1)
grid_1.fit(X_train, y_train)
pred_grid_1 = grid_1.predict(X_test)
matrix_1 = confusion_matrix(y_test, pred_grid_1)

result_1 = []
result_1.append(grid_1.best_params_)
result_1.append(grid_1.best_score_)
result_1.append(grid_1.score(X_test, y_test))
result_1.append(matrix_1)


# pipeline pour KMeans et SVC ensemble
pipeline_2 = make_pipeline(KMeans(), SVC())
# print(sorted(pipeline_2.get_params().keys()))

param_grid_2 = {'svc__C': [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10, 20, 50, 100, 1000],
                'svc__gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10, 20, 50, 75, 100],
                'kmeans__n_clusters': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
grid_2 = GridSearchCV(pipeline_2, param_grid=param_grid_2, cv=5, n_jobs=-1)
grid_2.fit(X_train, y_train)
pred_grid_2 = grid_2.predict(X_test)
matrix_2 = confusion_matrix(y_test, pred_grid_2)
result_2 = []
result_2.append(grid_2.best_params_)
result_2.append(grid_2.best_score_)
result_2.append(grid_2.score(X_test, y_test))

# pipeline pour KMeans et SGDClassifier ensemble
pipeline_3 = make_pipeline(KMeans(), SGDClassifier())
# print(sorted(pipeline_3.get_params().keys()))

param_grid_3 = {'sgdcclassifier__penalty': ['l1', 'l2', 'elasticnet'],
                'kmeans__n_clusters': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
grid_3 = GridSearchCV(pipeline_3, param_grid=param_grid_3, cv=5, n_jobs=-1)
grid_3.fit(X_train, y_train)
pred_grid_3 = grid_3.predict(X_test)
matrix_3 = confusion_matrix(y_test, pred_grid_3)
result_3 = []
result_3.append(grid_3.best_params_)
result_3.append(grid_3.best_score_)
result_3.append(grid_3.score(X_test, y_test))


# pipeline pour KMeans et DecisionTreeClassifier ensemble
pipeline_4 = make_pipeline(KMeans(), DecisionTreeClassifier())
# print(sorted(pipeline_4.get_params().keys()))

param_grid_4 = {'decisiontreeclassifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30],
                'kmeans__n_clusters': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
grid_4 = GridSearchCV(pipeline_4, param_grid=param_grid_4, cv=5, n_jobs=-1)
grid_4.fit(X_train, y_train)
pred_grid_4 = grid_4.predict(X_test)
matrix_4 = confusion_matrix(y_test, pred_grid_4)
result_4 = []
result_4.append(grid_4.best_params_)
result_4.append(grid_4.best_score_)
result_4.append(grid_4.score(X_test, y_test))

# pipeline pour KMeans et RamdomForestClassifier ensemble
pipeline_5 = make_pipeline(KMeans(), RandomForestClassifier())
print(sorted(pipeline_5.get_params().keys()))

mf = sqrt(len(feature_names))
param_grid_5 = {'randomforestclassifier__max_depth': [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30], 
                'randomforestclassifier__max_features': [3], 
                'randomforestclassifier__n_estimators': [5, 6, 7, 8, 9, 10, 50, 100, 250, 500, 1000]}
grid_5 = GridSearchCV(pipeline_5, param_grid=param_grid_5, cv=5, n_jobs=-1)
grid_5.fit(X_train, y_train)
pred_grid_5 = grid_5.predict(X_test)
matrix_5 = confusion_matrix(y_test, pred_grid_5)
result_5 = []
result_5.append(grid_5.best_params_)
result_5.append(grid_5.best_score_)
result_5.append(grid_5.score(X_test, y_test))

# pipeline pour KMeans et GradientBoostingClassifier ensemble
pipeline_6 = make_pipeline(KMeans(), GradientBoostingClassifier())
# print(sorted(pipeline_6.get_params().keys()))

param_grid_6 = {'gradientboostingclassifier__max_depth': [1, 2, 3, 4, 5], 
                'gradientboostingclassifier__n_estimators': [50, 100, 150, 200, 500, 1000],
                'gradientboostingclassifier__learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10],
                'kmeans__n_clusters': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
grid_6 = GridSearchCV(pipeline_6, param_grid=param_grid_6, cv=5, n_jobs=-1)
grid_6.fit(X_train, y_train)
pred_grid_6 = grid_6.predict(X_test)
matrix_6 = confusion_matrix(y_test, pred_grid_6)
result_6 = []
result_6.append(grid_6.best_params_)
result_6.append(grid_6.best_score_)
result_6.append(grid_6.score(X_test, y_test))

final_results = (result_1, result_2, result_3, result_4, result_5, result_6)"""

# {'decisiontreeclassifier__max_depth': 4, 'kmeans__n_clusters': 11}
# on passe de 11 à 9 car c'est le nombre de features maximal
dtreecl = DecisionTreeClassifier(max_depth=4)
kmeans = KMeans(n_clusters=9)
X_train_scaled = kmeans.fit_transform(X_train)
dtreecl.fit(X_train_scaled, y_train)
dtreecl_pred = dtreecl.predict(X_test)
# print("Test set predictions \n {}".format(dtreecl_pred))
