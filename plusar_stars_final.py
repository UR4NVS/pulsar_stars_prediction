# -*- coding: utf-8 -*-
"""
@author: alban

Dataset: Pavan Raj on kaggle.com
"""
import csv
from os.path import join, dirname
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd 
from math import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


pulsar_dataset = pd.read_csv(
        "C:\\Users\\alban\\Documents\\Pÿthon_ML\\Datasets\\data\\pulsar_stars\\pulsar_stars.csv")
pulsar_visualisation = pd.DataFrame(pulsar_dataset)

target_names = np.array(['other star', 'pulsar star'])
feature_names = list(pulsar_dataset.columns) 

X_train, X_test, y_train, y_test = train_test_split(pulsar_dataset.values, 
                                                    pulsar_dataset.target_class.values,
                                                    stratify=pulsar_dataset.target_class.values, 
                                                    random_state=42)

dtreecl = DecisionTreeClassifier(max_depth=4)
kmeans = KMeans(n_clusters=9)
X_train_scaled = kmeans.fit_transform(X_train)
dtreecl.fit(X_train_scaled, y_train)
dtreecl_pred = dtreecl.predict(X_test)
print("Test set predictions \n {}".format(dtreecl_pred))
