#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:01:11 2019

@author: christophenoblanc
"""

# This one is to use to work on Feature ingeenering
import math
from time import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import dvfdata

dep_selection="All"
model_name="DecisionTree"
df=dvfdata.loadDVF_Maisons(departement=dep_selection,refresh_force=False
                           ,add_commune=True,filterColsInsee=False)
df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
df_prepared=df_prepared.sample(n=700000, random_state=42)

# Split Train / Test
from sklearn.model_selection import train_test_split

# Exclude the Target predicted variable from to create the X and y
X_df = df_prepared.drop(columns='valeurfonc')
y = df_prepared['valeurfonc']
columns = X_df.columns

# Get list of columns by type
cat_cols= X_df.select_dtypes([np.object]).columns
num_cols = X_df.select_dtypes([np.number]).columns
#dvfdata.print_cols_infos(X_df)

# Split data Train & Test
X_train, X_test, y_train, y_test = train_test_split(X_df, y, random_state=42)

# Machine Learning Pipeline 
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing')
    ,OrdinalEncoder(categories=categories)
    #,StandardScaler()
    #,OneHotEncoder(categories=categories)
)

numeric_pipeline=make_pipeline(
    SimpleImputer(strategy='mean')
    #,StandardScaler()
)

# convenience function for combining the outputs of multiple transformer objects
# applied to column subsets of the original feature space
from sklearn.decomposition import PCA
#pca = PCA(n_components=5,random_state=42)
#pca = PCA(0.95,random_state=42)


preprocessing = make_column_transformer(
     # tuples of transformers and column selections
     (numeric_pipeline, num_cols)
    ,(category_pipeline,cat_cols)
    ,n_jobs=-1
    ,verbose=True
)

# Preprocessing
model = make_pipeline(preprocessing)
# Transform the Train & Test dataset
X_train_transformed=model.fit_transform(X_train)
X_test_transformed=model.transform(X_test)

# -------------------
# Permutation Features importance
# -------------------
import eli5
from eli5.sklearn import PermutationImportance

# -------------------
# Decision Tree
# -------------------
# Define the Model
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(max_depth=100,min_samples_leaf=50,random_state=42)

# First, fit the regressor with transformed X
reg.fit(X_train_transformed, y_train)
print("Regression Fit Done.")
# Then, use the Permutation to get Feature importance
perm = PermutationImportance(reg, random_state=42,n_iter=10)
# Using X_Test because we want to get the ones that generalize best
perm.fit(X_test_transformed, y_test) 
print("Permutation Importance Done.")
# Show the feature importance weights : 
print(eli5.format_as_text(eli5.explain_weights(perm,top=50,feature_names = X_test.columns.tolist())))


# -------------------
# Gradiant Boosting
# -------------------
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
reg=HistGradientBoostingRegressor(
    loss='least_squares', learning_rate=0.1, max_depth=None
        , scoring="neg_median_absolute_error", validation_fraction=0.1
        ,max_bins=255,n_iter_no_change=5, tol=1e-07, verbose=0
        ,min_samples_leaf=200
        ,max_iter=500)
# First, fit the regressor with transformed X
reg.fit(X_train_transformed, y_train)
print("Regression Fit Done.")
# Then, use the Permutation to get Feature importance
perm = PermutationImportance(reg, random_state=42,n_iter=10)
# Using X_Test because we want to get the ones that generalize best
perm.fit(X_test_transformed, y_test) 
print("Permutation Importance Done.")
# Show the feature importance weights : 
print(eli5.format_as_text(eli5.explain_weights(perm,top=50,feature_names = X_test.columns.tolist())))


# -------------------
# Random Forest
# -------------------
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(max_features=None,max_depth=None,min_samples_split=2
                          #,min_samples_leaf=None
                          ,n_estimators=300
                          ,random_state=42)
# First, fit the regressor with transformed X
reg.fit(X_train_transformed, y_train)
print("Regression Fit Done.")
# Then, use the Permutation to get Feature importance
perm = PermutationImportance(reg, random_state=42,n_iter=10)
# Using X_Test because we want to get the ones that generalize best
perm.fit(X_test_transformed, y_test) 
print("Permutation Importance Done.")
# Show the feature importance weights : 
print(eli5.format_as_text(eli5.explain_weights(perm,top=50,feature_names = X_test.columns.tolist())))

