#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:01:11 2019

@author: christophenoblanc
"""
import math
from time import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import dvfdata

# Get & Prepare DVF Data
dep_selection="92"
model_name="LinearRegression"
#dep_selection="77"
df=dvfdata.loadDVF_Maisons(departement=dep_selection,refresh_force=False
                           ,add_commune=False,filterColsInsee="Permutation")
df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
#df_prepared=df_prepared.sample(n=700000, random_state=42)

df_prepared = df_prepared.drop(columns=['departement','n_days','quarter','department_city_dist'])
columns = df_prepared.columns

# Split Train / Test
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

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
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
for i in range(len(categories)):
    categories[i] = ['missing' if x is np.nan else x for x in categories[i]]
category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing')
    ,OrdinalEncoder(categories=categories)
    #,OneHotEncoder(categories=categories,drop=‘first’)
    ,StandardScaler()
)

numeric_pipeline=make_pipeline(
    SimpleImputer(strategy='mean')
    ,StandardScaler()
)

# convenience function for combining the outputs of multiple transformer objects
# applied to column subsets of the original feature space
preprocessing = make_column_transformer(
     # tuples of transformers and column selections
     (numeric_pipeline, num_cols)
    ,(category_pipeline,cat_cols)
    #,(PolynomialFeatures(2,interaction_only=False),columns)
    ,n_jobs=-1
    #,verbose=True
)

# Define the Models
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
folds_num=5
reg=LinearRegression(fit_intercept=True,normalize=True,n_jobs=-1)
model = make_pipeline(preprocessing,reg)

estimators = [('LinearRegression', LinearRegression(fit_intercept=True,normalize=True,n_jobs=-1)),
              ('Theil-Sen', TheilSenRegressor(fit_intercept=True,n_jobs=-1,random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor(fit_intercept=True))]
results=[]
for name, estimator in estimators:
    # ------------------------------------------------------------------------
    # compute Cross-validation scores to check overfitting on train set
    # ------------------------------------------------------------------------
    t0 = time()
    model = make_pipeline(preprocessing,estimator)
    cross_val_scores=cross_val_score(model, X_train, y_train
                    ,scoring="neg_mean_absolute_error"
                    ,cv=folds_num,n_jobs=-1,verbose=20)
    print("CrossValidation score Done.")
    # --------------------
    # compute Test Scores
    # --------------------
    # Apply the Model on full Train dataset
    t1_fit_start=time()
    model.fit(X_train, y_train)
    print("Fit on Train. Done")
    # Predict on Test dataset
    t0_predict = time()
    y_test_predict=model.predict(X_test)
    t0_predict_end = time()
    print("Predict on Test. Done")
    
    # Prediction Score
    predict_score_mae=mean_absolute_error(y_test, y_test_predict)
    predict_score_rmse=math.sqrt(mean_squared_error(y_test, y_test_predict))
    #predict_score_msle=mean_squared_log_error(y_test, y_test_predict_non_negative)
    
    mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test, y_pred=y_test_predict)
    print("------------ Scoring ------------------")
    print("Model Name:",name)
    print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (-cross_val_scores.mean(), cross_val_scores.std() * 2))
    print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
    print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
    print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
    print("---------------------------------------")
    print("Done All in : %0.3fs" % (time() - t0))
    print("Done CrossVal in : %0.3fs" % (t1_fit_start - t0))
    print("Done Fit in : %0.3fs" % (t0_predict - t1_fit_start))
    print("Done Predict in : %0.3fs" % (t0_predict_end - t0_predict))
    print("---------------------------------------")
    
    results.append((name,y_test_predict,predict_score_mae,predict_score_rmse))
    

# Prediction vs True price
f, ax0 = plt.subplots(1, 1, sharey=True)
maxprice=1000000
ax0.scatter(y_test, y_test_predict,s=1)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('%s, MAE=%.2f, RMSE=%.2f' % (model_name,predict_score_mae,predict_score_rmse))
ax0.plot([0, maxprice], [0, maxprice], 'k-', color = 'lightblue')
ax0.plot([0, maxprice], [0, 0], 'k-', color = 'lightblue')
ax0.set_xlim([0, maxprice])
ax0.set_ylim([-150000, maxprice])

# -------------------
# Features importance
# -------------------
print(model.get_params())
features_selection=model['linearregression'].coef_
columns = X_df.columns

features_scores_ordering = np.argsort(features_selection)[::-1]
features_importances = features_selection[features_scores_ordering]
features_names = columns[features_scores_ordering]

features_df=pd.DataFrame(data=features_importances,index=features_names).reset_index()
features_df.columns = ['feature', 'score']

plt.figure(figsize=(6, 3))
top_x=10
x = np.arange(top_x)
plt.bar(x, features_df['score'][:top_x])
plt.xticks(x, features_df['feature'][:top_x], rotation=90, fontsize=10);

# -------------------
# Learning Curve
# -------------------
import dvf_learning_curve

reg=LinearRegression(fit_intercept=True,normalize=True)

model = make_pipeline(preprocessing,reg)

title = "Learning Curves ("+model_name+")"
dvf_learning_curve.plot_learning_curve(model, title, X_df, y
                    ,cv=5, n_jobs=-1,verbose=20
                    ,scoring="neg_mean_absolute_error"
                    ,train_sizes=np.linspace(.1, 1, 10))


