#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:01:11 2019

@author: christophenoblanc
"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import dvfdata
df=dvfdata.loadDVF_Maisons(departement='All',refresh_force=False,add_commune=True)
df_prepared=dvfdata.prepare_df(df,remove_categories=False)

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
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing')
    ,OrdinalEncoder(categories=categories)
    #,OneHotEncoder
)

numeric_pipeline=make_pipeline(
    SimpleImputer(strategy='mean')
    #,StandardScaler()
)

# convenience function for combining the outputs of multiple transformer objects
# applied to column subsets of the original feature space
preprocessing = make_column_transformer(
     # tuples of transformers and column selections
     (numeric_pipeline, num_cols)
    ,(category_pipeline,cat_cols)
    ,n_jobs=-1
    ,verbose=True
)

# Define the Model
from sklearn.tree import DecisionTreeRegressor
model_name="DecisionTree"
folds_num=5
tuned_parameters={'decisiontreeregressor__max_depth': [10,50,100,200,300]
                  ,'decisiontreeregressor__min_samples_leaf':[5,10,20,50]}

reg=DecisionTreeRegressor()
model = make_pipeline(preprocessing,reg)

# ------------------------------------------------------------------------
# Search hyper-parameters 
# ------------------------------------------------------------------------
# Search hyper-parameters 
print(model.get_params())
model_grid = GridSearchCV(model,tuned_parameters,scoring="neg_mean_absolute_error"
                          ,n_jobs=-1,cv=5,verbose=10)
model_grid.fit(X_train, y_train)
print("---------------------------------------")
print("Best parameters set found on train set:")
print(model_grid.best_params_)

df_gridcv = pd.DataFrame(model_grid.cv_results_)
gridcv_columns_to_keep = [
    'param_decisiontreeregressor__max_depth',
    'param_decisiontreeregressor__min_samples_leaf',
    'mean_test_score','std_test_score',
    'mean_fit_time','mean_score_time'
]
df_gridcv = df_gridcv[gridcv_columns_to_keep]
df_gridcv = df_gridcv.sort_values(by='mean_test_score', ascending=False)
print(df_gridcv)

# ------------------------------------------------------------------------
# compute Cross-validation scores to not over-fit on test set for hyper-parameter search
# ------------------------------------------------------------------------
reg=DecisionTreeRegressor(max_depth=100,min_samples_leaf=20)
model = make_pipeline(preprocessing,reg)

cross_val_scores=cross_val_score(model, X_train, y_train
                        , scoring="neg_mean_absolute_error",cv=folds_num,n_jobs=-1)

# Apply the Model on full Train dataset
model.fit(X_train, y_train)

# Predict on Test dataset
y_test_predict=model.predict(X_test)

# Prediction Score
predict_score_mae=mean_absolute_error(y_test, y_test_predict)
predict_score_rmse=math.sqrt(mean_squared_error(y_test, y_test_predict))
#predict_score_msle=mean_squared_log_error(y_test, y_test_predict_non_negative)

mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test, y_pred=y_test_predict)
print("------------ Scoring ------------------")
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (-cross_val_scores.mean(), cross_val_scores.std() * 2))
print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse * 2))
print("---------------------------------------")

f, ax0 = plt.subplots(1, 1, sharey=True)
ax0.scatter(y_test, y_test_predict,s=0.5)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('%s, MAE=%.2f, RMSE=%.2f' % (model_name,predict_score_mae,predict_score_rmse))
#ax0.set_xlim([0, 1000000])
#ax0.set_ylim([0, 1000000])
