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

# Get & Prepare DVF Data
import dvfdata
df=dvfdata.loadDVF_Maisons(departement='All',refresh_force=False,add_commune=True)
df_prepared=dvfdata.prepare_df(df,remove_categories=False)

# Split Train / Test
from sklearn.model_selection import cross_val_score
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
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='Unknown')
    ,OrdinalEncoder(categories=categories)
    #,OneHotEncoder(categories=categories)
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
    ,n_jobs=-1
    #,verbose=True
)

reg=LinearRegression(fit_intercept=True,normalize=True)
model = make_pipeline(preprocessing,reg)

# compute Cross-validation scores to not over-fit on test set for hyper-parameter search
folds_num=5
cross_val_scores=cross_val_score(model, X_train, y_train
                        , scoring="neg_median_absolute_error",cv=folds_num,n_jobs=-1)

# Apply the Model on full Train dataset
model.fit(X_train, y_train)

# Predict on Test dataset
y_test_predict=model.predict(X_test)

# Prediction Score
predict_score_mae=mean_absolute_error(y_test, y_test_predict)
predict_score_rmse=math.sqrt(mean_squared_error(y_test, y_test_predict))
#predict_score_rmsle=math.sqrt(mean_squared_log_error(y_test, y_test_predict))

mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test, y_pred=y_test_predict)
print("------------ Scoring ------------------")
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (-cross_val_scores.mean(), cross_val_scores.std() * 2))
print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
print("---------------------------------------")

f, ax0 = plt.subplots(1, 1, sharey=True)
ax0.scatter(y_test, y_test_predict,s=0.5)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('LinearRegression, MAE=%.2f, RMSE=%.2f' % (predict_score_mae,predict_score_rmse))
#ax0.set_xlim([0, 1000000])
#ax0.set_ylim([0, 1000000])

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

