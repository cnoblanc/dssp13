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

def prepare_df(DF):
    # Remove the extrem values
    selected_df=df[(df["valeurfonc"]<1000000) & (df["sterr"]<10000) & (df["nbpprinc"]<=10 ) & (df["nbpprinc"]>0) & (df["sbati"]<=500)]
    # Transform
    cat_cols= selected_df.select_dtypes([np.object]).columns
    X_drop = selected_df.drop(columns=cat_cols)
    #X_drop = selected_df.drop(columns=['quartier','commune','departement','communelabel','codepostal'])
    return(X_drop)

df_prepared=prepare_df(df)


# Split Train / Test
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Exclude the Target predicted variable from to create the X and y
X_df = df_prepared.drop(columns='valeurfonc')
y = df_prepared['valeurfonc']
columns = X_df.columns

# Get list of columns by type
cat_cols= X_df.select_dtypes([np.object]).columns
print(cat_cols)
for col in cat_cols:
    print("Column'",col,"' values (",len(X_df[col].unique()),") are:",X_df[col].unique()[:20])
num_cols = X_df.select_dtypes([np.number]).columns
print(num_cols)

# Split data Train & Test
X_train, X_test, y_train, y_test = train_test_split(X_df, y, random_state=42)


# Machine Learning Pipeline 
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='Unknown'),
    OrdinalEncoder(),
    #OneHotEncoder,
)
numeric_pipeline=SimpleImputer(strategy='mean')

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
ax0.set_title('LinearRegression, MAE=%.2f, RMSE=%.2f' % (predict_score_mae,predict_score_rmse))
#ax0.set_xlim([0, 1000000])
#ax0.set_ylim([0, 1000000])
