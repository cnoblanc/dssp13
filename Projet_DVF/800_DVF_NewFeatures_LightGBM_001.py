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

dep_selection="All"
model_name="LigthGBM"
#dep_selection="77"
df=dvfdata.loadDVF_Maisons(departement=dep_selection,refresh_force=False
                           ,add_commune=True,filterColsInsee="Permutation")

df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
df_prepared=df_prepared.sample(n=700000, random_state=42)

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
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing')
    ,OrdinalEncoder(categories=categories)
    #,OneHotEncoder(categories=categories)
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
    ,verbose=20
)

# Define the Model
from lightgbm import LGBMRegressor
folds_num=5

reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=50,num_leaves=40
                  ,learning_rate=0.06,n_estimators=2000)
model = make_pipeline(preprocessing,reg)

# ------------------------------------------------------------------------
# compute Cross-validation scores to check overfitting on train set
# ------------------------------------------------------------------------
t0 = time()
#reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
#                  ,max_depth=50,num_leaves=40
#                  ,learning_rate=0.06,n_estimators=2000)

#model = make_pipeline(preprocessing,reg)

#cross_val_scores=cross_val_score(model, X_train, y_train
#                        ,scoring="neg_mean_absolute_error"
#                        ,cv=folds_num,n_jobs=-1,verbose=20)
print("CrossValidation score Done.")
# ------------------------------------------------------------------------
# compute Test Scores
# ------------------------------------------------------------------------
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
#print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (-cross_val_scores.mean(), cross_val_scores.std() * 2))
print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t0))
print("Done CrossVal in : %0.3fs" % (t1_fit_start - t0))
print("Done Fit in : %0.3fs" % (t0_predict - t1_fit_start))
print("Done Predict in : %0.3fs" % (t0_predict_end - t0_predict))
print("---------------------------------------")

f, ax0 = plt.subplots(1, 1, sharey=True)
maxprice=1000000
ax0.scatter(y_test, y_test_predict,s=0.5)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('%s, MAE=%.2f, RMSE=%.2f' % (model_name,predict_score_mae,predict_score_rmse))
ax0.plot([0, maxprice], [0, maxprice], 'k-', color = 'lightblue')
ax0.set_xlim([0, maxprice])
ax0.set_ylim([0, maxprice])

# -------------------
# Permutation Features importance
# -------------------
import eli5
from eli5.sklearn import PermutationImportance

t0_feature_importance_start=time()
# Preprocessing
model = make_pipeline(preprocessing)
# Transform the Train & Test dataset
X_train_transformed=model.fit_transform(X_train)
print("Train set transformed Done: fit_transform")
X_test_transformed=model.transform(X_test)
print("Test set transformed Done: transform")

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
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t0_feature_importance_start))
print("---------------------------------------")

#--------------------------
# Add the month of mutation
#--------------------------
test_df=pd.to_datetime(df_prepared['datemut'])
df_prepared['year']=test_df.dt.year
df_prepared['month']=test_df.dt.month
df_prepared['quarter']=test_df.dt.quarter
df_prepared = df_prepared.drop(columns=['datemut'])
df_for_group=df_prepared[['year','month','valeurfonc']]

stats_year_month_mean=df_for_group.groupby(['year','month']).mean()
stats_year_month_mean.reset_index(inplace=True)
#stats_year_month_mean.head(10)

fig, axes = plt.subplots(figsize=(10, 5))
axes.set_title("Mean price over months")
sns.catplot(ax=axes,x="month", y="valeurfonc", hue="year", kind="bar"
            , data=stats_year_month_mean)