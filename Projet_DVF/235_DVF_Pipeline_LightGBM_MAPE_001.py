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
#df_prepared=df_prepared.sample(n=700000, random_state=42)

#df_prepared = df_prepared.drop(columns=['departement','n_days','quarter','department_city_dist'])
#columns = df_prepared.columns

# Split Train / Test
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import learning_curve

# Exclude the Target predicted variable from to create the X and y
X_df = df_prepared.drop(columns='valeurfonc')
y = df_prepared['valeurfonc']
#y = np.log1p(df_prepared['valeurfonc'])
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
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
for i in range(len(categories)):
    categories[i] = ['missing' if x is np.nan else x for x in categories[i]]
    #print(categories[i])
    
category_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing')
    ,OrdinalEncoder(categories=categories)
    #,OneHotEncoder(categories=categories,drop=‘first’)
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
from sklearn.linear_model import RANSACRegressor

folds_num=5
tuned_parameters={ #'max_depth':[20,30,40,50,70]
                  'num_leaves':[5,10,20]
                  ,'learning_rate':[0.03,0.04,0.05,0.06]
                  ,'n_estimators':[300,400,500]
                  #,'lgbmregressor__min_child_samples':[5,10,20,50]
                  #,'min_data_in_leaf'
                  }

# for 1 departement : 92
#reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
#                  ,max_depth=40,num_leaves=10,learning_rate=0.05,n_estimators=400)
reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=50,num_leaves=40,learning_rate=0.06,n_estimators=2000)


#model = make_pipeline(preprocessing,reg)

# ------------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------------
# Search hyper-parameters 
#print(model.get_params())

# Preprocessing
model = make_pipeline(preprocessing)
# Transform the Train & Test dataset
X_train_transformed=model.fit_transform(X_train)
X_test_transformed=model.transform(X_test)

# ------------------------------------------------------------------------
# Search hyper-parameters 
# ------------------------------------------------------------------------
# Search hyper-parameters 
#print(model.get_params())

# Define de scorer object to be the personalized MAPE
from mape_error_module import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

t0 = time()
tuned_parameters={ 'max_depth':[20,30,40,50,70]
                  ,'num_leaves':[5,10,20,40]
                  ,'learning_rate':[0.04,0.05,0.06]
                  ,'n_estimators':[500,1000,1500,2000]
                  #,'lgbmregressor__min_child_samples':[5,10,20,50]
                  #,'min_data_in_leaf'
                  }
model_grid=RandomizedSearchCV(estimator=reg,param_distributions=tuned_parameters
                          ,scoring=mape_scorer
                          ,refit=True
                          ,cv=2,verbose=20,n_iter=25)
model_grid.fit(X_train_transformed, y_train)
print("---------------------------------------")
print("Best parameters set found on train set:")
print(model_grid.best_params_)
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t0))
print("---------------------------------------")

# Save results in a DataFrame
df_gridcv = pd.DataFrame(model_grid.cv_results_)
gridcv_columns_to_keep = [
    'param_max_depth',
    'param_num_leaves',
    'param_learning_rate',
    'param_n_estimators',
    #'param_min_child_samples',
    'mean_test_score','std_test_score',
    'mean_fit_time','mean_score_time'
]
df_gridcv = df_gridcv[gridcv_columns_to_keep]
df_gridcv = df_gridcv.sort_values(by='mean_test_score', ascending=False)
#df_gridcv.to_parquet("data_parquet/GridSearch_"+dep_selection+"_"+model_name+".parquet", engine='fastparquet',compression='GZIP')

# ------------------------------------------------------------------------
# compute Cross-validation scores to check overfitting on train set
# ------------------------------------------------------------------------
t0 = time()
reg = model_grid.best_estimator_
#reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
#                  ,max_depth=50,num_leaves=40,learning_rate=0.06,n_estimators=2000)

cross_val_scores=cross_val_score(reg, X_train_transformed, y_train
                        ,scoring=mape_scorer
                        ,cv=folds_num,n_jobs=-1,verbose=20)
print("CrossValidation score Done.")

# ------------------------------------------------------------------------
# compute Test Scores
# ------------------------------------------------------------------------
# Apply the Model on full Train dataset
t1_fit_start=time()
reg.fit(X_train_transformed, y_train)
print("Fit on Train. Done")
# Predict on Test dataset
t0_predict = time()
y_test_predict=reg.predict(X_test_transformed)
t0_predict_end = time()
print("Predict on Test. Done")

# Prediction Score : transformation réciproque
y_test_recip=y_test
y_test_predict_recip=y_test_predict

#y_test_recip=np.expm1(y_test)
#y_test_predict_recip=np.expm1(y_test_predict)
#cross_val_scores_2=np.expm1(cross_val_scores)

predict_score_mae=mean_absolute_error(y_test_recip, y_test_predict_recip)
predict_score_rmse=math.sqrt(mean_squared_error(y_test_recip, y_test_predict_recip))
#predict_score_msle=mean_squared_log_error(y_test, y_test_predict_non_negative)

mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test_recip, y_pred=y_test_predict_recip)
print("------------ Scoring ------------------")
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

# Prediction vs True price
f, ax0 = plt.subplots(1, 1, sharey=True)
maxprice=1000000
#ax0.scatter(np.expm1(y_test), np.expm1(y_test_predict),s=0.5)
ax0.scatter(y_test, y_test_predict_recip,s=0.5)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('%s, MAE=%.2f, RMSE=%.2f' % (model_name,predict_score_mae,predict_score_rmse))
ax0.plot([0, maxprice], [0, maxprice], 'k-', color = 'lightblue')
ax0.set_xlim([0, maxprice])
ax0.set_ylim([0, maxprice])

# -------------------
# Features importance
# -------------------
features_selection=model['lgbmregressor'].feature_importances_
columns = X_df.columns
# select all features ordered by importance score
features_scores_ordering = np.argsort(features_selection)[::-1]
features_importances = features_selection[features_scores_ordering]
features_names = columns[features_scores_ordering]
# create DataFrame with Features Names & Score
features_df=pd.DataFrame(data=features_importances,index=features_names).reset_index()
features_df.columns = ['feature', 'score']
features_df.to_parquet("data_parquet/FeatureImportance_"+dep_selection+"_"+model_name+".parquet"
                       , engine='fastparquet',compression='GZIP')

plt.figure(figsize=(10, 3))
top_x=13
x = np.arange(top_x)
plt.bar(x, features_df['score'][:top_x])
plt.xticks(x, features_df['feature'][:top_x], rotation=90, fontsize=10);

# -------------------
# Learning Curve
# -------------------
import dvf_learning_curve

reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=50,num_leaves=40
                  ,learning_rate=0.06,n_estimators=2000)

model = make_pipeline(preprocessing,reg)

title = "Learning Curves ("+model_name+")"
dvf_learning_curve.plot_learning_curve(model, title, X_df, y
                    ,cv=5, n_jobs=-1,verbose=20
                    ,scoring="neg_mean_absolute_error"
                    ,train_sizes=np.linspace(.1, 1, 10))

X_df.shape

# -------------------
# Errors
# -------------------
X_test['abs_error']=(y_test_predict_recip-y_test).abs()
X_test['price']=y_test
X_test['price_predict']=y_test_predict_recip
X_test['mape']= 100*((y_test_predict_recip-y_test)/y_test).abs()
y_absolute_error=(y_test_predict_recip-y_test).abs()
X_test.sort_values(by='abs_error', ascending=False,inplace=True)

# Density of errors :
sns.distplot(X_test['abs_error'],kde=False, rug=True)
plt.suptitle("Hauts-de-Seine : Répartition erreurs de prédiction", fontsize=12,y=0.95) 
plt.show()
# -------------------
# Carte des erreurs
# -------------------
# Carte avec départements de France :
# Credit : https://perso.esiee.fr/~courivad/Python/15-geo.html
import folium as flm
from branca.colormap import linear

#d = {'CodeDep': [77, 93], 'Population': [300000, 500000]}
#df = pd.DataFrame(data=d)
#df_dict = df.set_index('CodeDep')['Population']

# For color map palette : http://colrd.com/palette/
#colormap = linear.YlGn_09.scale(df_results.mae.min(),df_results.mae.max())
#colormap = linear.Blues_09.scale(y_absolute_error.min(),y_absolute_error.max())
colormap = linear.YlOrRd_09.scale(y_absolute_error.min(),y_absolute_error.max())
colormap.caption = 'Prediction Price Absolute Error scale'
base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF/"
file=base_path+"map_saved/"+'lightGBM'

X_test.sort_values(by='abs_error', ascending=True,inplace=True)
# Create the map
centermap_lat=(X_test['geolat'].max()+X_test['geolat'].min())/2
centermap_long=(X_test['geolong'].max()+X_test['geolong'].min())/2
coords = (centermap_lat,centermap_long)
map = flm.Map(location=coords, tiles='OpenStreetMap', zoom_start=12)

for i in range(X_test.shape[0]):
    flm.Circle([X_test.iloc[i].geolat, X_test.iloc[i].geolong], 10\
            ,fill=True, fill_opacity=1\
            ,fill_color=colormap(X_test.iloc[i].abs_error)
            ,color=colormap(X_test.iloc[i].abs_error)) \
        .add_to(map)
        #.add_child(flm.Popup("Real Price is :",str(X_test.iloc[i].price),"; predicted is :",str(X_test.iloc[i].price_predict))).add_to(map)
        #.add_to(map)\
colormap.add_to(map)

fileName=base_path+"map_saved/"+"perf_ligntgbm.html"
map.save(outfile=fileName)
