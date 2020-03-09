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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor
folds_num=5

import dvfdata

model_name="LigthGBM"

base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF/"



def generateModelScores(departement,df_prepared):
    # Exclude the Target predicted variable from to create the X and y
    X_df = df_prepared.drop(columns='valeurfonc')
    y = df_prepared['valeurfonc']
    columns = X_df.columns
    # Get list of columns by type
    cat_cols= X_df.select_dtypes([np.object]).columns
    num_cols = X_df.select_dtypes([np.number]).columns
    # Split data Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, random_state=42)

    # Get the list of all possible categories from X_df
    categories = [X_df[column].unique() for column in X_df[cat_cols]]
    for i in range(len(categories)):
        categories[i] = ['missing' if x is np.nan else x for x in categories[i]]
    category_pipeline = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing')
        ,OrdinalEncoder(categories=categories)
        #,OneHotEncoder(categories=categories,drop=‘first’)
        #,StandardScaler()
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
    
    # Preprocessing
    model = make_pipeline(preprocessing)
    # Transform the Train & Test dataset
    X_train_transformed=model.fit_transform(X_train)
    print("Train set transformed Done: fit_transform")
    X_test_transformed=model.transform(X_test)
    print("Test set transformed Done: transform")

    train_len=X_train_transformed.shape[0]
    test_len=X_test_transformed.shape[0]

    reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=50,num_leaves=40
                  ,learning_rate=0.06,n_estimators=2000)
    
    # Keep Inliers
    X_train_inliers, y_train_inliers = dvfdata.inliers_split(reg,X_train_transformed, y_train
                ,exclude_outliers=0.5,scoring="neg_mean_absolute_error"
                ,inliers_rate=0.2,folds=5,random_state=42)
    
    # ------------------------------------------------------------------------
    # compute Cross-validation scores to check overfitting on train set
    # ------------------------------------------------------------------------
    t0 = time()

    # After GridSearch 
    reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=40,num_leaves=10
                  ,learning_rate=0.05,n_estimators=400)
    
    #model = make_pipeline(preprocessing,reg)
    
    cross_val_scores=cross_val_score(reg, X_train_inliers, y_train_inliers
                            ,scoring="neg_mean_absolute_error"
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
    total_time=time() - t0
    cross_val_mae=-cross_val_scores.mean()
    
    print("------------ Scoring ------------------")
    print("Departement=",departement)
    print("Cross-Validation Accuracy: %0.2f" % (cross_val_mae))
    print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
    print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
    print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
    print("---------------------------------------")
    print("Done All in : %0.3fs" % (total_time))
    print("Done CrossVal in : %0.3fs" % (t1_fit_start - t0))
    print("Done Fit in : %0.3fs" % (t0_predict - t1_fit_start))
    print("Done Predict in : %0.3fs" % (t0_predict_end - t0_predict))
    print("---------------------------------------")
    print("train_len=",train_len," ; test_len=",test_len)
    
    return cross_val_mae,mae, mape,rmse,total_time,train_len,test_len


df=dvfdata.loadDVF_Maisons(departement="Metropole",refresh_force=False
                       ,add_commune=True,filterColsInsee="Permutation")
df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
#df_prepared=df_prepared.sample(n=700000, random_state=42)

# remove departements where we do not have enough values
df_prepared=df_prepared[df_prepared['departement'] != '75']
columns = df_prepared.columns
dep_list = df_prepared['departement'].unique()
#dep_list=['01','02']

columns = ['dep','val_mae','mae', 'mape','rmse','total_time','train_len','test_len']
index=range(len(dep_list))
df_results = pd.DataFrame(index=index, columns=columns)
df_results[['val_mae','mae', 'mape','rmse','total_time','train_len','test_len']]=0.0
df_results['dep']=''


for i in range(len(dep_list)):
    print("---------------------------------------")
    print("call for i=",i)
    print("Start for departement=",dep_list[i])
    print("---------------------------------------")
    df_oneDep=df_prepared[df_prepared['departement']==dep_list[i]]
    df_oneDep=df_oneDep.drop(columns=['departement'])
    val_mae,mae, mape,rmse,total_time,train_len,test_len  \
            =generateModelScores(departement=dep_list[i],df_prepared=df_oneDep)
    df_results.iloc[i]['dep']=dep_list[i]
    df_results.iloc[i]['val_mae']=val_mae
    df_results.iloc[i]['mae']=mae
    df_results.iloc[i]['mape']=mape
    df_results.iloc[i]['rmse']=rmse
    df_results.iloc[i]['total_time']=total_time
    df_results.iloc[i]['train_len']=train_len
    df_results.iloc[i]['test_len']=test_len
    
print("---------------------------------------")
print("Save to local parquet file")
parquet_fileName=base_path+"keras_models_saved/df_results_AllDep_LightGBM"
df_results.to_parquet(parquet_fileName, engine='fastparquet',compression='GZIP') 

# Load Parquet
#df_results=pd.read_parquet(parquet_fileName, engine='fastparquet')

# Calculate the Mean of the cross-val MAE, the Test MAE & MAPE
df_results['weighted_val_mae']=df_results['val_mae']*df_results['train_len']
df_results['weighted_mae']=df_results['mae']*df_results['test_len']
df_results['weighted_mape']=df_results['mape']*df_results['test_len']
df_results['weighted_rmse']=df_results['rmse']*df_results['test_len']

print("------------ Final Scoring on full dataset------------------")
print("Cross-Validation Accuracy: %0.2f" % (df_results['weighted_val_mae'].sum()/df_results['train_len'].sum()))
print("Price diff error MAE: %0.2f" % (df_results['weighted_mae'].sum()/df_results['test_len'].sum()))
print("Percent of Price error MAPE: %0.2f" % (df_results['weighted_mape'].sum()/df_results['test_len'].sum()))
print("Price error RMSE: %0.2f" % (df_results['weighted_rmse'].sum()/df_results['test_len'].sum()))
print("---------------------------------------")
print("Done All in : %0.3fs" % (df_results['total_time'].sum()))

# Show the prediction performance by Departements
#plt.figure(figsize=(20, 20))
df_results.sort_values(by='mae', ascending=False,inplace=True)
df_results["below 40.000"] = (df_results["mae"] < 40000)
sns.set(style="whitegrid")
plt.figure(figsize=(5, 20))
ax = sns.barplot(x="mae", y="dep", hue="below 40.000",data=df_results
                 ,palette="Blues_d",dodge=False)
plt.title(model_name+' - Score MAE par département')
plt.xlabel('MAE (euros)')
plt.ylabel('Département')
plt.axvline(40000)
plt.show()


# Carte avec départements de France :
# Credit : https://perso.esiee.fr/~courivad/Python/15-geo.html
import folium as flm
import json
import geojson
#import geopandas
from branca.colormap import linear

#d = {'CodeDep': [77, 93], 'Population': [300000, 500000]}
#df = pd.DataFrame(data=d)
#df_dict = df.set_index('CodeDep')['Population']

# For color map palette : http://colrd.com/palette/
#colormap = linear.YlGn_09.scale(df_results.mae.min(),df_results.mae.max())
colormap = linear.Blues_09.scale(df_results.mae.min(),df_results.mae.max())
colormap.caption = 'MAE score scale'

df_dict = df_results.set_index('dep')['mae']

# read Departement geojson file
files = [ base_path+"map_saved/"+'departements.geojson' ]
file=base_path+"map_saved/"+'departements.geojson'

#f = open(file, 'r', encoding='utf8')
#json_data = json.loads(f.read())

f = open(file, 'r', encoding='utf8')
geojson_data=geojson.loads(f.read())

min_mae=df_dict.min()
drop_dep_indexes=[]
for i,feature in enumerate(geojson_data.features,0):
    #print(i,", feature value is :",feature["properties"]['code'])
    if feature["properties"]['code'] not in df_dict:
        print("will Drop dep:",feature["properties"]['code'] )
        drop_dep_indexes.append(i)
        # easier : add a min value for missing departements in df_dict
        df_dict=df_dict.set_value(feature["properties"]['code'],min_mae)
#print(drop_dep_indexes)   
#gdf = geopandas.read_file(f)

# Create the map
coords = (47.0,2.5)
map = flm.Map(location=coords, tiles='OpenStreetMap', zoom_start=6)

#folium.GeoJson(geo_json_data).add_to(map)
#folium.GeoJson(gdf).add_to(map)
flm.GeoJson(
    geojson_data,
    name='MAE',
    style_function=lambda feature :{'fillColor':colormap(df_dict[feature['properties']['code']])
                , 'fillOpacity':0.9, 'color':'#0096FF', 'weight':1, 'opacity':1}
    #lambda feature: {'fillColor': '#ffff00','color': '#2BA8FF','weight': 1}
   ,tooltip=flm.features.GeoJsonTooltip(fields=['nom','code'],labels=True,sticky=True)
    , show=False
).add_to(map)
colormap.add_to(map)

fileName=base_path+"map_saved/"+"perf_ligntgbm_departements.html"
map.save(outfile=fileName)

#test=geo_json_data['features']['properties']['code']
colormap

