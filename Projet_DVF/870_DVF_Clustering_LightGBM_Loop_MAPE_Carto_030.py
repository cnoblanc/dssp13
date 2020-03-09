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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from lightgbm import LGBMRegressor
folds_num=5

import dvfdata

model_name="LigthGBM"

base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF/"


df=dvfdata.loadDVF_Maisons(departement="Metropole",refresh_force=False
                       ,add_commune=True,filterColsInsee="Permutation")

df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
#df_prepared=df_prepared.sample(n=700000, random_state=42)

#df_prepared = df_prepared.drop(columns=['departement','n_days','quarter','department_city_dist'])
df_prepared = df_prepared.drop(columns=['departement'])
#columns = df_prepared.columns

print("---------------------------------------")
print("Start Pre-processing")
print("---------------------------------------")
X_df = df_prepared.drop(columns='valeurfonc')
#y = df_prepared['valeurfonc']
y = np.log1p(df_prepared['valeurfonc'])

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
    #,n_jobs=-1
    ,verbose=20
)

# Preprocessing
tranf_model = make_pipeline(preprocessing)
# Transform the Train & Test dataset
X_train_transformed=tranf_model.fit_transform(X_train)
print("Train set transformed Done: fit_transform")
X_test_transformed=tranf_model.transform(X_test)
print("Test set transformed Done: transform")

train_len=X_train_transformed.shape[0]
test_len=X_test_transformed.shape[0]

# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------

print("---------------------------------------")
print("Create models by clusters")
print("---------------------------------------")
from mape_error_module import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


from sklearn.cluster import KMeans

cluster_init="k-means++"
max_clusters=1

columns = ['cluster_cnt','init','inertia','cluster_time','crossval_mape','test_mae','test_mape','test_rmse','models_time']
index=range(max_clusters)
df_cluster_results = pd.DataFrame(index=index, columns=columns)
df_cluster_results[['inertia','cluster_cnt','cluster_time','crossval_mape','test_mae','test_mape','test_rmse','models_time']]=0.0
df_cluster_results['init']=cluster_init


clusters_cnt=7
print("---------------------------------------")
print("Start with %.0f clusters" % (clusters_cnt))
print("---------------------------------------")
# Define the kmeans model
kmeans_model=KMeans(init=cluster_init, n_clusters=clusters_cnt, n_init=10
             ,verbose=0,random_state=42) #,n_jobs=-1
t0 = time()
kmeans_model.fit(X_train_transformed)
print("K-Means Fit Done.")
cluster_train_pred=kmeans_model.predict(X_train_transformed)
#cluster_train_pred=clusters.labels_
print("K-Means Predict on Train Done.")
cluster_test_pred=kmeans_model.predict(X_test_transformed)
print("K-Means Predict on Test Done.")

# Cluster centers :
cluster_centers=kmeans_model.cluster_centers_
duration=time() - t0
print("------------ Clusters ------------------")
print("clusters_cnt=",clusters_cnt)
print("kmeans init :",cluster_init)
print("Sum of Inertia (Within - Intra Cluster): %0.2f" % (kmeans_model.inertia_))
print("---------------------------------------")
print("Done in : %0.3fs" % (duration))
print("---------------------------------------")
df_cluster_results.iloc[i]['cluster_cnt']=clusters_cnt
df_cluster_results.iloc[i]['inertia']=kmeans_model.inertia_
df_cluster_results.iloc[i]['cluster_time']=duration

# Set the Predicted Cluster
X_train['Cluster']=cluster_train_pred+1
X_test['Cluster']=cluster_test_pred+1

# ------------------------------------------------------------------------
# Carto
# ------------------------------------------------------------------------

min_long=-5
max_long=9.8
min_lat=41.2
max_lat=51.3
graphsize=11
delta_lat=+0.15 # use to move a little the background map to better fit scatter data.
img = plt.imread("maps_fonds/map_metropole.png")
extent = (min_long, max_long, min_lat+delta_lat, max_lat+delta_lat) # Use the boundaries for map background

fig, ax = plt.subplots(figsize=(graphsize,graphsize))
ax.set_xlim(min_long, max_long) # Define bundaries of axes in lat/long
ax.set_ylim(min_lat, max_lat)

im = ax.imshow(img, aspect='auto',extent=extent,zorder=5)

ax2=sns.scatterplot(data=X_test,x="geolong", y="geolat"
                    ,hue="Cluster"
                    ,ax=ax,zorder=10
                    ,legend="full",palette=sns.color_palette("hls", 7)
                    ,s=10
                    )

ax2.set_title("Répartition des données de Test sur les 7 Clusters")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
#ax2.legend(loc="best")
plt.show()

for i in range(7):
    fig, ax = plt.subplots(figsize=(graphsize,graphsize))
    ax.set_xlim(min_long, max_long) # Define bundaries of axes in lat/long
    ax.set_ylim(min_lat, max_lat)
    
    im = ax.imshow(img, aspect='auto',extent=extent,zorder=5)
    
    ax2=sns.scatterplot(data=X_test[X_test['Cluster']==i+1],x="geolong", y="geolat"
                        ,hue="Cluster"
                        ,ax=ax,zorder=10
                        ,legend="full",palette=sns.color_palette("hls", 1)
                        ,s=30
                        )
    
    ax2.set_title("Répartition des données de Test sur le Cluster %i" %(i+1))
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    #ax2.legend(loc="best")
    plt.show()


