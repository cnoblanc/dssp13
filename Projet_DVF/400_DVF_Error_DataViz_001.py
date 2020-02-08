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

# Split Train / Test
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split
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
folds_num=5

reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=50,num_leaves=40
                  ,learning_rate=0.06,n_estimators=2000)
model = make_pipeline(preprocessing,reg)

# ------------------------------------------------------------------------
# compute Cross-validation scores to check overfitting on train set
# ------------------------------------------------------------------------
t0 = time()
reg=LGBMRegressor(random_state=42,n_jobs=-1,silent=True
                  ,max_depth=50,num_leaves=40
                  ,learning_rate=0.06,n_estimators=2000)

model = make_pipeline(preprocessing,reg)

cross_val_pred=cross_val_predict(model, X_train, y_train
                        #,scoring="neg_mean_absolute_error"
                        ,cv=folds_num,n_jobs=-1,verbose=20)
print("CrossValidation predictions Done.")
print("Done  in : %0.3fs" % (time() - t0))

# Error Measures
mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_train, y_pred=cross_val_pred)
print("------------ Scoring ------------------")
print("CrossVal MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
print("CrossVal MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
print("CrossVal RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t0))
print("---------------------------------------")

f, ax0 = plt.subplots(1, 1, sharey=True)
ax0.scatter(y_train, cross_val_pred,s=0.5)
ax0.set_ylabel('Predicted Price')
ax0.set_xlabel('Real Price')
ax0.set_title('%s, MAE=%.2f, RMSE=%.2f' % (model_name,mae,rmse))


# ------------------------------------------------------------------------
# Prepare error data set
# ------------------------------------------------------------------------
df_errors=X_train.copy()
df_errors['realprice']=y_train
df_errors['predictedprice']=cross_val_pred
df_errors['priceerror']=(df_errors['predictedprice']-df_errors['realprice']).abs()

# Density of errors :
sns.distplot(df_errors['priceerror'],kde=False, rug=True)
plt.suptitle("France : Répartition erreurs de prédiction", fontsize=12,y=0.95) 
plt.show()

# ------------------------------------------------------------------------
# Errors <= 100 000 euros
# ------------------------------------------------------------------------
# What are the errors <= 100.000 euros ?
error_threshold=10000
selected_errors=df_errors[df_errors['priceerror']<=error_threshold]
s_mae,s_mae_std,s_mape,s_mape_std,_,_,_,_= dvfdata.get_predict_errors(y=selected_errors['realprice'], y_pred=selected_errors['predictedprice'])
print("---------------------------------------")
print("ERRORS <=",error_threshold)
print("Count and percentage: %s (%0.2f %%)" % (selected_errors.shape[0],100*selected_errors.shape[0]/X_train.shape[0]))
print("Selected Error MAE: %0.2f (+/- %0.2f)" % (s_mae, s_mae_std * 2))
print("Selected Error MAPE: %0.2f (+/- %0.2f)" % (s_mape, s_mape_std * 2))
print("---------------------------------------")
# Density of errors :
sns.distplot(selected_errors['priceerror'],kde=False, rug=False)
plt.suptitle("France : Répartition erreurs de prédiction (<=%s)"%(error_threshold), fontsize=12,y=0.95) 
plt.show()

# ------------------------------------------------------------------------
# Errors > 100 000 euros
# ------------------------------------------------------------------------
# What are the errors <= 100.000 euros ?
error_threshold=100000
selected_errors=df_errors[df_errors['priceerror']>error_threshold]
s_mae,s_mae_std,s_mape,s_mape_std,_,_,_,_= dvfdata.get_predict_errors(y=selected_errors['realprice'], y_pred=selected_errors['predictedprice'])
print("---------------------------------------")
print("ERRORS <=",error_threshold)
print("Count and percentage: %s (%0.2f %%)" % (selected_errors.shape[0],100*selected_errors.shape[0]/X_train.shape[0]))
print("Selected Error MAE: %0.2f (+/- %0.2f)" % (s_mae, s_mae_std * 2))
print("Selected Error MAPE: %0.2f (+/- %0.2f)" % (s_mape, s_mape_std * 2))
print("---------------------------------------")
# Density of errors :
sns.distplot(selected_errors['priceerror'],kde=False, rug=False)
plt.suptitle("France : Répartition erreurs de prédiction (>%s)"%(error_threshold), fontsize=12,y=0.95) 
plt.show()

# ------------------------------------------------------------------------
# DataViz Générale sur erreurs
# ------------------------------------------------------------------------
# Erreur vs prix réel
f, ax0 = plt.subplots(1, 1, sharey=True,figsize=(10, 5))
ax0.scatter(selected_errors['realprice'], selected_errors['priceerror'],s=0.5)
ax0.set_ylabel('Error')
ax0.set_xlabel('Real Price')
ax0.set_title('Error > 100 000 euros vs Real Price')

# Erreur vs surface bâtie
f, ax0 = plt.subplots(1, 1, sharey=True)
ax0.scatter(selected_errors['sbati'], selected_errors['priceerror'],s=0.5)
ax0.set_ylabel('Error')
ax0.set_xlabel('Surface batie')
ax0.set_title('Error > 100 000 euros vs surface batie')

# Erreur par surface de terrain
f, ax0 = plt.subplots(1, 1, sharey=True)
ax0.scatter(selected_errors['sterr'], selected_errors['priceerror'],s=0.5
            ,hue='Urbanité Ruralité')
ax0.set_ylabel('Error')
ax0.set_xlabel('Surface du terrain')
ax0.set_title('Error > 100 000 euros vs surface du terrain')

# Urbanité Ruralité vs error
sns.boxplot(x="priceerror", y="Urbanité Ruralité", data=selected_errors,
            whis="range", palette="vlag")


# Densité par géographie
# Select only Metropole
metropole_error=selected_errors[ ((selected_errors['geolong']<9.8) & (selected_errors['geolong']>-5))]
#sns.jointplot(x="geolong", y="geolat", data=metropole_error,s=0.5, kind="kde");
min_long=-5
max_long=9.8
min_lat=41.2
max_lat=51.3
graphsize=11
delta_lat=+0.15
base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF/"
img = plt.imread(base_path+"maps_fonds/map_metropole.png")
extent = (min_long, max_long, min_lat+delta_lat, max_lat+delta_lat) # Use the boundaries for map background

fig, ax = plt.subplots(figsize=(graphsize,graphsize))
ax.set_xlim(min_long, max_long) # Define bundaries of axes in lat/long
ax.set_ylim(min_lat, max_lat)
im = ax.imshow(img, aspect='auto',extent=extent,zorder=5)
ax2=metropole_error.plot.scatter(x="geolong",y="geolat",alpha=0.9,c="priceerror"
        ,s=0.1,cmap=plt.get_cmap("jet"),colorbar=True,norm=mpl.colors.Normalize()
        ,zorder=10
        ,ax=ax)
plt.show()

# BIAIS sur la quantité de transaction :
# Faire 9 graphiques de 156.000 transactions triés par l'erreur croissante.
min_long=-5
max_long=9.8
min_lat=41.2
max_lat=51.3
graphsize=11
delta_lat=+0.15
extent = (min_long, max_long, min_lat+delta_lat, max_lat+delta_lat) # Use the boundaries for map background

df_errors.sort_values(by=['priceerror'],inplace=True)
start=0
end=0
nb_of_grahes=4
total_rec=df_errors.shape[0]
for i in range(nb_of_grahes):
    end=(total_rec//nb_of_grahes)*(i+1)
    cut_df=df_errors.iloc[start:end]
        
    cut_metropole=cut_df[ ((cut_df['geolong']<9.8) & (cut_df['geolong']>-5))]
    error_min=np.min(cut_metropole["priceerror"])
    error_max=np.max(cut_metropole["priceerror"])
    
    # Show Map
    img = plt.imread(base_path+"maps_fonds/map_metropole.png")

    fig, ax = plt.subplots(figsize=(graphsize,graphsize))
    ax.set_xlim(min_long, max_long) # Define bundaries of axes in lat/long
    ax.set_ylim(min_lat, max_lat)
    ax.set_title("Error level. From %s to %s (error from %0.0f to %0.0f" % (start,end,error_min,error_max))
    im = ax.imshow(img, aspect='auto',extent=extent,zorder=5)
    ax2=cut_metropole.plot.scatter(x="geolong",y="geolat",alpha=0.9
            #,c="priceerror",cmap=plt.get_cmap("jet"),colorbar=True,norm=mpl.colors.Normalize()
            ,s=0.1,zorder=10,ax=ax)
    plt.show()

    start+=total_rec//nb_of_grahes


