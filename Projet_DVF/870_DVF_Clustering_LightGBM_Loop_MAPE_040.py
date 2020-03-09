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


df=dvfdata.loadDVF_Maisons(departement="All",refresh_force=False
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
y = df_prepared['valeurfonc']
#y = np.log1p(df_prepared['valeurfonc'])

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
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout,InputLayer,BatchNormalization

from mape_error_module import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

def build_nn_model(n_hidden=1,n_neurons=30,learning_rate=3e-3,input_shape=[21]):
    nn_model = Sequential()
    
    nn_model.add(InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        #nn_model.add(BatchNormalization())
        nn_model.add(Dense(n_neurons, kernel_initializer='normal'
                       #,kernel_regularizer=keras.regularizers.l1(0.001)
                       , activation='relu'))
        #nn_model.add(Dropout(0.2))

    #nn_model.add(BatchNormalization())
    #nn_model.add(Dense(100, kernel_initializer='normal', activation='linear'))
    nn_model.add(Dense(1,activation='linear'))
    #optimizer=keras.optimizers.Adadelta(lr=learning_rate)
    optimizer=keras.optimizers.Adadelta()
    
    # Compile model
    # model.compile(loss='mean_absolute_error', optimizer='adam')
    nn_model.compile(loss='mean_absolute_percentage_error', optimizer='adam'
                  ,metrics=['mae','mape'])
    return nn_model

def generateModelScores(X_train,y_train,X_test,y_test):
    train_len=X_train.shape[0]
    test_len=X_test.shape[0]
    folds_num=5
     

    BATCH_SIZE=train_len//100
    #BATCH_SIZE=32
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss' #,mode='min'
                                               , patience=10,min_delta=1)
    
    reg=keras.wrappers.scikit_learn.KerasRegressor(build_nn_model)
    
    tuned_parameters={ 'n_hidden':np.arange(3,30)
                  ,'n_neurons':np.arange(10,300)
                  #,'learning_rate':reciprocal(3e-4,3e-2)
                  }
    
    # ------------------------------------------------------------------------
    # Search hyper-parameters 
    # ------------------------------------------------------------------------
    # Search hyper-parameters 
    #print(model.get_params())
        
    if X_train.shape[0] > folds_num :
        t0 = time()
        scoring = {'mae': 'neg_mean_absolute_error', 'mape_score': mape_scorer}
        model_grid=RandomizedSearchCV(estimator=reg,param_distributions=tuned_parameters
                                  ,scoring=mape_scorer,refit=True
                                  ,cv=2,verbose=20,n_iter=1)
        
        model_grid.fit(X_train, y_train, verbose=1
                        , validation_data=(X_test, y_test)
                        #, callbacks=[early_stop,tensor_board_cb]
                        , callbacks=[early_stop]
                        , epochs=250, batch_size=BATCH_SIZE)
        
        optimised_reg = model_grid.best_estimator_
        print("---------------------------------------")
        print("Best parameters set found on train set:")
        print(model_grid.best_params_)
        print("---------------------------------------")
        print("Done All in : %0.3fs" % (time() - t0))
        print("---------------------------------------")
    else:
        optimised_reg=reg
        print("---------------------------------------")
        print("GridSearchCV not done as Train size = ",X_train.shape[0])
        print("---------------------------------------")

    # ------------------------------------------------------------------------
    # compute Cross-validation scores to check overfitting on train set
    # ------------------------------------------------------------------------
    t0 = time()
    
    if X_train.shape[0] > folds_num :
        cross_val_scores=cross_val_score(optimised_reg, X_train, y_train
                                ,scoring=mape_scorer
                                ,cv=folds_num,verbose=10)
        print("CrossValidation score Done.")
    else:
        y_train_pred=optimised_reg.fit(X_train, y_train).predict(X_train)
        cross_val_scores=np.abs(y_train - y_train_pred) / np.abs(y_train)
        #mean_absolute_error(y_train, y_train_pred, multioutput='raw_values')
        print("Train score Done.")


    # ------------------------------------------------------------------------
    # compute Test Scores
    # ------------------------------------------------------------------------
    # Apply the Model on full Train dataset
    t1_fit_start=time()
    optimised_reg.fit(X_train, y_train)
    print("Fit on Train. Done")
    
    # Predict on Train dataset
    t0_predict = time()
    #y_train_predict=optimised_reg.predict(X_train)
    
    y_train_predict = cross_val_predict(optimised_reg, X_train, y_train
                                , cv=folds_num,verbose=10)
    t0_predict_end = time()
    print("Predict on Train. Done")

    
    # Predict on Test dataset
    t0_predict = time()
    y_test_predict=optimised_reg.predict(X_test)
    t0_predict_end = time()
    print("Predict on Test. Done")
    
    # Prediction Score : transformation réciproque
    #y_test_recip=y_test
    #y_test_predict_recip=y_test_predict
    
    y_train_recip=np.expm1(y_train)
    y_train_predict_recip=np.expm1(y_train_predict)

    y_test_recip=np.expm1(y_test)
    y_test_predict_recip=np.expm1(y_test_predict)
    cross_val_scores_recip=np.expm1(cross_val_scores)
    
    #predict_score_mae=mean_absolute_error(y_test_recip, y_test_predict_recip)
    #predict_score_rmse=math.sqrt(mean_squared_error(y_test_recip, y_test_predict_recip))
    #predict_score_msle=mean_squared_log_error(y_test, y_test_predict_non_negative)
    
    t_mae,t_mae_std,t_mape, t_mape_std,t_mse,t_mse_std,t_rmse,t_rmse_std = dvfdata.get_predict_errors(y=y_train_recip, y_pred=y_train_predict_recip)

    mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test_recip, y_pred=y_test_predict_recip)
    total_time=time() - t0
    cross_val_mae=-cross_val_scores.mean()
    
    print("------------ Scoring ------------------")
    print("Train Accuracy: %0.2f" % (t_mape))
    #print("Cross-Validation Accuracy: %0.2f" % (cross_val_mae))
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
    
    return t_mape,mae, mape,rmse,total_time,train_len,test_len


from sklearn.cluster import KMeans

cluster_init="k-means++"
max_clusters=1

columns = ['cluster_cnt','init','inertia','cluster_time','crossval_mape','test_mae','test_mape','test_rmse','models_time']
index=range(max_clusters)
df_cluster_results = pd.DataFrame(index=index, columns=columns)
df_cluster_results[['inertia','cluster_cnt','cluster_time','crossval_mape','test_mae','test_mape','test_rmse','models_time']]=0.0
df_cluster_results['init']=cluster_init

for i in range(1):
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
    
    
    # Generate a specific model for each of the clusters
    t0 = time()
    
    # Prepare structure to store all individual results
    columns = ['cluster','val_mape','mae', 'mape','rmse','total_time','train_len','test_len']
    index=range(clusters_cnt)
    df_results = pd.DataFrame(index=index, columns=columns)
    df_results[['cluster','val_mape','mae', 'mape','rmse','total_time','train_len','test_len']]=0.0
    
    for clust in range(1): #range(clusters_cnt)
        print("---------------------------------------")
        print("Start for cluster=",clust)
        print("---------------------------------------")
        X_train_oneCluster=X_train_transformed[cluster_train_pred==clust]
        y_train_oneCluster=y_train[cluster_train_pred==clust]
        X_test_oneCluster=X_test_transformed[cluster_test_pred==clust]
        y_test_oneCluster=y_test[cluster_test_pred==clust]
        
        val_mape,mae, mape,rmse,total_time,train_len,test_len  \
                =generateModelScores(X_train_oneCluster,y_train_oneCluster,X_test_oneCluster,y_test_oneCluster)
                
        df_results.iloc[clust]['cluster']=clust
        df_results.iloc[clust]['val_mape']=val_mape
        df_results.iloc[clust]['mae']=mae
        df_results.iloc[clust]['mape']=mape
        df_results.iloc[clust]['rmse']=rmse
        df_results.iloc[clust]['total_time']=total_time
        df_results.iloc[clust]['train_len']=train_len
        df_results.iloc[clust]['test_len']=test_len
        
    # Calculate the Mean of the cross-val MAE, the Test MAE & MAPE
    df_results['weighted_val_mape']=df_results['val_mape']*df_results['train_len']
    df_results['weighted_mae']=df_results['mae']*df_results['test_len']
    df_results['weighted_mape']=df_results['mape']*df_results['test_len']
    df_results['weighted_rmse']=df_results['rmse']*df_results['test_len']
    
    df_cluster_results.iloc[i]['crossval_mape']=df_results['weighted_val_mape'].sum()/df_results['train_len'].sum()
    df_cluster_results.iloc[i]['test_mae']=df_results['weighted_mae'].sum()/df_results['test_len'].sum()
    df_cluster_results.iloc[i]['test_mape']=df_results['weighted_mape'].sum()/df_results['test_len'].sum()
    df_cluster_results.iloc[i]['test_rmse']=df_results['weighted_rmse'].sum()/df_results['test_len'].sum()
    
    print("------------ Final Scoring on full dataset------------------")
    print("Cross-Validation Accuracy: %0.2f" % (df_cluster_results.iloc[i]['crossval_mape']))
    print("Price diff error MAE: %0.2f" % (df_cluster_results.iloc[i]['test_mae']))
    print("Percent of Price error MAPE: %0.2f" % (df_cluster_results.iloc[i]['test_mape']))
    print("Price error RMSE: %0.2f" % (df_cluster_results.iloc[i]['test_rmse']))
    print("---------------------------------------")
    print("Done All in : %0.3fs" % (df_results['total_time'].sum()))
    
    duration_model=time() - t0
    df_cluster_results.iloc[i]['models_time']=duration_model


# Add Total_Time
df_cluster_results['total_time']=df_cluster_results['cluster_time']+df_cluster_results['models_time']

# Remove un-finished loops
df_cluster_results=df_cluster_results[df_cluster_results['models_time']!=0]
df_cluster_results.sort_values(by='cluster_cnt', ascending=True,inplace=True)

print("---------------------------------------")
print("Save to local parquet file")
parquet_fileName=base_path+"keras_models_saved/df_results_Clusters_LightGBM_GridSearch_040"
df_cluster_results.to_parquet(parquet_fileName, engine='fastparquet',compression='GZIP') 

# Load Parquet
parquet_fileName=base_path+"keras_models_saved/df_results_Clusters_LightGBM_GridSearch_040"
df_cluster_results=pd.read_parquet(parquet_fileName, engine='fastparquet')

print("Total Time is %f s" % (df_cluster_results['total_time'].sum()))

# Show the prediction performance by Cluster counts
fig, ax1 = plt.subplots(constrained_layout=True,figsize=(10, 5))
ax1.set_title(model_name+' - Score MAE et MAPE par # de clusters')
ax1.set_xticks(np.arange(1, df_cluster_results.shape[0]+1, step=1))
ax1.plot(df_cluster_results['cluster_cnt'], df_cluster_results['test_mae'], 'o-')
ax1.set_xlabel("Nombre de Clusters")
ax1.set_ylabel("Test score MAE")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:green'
ax2.set_ylabel('Test score MAPE (%)', color=color)  # we already handled the x-label with ax1
ax2.set_ylim([60, 80])
ax2.plot(df_cluster_results['cluster_cnt'], df_cluster_results['test_mape'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
