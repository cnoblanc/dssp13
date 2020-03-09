#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:01:11 2019

@author: christophenoblanc
"""
import math
import os
from time import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

import dvfdata

dep_selection="All"
model_name="Keras-TensorFlow"
root_log_dir=os.path.join(os.curdir,"keras_logs_board")
#dep_selection="77"
df=dvfdata.loadDVF_Maisons(departement=dep_selection,refresh_force=False
                           ,add_commune=True,filterColsInsee="Permutation")
df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
#df_prepared=df_prepared.sample(n=700000, random_state=42)

df_prepared=df_prepared.drop(columns=['departement'])

# Split Train / Test
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import learning_curve

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
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Get the list of all possible categories from X_df
categories = [X_df[column].unique() for column in X_df[cat_cols]]
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

# ------------------------------------------------------------------------
# Neuron Network
# ------------------------------------------------------------------------
# Define the Model
#from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout,InputLayer,BatchNormalization
from keras.callbacks import TensorBoard

print(tf.__version__)
#folds_num=5
def get_run_logdir():
    import time
    run_id=time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)

def build_nn_model(n_hidden=1,n_neurons=30,learning_rate=3e-3,input_shape=[21]):
    nn_model = Sequential()
    
    nn_model.add(InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        #nn_model.add(BatchNormalization())
        #nn_model.add(Dropout(0.2))
        nn_model.add(Dense(n_neurons, kernel_initializer='lecun_normal'
                       #,kernel_regularizer=keras.regularizers.l1(0.001)
                       , activation='selu'))
        
    #nn_model.add(BatchNormalization())
    #nn_model.add(Dropout(0.2))
    #nn_model.add(Dense(100, kernel_initializer='normal', activation='linear'))
    nn_model.add(Dense(n_neurons//2, kernel_initializer='lecun_normal'
                       #,kernel_regularizer=keras.regularizers.l1(0.001)
                       , activation='selu'))    
        
    nn_model.add(Dense(1,activation='linear'))
    #optimizer=keras.optimizers.Adadelta(lr=learning_rate)
    #optimizer=keras.optimizers.Adadelta()
    
    # Compile model
    # model.compile(loss='mean_absolute_error', optimizer='adam')
    nn_model.compile(loss='mean_absolute_percentage_error', optimizer='adam'
                  ,metrics=['mae','mape'])
    return nn_model

# TensorBoard
run_logdir=get_run_logdir()
tensor_board_cb=TensorBoard(run_logdir)


BATCH_SIZE=X_train_transformed.shape[0]//100
#BATCH_SIZE=32
early_stop = keras.callbacks.EarlyStopping(monitor='loss' #,mode='min'
                                           , patience=15,min_delta=1)

keras_reg=keras.wrappers.scikit_learn.KerasRegressor(build_nn_model)

# Run a first Estimation.
#nn_model=build_nn_model(n_hidden=1,n_neurons=50,learning_rate=3e-3,input_shape=[21])

#history=keras_reg.fit(X_train_transformed,y_train, verbose=1
#                , validation_data=(X_test_transformed, y_test)
#                , callbacks=[early_stop,tensor_board_cb]
#                , epochs=250, batch_size=BATCH_SIZE)

#score_test=keras_reg.score(X_test_transformed, y_test)
#y_test_predict=keras_reg.predict(X_test_transformed)

from mape_error_module import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# ------------------------------------------------------------------------
# Search hyper-parameters 
# ------------------------------------------------------------------------
# Search hyper-parameters 
#print(model.get_params())
#from scipy.stats import reciprocal
tuned_parameters={ 'n_hidden':np.arange(3,30)
                  ,'n_neurons':np.arange(10,400)
                  #,'learning_rate':reciprocal(3e-4,3e-2)
                  }

t0 = time()
#model_grid=RandomizedSearchCV(estimator=keras_reg,param_distributions=tuned_parameters
#                          ,scoring="neg_mean_absolute_error",refit=True
#                          ,cv=2,verbose=10,n_iter=25)
scoring = {'mae': 'neg_mean_absolute_error', 'mape_score': mape_scorer}
model_grid=RandomizedSearchCV(estimator=keras_reg,param_distributions=tuned_parameters
                          ,scoring=scoring,refit=mape_scorer
                          ,cv=5,verbose=20,n_iter=30)

model_grid.fit(X_train_transformed, y_train, verbose=1
                , validation_data=(X_test_transformed, y_test)
                #, callbacks=[early_stop,tensor_board_cb]
                , callbacks=[early_stop]
                , epochs=250, batch_size=BATCH_SIZE)

print("---------------------------------------")
print("Best parameters set found on train set:")
print(model_grid.best_params_)
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t0))
print("---------------------------------------")


# Save results in a DataFrame
df_gridcv = pd.DataFrame(model_grid.cv_results_)
gridcv_columns_to_keep = [
    #'param_max_depth',
    'param_n_hidden',
    #'param_learning_rate',
    'param_n_neurons',
    'mean_test_score','std_test_score',
    'mean_fit_time','mean_score_time'
]
df_gridcv['mape_score']=df_gridcv['mean_test_map_score']*100
df_gridcv['param_n_hidden']=df_gridcv['param_n_hidden'].astype(int)
df_gridcv['param_n_neurons']=df_gridcv['param_n_neurons'].astype(int)
#df_gridcv = df_gridcv[gridcv_columns_to_keep]
df_gridcv = df_gridcv.sort_values(by='mape_score', ascending=False)
df_gridcv.to_parquet("data_parquet/GridSearch_"+dep_selection+"_"+model_name+"_div100_mape.parquet", engine='fastparquet',compression='GZIP')


# DataViz des HyperParameters
f, ax0 = plt.subplots(1, 1, sharey=True)

ax0=df_gridcv.plot.scatter(x="param_n_hidden",y="param_n_neurons"
            #,alpha=0.9
            ,c="mape_score",cmap=plt.get_cmap("jet"),colorbar=True,norm=mpl.colors.Normalize()
            ,s=5)

ax0.set_ylabel('Number of Neurons per layer')
ax0.set_xlabel('Number of Hidden layers')
ax0.set_title('Model MAPE score per hyper-parameter')
plt.show()

# ------------------------------------------------------------------------
# compute Test Scores
# ------------------------------------------------------------------------
# Create the model

#best_reg = model_grid.best_estimator_.model
n_hidden=20
n_neurons=256
best_reg=build_nn_model(n_hidden=n_hidden,n_neurons=n_neurons,learning_rate=1e-3,input_shape=[21])

# Apply the Model on full Train dataset
#epoch_dots=tfdocs.modeling.EpochDots()
t1_fit_start=time()
history=best_reg.fit(X_train_transformed, y_train, verbose=1
                #,validation_split=0.2
                , validation_data=(X_test_transformed, y_test)
                ,callbacks=[early_stop]
                , epochs=500, batch_size=BATCH_SIZE)
print("Fit on Train. Done")
t1_fit_end=time()
print("---------------------------------------")
print("Done Fit in : %0.3fs" % (t1_fit_end - t1_fit_start))
print("---------------------------------------")

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
last_epoch=hist.tail(1)
print(last_epoch)

# Predict on Train dataset
t0_predict = time()
y_train_predict=best_reg.predict(X_train_transformed).flatten()
t0_predict_end = time()
print("Predict on Train. Done")

# Predict on Test dataset
t0_predict = time()
y_test_predict=best_reg.predict(X_test_transformed).flatten()
t0_predict_end = time()
print("Predict on Test. Done")


# Reciproque
#y_train_recip=np.expm1(y_train)
#y_train_predict_recip=np.expm1(y_train_predict)

#y_test_recip=np.expm1(y_test)
#y_test_predict_recip=np.expm1(y_test_predict)
#cross_val_scores_2=np.expm1(cross_val_scores)
y_train_recip=y_train
y_train_predict_recip=y_train_predict
y_test_recip=y_test
y_test_predict_recip=y_test_predict


# Prediction Score
predict_score_mae=mean_absolute_error(y_test_recip, y_test_predict_recip)
predict_score_rmse=math.sqrt(mean_squared_error(y_test_recip, y_test_predict_recip))
##predict_score_msle=mean_squared_log_error(y_test, y_test_predict_non_negative)

t_mae,t_mae_std,t_mape, t_mape_std,t_mse,t_mse_std,t_rmse,t_rmse_std = dvfdata.get_predict_errors(y=y_train_recip, y_pred=y_train_predict_recip)

mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test_recip, y_pred=y_test_predict_recip)
print("------------ Scoring ------------------")
print("Train Accuracy: %0.2f (from last epoch:%0.2f)" % (t_mape, last_epoch['mape']))
print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t1_fit_start))
#print("Done CrossVal in : %0.3fs" % (t1_fit_start - t0))
print("Done Fit in : %0.3fs" % (t1_fit_end - t1_fit_start))
print("Done Predict in : %0.3fs" % (t0_predict_end - t0_predict))
print("---------------------------------------")
best_reg.summary()

tf.keras.utils.plot_model(best_reg,show_shapes=True,show_layer_names=False
        ,expand_nested=True,rankdir='LR',dpi=96*1.4)
    
#print(history.history.keys())

#Graphs 
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Data': history}, metric = "mae")
plt.title('%s, MAE=%.2f' % (model_name,mae))
plt.ylabel('MAE (euros)')
plt.show()

plotter.plot({'Data': history}, metric = "mape")
plt.title('%s, MAPE=%.2f %%' % (model_name,mape))
plt.ylabel('MAPE (%)')
plt.show()

# Prediction vs True price
f, ax0 = plt.subplots(1, 1, sharey=True)
maxprice=1000000
ax0.scatter(y_test_recip, y_test_predict_recip,s=1)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('%s, MAPE=%.2f, MAE=%.2f' % (model_name,mape,mae))
ax0.plot([0, maxprice], [0, maxprice], 'k-', color = 'lightblue')
ax0.plot([0, maxprice], [0, 0], 'k-', color = 'lightblue')
ax0.set_xlim([0, maxprice])
ax0.set_ylim([0, maxprice])
ax0.set_ylim([-50000, maxprice])


