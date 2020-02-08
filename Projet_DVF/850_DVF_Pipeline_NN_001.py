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

dep_selection="92"
model_name="Keras-TensorFlow"
#dep_selection="77"
df=dvfdata.loadDVF_Maisons(departement=dep_selection,refresh_force=False
                           ,add_commune=True,filterColsInsee="Permutation")
df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
#df_prepared=df_prepared.sample(n=700000, random_state=42)

df_prepared=df_prepared.drop(columns=['departement'])

# Split Train / Test
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import learning_curve

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


# Define the Model
#from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout

print(tf.__version__)

#folds_num=5


# evaluate model
#reg = KerasRegressor(build_fn=baseline_model, verbose=0
#                     , epochs=15, batch_size=5)
#model = make_pipeline(preprocessing,reg)

# ------------------------------------------------------------------------
# compute Test Scores
# ------------------------------------------------------------------------
#reg = KerasRegressor(build_fn=baseline_model, verbose=1
#                     , epochs=30, batch_size=5)

#nn_model.add(Dropout(0.2))

nn_model = Sequential()
nn_model.add(Dense(50, input_dim=21, kernel_initializer='normal'
                   #,kernel_regularizer=keras.regularizers.l1(0.001)
                   , activation='elu'))
#nn_model.add(Dropout(0.2))
nn_model.add(Dense(80, kernel_initializer='normal'
                   #,kernel_regularizer=keras.regularizers.l1(0.001)
                   , activation='elu'))
nn_model.add(Dense(40, kernel_initializer='normal'
                   #,kernel_regularizer=keras.regularizers.l1(0.001)
                   , activation='elu'))
#nn_model.add(Dense(100, kernel_initializer='normal', activation='linear'))
nn_model.add(Dense(1,activation='linear'))
# Compile model
# model.compile(loss='mean_absolute_error', optimizer='adam')
nn_model.compile(loss='mean_absolute_error', optimizer='rmsprop'
              ,metrics=['mae','mape'])

#model = make_pipeline(preprocessing,reg)
# Apply the Model on full Train dataset

BATCH_SIZE=X_train_transformed.shape[0]//100
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss' #,mode='min'
                                           , patience=50,min_delta=50)
#epoch_dots=tfdocs.modeling.EpochDots()
t1_fit_start=time()
history=nn_model.fit(X_train_transformed, y_train, verbose=1
                ,validation_split=0.2
                #, validation_data=(X_test_transformed, y_test)
                ,callbacks=[early_stop]
                , epochs=1000, batch_size=BATCH_SIZE)
print("Fit on Train. Done")
t1_fit_end=time()
print("---------------------------------------")
print("Done Fit in : %0.3fs" % (t1_fit_end - t1_fit_start))
print("---------------------------------------")

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
last_epoch=hist.tail(1)
print(last_epoch)

# Predict on Test dataset
t0_predict = time()
y_test_predict=nn_model.predict(X_test_transformed).flatten()
t0_predict_end = time()
print("Predict on Test. Done")

# Prediction Score
predict_score_mae=mean_absolute_error(y_test, y_test_predict)
predict_score_rmse=math.sqrt(mean_squared_error(y_test, y_test_predict))
#predict_score_msle=mean_squared_log_error(y_test, y_test_predict_non_negative)

mae,mae_std,mape, mape_std,mse,mse_std,rmse,rmse_std = dvfdata.get_predict_errors(y=y_test, y_pred=y_test_predict)
print("------------ Scoring ------------------")
print("Cross-Validation Accuracy: %0.2f" % (last_epoch['val_mae']))
print("Price diff error MAE: %0.2f (+/- %0.2f)" % (mae, mae_std * 2))
print("Percent of Price error MAPE: %0.2f (+/- %0.2f)" % (mape, mape_std * 2))
print("Price error RMSE: %0.2f (+/- %0.2f)" % (rmse, rmse_std * 2))
print("---------------------------------------")
print("Done All in : %0.3fs" % (time() - t1_fit_start))
#print("Done CrossVal in : %0.3fs" % (t1_fit_start - t0))
print("Done Fit in : %0.3fs" % (t1_fit_end - t1_fit_start))
print("Done Predict in : %0.3fs" % (t0_predict_end - t0_predict))
print("---------------------------------------")
nn_model.summary()

tf.keras.utils.plot_model(nn_model,show_shapes=True,show_layer_names=False
        ,expand_nested=True,rankdir='LR',dpi=96*1.4)
    
#print(history.history.keys())

#Graphs 
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "mae")
plt.title('%s, MAE=%.2f' % (model_name,mae))
plt.ylabel('MAE (euros)')
plt.show()

plotter.plot({'Basic': history}, metric = "mape")
plt.title('%s, MAPE=%.2f %%' % (model_name,mape))
plt.ylabel('MAPE (%)')
plt.show()

# Prediction vs True price
f, ax0 = plt.subplots(1, 1, sharey=True)
maxprice=1000000
ax0.scatter(y_test, y_test_predict,s=1)
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('%s, MAE=%.2f, RMSE=%.2f' % (model_name,predict_score_mae,predict_score_rmse))
ax0.plot([0, maxprice], [0, maxprice], 'k-', color = 'lightblue')
ax0.set_xlim([0, maxprice])
ax0.set_ylim([0, maxprice])


