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


import dvfdata

dep_selection="All"
#dep_selection="77"
df=dvfdata.loadDVF_Maisons(departement=dep_selection,refresh_force=False
                           ,add_commune=True,filterColsInsee="Permutation")
df_prepared=dvfdata.prepare_df(df,remove_categories=False)
# Keep only random part of all records.
#df_prepared=df_prepared.sample(n=700000, random_state=42)

#df_prepared = df_prepared.drop(columns=['departement','n_days','quarter','department_city_dist'])
#columns = df_prepared.columns


df_prepared.rename(columns={'Taux Evasion Client': 'Taux_Evasion_Client'
                   , 'Nb Création Commerces': 'Nb_Creation_Commerces'
                   ,'Reg Moyenne Salaires Prof Intermédiaire Horaires':'Moyenne_Salaires_Prof_Intermediaire_Horaires'
                   ,'Urbanité Ruralité':'Urbanite_Ruralite'
                   ,'Nb Ménages':'Nb_Menages'
                   ,'Reg Moyenne Salaires Cadre Horaires':'Moyenne_Salaires_Cadre_Horaires'
                   ,'Taux de Hotel':'Taux_de_Hotel'
                   ,'Taux de Mineurs':'Taux_de_Mineurs'
                   ,'Taux de dentistes Libéraux BV':'Taux_de_dentistes_Liberaux'
                   ,'Nb Création Enteprises':'Nb_Creation_Enteprises'
                   ,'Dep Moyenne Salaires Horaires':'Moyenne_Salaires_Horaires'
                   ,'Taux de Occupants Résidence Principale':'Taux_de_Occupants_Residence_Principale'
                   ,'Taux de Homme':'Taux_de_Homme'}, inplace=True)

# Exclude the Target predicted variable from to create the X and y
X_df = df_prepared.drop(columns='valeurfonc')
y = df_prepared['valeurfonc']
#y = np.log1p(df_prepared['valeurfonc'])
columns = X_df.columns

# Get list of columns by type
cat_cols= X_df.select_dtypes([np.object]).columns
num_cols = X_df.select_dtypes([np.number]).columns
#dvfdata.print_cols_infos(X_df)

# Split Train / Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df, y, random_state=42)

# Machine Learning Pipeline 
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler


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

# ------------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------------

# Preprocessing
model = make_pipeline(preprocessing)
# Transform the Train & Test dataset
X_train_transformed=model.fit_transform(X_train)
X_test_transformed=model.transform(X_test)


X_train_transformed_df=pd.DataFrame(X_train_transformed,columns=columns)
X_train_transformed_df['valeurfonc']=np.round(y_train).astype(int)

X_test_transformed_df=pd.DataFrame(X_test_transformed,columns=columns)
X_test_transformed_df['valeurfonc']=np.round(y_test).astype(int)

X_df_transformed=X_train_transformed_df.append(pd.DataFrame(X_test_transformed_df))
#columns = X_df_transformed.columns

X_df_transformed.to_parquet("data_parquet/DVF_X_df_"+dep_selection+".parquet", engine='fastparquet',compression='GZIP')

print(columns)
