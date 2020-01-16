#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:04:50 2020

@author: christophenoblanc
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF"
TreeFile="data_parquet/FeatureImportance_All_DecisionTree.parquet"
ForestFile="data_parquet/FeatureImportance_All_RandomForest.parquet"

# Read Pre-Stored Feature Importance DF
print("Read TreeFile")
tree_fi_df=pd.read_parquet(TreeFile, engine='fastparquet')
print("Read ForestFile")
forest_fi_df=pd.read_parquet(ForestFile, engine='fastparquet')

# Calculate mean of Scores
fi_df = pd.merge(tree_fi_df, forest_fi_df, how='left', on=['feature'])
fi_df.columns = ['feature','score_tree','score_forest']
fi_df['score']=(fi_df['score_tree']+fi_df['score_forest'])/2
fi_df=fi_df.sort_values(by=['score'], ascending=False)

# Graph the best Features
plt.figure(figsize=(10, 3))
top_x=20
x = np.arange(top_x)
plt.bar(x, fi_df['score'][:top_x])
plt.xticks(x, fi_df['feature'][:top_x], rotation=90, fontsize=10);
