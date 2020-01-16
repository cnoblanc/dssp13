import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

class ChristopheGeoRegressor():
    def __init__(self,col_latitude="",col_longitude="",dist_weight=0.5,k=20):     
        self.col_latitude = col_latitude
        self.col_longitude = col_longitude
        self.k=k # How many Similar Neighbors we keep
        self.dist_weight=dist_weight # Weight 0<=w<=1 of the distance in the score to select similar neighbors

    def fit(self,X, y): # Train phase
        # From X, get columns lists
        print("fit : start")
        self.columns = np.array(X.columns)
        self.columns_geo = self.columns[(self.columns == self.col_latitude) | (self.columns == self.col_longitude)]
        self.columns_notgeo=X.columns.drop(self.columns_geo) 
        print("fit:copy X_train")
        self.X_train=X.copy()
        print("fit:copy y_train")
        self.y_train=y.copy()
        print("fit : Done")
    
    def predict(self,X):
        print("start predict:")
        print("- compute Similarities matrix")
        self.Similarities=cosine_similarity(X=X[self.columns_notgeo]
                            , Y=self.X_train[self.columns_notgeo], dense_output=True)
        print("- compute Distances matrix")
        self.Distances=pairwise_distances(X=X[self.columns_geo], Y=self.X_train[self.columns_geo]
                            , metric='euclidean', n_jobs=-1)
        print("- compute Normalized Distance (divide by the maximum value)")
        #max_distance=np.nanmax(self.Distances)
        #np.divide(self.Distances,max_distance)
        self.Distance_Norm=preprocessing.normalize(self.Distances, norm='l2')
        
        print("- compute Scores matrix")
        #self.Scores=np.add(  np.multiply(self.Similarities,1-self.dist_weight) \
        #                   , np.multiply(self.Distances, self.dist_weight) )
        
        #self.Scores=np.divide(self.Similarities,np.add(self.Distance_Norm,1))
        self.Scores=np.subtract(  np.multiply(self.Similarities,1-self.dist_weight) \
                                , np.multiply(self.Distances, self.dist_weight) )
        
        print("- select best scores")
        test_records=X.shape[0]
        y_hat = np.zeros(test_records)
        selected_indexes=np.zeros((test_records,self.k))
        
        n_per_100=test_records//100 # division entière
        
        for i in range(test_records):
            if i % n_per_100 ==0:
                print(i//100),
            df_scores=pd.DataFrame(self.Scores[i])
            df_scores.columns = ['score']
            selected_scores=df_scores.sort_values(by=['score'], ascending=False)[:self.k]
            selected_metrics=self.y_train.iloc[selected_scores.index]
            
            selected_indexes[i]=selected_scores.index
            y_hat[i]=selected_metrics.median()
            
        self.selected_indexes_=selected_indexes
        print("Predict:Done")
        
        return y_hat