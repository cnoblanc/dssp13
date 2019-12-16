from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
#import lightgbm as lgb

from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        #self.reg = RandomForestRegressor(n_estimators=100, max_depth=50, max_features=150)
        #self.reg = LinearRegression()
        #self.reg = Lasso(alpha=0.1)
        #self.reg = BayesianRidge()
        #self.reg  =GradientBoostingRegressor(learning_rate=0.2,n_estimators=680,max_depth=20)
        #self.reg = GradientBoostingRegressor(learning_rate=0.1,n_estimators=620,max_depth=22)
        #self.reg = xgb.XGBRegressor(objective ='reg:linear',nthreads=-1, colsample_bytree = 0.939621, learning_rate = 0.144394, \
        #        max_depth = 10, alpha = 10, n_estimators = 199,subsample=0.936487)
        #self.reg = xgb.XGBRegressor(objective ='reg:linear',nthreads=-1, colsample_bytree = 0.964245, learning_rate = 0.122122, \
        #        max_depth = 16, alpha = 10, n_estimators = 120,subsample=0.898837)
        
        self.reg=xgb.XGBRegressor(objective ='reg:linear',nthreads=-1,num_boost_round=1,
             n_estimators = 696, max_depth = 13, learning_rate=0.0472126 ,colsample_bytree = 0.945585, subsample=0.804857)

        
        #self.reg = lgb.LGBMRegressor(
        #    max_depth=15,
        #    learning_rate=0.08,
        #    n_estimators=8000,
        #    num_leaves=23,
        #    min_data_in_leaf=7,
        #)
        
    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
