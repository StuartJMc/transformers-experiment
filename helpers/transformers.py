from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import BinningProcess

import scipy
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.special import logit
from scipy.interpolate import UnivariateSpline

import pandas as pd
import numpy as np


class OptbinningScikit(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        try: 
            cat_cols=X.select_dtypes(include='category').columns.to_list()
        except:
            pass

        try:
            string_cols=X.select_dtypes(include='object').columns.to_list()
        except:
            pass

        cat_cols=cat_cols+string_cols
        self.features=X.columns.to_list()

        self.opt_transformer=BinningProcess(variable_names=self.features,categorical_variables =cat_cols).fit(X,y)

        return self


    def transform(self, X, y=None):
        # Perform arbitary transformation

        X_transform=self.opt_transformer.transform(X)
        return X_transform

    def get_feature_names_out(self,names):
        """"
            dummy function to work with scikitlearn api
        """
        return self.features





class LowessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,frac=1./3,use_logit=False,fit_spline=False):
        self.frac=frac
        self.use_logit=use_logit
        self.fit_spline=fit_spline

        self.col_smoothers={}


    def fit(self, X, y=None):

        self.features=X.columns.to_list()

        for col in self.features:
            
            self.col_smoothers[col]=self.fit_lowess_curve(X[col],y)

        return self


    def transform(self, X, y=None):

        X_copy=X.copy()

        for col in self.features:
            
            X_copy[col]=self.col_smoothers[col](X_copy[col])

        return X_copy

    def get_feature_names_out(self,names):
        """"
            dummy function to work with scikitlearn api
        """
        return self.features


    def fit_lowess_curve(self,x,y):
    
        lowess = sm.nonparametric.lowess
        z = lowess(y, x, frac= self.frac,it=0)
        
        z_unique=np.unique(z, axis=0)
    
        if self.use_logit==True:
            z_unique[:,1]=logit(z_unique[:,1])
        
        z_unique=z_unique[~np.isnan(z_unique).any(axis=1), :]
    
        if self.fit_spline==True:
            return UnivariateSpline(z_unique[:,0], z_unique[:,1])
        else:
            return interp1d(z_unique[:,0], z_unique[:,1], kind='quadratic',fill_value='extrapolate')