from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # Store all original columns
        self.cat_columns = None  # To store categorical column names
        self.encoders = {}

    def fit(self, X, y=None):
        # Identify and store categorical columns during fitting
        self.cat_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit encoders only for categorical columns
        for col in self.cat_columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        # Convert input to DataFrame with all original columns
        X_transformed = pd.DataFrame(X, columns=self.columns)  
        
        # Apply encoding only to categorical features
        for col in self.cat_columns:
            if col in X_transformed.columns:
                mapping = dict(zip(self.encoders[col].classes_, range(len(self.encoders[col].classes_))))
                X_transformed[col] = X_transformed[col].map(mapping).fillna(-1).astype(int)

        # Return DataFrame with all original columns
        return X_transformed

#Remove features which have only a unique value
class FeatureSelectorLabelled(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.unique_columns_ = X.nunique()
        self.columns_to_drop_ = self.unique_columns_[self.unique_columns_ == 1].index
        return self

    def transform(self, X):
        X_transformed = X.drop(self.columns_to_drop_, axis=1)
        return X_transformed
    
    #ANOVA used to select the top k relevant features. k=15 was used
class FeatureSelector_ANOVA(BaseEstimator, TransformerMixin):
    def __init__(self, k=15):
        self.k = k
        self.selected_features_ = None

    def fit(self, X, y=None):
        selector = SelectKBest(score_func=f_classif, k=self.k)
        selector.fit(X.astype(float), y)
        self.selected_features_ = X.columns[selector.get_support()]
        return self

    def transform(self, X, y=None):
        return X[self.selected_features_]
    
    #Reduce multicollinearity by removing highly correlated features.    
class CorrelationReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.correlated_features = None

    def fit(self, X, y=None):
        corr_matrix = X.corr()
        self.correlated_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    self.correlated_features.add(colname)
        return self

    def transform(self, X):
        return X.drop(columns=self.correlated_features, axis=1)
    
    





