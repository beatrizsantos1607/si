import sys
import os
import numpy as np
import pandas as pd
sys.path.append("/Users/utilizador/Documents/GitHub/si/src")

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.io.csv_file import read_csv



class SelectPercentile(Transformer):
    
    """
    Select a certain percentage of the features taking into account the F-score value.
    this is first we see the f-score of each feature and sorted that.
    after we choose a percentil that representes x % of this f-values sorted
    so we keep the features that indices have the f-value <= to the percentile
    
    Parameters
    -----------
    score_func:callable 
        taking the dataset and return a pair os array (F and p value)- allow analize the variance 
    percentile: int, deafult 50
        number that represents a percentage of the data/features to select 

    estimated parameters(given by the score_func)
    ---------------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    
    def __init__(self, score_func: callable= f_classification, percentile:int =50):
        self.score_func = score_func
        self.percentile = percentile
        self.F= None
        self.p= None
    
        if self.percentile > 100 or self.percentile < 0:
            raise ValueError("the value of percentile must be between 0 and 100")
    
    def _fit(self, dataset: Dataset):
        """
        It fits SelectPercentile to compute the F scores and p-values.
        
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        
        self.F, self.p = self.score_func(dataset)
        
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        
        """
        It selects the features according to the percentile.
        
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        
        Returns
        ----------
        dataset: Dataset
            A labeled dataset with the selected features.
            
        """
        
        percentile = np.percentile(self.F, self.percentile) # vai buscar o percentile do f values
        
        idxs = np.where(self.F > percentile)[0] # vai buscar os indices das features que tem f values

        features = np.array(dataset.features)[self.F > percentile] #vai buscar o nome das features
        
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=features, label=dataset.label)
        
        
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile to compute the F scores and p-values and then selects the features according to the percentile.
        
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        
        Returns
        ----------
        dataset: Dataset
            A labeled dataset with the selected features.
            
        """
        
        self.fit(dataset)
        return self.transform(dataset)