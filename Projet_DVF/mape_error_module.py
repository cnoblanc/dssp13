#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:21:42 2020

@author: christophenoblanc
"""

import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, **kwargs):
    """Mean absolute percentage error regression loss
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    
    Returns
    -------
    loss : float 
        MAPE output is non-negative floating point. The best value is 0.0.
        
    Examples
    --------
    >>> from mape_error_module import mean_absolute_percentage_error
    >>> from sklearn.metrics import make_scorer
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.32738...
    """
    
    abs_diff = 100*np.abs( (np.array(y_true) - np.array(y_pred)) / np.array(y_true) )
#/ np.array(y_true)
    return np.mean(abs_diff)