#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np
from scipy.stats import multiscale_graphcorr
from scipy._lib._util import MapWrapper
import warnings

######################################################################
# Scoring function

def k_sample_test(X, y,score_func="mgc"):
    """Nonparametric `K`-Sample Testing test statistic.
     
     A k_sample test tests equality in distribution among groups. Groups
    can be of different sizes, but must have the same dimensionality.
    There are not many non-parametric k_sample tests, and this version
    leverages the only k_sample multivariate independence test in scipy.
    
    Read more in the :ref:`User Guide <multivariate_feature_selection>`.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample vectors.
    y : ndarray of shape (n_samples,)
        The target vector.
    score_func : string that refers to a k_sample multivariate independence test from scipy
                The default and only existing test is multiscale graph correlation.
    
    Returns
    -------
    stat : float that refers to the computed k_sample test statistic
    
    Notes
    -----
    
    1) The k_sample testing problem can be thought of as a generalization of
    the two sample testing problem. 
    
    2) By manipulating the inputs of the k_sample test, we create
    concatenated versions of the inputs and a label matrix which are
    paired. Then, any multivariate nonparametric test can be performed on
    this data.
    """
    
    #extract data matrix of shape (_samples,_features) for each group
    k_array = np.unique(y)
    matrices = []
    for i in k_array:
        indices = np.where(y == i)[0] 
        if len(X.shape) == 1:
            xi = X[indices]
        else:
            xi = X[indices,:]
        matrices.append(xi)
    X = np.concatenate(matrices)
    #one hot encode y for multivariate independence test
    vs = []
    for i in range(len(np.unique(y))):
        n = matrices[i].shape[0]
        encode = np.zeros(shape=(n, len(matrices)))
        encode[:, i] = np.ones(shape=n)
        vs.append(encode)
    y = np.concatenate(vs)
    
    #mgc case
    if score_func == "mgc":
        warnings.filterwarnings("ignore")
        mgc = multiscale_graphcorr(X,y,reps = 0)
        stat = mgc.stat 
    return(stat)


# In[ ]:




