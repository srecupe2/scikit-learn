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

######################################################################
# Selector

class MultivariateFeatureSelector(SelectorMixin, BaseEstimator):    
    #Unparallelized
    def __init__(self, k=10):
        self.k = k
        
    def fit(self, X, y,workers = -1):
        features = np.arange(X.shape[1])
         
        if type(self.k) != int:
            raise ValueError("k features to select must be integer")
        if not 0 <= self.k <= X.shape[1]:
                raise ValueError("k features to select must be a non-negative integer that is less than or equal to n_features of X")
        if not X.shape[0] >= 5:
                raise ValueError("n_samples of data matrix X must be >= 5 in order to perform k_sample multivariate independence test")
        
        best_features = []
        while (len(best_features) < self.k):  
            X_new = np.array(X)
            scores = []
            for i in features:
                if np.var(X_new[:,i]) == 0:
                    stat = -1000.0
                else:   
                    if len(best_features)==0:
                        X_j = X_new[:,i] 
                        stat= k_sample_test(X_j,y)
                    else:
                        columns = best_features 
                        columns.append(i)
                        X_j = X_new[:,columns]
                        stat= k_sample_test(X_j,y)
                scores.append(stat)
            scores_index = np.zeros((len(features),2)) 
            scores_index[:,0] = features 
            scores_index[:,1] = scores 
            sorted_index = scores_index[scores_index[:, 1].argsort()] 
            best = sorted_index[len(scores)-1,0] 
            best_features.append(int(best))  
            features = np.delete(features,np.where(features == best))
        self.best_features_ = best_features
        self.features_ = np.arange(X.shape[1])
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self)
        return  np.array([x in self.best_features_ for x in self.features_])
    
    def _more_tags(self):
        return {"allow_nan": True,"requires_y": True}

