#!/usr/bin/env python
# coding: utf-8

# In[40]:


from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np
from scipy.stats import multiscale_graphcorr
from scipy.sparse import isspmatrix
from scipy._lib._util import MapWrapper
import warnings
from sklearn.utils.validation import check_is_fitted

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
    #default
    else:
        warnings.filterwarnings("ignore")
        mgc = multiscale_graphcorr(X,y,reps = 0)
        stat = mgc.stat 
    return(stat)

######################################################################
# Transformer

class MultivariateFeatureSelector(SelectorMixin, BaseEstimator):
    """ Transformer that performs forward selection.
    
    This feature selector adds features (forward selection) to
    form a feature subset. At each iteration, a parallel 
    operation occurs in which a multivariate independence test 
    is performed for each data matrix with the selected best 
    features and an additional feature not yet selected. The 
    additional feature associated with the highest multivariate
    independence test statistic is then chosen as the next best 
    feature. 
    
    Read more in the :ref:`User Guide <multivariate_feature_selection>`.
    
    Parameters
    ----------
    k: int, default=10
        amount of features to select. 
        
    Attributes
    ----------
    features_ : array, shape (n_features,)
        indices of all features in X
    
    best_features_ : array, shape (k,)
         indices of selected k best features of features_
         
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import MultivariateFeatureSelector
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = MultivariateFeatureSelector(k = 7).fit_transform(X, y)
    >>> X_new.shape
    (1797, 7)
    """
    
    def __init__(self, k=10):
        self.k = k
        
    class _Parallel:
        #helper class for calculating, in parallel, 
        #test statistic associated with
        #selected best features and an additional feature 
        def __init__(self, X_new, y,best_features):
            self.X_new = X_new
            self.y = y
            self.best_feat = best_features

        def __call__(self, index):
            if np.var(self.X_new[:,index]) == 0:
                stat = -1000.0
            else:   
                if len(self.best_feat)==0:
                    X_j =  self.X_new[:,index] 
                    stat= k_sample_test(X_j,self.y)
                else:
                    columns = self.best_feat 
                    columns.append(index)
                    X_j = self.X_new[:,columns]
                    stat= k_sample_test(X_j,self.y)
            return stat
        
        
    def fit(self, X, y,workers = -1):
        """Learn the features to select from X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.
        y : array-like of shape (n_samples,), default=None
            Target values. 
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if (isspmatrix(X) == True):
            X = X.toarray()
        
        features = np.arange(X.shape[1])
         
        if type(self.k) != int:
            raise ValueError("k features to select must be integer")
        if not 0 <= self.k <= X.shape[1]:
                raise ValueError("k features to select must be a non-negative integer that is less than or equal to n_features of X")
        if not X.shape[0] >= 5:
                raise ValueError("n_samples of data matrix X must be >= 5 in order to perform k_sample multivariate independence test")
        
        #loop to select k best features, 
        #each iteration adds next best feature
        best_features = []
        while (len(best_features) < self.k): 
            X_new = np.array(X)
            
            #Mapwrapper parallelizes test statistic calculations 
            #of selected best features and each additional feature.
            #size of operations in parallel per loop iteration is 
            #n_features - len(best_features)
            parallel = self._Parallel(X_new=X_new, y=y,best_features = best_features)
            with MapWrapper(workers) as mapwrapper:
                scores = list(mapwrapper(parallel, features)) 
            
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

