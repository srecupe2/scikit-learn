"""Multivariate features selection."""
# Authors: S. Panda, S. Recupero
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np
from scipy.stats import multiscale_graphcorr
from scipy.sparse import isspmatrix
import warnings
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed

######################################################################
# Scoring function

# The following is a rewriting of hyppo.ksample.KSample
# from hyppo.neurodata.io
def k_sample_test(X, y,score_func="mgc"):
    """Nonparametric `K`-Sample Testing test statistic.
     
    A k-sample test tests equality in distribution among groups. Groups
    can be of different sizes, but must have the same dimensionality.
    This implementation reduces the k-sample testing to an 
    independence testing problem, and leverages notable and powerful
    multivariate independence tests.
    
    Read more in the :ref:`User Guide <multivariate_feature_selection>`.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample vectors.
    y : ndarray of shape (n_samples,)
        The target vector.
    score_func : string that refers to a multivariate independence test from scipy
        The default and only existing test is multiscale graph correlation.
    
    Returns
    -------
    stat : float that refers to the computed k-sample test statistic
    
    Notes
    -----
    1. The k-sample testing problem can be thought of as a generalization of
    the two sample testing problem. 
    
    2. By manipulating the inputs of the k-sample test, we create
    concatenated versions of the inputs and a label matrix which are
    paired. Then, any multivariate nonparametric test can be performed on
    this data.
    
    3. Multivariate feature selection uses k-sample test score function to
    calculate a test statistic for each feature not already selected as a 
    best feature. For each feature in that sub-section, inputted is a data matrix 
    with best features selected and that additional feature.
    
    References
    ----------
    .. [1] Sambit Panda, Cencheng Shen, Ronan Perry, Jelle Zorn, Antoine Lutz, 
           Carey E. Priebe, and Joshua T. Vogelstein. Nonpar MANOVA via 
           Independence Testing. arXiv:1910.08883 [cs, stat], April 2021. 

    """
    # extract data matrix of shape (_samples,_features) for each group
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
    # one hot encode y for multivariate independence test
    vs = []
    for i in range(len(np.unique(y))):
        n = matrices[i].shape[0]
        encode = np.zeros(shape=(n, len(matrices)))
        encode[:, i] = np.ones(shape=n)
        vs.append(encode)
    y = np.concatenate(vs)
    
    # default, which is mgc case
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mgc = multiscale_graphcorr(X,y,reps = 0)
    stat = mgc.stat 
    return stat

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
        
    def _test_stat(self, X_new, y, best_features, index):
        # helper function for calculating, in parallel, 
        # test statistic associated with
        # selected best features and an additional feature 
        if np.var(X_new[:,index]) == 0:
            stat = -1.0
        else:   
            columns = best_features.copy() 
            columns.append(index)
            X_j = X_new[:,columns]
            stat = k_sample_test(X_j,y)
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
        if isspmatrix(X) == True:
            X = X.toarray()
        
        # array of indices that correspond to features,
        # at each iteration, the selected best feature
        # is removed from this array
        features = np.arange(X.shape[1])
        
        if np.isnan(X).any() == True:
            raise ValueError("existing multivariate independence tests in scipy do not allow nan")
        if type(self.k) is not int:
            raise TypeError("k is type {}, must be int".format(type(self.k)))
        if not 0 < self.k and self.k <= X.shape[1]:
                raise ValueError("k is {}, must be nonnegative <= number of features of X".format(self.k))
        if not X.shape[0] >= 5:
                raise ValueError("number of samples is {}, must be >= 5".format(X.shape[0]))
        
        # loop to select feature subset, 
        # each iteration adds next best feature as 
        # determined by the mulitivariate independence test
        # as we rank each additional feature by statistic
        best_features = []
        while len(best_features) < self.k: 
            X_new = np.array(X)
            
            # Parallel process for test statistic calculations 
            # of selected best features and each additional feature.
            # size of operations in parallel per loop iteration is 
            # n_features - len(best_features)
            scores = list(
                Parallel(n_jobs=workers)(
                    [
                        delayed(self._test_stat)(X_new,y,best_features,index)
                        for index in features
                    ]
                )
            )

            scores_index = np.column_stack((features,np.array(scores)))
            sorted_index = scores_index[scores_index[:,1].argsort()] 
            best = int(sorted_index[len(scores)-1,0])
            best_features.append(best) 
            features = np.delete(features,np.where(features == best)) 
        self.best_features_ = best_features
        self.features_ = np.arange(X.shape[1])
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self)
        return  np.array([x in self.best_features_ for x in self.features_])
    
    def _more_tags(self):
        return {"allow_nan": False,"requires_y": True}

