import pytest
import itertools
import numpy as np
from scipy import stats, sparse
from sklearn.feature_selection import MultivariateFeatureSelector, k_sample_test
from sklearn.utils._testing import assert_array_equal
from sklearn.datasets import make_classification

def test_k_sample_test():
    #Make sure k_sample_test produces test statistic as expected 
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    F = k_sample_test(X,y)
    assert (F >= -1)
    assert (F <= 1)

def test_sparse_support():
    #Make sure sparse data is supported
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )
    sequential_filter = MultivariateFeatureSelector(k = 4)
    sequential_filter.fit(sparse.csr_matrix(X),y)
    sequential_filter.transform(sparse.csr_matrix(X))
    

def test_multivariate_feature_selector():
    #Make sure that selector selects informative features
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )
    sequential_filter = MultivariateFeatureSelector(k = 5)
    sequential_filter.fit(X,y)
    support = sequential_filter.get_support()
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    assert_array_equal(support, gtruth)

def test_invalid_sample_size():
    #Test selector with invalid sample size
    X = np.array([[10, 20], [20, 20], [20, 30]])
    y = np.array([[1], [0], [0]])
    with pytest.raises(ValueError):
        MultivariateFeatureSelector(k=1).fit(X, y)

@pytest.mark.parametrize("k, expected",
                         [(-1, ValueError), (21, ValueError), (2.7, TypeError)])
def test_invalid_k(k, expected):
    #Test selector with invalid k input
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )
    with pytest.raises(expected):
        MultivariateFeatureSelector(k).fit(X, y)

def test_nan():
    # Test for nan, existing non-parametric multivariate independence tests in scipy
    # do not allow for nan
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )
    X[1,1] = np.nan
    with pytest.raises(ValueError):
        MultivariateFeatureSelector(k=1).fit(X, y)

def test_zero_variance_case():
    #Test for zero variance feature column case
    #exisiting multivariate independence test in scipy
    #Multiscale Graph Correlation does not support
    #zero variance column, so we are making sure
    #that the relevant accomodation works as intended
    X = np.array([[20, 20, 10], [20, 30, 10], [20, 30, 20], [20, 30, 20],[20, 30, 10]])
    y = np.array([[1], [0], [0], [0], [0]])
    multivariate =  MultivariateFeatureSelector(k=2)
    multivariate.fit(X, y)
    support_multivariate = multivariate.get_support()
    assert_array_equal(support_multivariate, np.array([False, True, True]))
    
def test_boundary_case():
    # Test boundary case, and always aim to select 1 feature.
    X = np.array([[10, 20], [20, 20], [20, 30], [20, 30],[20, 20]])
    y = np.array([[1], [0], [0], [0], [0]])
    multivariate =  MultivariateFeatureSelector(k=1)
    multivariate.fit(X, y)
    support_multivariate = multivariate.get_support()
    assert_array_equal(support_multivariate, np.array([True, False]))

