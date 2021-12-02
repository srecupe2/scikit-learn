#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import warnings
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats, sparse

import pytest


from sklearn.utils._testing import assert_almost_equal, _convert_container
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils import safe_mask

from sklearn.datasets import make_classification

def test_k_sample_test(): 
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    F = k_sample_test(X,y)
    F_sparse = k_sample_test(sparse.csr_matrix(X), y)
    assert (F >= -1)
    assert (F <= 1)
    assert_almost_equal(F_sparse, F, decimal = 3)

def test_multivariate_feature_selector():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )
sequential_filter = MultivariateFeatureSelector(4)
support = sequential_filter.get_support()
gtruth = np.zeros(20)
gtruth[:4] = 1
assert_array_equal(support, gtruth)

def test_invalid_sample_size():
    X = np.array([[10, 20], [20, 20], [20, 30]])
    y = np.array([[1], [0], [0]])
    with pytest.raises(ValueError):
        MultivariateFeatureSelector(k=1).fit(X, y)

def test_invalid_k():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )
    with pytest.raises(ValueError):
        MultivariateFeatureSelector(k=-1).fit(X, y)
    with pytest.raises(ValueError):
        MultivariateFeatureSelector(k=21).fit(X, y)
    
def test_boundary_case():
    # Test boundary case, and always aim to select 1 feature.
    X = np.array([[10, 20], [20, 20], [20, 30], [20,30],[20,20]])
    y = np.array([[1], [0], [0],[0],[0]])
    multivariate =  MultivariateFeatureSelector(k=1)
    multivariate.fit(X, y)
    support_multivariate = multivariate.get_support()
    assert_array_equal(support_multivariate, np.array([True, False]))

