#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
============================
Multivariate Feature Selection
============================
An example showing multivariate feature selection.
Noisy (non informative) features are added to the wine data and
multivariate feature selection is applied. For each feature, we plot the
the corresponding weights of an SVM prior to and after applying 
multivariate feature selection. In the total set of features, only the 
13 first ones are significant. We can see that prior to feature selection, 
the SVM assigns greater weight to the informative features, but there
is still a large amount of total weight assiged to the non-informative features. 
We can see from the after selection weights plot that multivariate feature selection
selects many of the informative features which results in a greater concentration of  
SVM total weight with the informative features, and will thus improve classification. 
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import MultivariateFeatureSelector, k_sample_test

# #############################################################################
# Import some data to play with
X, y = load_wine(return_X_y=True)

# Some noisy data not correlated
E = np.random.RandomState(42).uniform(0, 0.1, size=(X.shape[0], 50))

# Add the noisy data to the informative features
X = np.hstack((X, E))

X_indices = np.arange(X.shape[-1])

#split data to select features, and to evaluate classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

#classify without multivariate feature selection
clf = make_pipeline(MinMaxScaler(), LinearSVC())
clf.fit(X_train, y_train)
print(
    "Classification accuracy without selecting features: {:.3f}".format(
        clf.score(X_test, y_test)
    )
)

svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
svm_weights /= svm_weights.sum()

plt.bar(X_indices - 0.25, svm_weights, width=0.2, label="SVM weight")

#classifiy with multivariate feature selection
clf_selected = make_pipeline(MultivariateFeatureSelector(13),MinMaxScaler(), LinearSVC())
clf_selected.fit(X_train, y_train)
print(
    "Classification accuracy after multivariate feature selection: {:.3f}".format(
        clf_selected.score(X_test, y_test)
    )
)
selector = MultivariateFeatureSelector(13)
selector.fit(X_train, y_train)
svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()

plt.bar(
    X_indices[selector.get_support()] - 0.05,
    svm_weights_selected,
    width=0.2,
    label="SVM weights after selection",
)


plt.title("Comparing feature selection")
plt.xlabel("Feature number")
plt.yticks(())
plt.axis("tight")
plt.legend(loc="upper right")
plt.show()

