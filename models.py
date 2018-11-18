"""
Implement a Naive Bayes SVM!
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual='auto', verbose=0):
        self.C = C
        self.dual = dual
        self.verbose = verbose
        self._clf = None
        print("Creating model with C=%s" % C)

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def score(self, x, y):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.score(x.multiply(self._r), y)

    def predict_proba(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        if self.dual == 'auto':
            self.dual = x_nb.shape[0] <= x_nb.shape[1]
        self._clf = LogisticRegression(C=self.C, dual=self.dual, verbose=self.verbose)
        self._clf.fit(x_nb, y)
        return self
