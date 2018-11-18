"""
Implement a Naive Bayes SVM
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.svm import LinearSVC

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 clf = LinearSVC(),
                 clf__C = 1.0,
                 clf__dual = 'auto',
                 clf__verbose=0):

        set_params(
            clf = clf,
            clf__C = clf__C,
            clf__dual = clf__dual,
            clf__verbose, clf__verbose
        )

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['r_'])
        return self._clf.predict(x.multiply(self.r_))

    def score(self, x, y):
        check_is_fitted(self, ['r_'])
        return self._clf.score(x.multiply(self.r_), y)

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self.r_ = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self.r_)
        if self.dual == 'auto':
            self.dual = x_nb.shape[0] <= x_nb.shape[1]
        self._clf = LinearSVC(C=self.C, dual=self.dual, verbose=self.verbose)
        self._clf.fit(x_nb, y)
        return self
