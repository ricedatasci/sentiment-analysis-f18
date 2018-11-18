"""
Implement a TfidfVectorizer!
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class TfidfVectorizer(BaseEstimator, TransformerMixin):
    """
    Implements a TFIDF Vectorizer
    """

    def __init__(self, numwords):

        self.set_params(
            numwords = numwords
        )

        # A list mapping index -> word, can also be used as a vocabulary
        self.feature_names_ = None

        # The IDF vector
        self.idf_ = None

        pass

    def fit(self, X):
        """
        Fits this TFIDF vectorizer.  What you need to do:

        1. Figure out which words we need to use:
            * count occurences in each document, get the {self.numwords} most common
            * assign those words in order of frequency to self.feature_names_
        2. Build the self.idf_ vector:
            * count the number of documents containing each word
            * set idf of word i to log(N / 1 + # docs (i)) where N is number of docs
        """

        #### TODO: YOUR CODE HERE

        #### END YOUR CODE

        return self

    def transform(self, X):
        """
        Transforms the data provided in X.  What you need to do:

        1. Compute the term frequency for each document in X
            * count the number of times each word in self.feature_names_ is present
        2. Multiply each TF vector with self.idf_
        """

        check_is_fitted(self, ["feature_names_", "idf_"])

        X_feat = np.zeros((len(X), self.numwords))

        #### TODO: YOUR CODE HERE

        #### END YOUR CODE

        return X_feat
