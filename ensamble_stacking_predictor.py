import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class StackingBestPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, tree_clf, svm_clf, knn_clf):
        self.tree_clf = tree_clf
        self.svm_clf = svm_clf
        self.knn_clf = knn_clf

    def fit(self, X, y):
        return self.tree_clf

    def predict(self, X):
        return self.tree_clf.predict(X)
