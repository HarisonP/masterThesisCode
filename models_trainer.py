import numpy as np
from sklearn import tree
import pydotplus
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA

class ModelsTrainer:
    def __init__(self, features, scores):
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.features_names = [value['features_names'] for key, value in features.items()][0]
        self.X = [value['features_values'] for key, value in features.items()]

        self.min_max_scaler.fit(self.X)
        self.X_scaled01 = self.min_max_scaler.transform(self.X)
        self.Y = [scores[key] for key, value in features.items()]

        self.pca_and_scale_X()
        self.pca_and_scale_X_scaled()

        self.X_reduced = self.X_scaled01[:, self.features_above_tresh_hold]
        self.reduced_features_names = [self.features_names[index] for index in self.features_above_tresh_hold]

    def pca_and_scale_X_scaled(self):
        # this one is just like the svm
        # the third input
        # PCA 2
        FEATURE_TRESHHOLD = 5.0e-03
        pca = PCA()
        pca.fit(self.X_scaled01)
        # self.features_above_tresh_hold = np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0]

        pca.n_components = len(np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0])
        print(pca.n_components)

        self.X_pca_reduced_scaled = pca.fit_transform(self.X_scaled01)

    def pca_and_scale_X(self):
        # the second and the last of the input
        # PCA
        FEATURE_TRESHHOLD = 0.5
        pca = PCA()
        pca.fit(self.X)
        self.features_above_tresh_hold = np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0]
        pca.n_components = len( np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0])

        self.X_pca_reduced = pca.fit_transform(self.X)
        # transoform in [0 1]
        self.pca_min_max_scaler = preprocessing.MinMaxScaler()
        self.pca_min_max_scaler.fit(self.X_pca_reduced)
        self.X_pca_reduced = self.pca_min_max_scaler.transform(self.X_pca_reduced)

    def get_svm(self):
        # 1.07 ( 0.81)
        return svm.SVR(kernel="linear", C = 1, epsilon = 0.01)
        # Women SVM Cross Valid Error: 1.03 (+/- 0.77)
        # return svm.SVR(kernel="poly", degree=2, gamma = 0.2, C = 1, epsilon = 0.01, coef0=0.7, tol=1)

    def scale_features(self, features):
        return self.min_max_scaler.transform(features)

    def get_tree(self):
        return RandomForestRegressor(random_state=1, n_estimators=300)

    def get_knn(self):
        return KNeighborsRegressor(n_neighbors=5)

    def get_knn_radius(self):
        return RadiusNeighborsRegressor(radius=2.0)

    def train_svm(self, X, Y):
        return self.get_svm().fit(X, Y)

    def train_scaled_svm(self, X, Y):
        return self.get_svm().fit(X_scaled01, Y)

    def train_regression_tree(self, X, Y):
        return self.get_tree().fit(X, Y)

    def train_full_tree(self):
        return self.train_regression_tree(self.X, self.Y)

    def train_full_svm(self):
        return self.train_svm(self.X, self.Y)

    def print_regression_tree(self, reg_tree):

        dot_data = tree.export_graphviz(reg_tree, out_file=None,
                         feature_names=self.features_names,
                         filled=True, rounded=True,
                         special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("tree.pdf")

    def cross_val_scaled01_svm(self):
        clf = self.get_svm();
        # print(self.X_scaled01)
        scores = cross_val_score(clf, self.X_scaled01, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_unscaled_svm(self):
        clf = self.get_svm();
        scores = cross_val_score(clf, self.X, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_reduced_pca_features_svm(self):
        clf = self.get_svm();
        # print(self.X_pca_reduced)
        scores = cross_val_score(clf, self.X_pca_reduced, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_reduced_scaled_features_svm(self):
        clf = self.get_svm();
        scores = cross_val_score(clf, self.X_pca_reduced_scaled, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_reduced_features_svm(self):
        clf = self.get_svm();
        scores = cross_val_score(clf, self.X_reduced, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_tree(self):
        clf = self.get_tree();
        scores = cross_val_score(clf, self.X, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_reduced_scaled_features_tree(self):
        clf = self.get_tree();
        scores = cross_val_score(clf, self.X_pca_reduced_scaled, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_knn(self):
        clf = self.get_knn();
        scores = cross_val_score(clf, self.X, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_reduced_scaled_features_knn(self):
        clf = self.get_knn();
        scores = cross_val_score(clf, self.X_pca_reduced_scaled, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_reduced_features_knn(self):
        clf = self.get_knn();
        scores = cross_val_score(clf, self.X_reduced, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_knn_radius(self):
        clf = self.get_knn_radius();
        scores = cross_val_score(clf, self.X, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def mean_squared_error(self, regressior, test_set, test_scores):
        prediction = regressior.predict(test_set)
        return mean_squared_error(test_scores, prediction)

    def mean_abs_error(self, regressior,  test_set, test_scores):
        prediction = regressior.predict(test_set)
        return mean_absolute_error(test_scores, prediction)