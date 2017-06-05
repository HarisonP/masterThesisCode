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

class ModelsTrainer:
    def __init__(self, features, scores):
        self.min_max_scaler = preprocessing.MinMaxScaler()

        self.features_names = [value['features_names'] for key, value in features.items()][0]
        self.X = [value['features_values'] for key, value in features.items()]
        self.X_scaled01 = self.min_max_scaler.fit_transform(self.X)
        self.Y = [scores[key] for key, value in features.items()]

    def get_svm(self):
        return svm.SVR(kernel="linear", C = 1, epsilon = 0.01)
        # return svm.SVR(kernel="poly", degree=1, gamma = 0.001, C = 1)

    def scale_features(self, features):
        return self.min_max_scaler.fit_transform(features)
    def get_tree(self):
        return RandomForestRegressor(random_state=1, n_estimators=300)

    def get_knn(self):
        return KNeighborsRegressor(n_neighbors=15)

    def get_knn_radius(self):
        return RadiusNeighborsRegressor(radius=2.0)

    def train_svm(self, X, Y):
        return self.get_svm().fit(X, Y)

    def train_regression_tree(self, X, Y):
        return self.get_tree().fit(X, Y)

    def train_full_tree(self):
        return self.train_regression_tree(self.X, self.Y)

    def train_full_svm(self):
        return self.train_svm(self.X_scaled01, self.Y)

    def print_regression_tree(self, reg_tree):

        dot_data = tree.export_graphviz(reg_tree, out_file=None,
                         feature_names=self.features_names,
                         filled=True, rounded=True,
                         special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("tree.pdf")

    def squared_error_tree(self):
        TEST_SET_SIZE =  20 / 100
        test_set_index = int(len(self.X) * TEST_SET_SIZE)
        train_set = self.X[0 : test_set_index]
        train_set_scores = self.Y[0 : test_set_index]
        reg_tree = self.train_regression_tree(train_set, train_set_scores)

        return self.mean_squared_error(reg_tree, self.X[-test_set_index :], self.Y[-test_set_index : ])

    def abs_error_tree(self):
        TEST_SET_SIZE =  20 / 100
        test_set_index = int(len(self.X) * TEST_SET_SIZE)
        train_set = self.X[0 : test_set_index]
        train_set_scores = self.Y[0 : test_set_index]
        reg_tree = self.train_regression_tree(train_set, train_set_scores)

        return self.mean_abs_error(reg_tree, self.X[-test_set_index : ], self.Y[-test_set_index : ])

    def squared_error_svm(self):
        TEST_SET_SIZE =  20 / 100
        test_set_index = int(len(self.X) * TEST_SET_SIZE)
        train_set = self.X[0 : test_set_index]
        train_set_scores = self.Y[0 : test_set_index]
        reg_tree = self.train_svm(train_set, train_set_scores)

        return self.mean_squared_error(reg_tree, self.X[-test_set_index :], self.Y[-test_set_index : ])

    def abs_error_svm(self):
        TEST_SET_SIZE =  20 / 100
        test_set_index = int(len(self.X) * TEST_SET_SIZE)
        train_set = self.X[0 : test_set_index]
        train_set_scores = self.Y[0 : test_set_index]
        reg_tree = self.train_svm(train_set, train_set_scores)

        return self.mean_abs_error(reg_tree, self.X[-test_set_index : ], self.Y[-test_set_index : ])

    def cross_val_svm(self):
        clf = self.get_svm();
        # print(self.X_scaled01)
        scores = cross_val_score(clf, self.X_scaled01, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_tree(self):
        clf = self.get_tree();
        scores = cross_val_score(clf, self.X, self.Y, cv=10, scoring='neg_mean_absolute_error')
        return scores

    def cross_val_knn(self):
        clf = self.get_knn();
        scores = cross_val_score(clf, self.X, self.Y, cv=10, scoring='neg_mean_absolute_error')
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