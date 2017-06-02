import numpy as np
from sklearn import tree
import pydotplus
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class ModelsTrainer:
    def __init__(self, features, scores):
        TEST_SET_SIZE = 20 / 100

        self.train_set = {}
        self.test_set = {}

        self.train_scores = {}
        self.test_scores = {}

        self.test_set_size = int(TEST_SET_SIZE * len(features))

        print(self.test_set_size)

        for index, (key, value) in enumerate(features.items()):
            if (index > self.test_set_size):
                self.train_set[key] = value
                self.train_scores[key] = scores[key]
            else:
                self.test_set[key] = value
                self.test_scores[key] = scores[key]
        print(len(self.test_set))

    def train_regression_tree(self):
        X = [value['features_values'] for key, value in self.train_set.items()]
        Y = [self.train_scores[key] for key, value in self.train_set.items()]

        reg_tree = tree.DecisionTreeRegressor(max_depth=5)
        return reg_tree.fit(X, Y)

    def print_regression_tree(self, reg_tree):
        features_names = [value['features_names'] for key, value in self.train_set.items()][0]

        dot_data = tree.export_graphviz(reg_tree, out_file=None,
                         feature_names=features_names,
                         filled=True, rounded=True,
                         special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("iris.pdf")

    def mean_squared_error(self, regressior):
        X = [test_example['features_values'] for key, test_example in  self.test_set.items()]
        prediction = regressior.predict(X)
        scores = [self.test_scores[key] for key, value in self.test_set.items()]

        return mean_squared_error(scores, prediction)

    def mean_abs_error(self, regressior):
        X = [test_example['features_values'] for key, test_example in  self.test_set.items()]
        prediction = regressior.predict(X)
        scores = [self.test_scores[key] for key, value in self.test_set.items()]

        return mean_absolute_error(scores, prediction)