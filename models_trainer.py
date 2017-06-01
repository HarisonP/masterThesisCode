import numpy as np
from sklearn import tree
import pydotplus

class ModelsTrainer:
    def __init__(self, features, scores):
        self.features = features
        self.scores = scores

    def train_regression_tree(self):
        X = [value['features_values'] for key, value in self.features.items()]
        Y = [self.scores[key] for key, value in self.features.items()]

        reg_tree = tree.DecisionTreeRegressor()
        return reg_tree.fit(X, Y)

    def print_regression_tree(self, reg_tree):
        features_names = [value['features_names'] for key, value in self.features.items()][0]

        dot_data = tree.export_graphviz(reg_tree, out_file=None,
                         feature_names=features_names,
                         filled=True, rounded=True,
                         special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("iris.pdf")