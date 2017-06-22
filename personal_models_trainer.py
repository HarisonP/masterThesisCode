from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import pydotplus
import copy
import numpy as np
import matplotlib.pyplot as plt
import ntpath
from sklearn import tree

class PersonalModelTrainer:
    def __init__(self, pulbic_opinion_predictor, user_scores, features, user_name):
        self.public_opinion = pulbic_opinion_predictor
        self.user_scores = user_scores

        self.features_names = [value['features_names'] for key, value in features.items()][0]
        self.features_names.append("Public Opinion ")
        for key, value in features.items():
            public_score = pulbic_opinion_predictor.predict(value['features_values'])
            value['features_values'].append(public_score)

        self.X = [value['features_values'] for key, value in features.items()]
        self.Y = [user_scores[key] for key, value in features.items()]

        plt.boxplot(self.Y)
        # plt.show()
        plt.savefig('score_reports/graphics/' + ntpath.basename(user_name) + ".png")
        # plt.clf()
        self.user_name = user_name
        self.Y = [self.__create_classes(val) for val in self.Y]

        self.train_base_lane()


    def __create_classes(self, regression_score):
        SCORE_BORDED = 0.4
        if (regression_score < -SCORE_BORDED):
            return 0
        elif (regression_score >= -SCORE_BORDED and regression_score <= SCORE_BORDED):
            return 1
        else:
            return 2

    def train_personal_svm_predictor (self):
        self.svm = svm.SVC(kernel="linear", C = 1, class_weight="balanced", decision_function_shape="ovo")
        self.svm.fit(self.X, self.Y)
        return self.svm

    def get_tree(self):
        return DecisionTreeClassifier(max_depth=100)

    def print_presonal_tree(self):
        dot_data = tree.export_graphviz(self.get_tree().fit(self.X, self.Y), out_file=None,
                         feature_names=self.features_names,
                         filled=True, rounded=True,
                         special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("./predictor_reports/graphics/" + ntpath.basename(self.user_name )+ ".pdf")

    def train_personal_tree_predictor(self):
        self.forest = RandomForestClassifier(n_estimators=180, criterion="entropy",class_weight="balanced", random_state=1)
        self.forest.fit(self.X, self.Y)
        return self.forest

    def train_base_lane(self):
        self.baseline = DummyClassifier(random_state = 1, strategy='uniform')
        self.baseline.fit(self.X, self.Y)
        return self.baseline

    def base_accuracy(self, test_y, y_true):
        # copy...
        test_y = copy.deepcopy(test_y)

        for row in test_y:
            row.append(self.public_opinion.predict(np.array(row).reshape(1, -1)))

        y_pred = self.baseline.predict(test_y)
        y_true = [self.__create_classes(val) for val in y_true]
        return accuracy_score(y_true, y_pred)

    def forest_accuracy(self, test_y, y_true):

        test_y = copy.deepcopy(test_y)
        for row in test_y:
            row.append(self.public_opinion.predict(np.array(row).reshape(1, -1)))

        y_pred = self.forest.predict(test_y)
        y_true = [self.__create_classes(val) for val in y_true]
        print(y_pred)
        print(y_true)
        return accuracy_score(y_true, y_pred)

    def svm_accuracy(self, test_y, y_true):

        test_y = copy.deepcopy(test_y)
        for row in test_y:
            row.append(self.public_opinion.predict(np.array(row).reshape(1, -1)))

        y_pred = self.svm.predict(test_y)
        y_true = [self.__create_classes(val) for val in y_true]
        return accuracy_score(y_true, y_pred)