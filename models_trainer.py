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
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.dummy import DummyRegressor
from ensamble_stacking_predictor import StackingBestPredictor

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

        self.X_transformed = self.X_scaled01[:, self.features_above_threshold]
        self.transformed_features_names = [self.features_names[index] for index in self.features_above_threshold]

    def pca_and_scale_X_scaled(self):
        # this one is just like the svm
        # the third input
        # this is the right one for knn
        # PCA 2
        FEATURE_TRESHHOLD = 4.0e-03
        pca = PCA()
        pca.fit(self.X_scaled01)

        self.scaled_features_above_threshold = np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0]


        pca.n_components = len(np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0])
        # print(pca.n_components)

        self.X_pca_transformed_scaled = pca.fit_transform(self.X_scaled01)
        self.X_pca_scaled_filtered = [self.filter_pca_features(row) for row in self.X]
        self.pca = pca

        # print(self.scaled_features_above_threshold)
        # print(len(self.X_pca_scaled_filtered[0]))

    def pca_and_scale_X(self):
        # the second and the last of the input
        # PCA
        FEATURE_TRESHHOLD = 0.5
        pca = PCA()
        pca.fit(self.X)

        self.features_above_threshold = np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0]
        pca.n_components = len( np.where(pca.explained_variance_ > FEATURE_TRESHHOLD)[0])

        self.X_pca_transformed = pca.fit_transform(self.X)

        self.X_pca_filtered = [self.filter_pca_features(row) for row in self.X]

        # print(self.features_above_threshold)
        # print(len(self.X_pca_filtered[0]))

        # transoform in [0 1]
        self.pca_min_max_scaler = preprocessing.MinMaxScaler()
        self.pca_min_max_scaler.fit(self.X_pca_transformed)
        self.X_pca_transformed = self.pca_min_max_scaler.transform(self.X_pca_transformed)

        # print(len(self.X_pca_transformed[0]))
    def get_svm(self):
        # 1.07 ( 0.81)
        # the best for score_avr
        return svm.SVR(kernel="linear", C = 1, epsilon = 0.1)

        # return svm.SVR(kernel="linear", C = 1, epsilon = 0.01)


        # best for scores_scaled
        # return svm.SVR(kernel="poly", degree=1, gamma = 1, C = 1, epsilon = 0.01)

    def get_baseline(self):
        return DummyRegressor(strategy='mean')

    def scale_features(self, features):
        return self.min_max_scaler.transform(features)

    def filter_scaled_pca_features(self, features):
        return [features[i] for i in self.scaled_features_above_threshold]

    def filter_pca_features(self, features):
        return [features[i] for i in self.features_above_threshold]

    def get_ada_boost(self, clf, n_estimators):
        return AdaBoostRegressor(n_estimators=n_estimators, base_estimator=clf, loss="square")

    def get_forest(self):
        return RandomForestRegressor(random_state=1, n_estimators=300)

    def get_tree(self):
        return DecisionTreeRegressor(criterion="mse", max_depth=5)

    def get_knn(self):
        # the best for score_avr
        # return KNeighborsRegressor(n_neighbors=5, weights="uniform", p=2, algorithm="brute")
        return KNeighborsRegressor(n_neighbors=5, weights="distance", p=2, algorithm="brute")

    def train_svm(self, X, Y):
        return self.get_svm().fit(X, Y)

    def train_scaled_svm(self, X, Y):
        return self.get_svm().fit(self.X_scaled01, Y)

    def train_regression_tree(self, X, Y):
        return self.get_forest().fit(X, Y)

    def train_full_tree(self):
        return self.train_regression_tree(self.X, self.Y)

    def train_full_svm(self):
        return self.train_svm(self.X, self.Y)

    def train_scaled01_full_svm(self):
        return self.train_svm(self.X_scaled01, self.Y)

    def print_regression_tree(self, tree_image_name):

        dot_data = tree.export_graphviz(self.get_tree().fit(self.X, self.Y), out_file=None,
                         feature_names=self.features_names,
                         filled=True, rounded=True,
                         special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("./predictor_reports/graphics/" + tree_image_name + ".pdf")

    def train_cross_val(self, train_set, clf, scoring = 'neg_mean_absolute_error'):
        # print(self.X_scaled01)
        # scoring='neg_mean_absolute_error'
        scores = cross_val_score(clf, train_set, self.Y, cv=10, scoring=scoring)
        return scores

    # def cross_val_ensamble_best(self, scoring):
    #     clf = StackingBestPredictor(self.train_full_tree(), self.train_scaled01_full_svm(), self.train_scaled_fetures_knn())
    #     cross_val_score(clf, self.X, self.Y, cv=10, scoring=scoring)

    def cross_val_scaled01_svm(self, scoring):
        return self.train_cross_val(self.X_scaled01, self.get_svm(), scoring)

    def cross_val_unscaled_svm(self, scoring):
        return self.train_cross_val(self.X, self.get_svm(), scoring)

    def cross_val_transformed_pca_features_svm(self, scoring):
        return self.train_cross_val(self.X_pca_transformed, self.get_svm(), scoring)

    def cross_val_transformed_scaled_features_svm(self, scoring):
        return self.train_cross_val(self.X_pca_transformed_scaled, self.get_svm(), scoring)

    def cross_val_transformed_features_svm(self, scoring):
        return self.train_cross_val(self.X_transformed, self.get_svm(), scoring)

    def cross_val_pca_scaled_filtered_features_svm(self, scoring):
        return self.train_cross_val(self.X_pca_scaled_filtered, self.get_svm(), scoring)

    def cross_val_baseline(self, scoring):
        return self.train_cross_val(self.X_transformed, self.get_baseline(), scoring)

    def cross_val_ada_boost_svm(self, scoring):
        return self.train_cross_val(self.X_scaled01, self.get_ada_boost(self.get_svm(), 150), scoring)

    def cross_val_tree(self, scoring):
        return self.train_cross_val(self.X, self.get_forest(), scoring)

    def cross_val_transformed_scaled_features_tree(self, scoring):
        return self.train_cross_val(self.X_pca_transformed_scaled, self.get_forest(), scoring)

    def cross_val_transformed_features_tree(self, scoring):
        return self.train_cross_val(self.X_pca_transformed, self.get_forest(), scoring)

    def cross_val_pca_scaled_filtered_features_tree(self, scoring):
        return self.train_cross_val(self.X_pca_scaled_filtered, self.get_forest(), scoring)

    def cross_val_pca_filtered_features_tree(self, scoring):
        return self.train_cross_val(self.X_pca_filtered, self.get_forest(), scoring)

    def cross_val_ada_boost_tree(self, scoring):
        return self.train_cross_val(self.X, self.get_ada_boost(None, 150), scoring)

    def cross_val_knn(self, scoring):
        return self.train_cross_val(self.X, self.get_knn(), scoring)

    def cross_val_transformed_scaled_features_knn(self, scoring):
        return self.train_cross_val(self.X_pca_transformed_scaled, self.get_knn(), scoring)

    def train_scaled_fetures_knn(self):
        return self.get_knn().fit(self.X_pca_transformed_scaled, self.Y)

    def cross_val_transformed_features_knn(self, scoring):
        return self.train_cross_val(self.X_transformed, self.get_knn(), scoring)

    def cross_val_pca_filtered_features_knn(self, scoring):
        return self.train_cross_val(self.X_pca_filtered, self.get_forest(), scoring)

    def cross_val_pca_scaled_filtered_features_knn(self, scoring):
        return self.train_cross_val(self.X_pca_scaled_filtered, self.get_forest(), scoring)
