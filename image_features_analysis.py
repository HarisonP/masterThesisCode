import matplotlib.pyplot as plt
from face_feature_extractor import FaceFeatureExtractor
import json
from collections import OrderedDict
import numpy as np

def load_features_from_file(features_filename):
    with open(features_filename) as json_data:
        return json.load(json_data, object_pairs_hook=OrderedDict)

def box_plot_per_feature(features):
    feature_matrix = []
    feature_names = []

    for key , feature_dict in features.items():
        feature_names = feature_dict['features_names']
        feature_matrix.append(feature_dict['features_values'])

    for index, column in enumerate(np.asarray(feature_matrix).T):
        plt.boxplot(column)
        plt.savefig('feature_reports/graphics/' + feature_names[index] + ".png")
        plt.clf()

features = load_features_from_file('features.json')
features_men = load_features_from_file('features_men.json')
features_women = load_features_from_file('features_women.json')

box_plot_per_feature(features)

