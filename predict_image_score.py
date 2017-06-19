import argparse
from face_feature_extractor import FaceFeatureExtractor
from scores_extractor import ScoresExtractor
from models_trainer import ModelsTrainer
import json
from collections import OrderedDict
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())


def predict_score(features, scores):

    models_trainer_mixed = ModelsTrainer(features, scores)
    knn_predictor = models_trainer_mixed.train_scaled_fetures_knn()
    reg_tree = models_trainer_mixed.train_full_tree()
    feature_extractor = FaceFeatureExtractor(args["image"])
    img_features = feature_extractor.get_face_features()

    features_girl_scaled = models_trainer_mixed.scale_features(img_features['features_values'])
    features_girl_filtered = models_trainer_mixed.filter_scaled_pca_features(features_girl_scaled)

    knn_score = knn_predictor.predict(features_girl_filtered)
    tree_score = reg_tree.predict(img_features['features_values'])

    print("KNN: " ,knn_score)
    print("Tree: " ,tree_score)

def load_features_from_file(features_filename):
    with open(features_filename) as json_data:
        return json.load(json_data, object_pairs_hook=OrderedDict)

features = load_features_from_file('features.json')
features_men = load_features_from_file('features_men.json')
features_women = load_features_from_file('features_women.json')

scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores_avr = scores_extractor.extract_average_scores()
scores_scaled = scores_extractor.extract_z_scaled()
scores_z_avr = scores_extractor.get_z_scaled_average()

predict_score(features, scores_scaled)