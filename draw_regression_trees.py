from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from face_feature_extractor import FaceFeatureExtractor
from scores_extractor import ScoresExtractor
from models_trainer import ModelsTrainer
from personal_models_trainer import PersonalModelTrainer
import glob
import os
import json
from collections import OrderedDict
import math

def load_features_from_file(features_filename):
    with open(features_filename) as json_data:
        return json.load(json_data, object_pairs_hook=OrderedDict)

scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores_avr = scores_extractor.extract_average_scores()
scores_scaled = scores_extractor.extract_z_scaled()
scores_z_avr = scores_extractor.get_z_scaled_average()
# print(scores_avr, scores_scaled)

features = load_features_from_file('features.json')
features_men = load_features_from_file('features_men.json')
features_women = load_features_from_file('features_women.json')

models_trainer_mixed = ModelsTrainer(features, scores_scaled)
models_trainer_women = ModelsTrainer(features_women, scores_scaled)
models_trainer_men = ModelsTrainer(features_men, scores_scaled)

models_trainer_mixed.print_regression_tree("Mixed_tree")
models_trainer_women.print_regression_tree("Women_tree")
models_trainer_men.print_regression_tree("Men_tree")