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


def train_personal_model(features, scores):
    validation_imgs = glob.glob(os.path.realpath('./dataset/validation_set/*.jpg'))
    validation_imgs = [os.path.basename(img) for img in validation_imgs]
    features_filtered = {}
    features_for_validation = {}
    scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/personal_scores/*.txt')))

    for img in features.items():
        if not(img[0] in validation_imgs):
            features_filtered[img[0]] = img[1]
        else:
            features_for_validation[img[0]] = img[1]

    models_trainer_mixed = ModelsTrainer(features, scores)
    reg_tree = models_trainer_mixed.train_full_tree()
    personal_predictor = PersonalModelTrainer(reg_tree, scores_extractor.get_z_scaled_average(), features_filtered)
    personal_predictor.train_personal_tree_predictor()

    svm = personal_predictor.train_personal_svm_predictor()

    print(svm.coef_)
    test_scores = [scores[key] for key, value in features_for_validation.items()]
    test_features = [value['features_values'] for key, value in features_for_validation.items()]

    print("Baseline accuracy: ", personal_predictor.base_accuracy(test_features, test_scores))
    print("Random forest accuracy: ", personal_predictor.forest_accuracy(test_features, test_scores))
    print("SVM accuracy: ", personal_predictor.svm_accuracy(test_features, test_scores))

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--scores", required=True,
    help="path to the user scores")

args = vars(ap.parse_args())
all_images = glob.glob(os.path.realpath('./dataset/*.jpg'))
all_women = glob.glob(os.path.realpath('./dataset/women/*.jpg'))
all_men = glob.glob(os.path.realpath('./dataset/men/*.jpg'))


features = load_features_from_file('features.json')
features_men = load_features_from_file('features_men.json')
features_women = load_features_from_file('features_women.json')


scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores_avr = scores_extractor.extract_average_scores()
scores_scaled = scores_extractor.extract_z_scaled()
scores_z_avr = scores_extractor.get_z_scaled_average()

train_personal_model(features, scores_z_avr)