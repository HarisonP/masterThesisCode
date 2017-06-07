from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from face_feature_extractor import FaceFeatureExtractor
from scores_extractor import ScoresExtractor
from models_trainer import ModelsTrainer
import glob
import os
import json
from collections import OrderedDict
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# feature_extractor = FaceFeatureExtractor(args["image"])
# feature_extractor.get_face_features()
# feature_extractor.print_face_detected_with_shape()
# feature_extractor.print_features()


def output(models_trainer, prefix, dataset_size):
    print("Number of photos:", dataset_size)
    # tree_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_tree()])
    # tree_reduced_scaled_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_scaled_features_tree()])
    # print(prefix, "Tree Cross Valid Error: %0.2f (+/- %0.2f)" % (tree_scores.mean(), tree_scores.std() * 2))
    # print(prefix, "PCA reduced scaled01 Tree Corss Valid Error: %0.2f (+/- %0.2f)" % (tree_reduced_scaled_scores.mean(), tree_reduced_scaled_scores.std() * 2))

    # print(tree_scores)
    # print(models_trainer.abs_error_svm())

    svm_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_scaled01_svm()])
    # svm_pca_reduced_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_pca_features_svm()])
    # svm_unscaled = np.array([math.fabs(s) for s in models_trainer.cross_val_unscaled_svm()])
    # svm_reduced_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_scaled_features_svm()])

    # svm_reduced = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_features_svm()])

    # print(svm_scores)

    print(prefix, "scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std() * 2))

    # print(prefix, "PCA Reduced SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_pca_reduced_scores.mean(), svm_pca_reduced_scores.std() * 2))
    # unscaled
    # print(prefix, "PCA reduced scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_reduced_scaled.mean(), svm_reduced_scaled.std() * 2))
    # print(prefix, "reduced scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_reduced.mean(), svm_reduced.std() * 2))


    knn_score = np.array([math.fabs(s) for s in models_trainer.cross_val_knn()])
    knn_reduced_scaled_score = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_scaled_features_knn()])
    knn_reduced = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_features_svm()])
    # print(knn_score)
    print(prefix, "KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_score.mean(), knn_score.std() * 2))
    print(prefix, "PCA reduced scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_reduced_scaled_score.mean(), knn_reduced_scaled_score.std() * 2))
    print(prefix, "reduced scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_reduced.mean(), knn_reduced.std() * 2))

    # knn_score_radius = np.array([math.fabs(s) for s in models_trainer.cross_val_knn_radius()])
    # print(knn_score_radius)
    # print(prefix, "KNN radius Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_score_radius.mean(), knn_score_radius.std() * 2))


scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores = scores_extractor.extract_average_scores()
all_images = glob.glob(os.path.realpath('./dataset/*.jpg'))
all_women = glob.glob(os.path.realpath('./dataset/women/*.jpg'))
all_men = glob.glob(os.path.realpath('./dataset/men/*.jpg'))

features = {}
features_men = {}
features_women = {}

with open('features_men.json') as json_data:
    features_men = json.load(json_data, object_pairs_hook=OrderedDict)


with open('features_women.json') as json_data:
    features_women = json.load(json_data, object_pairs_hook=OrderedDict)

with open('features.json') as json_data:
    features = json.load(json_data, object_pairs_hook=OrderedDict)

# for filename in all_images:
#     print(filename)
#     next_feature_extractor = FaceFeatureExtractor(filename)
#     features[os.path.basename(filename)] = next_feature_extractor.get_face_features()


# for filename in all_women:
#     next_feature_extractor = FaceFeatureExtractor(filename)
#     features_women[os.path.basename(filename)] = next_feature_extractor.get_face_features()


# for filename in all_men:
#     next_feature_extractor = FaceFeatureExtractor(filename)
#     features_men[os.path.basename(filename)] = next_feature_extractor.get_face_features()


# with open('features_women.json', 'w') as outfile:
#     json.dump(features_women, outfile)

# with open('features_men.json', 'w') as outfile:
#     json.dump(features_men, outfile)


# with open('features.json', 'w') as outfile:
#     json.dump(features, outfile)

models_trainer_mixed = ModelsTrainer(features, scores)
models_trainer_women = ModelsTrainer(features_women, scores)

models_trainer_men = ModelsTrainer(features_men, scores)

output(models_trainer_mixed, "Mixed", len(features));
output(models_trainer_women, "Women", len(features_women));
output(models_trainer_men, "Men",  len(features_men));



# reg_tree = models_trainer_women.train_full_tree()
# reg_svm = models_trainer_women.train_full_svm()
# feature_extractor = FaceFeatureExtractor(args["image"])
# img_features = feature_extractor.get_face_features()

# features_girl = models_trainer_women.scale_features(img_features['features_values'])
# the_score = reg_tree.predict(img_features['features_values'])
# print(the_score)