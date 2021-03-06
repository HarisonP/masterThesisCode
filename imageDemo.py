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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())


def image_features_demo():
    feature_extractor = FaceFeatureExtractor(args["image"])
    feature_extractor.get_face_features()
    feature_extractor.print_face_detected_with_shape()
    feature_extractor.print_features()
# image_features_demo()

def output(models_trainer, prefix, dataset_size, error_type="neg_mean_absolute_error"):
    print("Number of photos:", dataset_size)
    print ("########## ErrorType", error_type," ##########")
    # models_trainer.cross_val_ensamble_best(error_type)
    # tree_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_tree(error_type)])
    # tree_pca_filtered = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_tree(error_type)])
    # tree_pca_filtered_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_tree(error_type)])

    # tree_ada_boost = np.array([math.fabs(s) for s in models_trainer.cross_val_ada_boost_tree(error_type)])
    # tree_reduced_scaled_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_scaled_features_tree(error_type)])
    # tree_scaled_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_features_tree(error_type)])

    # THE BEST OF TREE:
    # print(prefix, "Tree Cross Valid Error: %0.2f (+/- %0.2f)" % (tree_scores.mean(), tree_scores.std() * 2))
    # print(prefix, "Tree Cross pca_filtered Valid Error: %0.2f (+/- %0.2f)" % (tree_pca_filtered.mean(), tree_pca_filtered.std() * 2))
    # print(prefix, "Tree Cross pca_filtered_scaled Valid Error: %0.2f (+/- %0.2f)" % (tree_pca_filtered_scaled.mean(), tree_pca_filtered_scaled.std() * 2))

    # print(prefix, "PCA transformed scaled01 Tree Corss Valid Error: %0.2f (+/- %0.2f)" % (tree_reduced_scaled_scores.mean(), tree_reduced_scaled_scores.std() * 2))
    # print(prefix, "PCA transformed Tree Corss Valid Error: %0.2f (+/- %0.2f)" % (tree_scaled_scores.mean(), tree_scaled_scores.std() * 2))

    # print(prefix, "Tree Ada Boost Valid Error: %0.2f (+/- %0.2f)" % (tree_ada_boost.mean(), tree_ada_boost.std() * 2))

    # print(tree_scores)

    # svm_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_scaled01_svm(error_type)])
    # pca_svm_filtered_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_svm(error_type)])
    # svm_pca_reduced_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_pca_features_svm(error_type)])
    # svm_unscaled = np.array([math.fabs(s) for s in models_trainer.cross_val_unscaled_svm(error_type)])
    # svm_reduced_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_scaled_features_svm(error_type)])

    # svm_reduced = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_features_svm(error_type)])
    # svm_scores_ada_boost = np.array([math.fabs(s) for s in models_trainer.cross_val_ada_boost_svm(error_type)])

    # print(svm_scores)

    # THE BEST OF SVM:
    # print(prefix, "scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std() * 2))
    # print(prefix, "scaled01 PCA filtered SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (pca_svm_filtered_scores.mean(), pca_svm_filtered_scores.std() * 2))

    # print(prefix, "PCA transformed SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_pca_reduced_scores.mean(), svm_pca_reduced_scores.std() * 2))

    # print(prefix, "PCA transformed scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_reduced_scaled.mean(), svm_reduced_scaled.std() * 2))

    # print(prefix, "transformed scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_reduced.mean(), svm_reduced.std() * 2))
    # print(prefix, "scaled01 SVM Ada Boost Valid Error: %0.2f (+/- %0.2f)" % (svm_scores_ada_boost.mean(), svm_scores_ada_boost.std() * 2))



    # knn_score = np.array([math.fabs(s) for s in models_trainer.cross_val_knn(error_type)])
    # knn_pca_filtered = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_knn(error_type)])
    # knn_pca_filtered_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_knn(error_type)])

    # knn_reduced_scaled_score = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_scaled_features_knn(error_type)])
    # knn_reduced = np.array([math.fabs(s) for s in models_trainer.cross_val_reduced_features_svm(error_type)])
    # print(knn_reduced_scaled_score)
    # print(prefix, "KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_score.mean(), knn_score.std() * 2))
    # THE BEST OF KNN
    # print(prefix, "PCA transformed scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_reduced_scaled_score.mean(), knn_reduced_scaled_score.std() * 2))
    # print(prefix, "transformed scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_reduced.mean(), knn_reduced.std() * 2))
    # print(prefix, "filtered scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_pca_filtered_scaled.mean(), knn_pca_filtered_scaled.std() * 2))
    # print(prefix, "filtered KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_pca_filtered.mean(), knn_pca_filtered.std() * 2))

    base_line_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_baseline(error_type)])
    print(prefix, "BaseLine Cross Valid Error: %0.2f (+/- %0.2f)" % (base_line_scores.mean(), base_line_scores.std() * 2))

scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores_avr = scores_extractor.extract_average_scores()
scores_scaled = scores_extractor.extract_z_scaled()
scores_z_avr = scores_extractor.get_z_scaled_average()
# print(scores_avr, scores_scaled)

all_images = glob.glob(os.path.realpath('./dataset/*.jpg'))
all_women = glob.glob(os.path.realpath('./dataset/women/*.jpg'))
all_men = glob.glob(os.path.realpath('./dataset/men/*.jpg'))

features = {}
features_men = {}
features_women = {}


def load_features_from_file(features_filename):
    with open(features_filename) as json_data:
        return json.load(json_data, object_pairs_hook=OrderedDict)

def extract_and_load_to_file(features, features_men, features_women):
    for filename in all_images:
        print(filename)
        next_feature_extractor = FaceFeatureExtractor(filename)
        features[os.path.basename(filename)] = next_feature_extractor.get_face_features()

    for filename in all_women:
        features_women[os.path.basename(filename)] = features[os.path.basename(filename)]


    for filename in all_men:
        features_men[os.path.basename(filename)] = features[os.path.basename(filename)]

    with open('features_women.json', 'w') as outfile:
        json.dump(features_women, outfile)

    with open('features_men.json', 'w') as outfile:
        json.dump(features_men, outfile)


    with open('features.json', 'w') as outfile:
        json.dump(features, outfile)

features = load_features_from_file('features.json')
features_men = load_features_from_file('features_men.json')
features_women = load_features_from_file('features_women.json')

# extract_and_load_to_file(features, features_men, features_women)

def train_3_models(feature, scores, prefix):
    print(prefix)
    print("Start ======================= Start")
    models_trainer_mixed = ModelsTrainer(features, scores)
    models_trainer_women = ModelsTrainer(features_women, scores)
    models_trainer_men = ModelsTrainer(features_men, scores)

    output(models_trainer_mixed, "Mixed", len(features));
    output(models_trainer_women, "Women", len(features_women));
    output(models_trainer_men, "Men",  len(features_men));

    print("End ======================= End")

# train_3_models(features, scores_avr, "Scores Average")

# train_3_models(features, scores_scaled, "Scores Z-scaled")
# train_3_models(features, scores_z_avr, "Scores Z-scaled Avrg")

def predict_score(features, scores):

    models_trainer_mixed = ModelsTrainer(features, scores)
    knn_predictor = models_trainer_mixed.train_scaled_fetures_knn()
    reg_tree = models_trainer_mixed.train_full_tree()
    svm_01 = models_trainer_mixed.train_scaled01_full_svm()

    feature_extractor = FaceFeatureExtractor(args["image"])
    img_features = feature_extractor.get_face_features()

    features_girl_scaled = models_trainer_mixed.scale_features(img_features['features_values'])
    features_girl_filtered = models_trainer_mixed.filter_scaled_pca_features(features_girl_scaled)

    knn_score = knn_predictor.predict(features_girl_filtered)
    tree_score = reg_tree.predict(img_features['features_values'])
    svm_score = svm_01.predict(features_girl_scaled)

    print("KNN: " ,knn_score)
    print("Tree: " ,tree_score)
    print("SVM: ", svm_score)
    # for index, name in enumerate(models_trainer_mixed.features_names):
        # print(name, ": ", svm_01.coef_[0][index])
        # print(name, ": ", models_trainer_mixed.pca.explained_variance_[index])



predict_score(features_women, scores_scaled)
def train_personal_model(features, scores):
    validation_imgs = glob.glob(os.path.realpath('./dataset/validation_set/*.jpg'))
    validation_imgs = [os.path.basename(img) for img in validation_imgs]
    features_filtered = {}
    features_for_validation = {}
    scores_extractor = ScoresExtractor( glob.glob(os.path.realpath(args["scoreszx"])))

    for img in features.items():
        if not(img[0] in validation_imgs):
            features_filtered[img[0]] = img[1]
        else:
            features_for_validation[img[0]] = img[1]

    models_trainer_mixed = ModelsTrainer(features, scores)
    reg_tree = models_trainer_mixed.train_full_tree()
    personal_predictor = PersonalModelTrainer(reg_tree, scores_extractor.get_z_scaled_average(), features_filtered)
    personal_predictor.train_personal_tree_predictor()

    test_scores = [scores[key] for key, value in features_for_validation.items()]
    test_features = [value['features_values'] for key, value in features_for_validation.items()]

    print("Baseline:", personal_predictor.base_accuracy(test_features, test_scores))
    print("Forest:", personal_predictor.forest_accuracy(test_features, test_scores))



# train_personal_model(features, scores_z_avr)