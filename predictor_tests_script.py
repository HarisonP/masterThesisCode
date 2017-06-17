from scores_extractor import ScoresExtractor
from models_trainer import ModelsTrainer
import json
from collections import OrderedDict
import glob
import os
import numpy as np
import math

def output(models_trainer, prefix, dataset_size, error_type="neg_mean_absolute_error"):
    print("Number of photos:", dataset_size)
    print ("########## ErrorType", error_type," ##########")
    tree_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_tree(error_type)])
    tree_pca_filtered = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_tree(error_type)])
    tree_pca_filtered_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_tree(error_type)])

    # tree_ada_boost = np.array([math.fabs(s) for s in models_trainer.cross_val_ada_boost_tree(error_type)])
    tree_reduced_scaled_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_scaled_features_tree(error_type)])
    # THE BEST OF TREE:
    print(prefix, "Tree Cross Valid Error: %0.2f (+/- %0.2f)" % (tree_scores.mean(), tree_scores.std() * 2))
    print(prefix, "Tree Cross pca_filtered Valid Error: %0.2f (+/- %0.2f)" % (tree_pca_filtered.mean(), tree_pca_filtered.std() * 2))
    print(prefix, "Tree Cross pca_filtered_scaled Valid Error: %0.2f (+/- %0.2f)" % (tree_pca_filtered_scaled.mean(), tree_pca_filtered_scaled.std() * 2))

    print(prefix, "PCA transformed scaled01 Tree Corss Valid Error: %0.2f (+/- %0.2f)" % (tree_reduced_scaled_scores.mean(), tree_reduced_scaled_scores.std() * 2))
    # print(prefix, "Tree Ada Boost Valid Error: %0.2f (+/- %0.2f)" % (tree_ada_boost.mean(), tree_ada_boost.std() * 2))

    # print(tree_scores)

    svm_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_scaled01_svm(error_type)])
    pca_svm_filtered_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_svm(error_type)])
    svm_pca_reduced_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_pca_features_svm(error_type)])
    # svm_unscaled = np.array([math.fabs(s) for s in models_trainer.cross_val_unscaled_svm(error_type)])
    svm_reduced_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_scaled_features_svm(error_type)])

    svm_reduced = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_features_svm(error_type)])
    # svm_scores_ada_boost = np.array([math.fabs(s) for s in models_trainer.cross_val_ada_boost_svm(error_type)])

    # print(svm_scores)

    # THE BEST OF SVM:
    print(prefix, "scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std() * 2))
    print(prefix, "scaled01 PCA filtered SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (pca_svm_filtered_scores.mean(), pca_svm_filtered_scores.std() * 2))

    print(prefix, "PCA transformed SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_pca_reduced_scores.mean(), svm_pca_reduced_scores.std() * 2))

    print(prefix, "PCA transformed scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_reduced_scaled.mean(), svm_reduced_scaled.std() * 2))

    print(prefix, "transformed scaled01 SVM Cross Valid Error: %0.2f (+/- %0.2f)" % (svm_reduced.mean(), svm_reduced.std() * 2))
    # print(prefix, "scaled01 SVM Ada Boost Valid Error: %0.2f (+/- %0.2f)" % (svm_scores_ada_boost.mean(), svm_scores_ada_boost.std() * 2))



    knn_score = np.array([math.fabs(s) for s in models_trainer.cross_val_knn(error_type)])
    knn_pca_filtered = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_knn(error_type)])
    knn_pca_filtered_scaled = np.array([math.fabs(s) for s in models_trainer.cross_val_pca_scaled_filtered_features_knn(error_type)])

    knn_reduced_scaled_score = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_scaled_features_knn(error_type)])
    knn_reduced = np.array([math.fabs(s) for s in models_trainer.cross_val_transformed_features_svm(error_type)])
    # print(knn_reduced_scaled_score)
    print(prefix, "KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_score.mean(), knn_score.std() * 2))
    # THE BEST OF KNN
    print(prefix, "PCA transformed scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_reduced_scaled_score.mean(), knn_reduced_scaled_score.std() * 2))
    print(prefix, "transformed scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_reduced.mean(), knn_reduced.std() * 2))
    print(prefix, "filtered scaled01 KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_pca_filtered_scaled.mean(), knn_pca_filtered_scaled.std() * 2))
    print(prefix, "filtered KNN Cross Valid Error: %0.2f (+/- %0.2f)" % (knn_pca_filtered.mean(), knn_pca_filtered.std() * 2))

    base_line_scores = np.array([math.fabs(s) for s in models_trainer.cross_val_baseline(error_type)])
    print(prefix, "BaseLine Cross Valid Error: %0.2f (+/- %0.2f)" % (base_line_scores.mean(), base_line_scores.std() * 2))

def load_features_from_file(features_filename):
    with open(features_filename) as json_data:
        return json.load(json_data, object_pairs_hook=OrderedDict)

features = load_features_from_file('features.json')
features_men = load_features_from_file('features_men.json')
features_women = load_features_from_file('features_women.json')

# extract_and_load_to_file(features, features_men, features_women)

def train_3_models(feature, scores, prefix, error_type="neg_mean_absolute_error"):
    print("!!!", prefix, "!!!!")
    print("Start ======================= Start")
    models_trainer_mixed = ModelsTrainer(features, scores)
    models_trainer_women = ModelsTrainer(features_women, scores)
    models_trainer_men = ModelsTrainer(features_men, scores)

    output(models_trainer_mixed, "Mixed", len(features), error_type);
    output(models_trainer_women, "Women", len(features_women), error_type);
    output(models_trainer_men, "Men",  len(features_men), error_type);

    print("!!!", prefix, "!!!!")
    print("End ======================= End")

scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores_avr = scores_extractor.extract_average_scores()
scores_scaled = scores_extractor.extract_z_scaled()
scores_z_avr = scores_extractor.get_z_scaled_average()

train_3_models(features, scores_avr, "Scores Average")
print("\n\n\n")
train_3_models(features, scores_scaled, "Scores Z-scaled")
print("\n\n\n")
train_3_models(features, scores_z_avr, "Scores Z-scaled Avrg")
print("\n\n\n")

train_3_models(features, scores_avr, "Scores Average", "neg_mean_squared_error")
print("\n\n\n")
train_3_models(features, scores_scaled, "Scores Z-scaled","neg_mean_squared_error")
print("\n\n\n")
train_3_models(features, scores_z_avr, "Scores Z-scaled Avrg", "neg_mean_squared_error")
print("\n\n\n")

