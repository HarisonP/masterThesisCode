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

# def rect_to_bb(rect):
#     # take a bounding predicted by dlib and convert it
#     # to the format (x, y, w, h) as we would normally do
#     # with OpenCV
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y

#     # return a tuple of (x, y, w, h)
#     return (x, y, w, h)

# def shape_to_np(shape, dtype="int"):
#     print('shapeTonp')
#     # initialize the list of (x, y)-coordinates
#     coords = np.zeros((68, 2), dtype=dtype)

#     # loop over the 68 facial landmarks and convert them
#     # to a 2-tuple of (x, y)-coordinates
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)

#     # return the list of (x, y)-coordinates
#     return coords



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# feature_extractor = FaceFeatureExtractor(args["image"])
# feature_extractor.get_face_features()
# feature_extractor.print_face_detected_with_shape()
# feature_extractor.print_features()



scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores = scores_extractor.extract_average_scores()
all_images = glob.glob(os.path.realpath('./dataset/*.jpg'))

features = {}
with open('features.json') as json_data:
    features = json.load(json_data)

# for filename in all_images:
#     next_feature_extractor = FaceFeatureExtractor(filename)
#     features[os.path.basename(filename)] = next_feature_extractor.get_face_features()

# with open('features.json', 'w') as outfile:
#     json.dump(features, outfile)


models_trainer = ModelsTrainer(features, scores)
reg_tree = models_trainer.train_regression_tree()
models_trainer.print_regression_tree(reg_tree)
svm = models_trainer.train_svm()

# print(models_trainer.mean_squared_error(reg_tree))
print("Tree: ", models_trainer.mean_abs_error(reg_tree))
print("SVM: ", models_trainer.mean_abs_error(svm))
# print(reg_tree.predict(features['0_natalie_dormer.jpg']['features_values']))
# print(scores['0_natalie_dormer.jpg'])