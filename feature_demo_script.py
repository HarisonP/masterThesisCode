import argparse
from face_feature_extractor import FaceFeatureExtractor
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

def image_features_demo():
    feature_extractor = FaceFeatureExtractor(args["image"])
    feature_extractor.get_face_features()
    feature_extractor.print_face_detected_with_shape()

    feature_extractor.print_features()
    with open('./feature_reports/' + os.path.basename(args["image"]) + ".txt", 'w+') as report:
        report.write(feature_extractor.features_for_printinig())


image_features_demo()