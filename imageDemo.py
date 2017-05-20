from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from face_feature_extractor import FaceFeatureExtractor


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

feature_extractor = FaceFeatureExtractor(args["image"])
feature_extractor.print_face_detected_with_shape()
print(feature_extractor.get_face_features())
