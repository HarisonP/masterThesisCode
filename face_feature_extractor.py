from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import geometry_helper
import constants

class FaceFeatureExtractor:

    def __init__(self, image_path):

        face_cascade = cv2.CascadeClassifier('precompiled_models/haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.full_face_detector = face_cascade
        self.predictor = dlib.shape_predictor(constants.DETECTOR_MODEL)

        image = cv2.imread(image_path)
        self.image = imutils.resize(image, width=constants.FACE_IMAGE_SIZE)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.features = { 'features_names': [], 'features_values': [] }
        self.__detect_faces()
        self.__detect_shapes()
        self.__extract_important_shape_points()

    def __detect_faces(self):
        rects = self.detector(self.gray, 1)
        faces = self.full_face_detector.detectMultiScale(self.gray, 1.3, 5)

        if(len(rects) > 1 or len(faces) > 1):
            raise Exception("More than one face detected!!!")

        self.rect = rects[0]
        self.full_face = faces[0]

    def __detect_shapes(self):
        shapes = []
        shape = self.predictor(self.gray, self.rect)
        self.shape = face_utils.shape_to_np(shape)

    def __get_scale_factor(self):
        return 1 - (self.face_width * self.face_height) / (constants.FACE_IMAGE_SIZE * constants.FACE_IMAGE_SIZE)

    def __extract_important_shape_points(self):
        self.point_between_the_eyebrows = geometry_helper.middle_point_between(self.shape[constants.LEFT_EYEBROW_INNER_POINT_INDEX],
                                                                               self.shape[constants.RIGTH_EYEBROW_INNER_POINT_INDEX])

        self.heightest_face_point = [self.point_between_the_eyebrows[0], self.full_face[1]]

        self.face_height = geometry_helper.point_distance(self.heightest_face_point,
                                                          self.shape[constants.LOWEST_CHIN_POINT_INDEX])


        self.face_width = geometry_helper.point_distance(self.shape[constants.LEFT_EAR],
                                                         self.shape[constants.RIGHT_EAR])

    def __scale_distance(self, distance):
        return round(distance * self.__get_scale_factor(), 2)

    def add_feature(self, name, val):
        self.features['features_names'].append(name)
        self.features['features_values'].append(val);

    def extract_face_sizes(self):

        self.add_feature("Relative Height", self.__scale_distance(self.face_height))
        self.add_feature("Relative Eye Level Width", self.__scale_distance(self.face_width))

        self.add_feature("Eye level Width To Height", self.face_width / self.face_height)

        chin_width = geometry_helper.point_distance(self.shape[constants.LEFT_CHIN_START],
                                                    self.shape[constants.RIGHT_CHIN_START])

        self.add_feature("Relative Chin Width", self.__scale_distance(chin_width))
        self.add_feature("Eye level Width to Chin Level Width", chin_width / self.face_width)
        self.add_feature("Chin Level Width to Height", chin_width / self.face_height)

        chin_to_nose_end_height = geometry_helper.point_distance(self.point_between_the_eyebrows,
                                                             self.shape[constants.LOWEST_CHIN_POINT_INDEX])

        self.add_feature("Relative Chin to nose Height", self.__scale_distance(chin_to_nose_end_height))
        self.add_feature("Chin to nose Height to Height ", chin_to_nose_end_height / self.face_height)
        self.add_feature("Relative size of the forhead", self.__scale_distance(self.face_height - chin_to_nose_end_height))

    def print_face_detected_with_shape(self):
        # loop over the face detections
        (x, y, w, h) = face_utils.rect_to_bb(self.rect)
        (xf, yf, wf, hf) = self.full_face

        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(self.image, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
        print(w, h)
        # show the face number
        cv2.putText(self.image, "Face #{}".format(1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in self.shape:
            cv2.circle(self.image, (x, y), 1, (0, 0, 255), -1)

        # point_between_the_eyebrows =
        mid_point = geometry_helper.middle_point_between(self.shape[constants.LEFT_EYEBROW_INNER_POINT_INDEX],
                                                         self.shape[constants.RIGTH_EYEBROW_INNER_POINT_INDEX] )
        cv2.circle(self.image, (mid_point[0], mid_point[1]), 1, (255, 0, 0), -1)
        # show the output image with the face detections + facial landmarks

        cv2.circle(self.image, (self.heightest_face_point[0], self.heightest_face_point[1] - 10), 5, (255, 155, 0), -1)

        cv2.imshow("Output", self.image)
        cv2.waitKey(0)

    def get_face_features(self):
        self.extract_face_sizes()
        return self.features