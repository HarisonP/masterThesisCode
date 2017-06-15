from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import geometry_helper
import face_extractor_constants as constants
from sklearn.metrics import mean_squared_error

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

        self.left_pupils = geometry_helper.middle_point_between_n([self.shape[i] for i in range(constants.LEFT_EYE_LEFTEST_POINT_INDEX,
                                                                                                constants.LEFT_EYE_LAST_POINT_INDEX)])

        self.rigth_pupils = geometry_helper.middle_point_between_n([self.shape[i] for i in range(constants.RIGHT_EYE_LEFTEST_POINT_INDEX,
                                                                                                 constants.RIGHT_EYE_LAST_POINT_INDEX)])

    def __scale_distance(self, distance):
        return round(distance * self.__get_scale_factor(), 2)

    def add_feature(self, name, val):
        self.features['features_names'].append(name)
        self.features['features_values'].append(val);

    def extract_eyes_features(self):
        distance_between_pulips = geometry_helper.point_distance(self.left_pupils, self.rigth_pupils)
        self.add_feature("Relative distance between Pupils", self.__scale_distance(distance_between_pulips))
        self.add_feature("Distance between Pupils to eye level Widht", distance_between_pulips / self.face_width)

        small_distance_between_eyes = geometry_helper.point_distance(self.shape[constants.LEFT_EYE_RIGHTEST_POINT_INDEX],
                                                                     self.shape[constants.RIGHT_EYE_LEFTEST_POINT_INDEX])

        self.add_feature("Relative inner distance between eyes", self.__scale_distance(small_distance_between_eyes))
        self.add_feature("Inner Distance between eyes to eye level Widht", small_distance_between_eyes / self.face_width)

        big_distance_between_eyes = geometry_helper.point_distance(self.shape[constants.LEFT_EYE_LEFTEST_POINT_INDEX],
                                                                   self.shape[constants.RIGHT_EYE_RIGHTEST_POINT_INDEX])

        self.add_feature("Relative outer distance between eyes", self.__scale_distance(big_distance_between_eyes))
        self.add_feature("Outer Distance between eyes to eye level Widht", big_distance_between_eyes / self.face_width)

        right_eye_height = self.__scale_distance(self.get_right_eye_height())
        left_eye_height = self.__scale_distance(self.get_left_eye_height())
        self.add_feature("Right Eye Height", right_eye_height)
        self.add_feature("Left Eye Height", left_eye_height)

        right_eye_width = self.__scale_distance(self.get_right_eye_width())
        left_eye_width = self.__scale_distance(self.get_left_eye_width())

        self.add_feature("Right Eye Width", right_eye_width)
        self.add_feature("Left Eye Width", left_eye_width)

        left_eye_size = left_eye_height * left_eye_width
        right_eye_size = right_eye_height * right_eye_width

        self.add_feature("Left Eye Size", left_eye_size)
        self.add_feature("Right Eye Size", right_eye_size)
        # TODO calc this not with the scaled sizes
        self.add_feature("Left Eye Size to Right Eye Size", left_eye_size / right_eye_size)

    def extract_face_sizes(self):

        self.add_feature("Relative Height", self.__scale_distance(self.face_height))
        self.add_feature("Relative Eye Level Width", self.__scale_distance(self.face_width))

        self.add_feature("Eye level Width To Height", self.face_width / self.face_height)

        chin_width = geometry_helper.point_distance(self.shape[constants.LEFT_CHIN_START],
                                                    self.shape[constants.RIGHT_CHIN_START])

        self.add_feature("Relative Chin Width", self.__scale_distance(chin_width))
        self.add_feature("Eye level Width to Chin Level Width", chin_width / self.face_width)
        self.add_feature("Chin Level Width to Height", chin_width / self.face_height)


        chin_height = geometry_helper.point_distance(self.shape[constants.HIGHTEST_CHIN_POINT_INDEX],
                                                     self.shape[constants.LOWEST_CHIN_POINT_INDEX])

        self.add_feature("Relative Chin height", self.__scale_distance(chin_height));
        self.add_feature("Chin Height to Height ", chin_height / self.face_height)

        chin_to_nose_end_height = geometry_helper.point_distance(self.point_between_the_eyebrows,
                                                                 self.shape[constants.LOWEST_CHIN_POINT_INDEX])

        self.add_feature("Relative Chin to nose Height", self.__scale_distance(chin_to_nose_end_height))
        self.add_feature("Chin to nose Height to Height ", chin_to_nose_end_height / self.face_height)
        self.add_feature("Chin Height to Chin to nose Height", chin_height / chin_to_nose_end_height)

        self.add_feature("Relative size of the forhead", self.__scale_distance(self.face_height - chin_to_nose_end_height))

        self.add_feature("Checkbown width", self.__scale_distance(self.face_width - chin_width))
        self.add_feature("Checkbown width to face width", (self.face_width - chin_width) / self.face_width)

    def extract_eyebrows_features(self):
        # TODO use min between heights of the eyebrow points
        left_eyebrow_height = self.shape[constants.LEFT_EYEBROW_OUTER_POINT_INDEX][1] - self.shape[constants.LEFT_EYEBROW_HEIGHEST_POINT_INDEX][1]
        right_eyebrow_height = self.shape[constants.RIGHT_EYEBROW_OUTER_POINT_INDEX][1] - self.shape[constants.RIGHT_EYEBROW_HEIGHEST_POINT_INDEX][1]
        self.add_feature("Left eyebrow height", self.__scale_distance(left_eyebrow_height))
        self.add_feature("Right eyebrow height", self.__scale_distance(right_eyebrow_height))

    def extract_nose_features(self):
        nose_width = geometry_helper.point_distance(self.shape[constants.NOSE_LEFTEST_POINT_INDEX],
                                                    self.shape[constants.NOSE_RIGHTEST_POINT_INDEX])

        self.add_feature("Nose width at nostrils", self.__scale_distance(nose_width))
        self.add_feature("Nose widht to face widths", nose_width / self.face_width)

        nose_form = (self.shape[constants.NOSE_MIDDLE_POINT_INDEX][1] - self.shape[constants.NOSE_LEFTEST_POINT_INDEX][1])
        nose_form += (self.shape[constants.NOSE_MIDDLE_POINT_INDEX][1] - self.shape[constants.NOSE_RIGHTEST_POINT_INDEX][1])

        self.add_feature("Nose form", self.__scale_distance(nose_form));

        nose_height = geometry_helper.point_distance(self.shape[constants.NOSE_MIDDLE_POINT_INDEX],
                                                    self.shape[constants.NOSE_HEIGHTES_POINT_INDEX])

        self.add_feature("Nose height", self.__scale_distance(nose_height))
        self.add_feature("Nose Size", self.__scale_distance(nose_height * nose_width))

        left_nostril_size = geometry_helper.point_distance(self.shape[constants.NOSE_MIDDLE_POINT_INDEX],
                                                           self.shape[constants.NOSE_LEFTEST_POINT_INDEX])


        right_nostril_size = geometry_helper.point_distance(self.shape[constants.NOSE_MIDDLE_POINT_INDEX],
                                                           self.shape[constants.NOSE_RIGHTEST_POINT_INDEX])

        self.add_feature("Left nostril size", self.__scale_distance(left_nostril_size))
        self.add_feature("Right nostril size", self.__scale_distance(right_nostril_size))
        self.add_feature("Left nostril size to right nostril size", left_nostril_size / right_nostril_size)

    def extract_mouth_features(self):
        lower_lip_height = geometry_helper.point_distance(self.shape[constants.MOUTH_LOWEST_POINT_INDEX],
                                                          self.shape[constants.LOWER_LIP_END_POINT_INDEX])

        upper_lip_height = geometry_helper.point_distance(self.shape[constants.MOUTH_HIGHEST_POINT_INDEX],
                                                          self.shape[constants.UPPER_LIP_END_POINT_INDEX])

        self.add_feature("Lower lip relative height", self.__scale_distance(lower_lip_height))
        self.add_feature("Upper lip relative height", self.__scale_distance(upper_lip_height))
        self.add_feature("Lower lip to upper lip heights", self.__scale_distance(lower_lip_height / upper_lip_height));

        mouth_width = geometry_helper.point_distance(self.shape[constants.MOUTH_LEFT_CORNER_INDEX],
                                                     self.shape[constants.MOUTH_RIGHT_CORNER_INDEX])

        self.add_feature("Mouth width", self.__scale_distance(mouth_width))

        left_checkbown_height = geometry_helper.point_distance(self.shape[constants.MOUTH_LEFT_CORNER_INDEX],
                                                               self.shape[constants.LEFT_EYE_LAST_POINT_INDEX - 1])
        right_checkbown_height = geometry_helper.point_distance(self.shape[constants.MOUTH_RIGHT_CORNER_INDEX],
                                                               self.shape[constants.RIGHT_EYE_LAST_POINT_INDEX - 1])

        self.add_feature("Left checkbown height", self.__scale_distance(left_checkbown_height))
        self.add_feature("Right checkbown height", self.__scale_distance(right_checkbown_height))


        left_jaw_size = geometry_helper.point_distance(self.shape[constants.LEFT_CHIN_START],
                                                        self.shape[constants.MOUTH_LEFT_CORNER_INDEX])

        right_jaw_size = geometry_helper.point_distance(self.shape[constants.RIGHT_CHIN_START],
                                                        self.shape[constants.MOUTH_RIGHT_CORNER_INDEX])

        self.add_feature("Left jaw size", self.__scale_distance(left_jaw_size))
        self.add_feature("Right jaw size", self.__scale_distance(right_jaw_size))
        self.add_feature("Left checkbown height to left jaw size", left_checkbown_height / left_jaw_size)
        self.add_feature("Right checkbown height to right jaw size", right_checkbown_height / right_jaw_size)

    def extract_skin_smoothness(self):
        (x, y, w, h) = face_utils.rect_to_bb(self.rect)
        face = self.gray[y : h + y, x : x + w]
        face_mean = face.mean()
        avr_face_matrix = np.full((len(face),len(face[0])), face_mean, dtype=np.float)

        self.add_feature("Skin Smoothenss", mean_squared_error(avr_face_matrix, face))

    def extract_symmetricity(self):
        (x, y, w, h) = face_utils.rect_to_bb(self.rect)

        left_face_rect = self.gray[y : h + y, x : x + int(w / 2)]
        rigth_face_rect = self.gray[y : h + y, x + int(w / 2) : x + w]

        if ( len(rigth_face_rect[0]) > len(left_face_rect[0])):
            rigth_face_rect = [l[1:] for l in rigth_face_rect]
        elif ( len(rigth_face_rect[0]) > len(left_face_rect[0])):
            left_face_rect = [l[1:] for l in left_face_rect]

        # print(len(left_face_rect[0]), len(rigth_face_rect[0]))
        # print(mean_squared_error(left_face_rect, rigth_face_rect))

        self.add_feature('Symmeticity',mean_squared_error(left_face_rect, rigth_face_rect) )
        # cv2.imshow("Output", left_face_rect)
        # cv2.imshow("Output", rigth_face_rect)

    def print_face_detected_with_shape(self):
        # loop over the face detections
        (x, y, w, h) = face_utils.rect_to_bb(self.rect)
        (xf, yf, wf, hf) = self.full_face
        (hair_rect_x, hair_rect_y, hair_rect_width, hair_rect_height) = self.hair_color_rect

        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(self.image, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)

        # hair color rect.
        cv2.rectangle(self.image, (hair_rect_x, hair_rect_y),
                     (hair_rect_x + hair_rect_width, hair_rect_y - hair_rect_height),
                     (0, 0, 255), 2)
        # skin color rect

        cv2.rectangle(self.image, (hair_rect_x, hair_rect_y),
                     (hair_rect_x + hair_rect_width, hair_rect_y + hair_rect_height),
                     (0, 255, 255), 2)

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

        cv2.circle(self.image, (self.left_pupils[0], self.left_pupils[1]), 3, (100, 200, 255), -1)
        cv2.circle(self.image, (self.rigth_pupils[0], self.rigth_pupils[1]), 3, (100, 200, 255), -1)

        cv2.imshow("Output", self.image)
        cv2.waitKey(0)

    def get_face_features(self):
        self.extract_face_sizes()
        self.extract_eyes_features()
        self.extract_eyebrows_features()
        self.extract_nose_features()
        self.extract_mouth_features()
        self.extract_hair_color()
        self.extract_skintone_feature()
        self.extract_symmetricity()
        self.extract_skin_smoothness()
        return self.features

    def extract_hair_color(self):
        hair_rect_width = int(self.face_width / 2)
        hair_rect_height = int(self.face_height / 5)

        hair_rect_x = int(self.heightest_face_point[0] - hair_rect_width / 2)
        hair_rect_y = int(self.heightest_face_point[1])


        rect_height = max(hair_rect_y - hair_rect_height, 0)
        rect_width = min(hair_rect_x + hair_rect_width, constants.FACE_IMAGE_SIZE)

        rect = self.gray[rect_height: hair_rect_y, hair_rect_x:rect_width]

        self.add_feature("Hair Grayscaled color", rect.mean())
        self.hair_color_rect = [hair_rect_x, hair_rect_y, hair_rect_width, hair_rect_height]

    # Must be called after extract_hair_color
    def extract_skintone_feature(self):
        (hair_rect_x, hair_rect_y, hair_rect_width, hair_rect_height) = self.hair_color_rect

        rect_height = min(hair_rect_y + hair_rect_height, constants.FACE_IMAGE_SIZE)
        rect_width = min(hair_rect_x + hair_rect_width, constants.FACE_IMAGE_SIZE)

        rect = self.gray[hair_rect_y: rect_height, hair_rect_x:rect_width]
        self.add_feature("Skintone Grayscaled color", rect.mean())

    def features_for_printinig(self):
        formatted = ""
        for i, feature_val in enumerate(self.features['features_values']):
            formatted = formatted + str(i) + " " + self.features['features_names'][i] + ": " + str(feature_val) + "\n";

        return formatted

    def print_features(self):
        print(self.features_for_printinig())
    # TODO: implement good get interface for the important features
    def get_left_eye_height(self):
        height1 = geometry_helper.point_distance(self.shape[constants.LEFT_EYE_LAST_POINT_INDEX],
                                                 self.shape[constants.LEFT_EYE_LEFTEST_POINT_INDEX + 1])

        height2 = geometry_helper.point_distance(self.shape[constants.LEFT_EYE_LAST_POINT_INDEX - 1],
                                                 self.shape[constants.LEFT_EYE_LEFTEST_POINT_INDEX + 2])

        return (height1 + height2) / 2

    def get_right_eye_height(self):
        height1 = geometry_helper.point_distance(self.shape[constants.RIGHT_EYE_LAST_POINT_INDEX],
                                                 self.shape[constants.RIGHT_EYE_LEFTEST_POINT_INDEX + 1])

        height2 = geometry_helper.point_distance(self.shape[constants.RIGHT_EYE_LAST_POINT_INDEX - 1],
                                                 self.shape[constants.RIGHT_EYE_LEFTEST_POINT_INDEX + 2])

        return (height1 + height2) / 2

    def get_left_eye_width(self):
        return geometry_helper.point_distance(self.shape[constants.LEFT_EYE_LEFTEST_POINT_INDEX],
                                              self.shape[constants.LEFT_EYE_RIGHTEST_POINT_INDEX])


    def get_right_eye_width(self):
        return geometry_helper.point_distance(self.shape[constants.RIGHT_EYE_LEFTEST_POINT_INDEX],
                                              self.shape[constants.RIGHT_EYE_RIGHTEST_POINT_INDEX])