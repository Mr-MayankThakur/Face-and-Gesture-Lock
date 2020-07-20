# source : http://dlib.net/face_recognition.py.html

import dlib
import numpy as np
from functools import partial


def recognize_face(img, face_recognizer, shape_predictor, bounding_box_coordinates):
    """
    Converts the cropped face image into 128D Feature vector
    Note: image size should be 128x128x3

    :param img: numpy array
        The image should be cropped image

    :param face_recognizer: dlib.face_recognition_model_v1
        pre-inititlized dlib_face_recognition_resnet_model_v1

    :param shape_predictor: dlib.shape_predictor
        pre-inititlized dlib shape predictor

    :param bounding_box_coordinates: [x1, y1, x2, y2]
        (x1, y1)-----
           |        |
           |        |
           -----(x2, y2)

    :return: numpy array
        128D Feature vector representing the face img provided
    """
    shape = shape_predictor(img, dlib.rectangle(*bounding_box_coordinates))

    # Let's generate the aligned image using get_face_chip
    face_chip = dlib.get_face_chip(img, shape)

    # Now we simply pass this chip (aligned image) to the api
    return np.array(face_recognizer.compute_face_descriptor(face_chip))


def initialize_models(face_recognizer_path, shape_predictor_path):
    """
    initilaizes the models which takes approx 400ms

    :param face_recognizer_path: str
        path to dlib_face_recognition_resnet_model_v1.dat

    :param shape_predictor_path: str
        path to shape_predictor_5_face_landmarks.dat

    :return: dlib.face_recognition_model_v1, dlib.shape_predictor
    """
    # defining the models and initializing them
    return dlib.face_recognition_model_v1(face_recognizer_path), dlib.shape_predictor(shape_predictor_path)


def compare_face(vec1, vec2):
    """
    Returns Euclidean Distance between two feature vectors

    :param vec1: numpy array
        Feature vector 1
    :param vec2: numpy array
        Feature vector 2
    :return: float
        Euclidean distance between the feature vectors
    """
    return np.linalg.norm(vec1, vec2)


def generate_face_compare_fn(saved_feature_vector):
    """
    Returns compare face function with hard-coding already saved feature encoding.

    :param saved_feature_vector: numpy array
    :return: partial functool wrapper for compare face function
    """
    return partial(compare_face, saved_feature_vector)
