import pickle
import customtkinter
from PIL import ImageTk, Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import skimage.io as IO
import os
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import threading
import pandas as pd

from time import sleep

model = pickle.load(open(
    "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\\voting_clf.pkl", "rb"))

IMAGE_PATH = "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\OIP.jpg"

base_options = python.BaseOptions(
    model_asset_path="D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\\face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def detect(path):
    image = mp.Image.create_from_file(path)

    # print(image)
    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(
        image.numpy_view(), detection_result)

    return annotated_image, detection_result


path = "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\\0074.jpg"
image, result = detect(path)

face_blendshapes_names = [
    face_blendshapes_category.category_name for face_blendshapes_category in result.face_blendshapes[0]]
face_blendshapes_scores = [
    face_blendshapes_category.score for face_blendshapes_category in result.face_blendshapes[0]]


needed_names = [face_blendshapes_names[i] for i in [1, 2, 3, 4, 5, 9, 10, 11,
                                                    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 44, 45, 46, 47, 48, 49]]

needed_scores = [face_blendshapes_scores[i] for i in [1, 2, 3, 4, 5, 9, 10,
                                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 44, 45, 46, 47, 48, 49]]


def predict(path=None):
    global IMAGE_PATH
    global face_blendshapes_scores

    if path is None:
        path = IMAGE_PATH

    print(path)

    image, result = detect(path)

    face_blendshapes_scores = [
        face_blendshapes_category.score for face_blendshapes_category in result.face_blendshapes[0]]

    # needed_scores = [face_blendshapes_scores[i] for i in [1, 2, 3, 4, 5, 9, 10,
    #                                                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 44, 45, 46, 47, 48, 49]]

    needed_scores = [face_blendshapes_scores[i] for i in [1, 2, 3, 4, 5, 9, 10,
                                                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25]]

    # print(IMAGE_PATH)
    probas = model.predict_proba([needed_scores])

    return model.predict([needed_scores]), probas
    # if model.predict([needed_scores]) == 1:
    #     print("Engaged")
    # else:
    #     print("Not Engaged")

    # global PREDICTION
    # PREDICTION = f"{probas[0,1]*100}% Engaged"
