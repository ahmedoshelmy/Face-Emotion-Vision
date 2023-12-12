#############################################needs modifications
#just for testing





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


import pandas as pd

model = pickle.load(open(
    "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\\random_forest.pkl", "rb"))

IMAGE_PATH = "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\OIP.jpg"

base_options = python.BaseOptions(
    model_asset_path="D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\\face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)

detector = vision.FaceLandmarker.create_from_options(options)


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


def imageChange_callback():
    img2 = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH),
                                  size=(450, 450))

    image_label.configure(image=img2)


def upload_file():
    file_path = customtkinter.filedialog.askopenfilename(title="Select an image", filetypes=[(
        'all files', '*.*'), ('png files', '*.png'), ('jpeg files', '*.jpeg'), ('jpg files', '*.jpg')])

    global IMAGE_PATH
    IMAGE_PATH = file_path
    imageChange_callback()


def preprocess_image():
    orig_img = cv2.imread(IMAGE_PATH)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    orig_img = cv2.resize(orig_img, (224, 224))
    orig_img = orig_img/255

    image = np.expand_dims(orig_img, axis=0)

    return image


def predict():
    global IMAGE_PATH
    global face_blendshapes_scores

    image, result = detect(IMAGE_PATH)

    face_blendshapes_scores = [
        face_blendshapes_category.score for face_blendshapes_category in result.face_blendshapes[0]]

    needed_scores = [face_blendshapes_scores[i] for i in [1, 2, 3, 4, 5, 9, 10,
                                                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 44, 45, 46, 47, 48, 49]]

    print(IMAGE_PATH)
    if model.predict([needed_scores]) == 1:
        print("Engaged")
    else:
        print("Not Engaged")


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


app = customtkinter.CTk()
app.geometry("1080x720")

app.title("Is student Engaged?")


title = customtkinter.CTkLabel(app, text="Predict Weather the student is Engaged or Not",
                               font=customtkinter.CTkFont(family='Arial', size=20, weight='bold'))
title.pack(padx=10, pady=(40, 80))

mainFrame = customtkinter.CTkFrame(app, fg_color="transparent")
mainFrame.pack(fill="x", expand=True, anchor="n")


my_image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH),
                                  size=(450, 450))


imageButtonsFrame = customtkinter.CTkFrame(
    mainFrame, width=450, height=500, fg_color="transparent")
imageButtonsFrame.pack_propagate(False)
imageButtonsFrame.pack(side="right", padx=(10))

image_label = customtkinter.CTkLabel(
    imageButtonsFrame, image=my_image, text="")  # display image with a CTkLabel
image_label.pack()

buttonsFrame = customtkinter.CTkFrame(
    imageButtonsFrame, fg_color="transparent")
buttonsFrame.pack(pady=10, anchor="n")

uploadButton = customtkinter.CTkButton(
    buttonsFrame, text="Upload an image", command=upload_file)
uploadButton.pack(side="left", padx=(5, 5))

predictButton = customtkinter.CTkButton(
    buttonsFrame, text="Predict", command=predict)
predictButton.pack(side="right", padx=(5, 5))


app.mainloop()
