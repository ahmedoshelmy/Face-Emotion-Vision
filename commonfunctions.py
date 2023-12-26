

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb

import cv2


# Edges
from skimage.filters import sobel_h, sobel, sobel_v, roberts, prewitt

# Show the figures / plots inside the notebook


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def walk_through_dir(dir_path):

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


#################################################### MediaPipe ##############################################################

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


def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [
        face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [
        face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores,
                  label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(),
                 patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

    ####################################### Haar #########################################


def process_image(path):
    img = cv2.imread(path)
    height = 224
    if img.any() == None:
        return None, False
    if img.all() != None:
        width = img.shape[1]*height/img.shape[0]
        img = cv2.resize(img, (int(width), height), None, 0.5,
                         0.5, interpolation=cv2.INTER_AREA)
        return img, True
    else:
        return None, False


eye_glasses_haarCascade = cv2.CascadeClassifier(
    'D:\GitHub Repos\Face-Emotion-Vision\opencv\haarcascade_eye_tree_eyeglasses.xml')
face_haarCascade = cv2.CascadeClassifier(
    "D:\GitHub Repos\Face-Emotion-Vision\opencv\haarcascade_frontalface_default.xml")
eye_haarCascade = cv2.CascadeClassifier(
    "D:\GitHub Repos\Face-Emotion-Vision\opencv\haarcascade_eye.xml")


def get_faces_with_eyes(image):
    copy = np.copy(image)
    faces = []
    colorfaces = []
    response = []
    faces_with_edges = []

    if image is None:
        return faces, colorfaces, faces_with_edges, response

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_haarCascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in face_coordinates:
        face = gray_img[y:y + h, x:x + w]
        colorFace = copy[y:y + h, x:x + w]

        eye_coordinates = eye_haarCascade.detectMultiScale(face, 1.1)
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      color=(255, 0, 0), thickness=2)

        eye_coordinates2 = eye_glasses_haarCascade.detectMultiScale(face, 1.1)

        if max(len(eye_coordinates), len(eye_coordinates2)) < 2:
            response.append(
                f"just detected the face with {max(len(eye_coordinates), len(eye_coordinates2))} eyes")
            faces_with_edges.append(face)
            colorfaces.append(colorFace)
            faces.append(face)
            continue

        final_eye_coordinates = eye_coordinates

        if len(eye_coordinates2) > len(eye_coordinates):
            final_eye_coordinates = eye_coordinates2

        for (x2, y2, w2, h2) in final_eye_coordinates:
            eye_center = (x+x2 + w2 // 2, y+y2 + h2 // 2)
            eye_radius = int(round((w2 + h2) * 0.25))
            cv2.circle(image, center=eye_center, radius=eye_radius,
                       color=(255, 255, 0))

        faces_with_edges.append(image[y:y + h, x:x + w, :])
        faces.append(face)
        colorfaces.append(colorFace)
        response.append("Success")

    return faces, colorfaces, faces_with_edges, response
