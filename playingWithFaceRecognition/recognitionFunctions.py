from skimage.morphology import disk
import skimage as sk
from typing import Final
from pathlib import Path
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

models = {
    1: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\KNN_final.pkl", "rb")),
    2: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\KNN_hog.pkl", "rb")),
    3: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\KNN_lbp.pkl", "rb")),
    4: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\rnd_final.pkl", "rb")),
    5: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\rnd_hog.pkl", "rb")),
    6: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\rnd_lbp.pkl", "rb")),
    7: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\SVM_final.pkl", "rb")),
    8: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\SVM_hog.pkl", "rb")),
    9: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\SVM_lbp.pkl", "rb")),
    10: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\voting_clf_hard_final.pkl", "rb")),
    11: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\voting_clf_hard_hog.pkl", "rb")),
    12: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\voting_clf_hard_lbp.pkl", "rb")),
    13: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\voting_clf_soft_final.pkl", "rb")),
    14: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\voting_clf_soft_hog.pkl", "rb")),
    15: pickle.load(open(
        "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\voting_clf_soft_lbp.pkl", "rb")),
}


IMAGE_PATH = "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\OIP.jpg"

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def processImage(
    image,
    rem_noise=False,
    gamma_corre=False,
    hist_eq=False,
    edge_det=False,
    resize=False,
    showProgress=False,
    gamma=0.5,
    median_kernel=disk(3),
    edge_det_method="sobel",
    target_size=(48, 48),
):
    # Read the images
    img = np.copy(image)

    # Convert the images to grayscale (1 channel)
    img = convert_to_grayscale(img)

    # Resize the images
    if resize:
        img = resize_image(img, target_size)

    # Apply the median filter
    if rem_noise:
        img = median(img, median_kernel)

    # Apply histogram equalization
    if hist_eq:
        img = histogram_equalization(img)

    # Apply gamma correction
    if gamma_corre:
        img = gamma_correction(img, gamma)

    # Apply edge detection
    if edge_det:
        img = detect_edges(img, edge_det_method)

    return img


# def processImg(
#     image,
#     rem_noise=False,
#     gamma_corre=False,
#     hist_eq=False,
#     edge_det=False,
#     gamma=0.5,
#     median_kernel=disk(3),
#     edge_det_method="sobel"
# ):

#     img = np.copy(image)
#     # Convert the images to grayscale (1 channel)
#     img = convert_to_grayscale(img)

#     # Apply the median filter
#     if rem_noise:
#         img = median(img, median_kernel)

#     # Apply histogram equalization
#     if hist_eq:
#         img = histogram_equalization(img)

#     # Apply gamma correction
#     if gamma_corre:
#         img = gamma_correction(img, gamma)
#         # Show the images after applying gamma correction

#     # Apply edge detection
#     if edge_det:
#         img = detect_edges(img, edge_det_method)
#         # Show the images after applying edge detection

#     return img


def convert_to_grayscale(img):
    return sk.color.rgb2gray(img)


def median(img, kernel):
    return sk.filters.median(img, kernel)


def gamma_correction(img, gamma):
    return sk.exposure.adjust_gamma(img, gamma=gamma)


def histogram_equalization(img):
    return sk.exposure.equalize_hist(img)


def detect_edges(img, method="sobel"):
    if method == "sobel":
        return sk.filters.sobel(img)
    elif method == "canny":
        return sk.feature.canny(img)
    else:
        return sk.filters.sobel(img)


def resize_image(img, target_size):
    return sk.transform.resize(img, target_size)


def compute_gradient(img: np.ndarray, grad_filter: np.ndarray) -> np.ndarray:

    ts = grad_filter.shape[0]

    new_img = np.zeros((img.shape[0] + ts - 1, img.shape[1] + ts - 1))

    new_img[int((ts-1)/2.0):img.shape[0] + int((ts-1)/2.0),
            int((ts-1)/2.0):img.shape[1] + int((ts-1)/2.0)] = img

    result = np.zeros((new_img.shape))

    for r in np.uint16(np.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
        for c in np.uint16(np.arange((ts-1)/2.0, img.shape[1]+(ts-1)/2.0)):
            curr_region = new_img[r-np.uint16((ts-1)/2.0):r+np.uint16((ts-1)/2.0)+1,
                                  c-np.uint16((ts-1)/2.0):c+np.uint16((ts-1)/2.0)+1]
            curr_result = curr_region * grad_filter
            score = np.sum(curr_result)
            result[r, c] = score

    result_img = result[np.uint16((ts-1)/2.0):result.shape[0]-np.uint16((ts-1)/2.0),
                        np.uint16((ts-1)/2.0):result.shape[1]-np.uint16((ts-1)/2.0)]

    return result_img


def compute_gradient_magnitude(horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:

    return np.sqrt(np.power(horizontal_gradient, 2) + np.power(vertical_gradient, 2))


def compute_gradient_direction(horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:

    rad = np.arctan(vertical_gradient/(horizontal_gradient + 1e-5))
    deg = np.rad2deg(rad) % 180
    return deg


def find_nearest_bins(curr_direction: float, hist_bins: np.ndarray) -> (int, int):

    diff = np.abs(hist_bins - curr_direction)
    if curr_direction < hist_bins[0]:
        if (abs(curr_direction - hist_bins[0]) <= abs(180 + curr_direction - hist_bins[-1])):
            return (0, len(hist_bins) - 1)
        else:
            return (len(hist_bins) - 1, 0)
    elif curr_direction > hist_bins[-1]:
        if (abs(curr_direction - hist_bins[-1]) <= abs(180 + hist_bins[0] - curr_direction)):
            return (len(hist_bins) - 1, 0)
        else:
            return (0, len(hist_bins) - 1)
    else:
        minidx = np.argmin(diff)
        if curr_direction <= hist_bins[minidx]:
            return (minidx, minidx - 1)
        elif curr_direction > hist_bins[minidx]:
            return (minidx, minidx + 1)


def update_histogram_bins(
    HOG_cell_hist: np.ndarray,
    curr_direction: float,
    curr_magnitude: float,
    first_bin_idx: int,
    second_bin_idx: int,
    hist_bins: np.ndarray
) -> None:

    bin_size = hist_bins[1] - hist_bins[0]
    first_bin_contribution = (
        abs(curr_direction - hist_bins[second_bin_idx]) / bin_size) * curr_magnitude
    second_bin_contribution = (
        abs(curr_direction - hist_bins[first_bin_idx]) / bin_size) * curr_magnitude
    HOG_cell_hist[first_bin_idx] += first_bin_contribution
    HOG_cell_hist[second_bin_idx] += second_bin_contribution


def calculate_histogram_per_cell(
    cell_direction: np.ndarray,
    cell_magnitude: np.ndarray,
    hist_bins: np.ndarray
) -> np.ndarray:

    histogram = np.zeros(len(hist_bins))
    for i in range(cell_direction.shape[0]):
        for j in range(cell_direction.shape[1]):
            first_bin_idx, second_bin_idx = find_nearest_bins(
                cell_direction[i][j], hist_bins)
            update_histogram_bins(
                histogram, cell_direction[i][j], cell_magnitude[i][j], first_bin_idx, second_bin_idx, hist_bins)
    return histogram


def compute_hog_features(image: np.ndarray) -> np.ndarray:

    # Define gradient masks
    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1], [0], [1]])

    # Compute gradients
    horizontal_gradient = compute_gradient(image, horizontal_mask)
    vertical_gradient = compute_gradient(image, vertical_mask)

    # Compute gradient magnitude and direction
    grad_magnitude = compute_gradient_magnitude(
        horizontal_gradient, vertical_gradient)
    grad_direction = compute_gradient_direction(
        horizontal_gradient, vertical_gradient)

    # Define histogram bins
    hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    # Compute histograms for each cell
    cells_histogram = np.zeros((16, 8, 9))
    for r in range(0, grad_magnitude.shape[0], 8):
        for c in range(0, grad_magnitude.shape[1], 8):
            cell_direction = grad_direction[r:r+8, c:c+8]
            cell_magnitude = grad_magnitude[r:r+8, c:c+8]
            cells_histogram[int(r / 8), int(c / 8)] = calculate_histogram_per_cell(
                cell_direction, cell_magnitude, hist_bins)

    # Normalize and concatenate histograms
    features_list = []
    for r in range(cells_histogram.shape[0] - 1):
        for c in range(cells_histogram.shape[1] - 1):
            histogram_16x16 = np.reshape(cells_histogram[r:r+2, c:c+2], (36,))
            histogram_16x16_normalized = histogram_16x16 / \
                (np.linalg.norm(histogram_16x16) + 1e-5)
            features_list.append(histogram_16x16_normalized)

    return np.concatenate(features_list, axis=0)


def get_feature_list_from_paths(imgs) -> list[np.ndarray]:

    resizedMss = [sk.transform.resize(img, (128, 64)) for img in imgs]

    hog_features_list = [compute_hog_features(image) for image in resizedMss]
    return hog_features_list


def calculate_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    return np.mean((feat1 - feat2) ** 2)


def ComputeLBP(x, y, arr):
    f = s = -1
    value = 0
    height, width = arr.shape
    for i in range(8):
        if (i == 3 or i == 7):
            f = 0
        elif (i > 3):
            f = 1
        if (i == 1 or i == 5):
            s = 0
        elif (i > 1 and i < 5):
            s = 1
        else:
            s = -1
        if (x+f > -1 and x+f < height and y+s > -1 and y+s < width):
            value += pow(2, 7-i) if arr[x+f, y+s] > arr[x, y] else 0
    return value


def LBP(arr):
    tempArr = np.copy(arr)
    hight, width = tempArr.shape
    for i in range(hight):
        for j in range(width):
            tempArr[i, j] = ComputeLBP(i, j, tempArr)
    return tempArr


def get_LBP_features(imagesList):
    imList = []
    for image, index in zip(imagesList, range(len(imagesList))):
        imList.append(LBP(image)/255)
        # print(index)

    return imList


def predict(img, num):

    hog: list[np.ndarray] = get_feature_list_from_paths([img])
    lbp = get_LBP_features([img])
    lbp = [np.ravel(i) for i in lbp]

    final = []

    for l, h in zip(lbp, hog):
        final.append(np.concatenate([l, h], axis=0))

    needed_model = models[num]

    if num == 1 or num == 4 or num == 7 or num == 10 or num == 13:
        return needed_model.predict(final)
    elif num == 2 or num == 5 or num == 8 or num == 11 or num == 14:
        return needed_model.predict(hog)
    elif num == 3 or num == 6 or num == 9 or num == 12 or num == 15:
        return needed_model.predict(lbp)
    

