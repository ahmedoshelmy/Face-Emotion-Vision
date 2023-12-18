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

model = pickle.load(open(
    "D:\GitHub Repos\Face-Emotion-Vision\playingWithFaceRecognition\\rnd.pkl", "rb"))

IMAGE_PATH = "D:\GitHub Repos\Face-Emotion-Vision\studentEngagementModel\OIP.jpg"

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def processImg(
    image,
    rem_noise=False,
    gamma_corre=False,
    hist_eq=False,
    edge_det=False,
    gamma=0.5,
    median_kernel=disk(3),
    edge_det_method="sobel"
):

    img = np.copy(image)
    # Convert the images to grayscale (1 channel)
    img = convert_to_grayscale(img)

    # Apply the median filter
    if rem_noise:
        img = median(img, median_kernel)

    # Apply histogram equalization
    if hist_eq:
        img = histogram_equalization(img)

    # Apply gamma correction
    if gamma_corre:
        img = gamma_correction(img, gamma)
        # Show the images after applying gamma correction

    # Apply edge detection
    if edge_det:
        img = detect_edges(img, edge_det_method)
        # Show the images after applying edge detection

    return img


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







def predict(img):

    hog: list[np.ndarray] = get_feature_list_from_paths([img])

   
    return model.predict(hog)
    
