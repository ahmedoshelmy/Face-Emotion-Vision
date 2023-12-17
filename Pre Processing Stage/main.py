import os
from pathlib import Path
from typing import Final
import cv2

import numpy as np
import skimage as sk
from skimage.morphology import disk

import commonfunctions

def process(
        folder, 
        rem_noise=False, 
        gamma_corre=False, 
        hist_eq=False, 
        edge_det=False,
        gamma=0.5,
        median_kernel=disk(3),
        edge_det_method="sobel"
    ):
    # Read the images
    images = read_images_from_folder(folder)
    # Convert the images to grayscale (1 channel)
    images = [convert_to_grayscale(img) for img in images]
    # Show the original images
    commonfunctions.show_images(images, titles=["Original Image" for i in range(len(images))])
    # Apply the median filter
    if rem_noise:
        images = [median(img, median_kernel) for img in images]
        # Show the images after applying the median filter
        commonfunctions.show_images(images, titles=["Median Filter" for i in range(len(images))])
    # Apply histogram equalization
    if hist_eq:
        images = [histogram_equalization(img) for img in images]
        # Show the images after applying histogram equalization
        commonfunctions.show_images(images, titles=["Histogram Equalization" for i in range(len(images))])
    # Apply gamma correction
    if gamma_corre:
        images = [gamma_correction(img, gamma) for img in images]
        # Show the images after applying gamma correction
        commonfunctions.show_images(images, titles=["Gamma Correction" for i in range(len(images))])
    # Apply edge detection
    if edge_det:
        images = [detect_edges(img, edge_det_method) for img in images]
        # Show the images after applying edge detection
        commonfunctions.show_images(images, titles=["Edge Detection" for i in range(len(images))])

    return images

def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = sk.io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

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