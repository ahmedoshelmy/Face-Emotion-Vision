import os
from pathlib import Path
from typing import Final
import cv2

import numpy as np
import skimage as sk
from skimage.morphology import disk

import commonfunctions

def process(folder, rem_noise=False, gamma_corre=False, hist_eq=False):
    # Read the images
    images = read_images_from_folder(folder)
    # Convert the images to grayscale (1 channel)
    images = [convert_to_grayscale(img) for img in images]
    # Show the original images
    commonfunctions.show_images(images)
    # Apply the median filter
    images = [median(img, disk(3)) for img in images]
    # Show the images after applying the median filter
    commonfunctions.show_images(images)
    # Apply gamma correction
    images = [gamma_correction(img, 0.5) for img in images]
    # Show the images after applying gamma correction
    commonfunctions.show_images(images)
    # Apply histogram equalization
    images = [histogram_equalization(img) for img in images]
    # Show the images after applying histogram equalization
    commonfunctions.show_images(images)
    # Apply edge detection
    images = [detect_edges(img) for img in images]
    # Show the images after applying edge detection
    commonfunctions.show_images(images)

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

def detect_edges(img):
    return sk.feature.canny(img)