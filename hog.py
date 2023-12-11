import os
from pathlib import Path
from typing import Final

import numpy as np
import skimage as sk

import commonfunctions

def read_image(image_path: Path) -> np.ndarray:
    """Reads an image from a file, converts it to grayscale if it's 
    in RGB, and resizes it.

    The function uses the skimage library to perform its operations. It first reads
    the image from the specified file path. If the image is in RGB format, it is 
    converted to grayscale. Finally, the image is resized to a fixed size of 128x64 
    pixels.

    Parameters:
    - image_path (str): The file path of the image to be read.

    Returns:
    np.ndarray: The processed image as a NumPy array. The image will be in grayscale
                and of size 128x64 pixels.
    """
    img: np.ndarray = sk.io.imread(image_path)
    if len(img.shape) == 3:
        img = sk.color.rgb2gray(img)
    img = sk.transform.resize(img, (128, 64))

    return img

def compute_gradient(img: np.ndarray, grad_filter: np.ndarray) -> np.ndarray:
    """Computes the gradient of an image using a convolution with a gradient filter.

    This function applies a specified gradient filter to the input image to compute its gradient.
    It handles images of arbitrary size, padding the original image before applying the filter
    and then cropping the result back to the original image size.

    Parameters:
    - img (np.ndarray): The input image as a 2D NumPy array.
    - grad_filter (np.ndarray): The gradient filter as a 2D NumPy array, typically a small kernel.

    Returns:
    np.ndarray: The gradient image as a 2D NumPy array, of the same size as the input image.
    """
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
    """Computes the magnitude of the gradient from the horizontal and vertical gradient components.

    This function calculates the gradient magnitude at each pixel of an image, which is
    useful in edge detection algorithms. The gradient magnitude is calculated as the square root
    of the sum of the squares of the horizontal and vertical gradients.

    Parameters:
    - horizontal_gradient (np.ndarray): A 2D NumPy array representing the horizontal gradient of the image.
    - vertical_gradient (np.ndarray): A 2D NumPy array representing the vertical gradient of the image.

    Hints:
        - Use np.power() to square the gradients.
        - Use np.sqrt() to compute the square root of the sum of the squares.

    Returns:
    np.ndarray: A 2D NumPy array representing the gradient magnitude of the image.
    """
    return np.sqrt(np.power(horizontal_gradient, 2) + np.power(vertical_gradient, 2))

def compute_gradient_direction(horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:
    """Computes the direction of the gradient from the horizontal and vertical gradient components.

    This function calculates the angle of the gradient at each pixel of an image. The gradient direction
    is given by the arctangent of the vertical gradient divided by the horizontal gradient. A small value 
    (1e-5) is added to the horizontal gradient to avoid division by zero. The resulting angles are 
    converted from radians to degrees and normalized to the range [0, 180) degrees.

    Parameters:
    - horizontal_gradient (np.ndarray): A 2D NumPy array representing the horizontal gradient of the image.
    - vertical_gradient (np.ndarray): A 2D NumPy array representing the vertical gradient of the image.

    Hints:
        - Use np.arctan() to compute the arctangent of the gradients.
        - Use np.rad2deg() to convert the angles from radians to degrees.
        - Use np.mod() or % to normalize the angles to the range [0, 180) degrees.

    Returns:
    np.ndarray: A 2D NumPy array representing the gradient direction of the image, in degrees.
    """
    rad = np.arctan(vertical_gradient/(horizontal_gradient + 1e-5))
    deg = np.rad2deg(rad) % 180
    return deg

def find_nearest_bins(curr_direction: float, hist_bins: np.ndarray) -> (int, int):
    """TODO: Implement this function to find the two histogram bins nearest to the current direction.

    Parameters:
    - curr_direction (float): The current gradient direction at a pixel in the cell.
    - hist_bins (np.ndarray): An array of histogram bin edge values.

    Returns:
    - (int, int): A tuple containing the indices of the two nearest bins.

    Hints for implementation:
    1. Calculate the absolute difference between the current direction and each histogram bin.
    2. Handle edge cases where the direction is less than the first bin or greater than the last bin.
    3. In other cases, find the bin with the minimum difference (the closest bin).
    4. Determine the second closest bin considering the circular nature of the histogram.
    """
    diff = np.abs(hist_bins - curr_direction)
    if curr_direction < hist_bins[0]:
        if(abs(curr_direction - hist_bins[0]) <= abs(180 + curr_direction - hist_bins[-1])):
            return (0, len(hist_bins) - 1)
        else:
            return (len(hist_bins) - 1, 0)
    elif curr_direction > hist_bins[-1]:
        if(abs(curr_direction - hist_bins[-1]) <= abs(180 + hist_bins[0] - curr_direction)):
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
    """TODO: Implement this function to update the histogram bins based on the current direction and magnitude.

    Parameters:
    - HOG_cell_hist (np.ndarray): The histogram array to be updated.
    - curr_direction (float): The current gradient direction at a pixel in the cell.
    - curr_magnitude (float): The current gradient magnitude at a pixel in the cell.
    - first_bin_idx (int): The index of the first nearest bin.
    - second_bin_idx (int): The index of the second nearest bin.
    - hist_bins (np.ndarray): An array of histogram bin edge values.

    Hints for implementation:
    1. Calculate the size of each bin in the histogram.
    2. Compute the proportional contribution to the first and second nearest bins.
    3. Update the corresponding bins in the histogram with these contributions.
    """
    bin_size = hist_bins[1] - hist_bins[0]
    first_bin_contribution = (abs(curr_direction - hist_bins[second_bin_idx]) / bin_size) * curr_magnitude
    second_bin_contribution = (abs(curr_direction - hist_bins[first_bin_idx]) / bin_size) * curr_magnitude
    HOG_cell_hist[first_bin_idx] += first_bin_contribution
    HOG_cell_hist[second_bin_idx] += second_bin_contribution


def calculate_histogram_per_cell(
        cell_direction: np.ndarray, 
        cell_magnitude: np.ndarray, 
        hist_bins: np.ndarray
    ) -> np.ndarray:
    """TODO: Implement this function to calculate the Histogram of Oriented Gradients (HOG) for a single cell.

    Parameters:
    - cell_direction (np.ndarray): A 2D array of gradient directions for each pixel in the cell.
    - cell_magnitude (np.ndarray): A 2D array of gradient magnitudes for each pixel in the cell.
    - hist_bins (np.ndarray): A 1D array defining the bin edges for the histogram.

    Returns:
    - np.ndarray: A 1D array representing the computed HOG for the cell.

    Hints for implementation:
    1. Initialize a zero-filled numpy array for the histogram.
    2. Iterate over each pixel in the cell.
    3. For each pixel, find the nearest bins using the 'find_nearest_bins' function.
    4. Update the histogram using the 'update_histogram_bins' function.
    5. Return the final histogram array.
    """
    histogram = np.zeros(len(hist_bins))
    for i in range(cell_direction.shape[0]):
        for j in range(cell_direction.shape[1]):
            first_bin_idx, second_bin_idx = find_nearest_bins(cell_direction[i][j], hist_bins)
            update_histogram_bins(histogram, cell_direction[i][j], cell_magnitude[i][j], first_bin_idx, second_bin_idx, hist_bins)
    return histogram

def compute_hog_features(image: np.ndarray) -> np.ndarray:
    """Computes the Histogram of Oriented Gradients (HoG) features for an image.

    The function first computes the horizontal and vertical gradients of the image.
    It then calculates the gradient magnitude and direction at each pixel. The image is
    divided into cells, and for each cell, a histogram of gradient directions is computed,
    weighted by the gradient magnitudes. These histograms are then concatenated to form the
    final HOG feature descriptor of the image.

    Parameters:
    image (np.ndarray): A 2D NumPy array representing the input image.

    Returns:
    np.ndarray: A 1D NumPy array representing the HOG feature descriptor of the image.
    """
    # Define gradient masks
    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1], [0], [1]])

    # Compute gradients
    horizontal_gradient = compute_gradient(image, horizontal_mask)
    vertical_gradient = compute_gradient(image, vertical_mask)

    # Compute gradient magnitude and direction
    grad_magnitude = compute_gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = compute_gradient_direction(horizontal_gradient, vertical_gradient)

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
            histogram_16x16_normalized = histogram_16x16 / (np.linalg.norm(histogram_16x16) + 1e-5)
            features_list.append(histogram_16x16_normalized)

    return np.concatenate(features_list, axis=0)

def get_feature_list_from_paths(image_fns: list[Path]) -> list[np.ndarray]:
    """Reads a list of images from their file paths and computes their HoG features.

    This function takes a list of file paths pointing to images, reads each image, 
    and computes the Histogram of Oriented Gradients (HoG) features for each image. 
    It uses the 'read_image' function to read and preprocess the images and the 
    'compute_hog_features' function to compute the HoG features. 

    Parameters:
    image_fns (list[Path]): A list of Paths where each one is a Path to an image.

    Returns:
    list[np.ndarray]: A list of NumPy arrays, where each array represents the HoG feature
        descriptor of a corresponding image in the input list.
    """
    images_list = [read_image(image_fn) for image_fn in image_fns]
    hog_features_list = [compute_hog_features(image) for image in images_list]
    return hog_features_list

def calculate_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Calculates the mean squared distance between two feature vectors.

    This function computes the mean squared difference between each corresponding element
    of the two input feature vectors. This distance measure is useful for comparing the 
    similarity between two sets of features in many applications, such as machine learning.

    Parameters:
        - feat1 (np.ndarray): The first feature vector as a NumPy array.
        - feat2 (np.ndarray): The second feature vector as a NumPy array.

    Returns:c
        - float: The mean squared distance between the two feature vectors.
    """
    return np.mean((feat1 - feat2) ** 2)