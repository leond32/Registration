import numpy as np
import cv2

def median_filter(image, kernel_size=11):
    return cv2.medianBlur(image, kernel_size)

def remove_black_border(input_array, threshold=40):
    # Ensure input is a 2D grayscale image
    if len(input_array.shape) != 2:
        raise ValueError("Input array must be a 2D grayscale image")
    median_array = median_filter(input_array, kernel_size=21)
    # Create a mask of the pixels that are not black (or within the threshold)
    mask = median_array > threshold

    # Find the coordinates of non-black pixels
    coords = np.argwhere(mask)

    # If there are no non-black pixels, return the original array
    if coords.size == 0:
        return input_array

    # Get the min and max coordinates for cropping
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the array to these coordinates
    cropped_array = input_array[y_min:y_max+1, x_min:x_max+1]

    return cropped_array


