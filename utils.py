import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from skimage import transform
import os
from tqdm import tqdm
import math

def load_mnist():
    """
    Definition:
    Loads MNIST data and concatenates train and test sets into singular
    outputs

    Returns: 
    X (np.array) : Array of 28 x 28 images
    Y (np.array) : Class labels withe expanded dim.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate([x_train, x_test], axis=0)
    Y = np.concatenate([y_train, y_test], axis=0)
    Y = np.expand_dims(Y, -1)

    return X, Y

def to_center_coordinates(x_min, y_min, x_max, y_max):
    """
    Definition:
    Converts top-left and bottom-right coordinates of a bounding box to
    center coordinates with width and height.

    Parameters:
    x_min (float) : X coordinate of top-left corner
    y_min (float) : Y coordinate of top-left corner
    x_max (float) : X coordinate of bottom-right corner
    y_max (float) : X coordinate of bottom-right corner

    Returns:
    list: [center_x, center_y, width, height]
    """
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    return [center_x, center_y, width, height]


def to_corner_coordinates(center_x, center_y, width, height):
    """
    Definition:
    Converts center coordinates with width and height to
    top-left and bottom-right coordinates of a bounding box.

    Parameters:
    center_x (float) : X coordinates of the center 
    center_y (float) : Y coordinates of the center 
    width (float)    : Width of the bounding box
    height (float)   : Height of the bounding box

    Returns:
    list: [x_min, y_min, x_max, y_max]
    """
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    
    return [x_min, y_min, x_max, y_max]

def find_bbox(object, 
              corner_coordinates=True):
    """
    Definition:
    Locates the bounding box pixel coordinates of the non-zero pixels
    in the image. Can return in corner coordinates or center, width, height.

    Parameters:
    object (np.array)         : 2D image array
    corner_coordinates (bool) : Specifies output format
    
    Returns:
    np.array : [x_min, y_min, x_max, y_max]
    or
    np.array : [center_x, center_y, width, height]
    """
    # Find the non-zero pixels
    coords = np.column_stack(np.where(object > 0)).astype(float)
    coords[:, 0] = coords[:, 0] / object.shape[0]
    coords[:, 1] = coords[:, 1] / object.shape[1]
    # Find the top_left bottom_right bounding box coordinates
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # return specified label-type (corner coordinates) 
    if corner_coordinates:
        return np.array([x_min, y_min, x_max, y_max])
    # (center-coordinate, width, height)
    else:
        return np.array(to_center_coordinates(x_min, y_min, x_max, y_max))
    