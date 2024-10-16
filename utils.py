import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from skimage import transform
import os
from tqdm import tqdm
import math
import copy

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
    object (np.array)         : 2D MNIST image array
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
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # return specified label-type (corner coordinates) 
    if corner_coordinates:
        return np.array([x_min, y_min, x_max, y_max])
    # (center-coordinate, width, height)
    else:
        return np.array(to_center_coordinates(x_min, y_min, x_max, y_max))
    
def generate_noisy_image(image_size=(128, 128),
                         noise_intensity = 128):
    """
    Definition:
    Creates an background image with random noise and returns as a numpy array.

    Parameters:
    image_size ((int , int)) : the set height and width of returned image
    noise_intensity (int)    : the scalar intensity value for the background noise

    Returns:
    random_image (np.array) : 2D np.array with shape size and random values 
                              from 0 to intensity
    """
    # Generate a random array of shape size with values between 0 and 255
    random_image = np.random.randint(0, noise_intensity, image_size, dtype=np.uint8)
    return random_image

def choose_regions_to_populate(max_objects = 8,
                               grid_rows = 4,
                               grid_cols = 4):
    """
    Definition:
    Randomly chooses up to the max_objects regions based on the allowable grid

    Parameters:
    max_objects (int) : upper limit of chosen regions
    grid_rows (int)   : number of rows the image is broken down into
    grid_cols (int)   : number of cols the image is broken down into
    
    Returns:
    regions (np.array) : 1D array of random values of length between 0 and max_objects
    """
    num_objects = np.random.choice(range(max_objects), 1)
    regions = np.random.choice(range(1, grid_cols * grid_rows + 1), 
                               num_objects, 
                               replace=False)
    return regions
    
def grab_x_bbox_region(object, 
                       label,
                       corner_coordinates=True):
    """
    Definition
    Returns the object bbox subsection

    Parameters:
    object (np.array)         : 2D MNIST image array
    label (np.array)          : associated class and bbox label with input object
    corner_coordinates (bool) : defines what bbox coordinate system is in use
    
    Returns:
    object (np.array) : bbox subsection of MNIST object
    """   
    if corner_coordinates:
        x_min, y_min, x_max, y_max = label[1:]
    else:
        center_x, center_y, width, height = label[1:]
        x_min, y_min, x_max, y_max = to_corner_coordinates(center_x, center_y, width, height)

    # Upscale from normalized coordinates
    x_min = int(np.floor(x_min * object.shape[0]))
    y_min = int(np.floor(y_min * object.shape[1]))
    x_max = int(np.ceil(x_max * object.shape[0]))
    y_max = int(np.ceil(y_max * object.shape[1]))

    return object[y_min:y_max, x_min:x_max]

def add_object_to_image(image,
                        region_of_interest,
                        object,
                        label,
                        object_num,
                        grid_rows = 4,
                        grid_cols = 4,
                        scale_value = 1,
                        corner_coordinates=True):
    """
    Definition:
    Overlays the object onto the image centered in the region of interest. The object may be
    scaled up in size.

    Parameters:
    image (np.array)          : current image being created
    region_of_interest (int)  : region num of image grid to center object in
    object (np.array)         : 2D MNIST image array
    label (np.array)          : associated class and bbox label with input object
    object_num (int)          : nth object being added to image (used for tracking in wrapper function)
    grid_rows (int)           : number of rows the image is broken down into
    grid_cols (int)           : number of cols the image is broken down into
    scale_value (float)       : scaler for object size
    corner_coordinates (bool) : defines what bbox coordinate system is in use

    Returns:
    image (np.array)    : updated image with new overlayed object
    added_object (dict) : dict with class, true object coordinates on image, and normalized coordinates
    """
    # Determine the size of a region based on chosen image grid
    region_x = int(image.shape[1] / grid_rows)
    region_y = int(image.shape[0] / grid_cols)
    # Randomly choose a center point within the size of one grid region
    y_center = np.random.randint(0, region_y + 1, 1)
    x_center = np.random.randint(0, region_x + 1, 1)
    # Offset center to the chosen region of interest 

    y_center += ((region_of_interest - 1)// grid_cols) * region_y
    x_center += ((region_of_interest - 1) % grid_cols) * region_x
    
    # Grab the bbox area of the input object for overlaying onto the image
    bbox_object = grab_x_bbox_region(object,
                                     label,
                                     corner_coordinates=corner_coordinates)
   
    m, n = bbox_object.shape

    # Find all edge regions on graph such that no scaling will happen in these regions due to potential to be placed outside image
    edge_regions = list(range(1, grid_cols + 1))
    edge_regions += list(range(grid_cols + 1, (grid_rows - 1) * grid_cols , grid_cols))
    edge_regions += list(range(grid_cols * 2, ((grid_rows - 1) * grid_cols  + 1), grid_cols))
    edge_regions += list(range((grid_rows - 1) * grid_cols + 1, grid_rows * grid_cols + 1))

    if scale_value > min(region_x / n, region_y / m) * 2 :
        scale_value = (math.floor(min(region_x / n, region_y / m) * 20) / 10) - 0.5
    
    if scale_value != 1: # and region_of_interest not in edge_regions:
        bbox_object = transform.rescale(bbox_object, 
                                        scale_value,
                                        mode = 'constant',
                                        cval = 0,
                                        anti_aliasing=False,
                                        preserve_range=True)
    # Find new size of bbox object
    m, n = bbox_object.shape
    
    # Calculate image location for bbox object
    y_min = int((x_center[0] - n // 2))
    x_min = int((y_center[0] - m // 2))
    y_max = int(y_min + n)
    x_max = int(x_min + m)

    if y_min < 0:
        up_shift = 0 - y_min
        y_min, y_max = y_min + up_shift, y_max + up_shift
    elif y_max >= image.shape[1]:
        down_shift = (y_max - image.shape[1]) + 1
        y_min, y_max = y_min - down_shift, y_max - down_shift  

    if x_min < 0:
        r_shift = 0 - x_min
        x_min, x_max = x_min + r_shift, x_max + r_shift
    elif x_max >= image.shape[0]:
        l_shift = (x_max - image.shape[0]) + 1
        x_min, x_max = x_min - l_shift, x_max - l_shift

    image[x_min:x_max, y_min:y_max] = np.maximum(image[x_min:x_max, y_min:y_max], bbox_object)

    N, M = image.shape

    if corner_coordinates:
        added_object = {object_num : {'class' : int(label[0]),
                                      'bbox_true' : [y_min,     x_min,     y_max,     x_max    ],
                                      'bbox_norm' : [y_min / M, x_min / N, y_max / M, x_max / N]}
                        }
    else:
        center_x, center_y, width, height = to_center_coordinates(y_min, x_min, y_max, x_max)
        added_object = {object_num : {'class' : int(label[0]),
                                      'bbox_true' : [center_x,     center_y,     width,     height     ],
                                      'bbox_norm' : [center_x / M, center_y / N, width / M, height / N]}}

    return image, added_object

def check_overlap(bbox1, 
                  bbox2,
                  corner_coordinates=True):
    """
    Definition:
    Check if two bounding boxes overlap.

    Parameters:
    bbox1 (np.array)          : bbox coordinates of first object    
    bbox2 (np.array)          : bbox coordinates of second object
    corner_coordinates (bool) : defines what bbox coordinate system is in use

    Returns:
    overlap (bool) : True if the boxes overlap, False otherwise
    """
    # Unpack the coordinates
    if corner_coordinates:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
    else:
        center1_x, center1_y, width1, height1 = bbox1
        x1_min, y1_min, x1_max, y1_max = to_corner_coordinates(center1_x, center1_y, width1, height1)
        center2_x, center2_y, width2, height2 = bbox2
        x2_min, y2_min, x2_max, y2_max = to_corner_coordinates(center2_x, center2_y, width2, height2)

    # Check if the boxes do not overlap
    if x1_max < x2_min or x2_max < x1_min:  # One box is to the left of the other
        return False
    if y1_max < y2_min or y2_max < y1_min:  # One box is above the other
        return False

    # If none of the above conditions are true, the boxes overlap
    return True

def draw_grid_on_image(image, 
                       grid_rows = 4, 
                       grid_cols = 4):
    """
    Definition
    Draws black grid onto image based on input grid specs

    Parameters:
    image (np.array) : image to draw grid on
    grid_rows (int)  : number of rows the image is broken down into
    grid_cols (int)  : number of cols the image is broken down into

    Returns:
    image (np.array) : image with gridlines
    """
    # Get the shape of the image
    height, width = image.shape
    
    # Calculate the step size for rows and columns
    row_step = height // grid_rows
    col_step = width // grid_cols

    # Set grid lines to zero (black)
    for i in range(1, grid_rows):
        image[i * row_step, :] = 0  # Horizontal lines
    for j in range(1, grid_cols):
        image[:, j * col_step] = 0  # Vertical lines

    return image

def create_image(objects,
                 labels,
                 image_size = (128, 128),
                 noise_intensity = 180,
                 grid_rows = 4,
                 grid_cols = 4,
                 max_objects = 8,
                 max_scaling = 2.5,
                 add_gridlines = False,
                 allow_overlap = False,
                 corner_coordinates=True):
    """
    Definition:
    Create an image for the output dataset

    Parameters:
    objects (np.array)         : all images of MNIST dataset
    label (np.array)           : all associated classes and bbox labels of MNIST dataset
    image_size ((int , int))   : the set height and width of returned image
    noise_intensity (int)      : the scalar intensity value for the background noise
    grid_rows (int)            : number of rows the image is broken down into
    grid_cols (int)            : number of cols the image is broken down into
    max_objects (int)          : upper limit of objects to be added to image
    max_scaling (float)        : upper limit of size scalar for objects
    add_gridlines (bool)       : adds gridlines to image if True
    allow_overlap (bool)       : removes added object if it overlaps with another object if False
    corner_coordinates (bool)  : defines what bbox coordinate system is in use

    Returns:
    image (np.array)     : finished created image
    added_objects (dict) : dict with all object class, true object coordinates on 
                           image, and normalized coordinates
    """
    image = generate_noisy_image(image_size = image_size,
                                 noise_intensity = noise_intensity)
    
    data_size = len(objects)
    scaling_options = np.arange(1, max_scaling + 0.125, 0.125)

    regions_to_populate = choose_regions_to_populate(max_objects=max_objects,
                                                     grid_rows = grid_rows,
                                                     grid_cols = grid_cols)
    
    if add_gridlines:
        image = draw_grid_on_image(image, 
                                   grid_rows = grid_rows, 
                                   grid_cols = grid_cols)
        
    added_objects = {}

    for object_num, region in enumerate(regions_to_populate):
       
        index = np.random.randint(0, data_size)
        scaler = np.random.choice(scaling_options)

        temp_image, object_to_add = add_object_to_image(image = copy.deepcopy(image),
                                                        region_of_interest = region,
                                                        object = objects[index],
                                                        label = labels[index],
                                                        object_num = object_num,
                                                        grid_rows = grid_rows,
                                                        grid_cols = grid_cols,
                                                        scale_value = scaler,
                                                        corner_coordinates=corner_coordinates)
        
        overlap = False
        if object_num > 0 and not allow_overlap:
            added_objects_list = list(added_objects.values())
            for added_object in added_objects_list:
                overlap = check_overlap(object_to_add[object_num]['bbox_true'],
                                        added_object['bbox_true'],
                                        corner_coordinates=corner_coordinates)
                if overlap:
                    break
        
        if not overlap:
            added_objects.update(object_to_add)
            image = temp_image
        
        del temp_image

    return image, added_objects

def add_bboxes_to_image(image, 
                        added_objects, 
                        label_color_map,
                        corner_coordinates=True):
    """
    Definition
    Convert image to RGB and overlay bboxes colored by class

    Parameters:
    image (np.array)           : image
    added_objects (dict)       : dictionary with all object and bbox information
    label_color_map (dict)     : dictionary with color key by class
    corner_coordinates (bool)  : defines what bbox coordinate system is in use

    Returns:
    image (np.array) : RGB image with bboxes
    """    
    image = np.stack([image, image, image], axis=-1)

    for _ , value in added_objects.items():
        
        if corner_coordinates:
            x_min, y_min, x_max, y_max = value['bbox_true']
        else:
            center_x, center_y, width, height = value['bbox_true']
            x_min, y_min, x_max, y_max = to_corner_coordinates(center_x, center_y, width, height)

        image = cv2.rectangle(image, 
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),  
                              color=label_color_map[value['class']], 
                              thickness=1)
    
    return image