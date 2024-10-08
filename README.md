# MNIST Object Detection Dataset Creator

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/github/license/Matthew-Weisberg/MNIST_ObjectDetection_Dataset)
![Issues](https://img.shields.io/github/issues/Matthew-Weisberg/MNIST_ObjectDetection_Datasett)
![Stars](https://img.shields.io/github/stars/Matthew-Weisberg/MNIST_ObjectDetection_Dataset)

## Overview

This repository contains code to generate an **object detection dataset** from the popular [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. MNIST is traditionally used for image classification tasks, but this project repurposes it to create synthetic images where multiple digits appear, along with corresponding bounding box annotations for object detection. The goal is to take an incredibly simple dataset and utilize it in a unique way that showcases many aspects of machine learning proficiency. The created dataset will used by a custom YOLO model trained scratch.

## Features

- **Customizable dataset**: Allows for specification of the following parameters:
  1. Size of each image
  2. Maximum number of digits per image
  3. Maximum upscaling of digits
  4. The number of grid rows and columns in which can contain one digit (CNN object detection algorithm such as YOLO)
  5. The background noise intensity of each image
  6. Whether digits are allowed to overlap within each image
- **Bounding boxes**: Automatically generates bounding boxes for each digit in the image.
- **Multiple formats**: Export annotations in formats compatible with popular object detection frameworks such as YOLO, COCO, or Pascal VOC.

## Example Image

Here is an example of a generated image with bounding boxes around the digits:

![Example Image](https://github.com/user-attachments/assets/125a3c61-4ab9-45f5-9c21-57353bd9b998)

## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries: 
  - `tensorflow`
  - `numpy`
  - `opencv-python`
  - `matplotlib`
