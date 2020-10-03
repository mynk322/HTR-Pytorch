import os

import cv2
import numpy as np
import random as rng
import matplotlib.pyplot as plt

def segmentImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    form = cv2.GaussianBlur(image, ksize=(49, 49), sigmaX = 11, sigmaY = 11)
    ret, form = cv2.threshold(form, 215, 255, cv2.THRESH_BINARY)
    form = 255 - form

    _, labels_im = cv2.connectedComponents(form)

    min_x = {}
    max_x = {}
    min_y = {}
    max_y = {}

    for i in range(labels_im.shape[0]):
        for j in range(labels_im.shape[1]):
            if not labels_im[i][j] == 0:
                if labels_im[i][j] not in min_x.keys():
                    min_x[labels_im[i][j]] = i
                    max_x[labels_im[i][j]] = i
                    min_y[labels_im[i][j]] = j
                    max_y[labels_im[i][j]] = j
                else:
                    min_x[labels_im[i][j]] = min(min_x[labels_im[i][j]], i)
                    max_x[labels_im[i][j]] = max(max_x[labels_im[i][j]], i)
                    min_y[labels_im[i][j]] = min(min_y[labels_im[i][j]], j)
                    max_y[labels_im[i][j]] = max(max_y[labels_im[i][j]], j)
    
    bounding_boxes = []
    for i in range(1, len(min_x.keys()) + 1):
        bounding_boxes.append((min_x[i], max_x[i], min_y[i], max_y[i]))

    bounding_boxes = sorted(bounding_boxes, key = lambda entry: entry[0])

    current_window_min = -1
    current_window_max = -1
    current_window_str = -1

    for i in range(len(bounding_boxes)):
        if current_window_str == -1:
            current_window_min = bounding_boxes[i][0]
            current_window_max = bounding_boxes[i][1]
            current_window_str = i
            continue
        if (i == len(bounding_boxes) - 1) or (not ((bounding_boxes[i][0] >= current_window_min and bounding_boxes[i][0] <= current_window_max) or (bounding_boxes[i][1] >= current_window_min and bounding_boxes[i][1] <= current_window_max))):
            if i == len(bounding_boxes) - 1:
                i = i + 1
            bounding_boxes[current_window_str:i] = sorted(bounding_boxes[current_window_str:i], key = lambda entry: entry[2])
            if i == len(bounding_boxes):
                i = i - 1
            current_window_min = bounding_boxes[i][0]
            current_window_max = bounding_boxes[i][1]
            current_window_str = i
            
        if (bounding_boxes[i][0] >= current_window_min and bounding_boxes[i][0] <= current_window_max) or (bounding_boxes[i][1] >= current_window_min and bounding_boxes[i][1] <= current_window_max):
            current_window_min = min(current_window_min, bounding_boxes[i][0])
            current_window_max = min(current_window_max, bounding_boxes[i][1])
    
    segments = []
    for i in range(len(bounding_boxes)):
        segments.append(image[bounding_boxes[i][0]:bounding_boxes[i][1], bounding_boxes[i][2]:bounding_boxes[i][3]])

    return segments
        