import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import pandas as pd
import os
from PIL import Image
import math
import sys
import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
image = cv2.imread("/Users/issakovakamilla/Desktop/Papers/Thesis/pics/668.jpg")

image = cv2.resize(image, (600,500))
# Convert the image color to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect the face
rects = detector(gray, 1)
# Detect landmarks for each face
for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# Display the image
cv2.imshow('Landmark Detection', image)
cap.release()