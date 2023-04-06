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

path_to_folder = '/Users/issakovakamilla/Desktop/Papers/Thesis/one'
files = [f for f in os.listdir(path_to_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
#files = os.listdir(path_to_folder)
#sorted_files = sorted(files)

for filename in sorted_files:
    #if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".webp") or filename.endswith(".png"):
        image_path = os.path.join(path_to_folder, filename)
        image = dlib.load_rgb_image(image_path)

        #Load the Shape_predictor_68_face_landmarks shape predictor
        predictor = dlib.shape_predictor('/Users/issakovakamilla/PycharmProjects/FaceSymmetry/venv/shape_predictor_68_face_landmarks.dat')
        # Load the dlib face detector
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)

        # ---------------------------------------------------------------------------------------------------------------------------------
        # image = cv2.imread("/Users/issakovakamilla/Desktop/Papers/Thesis/Photos/britney.jpg")
        def symm_code():
            for face in faces:
                landmarks = predictor(image, face)
#2 of the main neoclassical canons described in the literature
                left_face_x = landmarks.part(1).x
                left_face_y = landmarks.part(1).y
                right_face_x = landmarks.part(15).x
                right_face_y = landmarks.part(15).y
                face_width = math.sqrt((left_face_x - right_face_x) ** 2 + (left_face_y - right_face_y) ** 2)
                left_nose_x = landmarks.part(31).x
                left_nose_y = landmarks.part(31).y
                right_nose_x = landmarks.part(35).x
                right_nose_y = landmarks.part(35).y
                nose_dist = math.sqrt((left_nose_x - right_nose_x) ** 2 + (left_nose_y - right_nose_y) ** 2)
                facenosedev = math.sqrt((face_width - 4*(nose_dist)) ** 2)

                left_eye_corner_out_x = landmarks.part(36).x
                left_eye_corner_out_y = landmarks.part(36).y
                left_eye_corner_in_x = landmarks.part(39).x
                left_eye_corner_in_y = landmarks.part(39).y
                right_eye_corner_out_x = landmarks.part(45).x
                right_eye_corner_out_y = landmarks.part(45).y
                right_eye_corner_in_x = landmarks.part(42).x
                right_eye_corner_in_y = landmarks.part(42).y
                # Calculating the size of an eye cut:
                left_eye_size = math.sqrt((left_eye_corner_out_x - left_eye_corner_in_x) ** 2 + (
                            left_eye_corner_out_y - left_eye_corner_in_y) ** 2)
                right_eye_size = math.sqrt((right_eye_corner_out_x - right_eye_corner_in_x) ** 2 + (
                            right_eye_corner_out_y - right_eye_corner_in_y) ** 2)
                av_eye_size = math.sqrt(((left_eye_size + right_eye_size)/2) ** 2)
                intereye = math.sqrt((left_eye_corner_in_x - right_eye_corner_in_x) ** 2 + (
                            left_eye_corner_in_y - right_eye_corner_in_y) ** 2)
                eyemideyedev = math.sqrt((av_eye_size - intereye) ** 2)

                canonsumdev = sum([facenosedev, eyemideyedev])
                print(canonsumdev)
        symm_code()