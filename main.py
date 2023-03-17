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

path_to_folder = "/Users/issakovakamilla/Desktop/Papers/Thesis/pics"
files = [f for f in os.listdir(path_to_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
#files = os.listdir(path_to_folder)
#sorted_files = sorted(files)

for filename in sorted_files:
    #if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".webp") or filename.endswith(".png"):
        image_path = os.path.join(path_to_folder, filename)
        image = dlib.load_rgb_image(image_path)

        #Load the Shape_predictor_68_face_landmarks shape predictor
        predictor = dlib.shape_predictor('/Users/issakovakamilla/PycharmProjects/FaceSymmetry/venv'
                                             '/shape_predictor_68_face_landmarks.dat')
        # Load the dlib face detector
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)

        # ---------------------------------------------------------------------------------------------------------------------------------
        # image = cv2.imread("/Users/issakovakamilla/Desktop/Papers/Thesis/Photos/britney.jpg")

        def symm_code():
            for face in faces:
                landmarks = predictor(image, face)

            left_jaw_x = landmarks.part(4).x
            left_jaw_y = landmarks.part(4).y
            right_jaw_x = landmarks.part(12).x
            right_jaw_y = landmarks.part(12).y
            mid_jaw_x = landmarks.part(66).x
            mid_jaw_y = landmarks.part(66).y
            left_jaw_dist = math.sqrt((left_jaw_x - mid_jaw_x) ** 2 + (left_jaw_y - mid_jaw_y) ** 2)
            right_jaw_dist = math.sqrt((right_jaw_x - mid_jaw_x) ** 2 + (right_jaw_y - mid_jaw_y) ** 2)
            # Jaw-midpoint deviation, pairwise (right side from left side):
            jaw_dev = math.sqrt((left_jaw_dist - right_jaw_dist) ** 2)

            left_lip_x = landmarks.part(48).x
            left_lip_y = landmarks.part(48).y
            right_lip_x = landmarks.part(54).x
            right_lip_y = landmarks.part(54).y
            mid_lip_x = landmarks.part(66).x
            mid_lip_y = landmarks.part(66).y
            left_lip_dist = math.sqrt((left_lip_x - mid_lip_x) ** 2 + (left_lip_y - mid_lip_y) ** 2)
            right_lip_dist = math.sqrt((right_lip_x - mid_lip_x) ** 2 + (right_lip_y - mid_lip_y) ** 2)
            # Lip-midpoint deviation, pairwise:
            lip_dev = math.sqrt((left_lip_dist - right_lip_dist) ** 2)

            left_nose_x = landmarks.part(31).x
            left_nose_y = landmarks.part(31).y
            right_nose_x = landmarks.part(35).x
            right_nose_y = landmarks.part(35).y
            mid_nose_x = landmarks.part(33).x
            mid_nose_y = landmarks.part(33).y
            left_nose_dist = math.sqrt((left_nose_x - mid_nose_x) ** 2 + (left_nose_y - mid_nose_y) ** 2)
            right_nose_dist = math.sqrt((right_nose_x - mid_nose_x) ** 2 + (right_nose_y - mid_nose_y) ** 2)
            # Nose-midpoint deviation, pairwise:
            nose_dev = math.sqrt((left_nose_dist - right_nose_dist) ** 2)

            left_skull_x = landmarks.part(0).x
            left_skull_y = landmarks.part(0).y
            right_skull_x = landmarks.part(16).x
            right_skull_y = landmarks.part(16).y
            mid_skull_x = landmarks.part(28).x
            mid_skull_y = landmarks.part(28).y
            left_skull_dist = math.sqrt((left_skull_x - mid_skull_x) ** 2 + (left_skull_y - mid_skull_y) ** 2)
            right_skull_dist = math.sqrt((right_skull_x - mid_skull_x) ** 2 + (right_skull_y - mid_skull_y) ** 2)
            # Skull-midpoint deviation, pairwise:
            skull_dev = math.sqrt((left_skull_dist - right_skull_dist) ** 2)

            left_eye_corner_out_x = landmarks.part(36).x
            left_eye_corner_out_y = landmarks.part(36).y
            left_eye_corner_in_x = landmarks.part(39).x
            left_eye_corner_in_y = landmarks.part(39).y
            right_eye_corner_out_x = landmarks.part(45).x
            right_eye_corner_out_y = landmarks.part(45).y
            right_eye_corner_in_x = landmarks.part(42).x
            right_eye_corner_in_y = landmarks.part(42).y
            # Calculating the size of an eye cut:
            left_eye_size = math.sqrt(
                (left_eye_corner_out_x - left_eye_corner_in_x) ** 2 + (left_eye_corner_out_y - left_eye_corner_in_y) ** 2)
            right_eye_size = math.sqrt(
                (right_eye_corner_out_x - right_eye_corner_in_x) ** 2 + (
                        right_eye_corner_out_y - right_eye_corner_in_y) ** 2)
            # Eye size difference, pairwise:
            eye_size_diff = math.sqrt((left_eye_size - right_eye_size) ** 2)

            left_brow_mid_x = landmarks.part(19).x
            left_brow_mid_y = landmarks.part(19).y
            right_brow_mid_x = landmarks.part(24).x
            right_brow_mid_y = landmarks.part(24).y
            left_eye_up_x = landmarks.part(38).x
            left_eye_up_y = landmarks.part(38).y
            right_eye_up_x = landmarks.part(44).x
            right_eye_up_y = landmarks.part(44).y
            # Calculating deeb -  distance between eye and eyebrow
            left_deeb = math.sqrt((left_brow_mid_x - left_eye_up_x) ** 2 + (left_brow_mid_y - left_eye_up_y) ** 2)
            right_deeb = math.sqrt((right_brow_mid_x - right_eye_up_x) ** 2 + (right_brow_mid_y - right_eye_up_y) ** 2)
            # Deeb deviation, pairwise:
            deeb_dev = math.sqrt((left_deeb - right_deeb) ** 2)

            # Overall symmetry index - Sum of pairwise deviations of face feature points and eye cut difference:
            # Since sum() takes 2 arg at most, paired sums:
            sum1 = sum([jaw_dev, lip_dev])
            sum2 = sum([nose_dev, skull_dev])
            sum3 = sum([deeb_dev, eye_size_diff])

            sum4 = sum([sum1, sum2])
            SymmIndex = sum([sum4, sum3])
            print(SymmIndex)#
        symm_code()