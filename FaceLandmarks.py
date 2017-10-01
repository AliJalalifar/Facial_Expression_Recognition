import dlib
import cv2
import glob
import math
import numpy
import numpy as np
import os
from math import atan2, degrees, pi

#dlib detector
from skimage.feature import hog
from sklearn.externals import joblib

detector = dlib.get_frontal_face_detector()
#loading dlib shape predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]

for emotion in emotions:
    for img_path in glob.glob(os.path.join("dataset\\%s\\*" % emotion)):
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        dets = detector(im, 1)
        for k, d in enumerate(dets):
            # predicting face parts
            parts = predictor(im, d).parts()
            landmarks = np.matrix([[p.x, p.y] for p in parts])
            mouth = np.zeros(102)
            mouth[0:102] = landmarks[17:68].flatten()
            joblib.dump(mouth, "Parts\\" + img_path[:-4]+ ".kp")
            print "Parts\\" + img_path[:-4]+ ".kp" + " Created!"
