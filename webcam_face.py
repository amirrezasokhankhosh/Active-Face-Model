#!/usr/bin/python

import dlib
import cv2
import time
import numpy as np
# import scipy
# from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

i = 0

face_list = []


while rval:

    rval, img = vc.read()
    # print(rval)
    i += 1
    if i % 3 != 1:
        pass
    else:

        dets = detector(img, 1)

    for k, d in enumerate(dets):

        shape = predictor(img, d)

        X = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])

        for x in X:
            cv2.circle(img, (x[0], x[1]), 2, (0, 0, 255))

    cv2.imshow('', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        print('X =', X)
        face_list.append(X)


print(face_list)
