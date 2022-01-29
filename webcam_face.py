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
        theta = np.radians(180)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        Z = X @ R
        for x in X:
            cv2.circle(img, (x[0], x[1]), 2, (0, 0, 255))

    cv2.imshow('', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        # print('X =', X)
        face_list.append(Z)


base_face = face_list[0]
affine_registered_list = []
# averaging faces - affine register
for i in range(1, len(face_list)):
    face = face_list[i]
    mean = np.mean(face, axis=0)
    of = face - mean
    bf = base_face - np.mean(base_face, axis=0)
    new_face = np.c_[of, np.zeros((68, 2))]
    new_face = np.repeat(new_face, repeats=2, axis=0)
    new_face[1::2, [2, 0]] = new_face[1::2, [0, 2]]
    new_face[1::2, [3, 1]] = new_face[1::2, [1, 3]]

    new_base_face = np.ravel(bf)
    y = np.linalg.lstsq(new_face, new_base_face, rcond=None)[0]
    registered_face = new_face @ y
    t = np.reshape(registered_face, (68, 2))
    affine_registered_list.append(t)
    plt.plot(t[:, 0], t[:, 1], 'o', color='k', markersize=3)
    plt.plot(bf[:, 0], bf[:, 1], 'o', color='r', markersize=3)
    plt.show()

# averaging faces - similarity register
similarity_registered_list = []
for i in range(1, len(face_list)):
    face = face_list[i]
    mean = np.mean(face, axis=0)
    of = face - mean
    bf = base_face - np.mean(base_face, axis=0)
    new_face = np.repeat(of, repeats=2, axis=0)
    new_face[1::2, 0] = new_face[1::2, 0] * -1
    new_face[1::2, [0, 1]] = new_face[1::2, [1, 0]]

    new_base_face = np.ravel(bf)
    y = np.linalg.lstsq(new_face, new_base_face, rcond=None)[0]
    registered_face = new_face @ y
    t = np.reshape(registered_face, (68, 2))
    similarity_registered_list.append(t)
    plt.plot(t[:, 0], t[:, 1], 'o', color='b', markersize=3)
    plt.plot(bf[:, 0], bf[:, 1], 'o', color='r', markersize=3)
    plt.show()



