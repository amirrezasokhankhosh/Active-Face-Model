#!/usr/bin/python

from email.mime import base
import dlib
import cv2
import time
import numpy as np
# import scipy
# from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from edges import edges


def collect_faces():
    faces = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    for i in range(1, 44):
        path = 'C:\\Users\\amirr\\Desktop\\Active-Face-Model\\photos\\' + str(i) + '.jpg'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = predictor(img, d)

            X = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
            theta = np.radians(180)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            Z = X @ R
            faces.append(Z)

    return faces


def plot_face(X, color='b'):
    "plots a face"

    plt.plot(X[:, 0], X[:, 1], 'o', color=color)

    for edge in edges:
        i, j = edge  # edge from node i to node j

        xi = X[i, 0]
        yi = X[i, 1]
        xj = X[j, 0]
        yj = X[j, 1]
        # draw a line between X[i] and X[j]
        plt.plot((xi, xj), (yi, yj), '-', color=color)


def register_affine_face(base_face, face):
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
    return np.reshape(registered_face, (68, 2))


def registered_similarity_face(base_face, face):
    mean = np.mean(face, axis=0)
    of = face - mean
    bf = base_face - np.mean(base_face, axis=0)
    new_face = np.repeat(of, repeats=2, axis=0)
    new_face[1::2, 0] = new_face[1::2, 0] * -1
    new_face[1::2, [0, 1]] = new_face[1::2, [1, 0]]

    new_base_face = np.ravel(bf)
    y = np.linalg.lstsq(new_face, new_base_face, rcond=None)[0]
    registered_face = new_face @ y
    return np.reshape(registered_face, (68, 2))


def PCA(X, avg, k):
    Z = np.zeros((136, len(face_list)))
    for i in range(len(face_list)):
        Z[:, i] = X[i].reshape((1, 136)) - avg.reshape((1, 136))
    U, sigma, V = np.linalg.svd(Z, full_matrices=False)
    U, sigma, V = U[:, 0:k], sigma[0:k], V[0:k, :]
    return U, sigma, V


def animate(U, sigma, avg, k, color='b'):
    for i in range(k):
        sig = sigma[i]
        rng = np.linspace(-sig, sig, 5)
        for alpha in rng:
            plt.cla()
            A = avg + (alpha * U[:, i].reshape(68, 2))
            plot_face(A, color)
            plt.draw()
            plt.pause(.5)


def transfer(avg, U, X):
    registered_x = register_affine_face(avg, X)
    answer = np.linalg.lstsq(U, (avg.reshape((136, 1)) - registered_x.reshape((136, 1))), rcond=None)[0]
    transfered_face = avg.reshape((136, 1)) + U @ answer
    plot_face(transfered_face.reshape((68, 2)), )
    plt.show()


face_list = collect_faces()
for i in range(5):
    plot_face(face_list[i], color='g')
    plt.show()
base_face = face_list[0]
team_mate = face_list.pop()
print(len(face_list))
# averaging faces - affine register
affine_registered_list = []
affine_registered_list.append(base_face)
for i in range(1, len(face_list)):
    bf = base_face - np.mean(base_face, axis=0)
    face = face_list[i]
    t = register_affine_face(base_face, face)
    affine_registered_list.append(t)
    # plot_face(t, color='r')
    # plot_face(bf, color='k')
    # plt.show()


# averaging faces - similarity register
similarity_registered_list = []
similarity_registered_list.append(base_face)

for i in range(1, len(face_list)):
    bf = base_face - np.mean(base_face, axis=0)
    face = face_list[i]
    t = registered_similarity_face(base_face, face)
    similarity_registered_list.append(t)
    # plot_face(t, color='b')
    # plot_face(bf, color='k')
    # plt.show()

avg = sum(similarity_registered_list) / len(similarity_registered_list)
plot_face(avg, color='orange')
plt.show()

# TODO: Change k
k = 16
affine_avg = sum(affine_registered_list) / len(affine_registered_list)
U, sigma, V = PCA(similarity_registered_list, avg, k)
# animate(U, sigma, avg, k)
transfer(affine_avg, U, team_mate)
