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

    for i in range(1, 45):
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
    
    X_p, Y_p = np.copy(of[:, 0]).reshape((68, 1)), np.copy(of[:, 1]).reshape((68, 1))
    X, Y = np.copy(bf[:, 0]).reshape((68, 1)), np.copy(bf[:, 1]).reshape((68, 1)) 
    
    equations = np.array([[np.dot(X_p.T, X_p)[0,0], np.dot(X_p.T, Y_p)[0,0], 0, 0],
                          [np.dot(X_p.T, Y_p)[0,0], np.dot(Y_p.T, Y_p)[0,0], 0, 0],
                          [0, 0, np.dot(X_p.T, X_p)[0,0], np.dot(X_p.T, Y_p)[0,0]],
                          [0, 0, np.dot(X_p.T, Y_p)[0,0], np.dot(Y_p.T, Y_p)[0,0]]])
    outcomes = np.stack((np.dot(X_p.T, X)[0,0], np.dot(Y_p.T, X)[0,0], np.dot(X_p.T, Y)[0,0], np.dot(Y_p.T, Y)[0,0]))
    a, b, c, d = np.linalg.solve(equations, outcomes.T)
    A = np.array([[a, b], [c, d]])
    registered_face = (A @ of.T).T
    return registered_face

def register_similarity_face(base_face, face):
    mean = np.mean(face, axis=0)
    of = face - mean
    bf = base_face - np.mean(base_face, axis=0)
    
    X_p, Y_p = np.copy(of[:, 0]).reshape((68, 1)), np.copy(of[:, 1]).reshape((68, 1))
    X, Y = np.copy(bf[:, 0]).reshape((68, 1)), np.copy(bf[:, 1]).reshape((68, 1)) 
    
    a = (np.dot(X_p.T, X)[0,0] + np.dot(Y_p.T, Y)[0,0]) / (np.dot(X_p.T, X_p)[0,0] + np.dot(Y_p.T, Y_p)[0,0])
    b = (np.dot(X_p.T, Y)[0,0] - np.dot(Y_p.T, X)[0,0]) / (np.dot(X_p.T, X_p)[0,0] + np.dot(Y_p.T, Y_p)[0,0])
    A = np.array([[a, -b], [b, a]])
    registered_face = (A @ of.T).T
    return registered_face


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
                plt.pause(0.5)


def transfer(avg, U, X):
    registered_x = register_affine_face(avg, X)
    answer = np.linalg.lstsq(U, (registered_x.reshape((136, 1)) - avg.reshape((136, 1))), rcond=None)[0]
    transfered_face = avg.reshape((136, 1)) + U @ answer
    plt.cla()
    plot_face(registered_x, color='r')
    plt.show()
    plot_face(transfered_face.reshape((68, 2)), color='orange')
    plt.show()


face_list = collect_faces()

base_face = face_list[0]
team_mate = face_list.pop()

for i in range(5):
    plot_face(face_list[i], color='g')
    plt.show()

# averaging faces - affine register
affine_registered_list = []
affine_registered_list.append(base_face)
for i in range(1, len(face_list)):
    bf = base_face - np.mean(base_face, axis=0)
    face = face_list[i]
    t = register_affine_face(base_face, face)
    affine_registered_list.append(t)

for i in range(1, 6):
    plot_face(affine_registered_list[i], color='r')
    bf = base_face - np.mean(base_face, axis=0)
    plot_face(bf, color='k')
    plt.show()


# averaging faces - similarity register
similarity_registered_list = []
similarity_registered_list.append(base_face)

for i in range(1, len(face_list)):
    bf = base_face - np.mean(base_face, axis=0)
    face = face_list[i]
    t = register_similarity_face(base_face, face)
    similarity_registered_list.append(t)

for i in range(1, 6):
    plot_face(similarity_registered_list[i], color='b')
    bf = base_face - np.mean(base_face, axis=0)
    plot_face(bf, color='k')
    plt.show()
    

affine_avg = sum(affine_registered_list) / len(similarity_registered_list)
plot_face(affine_avg, color='b')
plt.show()

# TODO: Change k
k = 16

U, sigma, V = PCA(affine_registered_list, affine_avg, k)

animate(U, sigma, affine_avg, k)

transfer(affine_avg, U, team_mate)
