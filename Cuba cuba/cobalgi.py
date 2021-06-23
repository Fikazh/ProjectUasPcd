import numpy as np
import glob
from PIL import Image
from skimage.io import imread, imsave
import pandas as pd
import os
from sklearn import svm


def traning(datasets):
    feature_dataset = []
    y = [1, 1, 1, 1, 1]
    for data in datasets:
        feature_dataset.append(extract_feature(data))
    X = np.vstack(feature_dataset)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    return clf


def extract_feature(img):
    img = Image.open(img).convert('L')
    gl_0 = glcm(img, 0)
    gl_45 = glcm(img, 45)
    gl_90 = glcm(img, 90)
    gl_135 = glcm(img, 135)
    feature = np.array(
        [np.average([contrast(gl_0), energy(gl_0), homogenity(gl_0), entropy(gl_0)]),
         np.average([contrast(gl_45), energy(gl_45),
                     homogenity(gl_45), entropy(gl_45)]),
         np.average([contrast(gl_90), energy(gl_90),
                     homogenity(gl_90), entropy(gl_90)]),
         np.average([contrast(gl_135), energy(gl_135), homogenity(gl_135), entropy(gl_135)])])
    print(feature)
    return feature


def contrast(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += matrix[i][j] * np.power(i-j, 2)
    return res


def energy(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += np.power(matrix[i][j], 2)
    return res


def homogenity(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += matrix[i][j] / (1 + np.power(i-j, 2))
    return res


def entropy(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            if matrix[i][j] > 0:
                res += matrix[i][j] * np.log2(matrix[i][j])
    return res


def glcm(img, degree):
    img = img.resize([128, 128], Image.NEAREST)
    arr = np.array(img)
    res = np.zeros((arr.max() + 1, arr.max() + 1), dtype=int)
    width, height = arr.shape
    if degree == 0:
        for i in range(width - 1):
            for j in range(height):
                res[arr[j, i+1], arr[j, i]] += 1

    elif degree == 45:
        for i in range(width - 1):
            for j in range(1, height):
                res[arr[j-1, i+1], arr[j, i]] += 1

    elif degree == 90:
        for i in range(width):
            for j in range(1, height):
                res[arr[j-1, i], arr[j, i]] += 1

    elif degree == 135:
        for i in range(1, width):
            for j in range(1, height):
                res[arr[j-1, i-1], arr[j, i]] += 1

    else:
        print("Sudut tidak valid")
    return res


datasets = glob.glob('../Datasets/Train/[0-9].jpg')
classifier = traning(datasets)
print(classifier.predict([extract_feature('../Datasets/Beracun/7.jpg')]))
