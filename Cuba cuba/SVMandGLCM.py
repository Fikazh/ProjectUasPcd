import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from PIL import Image
from skimage.io import imread, imsave
from sklearn import svm

categories = ['Beracun', 'BisaDimakan']

data = []
dir = os.getcwd()


def TulisDataset(name, ext):
    pick_in = open(name, 'wb')
    pickle.dump(ext, pick_in)
    pick_in.close()


def BacaDataset(name):
    pick_in = open(name, 'rb')
    ext = pickle.load(pick_in)
    pick_in.close()
    return ext


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


def SetupDataset():
    feature = []
    labels = []
    for category in categories:
        path = os.path.join(dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            # image = cv.imread(imgpath, 0)
            # image = extract_feature(image)

            # feature_matrix = np.array().flatten()
            # feature.append()
            # X = np.vstack(extract_feature(imgpath))
            data.append([extract_feature(imgpath), label])

    TulisDataset("jamurDataset3.pickle", data)


def SVM():
    data1 = BacaDataset("jamurDataset3.pickle")
    features = []
    labels = []

    # print(data1)

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    # print(xtrain)
    # print()
    # print(xtest)
    # print(ytrain)
    # print(ytest)
    # Membuat model

    model = SVC(C=1, kernel='linear', gamma='auto')
    model.fit(xtrain, ytrain)
    TulisDataset('model1.sav', model)
    # model1 = SVC(C=1, kernel='linear', gamma='auto')
    # model1 = BacaDataset('model3.sav')

    # prediksi = model1.predict(xtest)
    # akurasi = model1.score(xtest, ytest)

    # print('Akurasi: ', akurasi)
    # print("Precision:", metrics.precision_score(ytest, prediksi))
    # print("Recall:", metrics.recall_score(ytest, prediksi))

    # for i in range(14):
    #     print('Prediksi: ', categories[prediksi[i]])
    # jamoer = xtest[i].reshape(200, 200)
    # plt.imshow(jamoer, cmap='gray')
    # plt.show()


# SetupDataset()
SVM()
