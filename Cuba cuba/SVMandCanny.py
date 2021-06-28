import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd

# categories = ['Beracun', 'BisaDimakan']
categories = ['Backup gambar\Beracun noBG', 'Backup gambar\BisaDimakan noBG']
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


def SetupDataset():
    for category in categories:
        path = os.path.join(dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = cv.imread(imgpath, 0)
            image = cv.resize(image, (200, 200))
            ret, th1 = cv.threshold(
                image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

            imgCanny = cv.Canny(th1, 200, 200)

            feature_matrix = np.array(imgCanny).flatten()
            data.append([feature_matrix, label])

    TulisDataset("jamurDataset4.pickle", data)


def SVM():
    data1 = BacaDataset("jamurDataset4.pickle")

    features = []
    labels = []

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.30)

    # Membuat model

    # model = SVC(C=1, kernel='linear', gamma='auto')
    # model.fit(xtrain, ytrain)
    # TulisDataset('model4.sav', model)

    model1 = BacaDataset('model4.sav')

    prediksi = model1.predict(xtest)
    akurasi = model1.score(xtest, ytest)

    print('Akurasi: ', akurasi)
    # print("Precision:", metrics.precision_score(ytest, prediksi))
    # print("Recall:", metrics.recall_score(ytest, prediksi))

    for i in range(2):
        print('Prediksi: ', categories[prediksi[i]])
        jamoer = xtest[i]
        cv.imshow('j', jamoer)
        cv.waitKey(0)
        cv.destroyAllWindows()


# SetupDataset()
SVM()


# img = cv.imread('Backup gambar\\Beracun noBG\\2-removebg-preview.png')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# # noise removal
# kernel = np.ones((3, 3), np.uint8)
# opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# # sure background area
# sure_bg = cv.dilate(opening, kernel, iterations=3)
# # Finding sure foreground area
# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg, sure_fg)
# # Marker labelling
# ret, markers = cv.connectedComponents(sure_fg)
# cv.imshow("a", markers)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0
# markers = cv.watershed(img, markers)
# img[markers == -1] = [255, 0, 0]


# cv.waitKey(0)
# cv.destroyAllWindows()
