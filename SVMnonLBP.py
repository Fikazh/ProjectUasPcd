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


def SetupDataset():
    for category in categories:
        path = os.path.join(dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = cv.imread(imgpath, 0)
            image = cv.resize(image, (200, 200))

            feature_matrix = np.array(image).flatten()
            data.append([feature_matrix, label])

    df = pd.DataFrame(data=data).T
    df.to_csv("jamur1.csv")
    # TulisDataset("jamurDataset1.pickle", data)


def SVM():
    # data1 = BacaDataset("jamurDataset1.pickle")
    data1 = pd.read_csv('jamur1.csv', index_col=1)

    print(data1)

    features = []
    labels = []

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.3)

    # print(xtrain)
    # print()
    # print(xtest)
    # print(ytrain)
    # print(ytest)

    # Membuat model

    # model = SVC(C=1, kernel='poly', gamma='auto')
    # model.fit(xtrain, ytrain)
    # TulisDataset('model1.sav', model)

    # model1 = BacaDataset('model1.sav')

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
