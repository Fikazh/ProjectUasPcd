import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

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
            image = cv.resize(image, (50, 50))

            feature_matrix = np.array(image).flatten()
            data.append([feature_matrix, label])

    TulisDataset("jamurDataset1.pickle", data)


def SVM():
    data1 = BacaDataset("jamurDataset1.pickle")
    features = []
    labels = []

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    # Membuat model

    # model = SVC(C=1, kernel='poly', gamma='auto')
    # model.fit(xtrain, ytrain)
    # TulisDataset('model1.sav', model)

    model1 = BacaDataset('model1.sav')

    prediksi = model1.predict(xtest)
    akurasi = model1.score(xtest, ytest)

    print('Akurasi: ', akurasi)
    print("Precision:", metrics.precision_score(ytest, prediksi))
    print("Recall:", metrics.recall_score(ytest, prediksi))

    # for i in range(13):
    # print('Prediksi: ', categories[prediksi[i]])
    # jamoer = xtest[i].reshape(50, 50)
    # plt.imshow(jamoer, cmap='gray')
    # plt.show()


# SetupDataset()
SVM()
