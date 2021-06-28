import numpy as np
from numpy import mean
import cv2 as cv
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut

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
            image = cv.imread(imgpath, cv.IMREAD_COLOR)
            image = cv.resize(image, (200, 200))
            rgbImg = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            hsvImg = cv.cvtColor(rgbImg, cv.COLOR_BGR2HSV)
            hsvFeature = [np.mean(hsvImg), np.std(hsvImg)]
            # hsvhist, bins = np.histogram(hsv.ravel(), 256, [0, 256])
            # hsvhist = np.transpose(hsvhist[0:18, np.newaxis])
            # hsvhist = hsvhist.astype("float")

            feature_matrix = np.array(hsvImg).flatten()
            data.append([feature_matrix, label])

    TulisDataset("jamurDataset5.pickle", data)


def getScore(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)
    return model.score(xtest, ytest)


def evaluate_model(cv, model, X, y):
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()


def SVM():
    data1 = BacaDataset("jamurDataset5.pickle")

    features = []
    labels = []

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.3)

    # Membuat model

    model = SVC(C=1, kernel='poly', gamma='auto')
    model.fit(xtrain, ytrain)
    TulisDataset('model5.sav', model)

    model1 = BacaDataset('model5.sav')
    prediksi = model1.predict(xtest)
    testData = ytest
    means, mins, maxs = list(), list(), list()

#    # Kfold
    for i in range(2, 16):
        cv = KFold(n_splits=i, shuffle=True, random_state=1)
        k_mean, k_min, k_max = evaluate_model(cv, model1, features, labels)
        scores = cross_val_score(
            model1, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        print(scores)
        print('> folds=%d, accuracy(mean)=%.3f (min = %.3f,max = %.3f)' %
              (i, k_mean, k_min, k_max))
        means.append(k_mean)
        # menyimpan min and max terkait dengan mean
        mins.append(k_mean - k_min)
        maxs.append(k_max - k_mean)

    # akurasi = model1.score(xtest, ytest)

    # print('Akurasi: ', akurasi)
    # print("Precision:", metrics.precision_score(ytest, prediksi))
    # print("Recall:", metrics.recall_score(ytest, prediksi))

    for i in range(14):
        print('Prediksi: ', categories[prediksi[i]],
              'Data test:', categories[testData[i]])
    jamoer = xtest[i]
    plt.imshow(jamoer, cmap='gray')
    plt.show()


SetupDataset()
SVM()
