import numpy as np
import cv2 as cv
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

categories = ['Backup gambar\Beracun noBG', 'Backup gambar\BisaDimakan noBG']
# categories = ['Beracun', 'BisaDimakan']

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

            feature_matrix = np.array(image).flatten()
            data.append([feature_matrix, label])

    TulisDataset("jamurDataset1.pickle", data)


def getScore(cvK, model, X, y):
    # evaluate the model
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cvK, n_jobs=-1)
    # return scores
    return scores, scores.mean(), scores.min(), scores.max()


def SVMandKfoldCV():
    data1 = BacaDataset("jamurDataset1.pickle")

    features = []
    labels = []

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    # Membuat model
    model1 = SVC(C=1, kernel='poly', gamma='auto')
    means, mins, maxs = list(), list(), list()

    # Kfold
    cvK = KFold(n_splits=5, shuffle=True, random_state=42)
    perIndx = 1
    for train_index, test_index in cvK.split(features):
        # split data train dan test dengan kfold.split
        X_train, X_test = np.array(
            features)[train_index], np.array(features)[test_index]
        y_train, y_test = np.array(
            labels)[train_index], np.array(labels)[test_index]
        print("TRAIN:", train_index, "TEST:", test_index)
        model1.fit(X_train, y_train)

        # cros validation
        scores, k_mean, k_min, k_max = getScore(
            cvK, model1, X_test, y_test)
        print('Isi Score = ', scores)
        print('> Percobaan ke :%d, accuracy(mean)=%.3f (min = %.3f,max = %.3f)' %
              (perIndx, k_mean, k_min, k_max))
        perIndx += 1

        # for i in range(0, len(test_index)):
        #     jamoer = X_test[i].reshape(200, 200)
        #     plt.imshow(jamoer, cmap='gray')
        #     plt.show()

        y_pred = cross_val_predict(model1, X_test, y_test, cv=cvK)
        print(y_test)
        print(y_pred)
        # menampilkan benar tidaknya prediksi (satuan) dalam confusion matrix
        print(confusion_matrix(y_pred=y_pred, y_true=y_test))
        # menampilkan benar tidaknya prediksi (presentase) dalam confusion matrix
        matrix = plot_confusion_matrix(model1, X=X_test, y_true=y_test,
                                       display_labels=categories,
                                       cmap=plt.cm.Blues,
                                       normalize='true')
        plt.title('Confusion matrix for our classifier')
        plt.show()


SetupDataset()
SVMandKfoldCV()
