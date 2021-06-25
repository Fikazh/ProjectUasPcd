import numpy as np
from numpy import mean
import cv2 as cv
from matplotlib import pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, StratifiedKFold

categories = ['Beracun', 'BisaDimakan']
data = []
dir = os.getcwd()


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1

    except:
        pass

    return new_value


def lbp_calculated_pixel(img, x, y):

    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))

    # top
    val_ar.append(get_pixel(img, center, x-1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))

    # left
    val_ar.append(get_pixel(img, center, x, y-1))

    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


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

            # Perubahan image menjadi Tekstur
            height, width = image.shape
            img_lbp = np.zeros((height, width), np.uint8)
            for i in range(0, height):
                for j in range(0, width):
                    img_lbp[i, j] = lbp_calculated_pixel(image, i, j)

            feature_matrix = np.array(img_lbp).flatten()
            data.append([feature_matrix, label])

    TulisDataset("jamurDataset2.pickle", data)


def evaluate_model(cv, model, X, y):
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()


def SVM():
    data2 = BacaDataset("jamurDataset2.pickle")
    features = []
    labels = []

    for feature, label in data2:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    # Membuat model

    model = SVC(C=1, kernel='poly', gamma='auto')
    model.fit(xtrain, ytrain)
    TulisDataset('model2.sav', model)

    model2 = BacaDataset('model2.sav')
    prediksi = model2.predict(xtest)
    testData = ytest
    akurasi = model2.score(xtest, ytest)

    means, mins, maxs = list(), list(), list()

    # Kfold
    for i in range(2, 16):
        cv = KFold(n_splits=i, shuffle=True, random_state=1)
        k_mean, k_min, k_max = evaluate_model(cv, model2, features, labels)
        scores = cross_val_score(
            model2, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        print(scores)
        print('> folds=%d, accuracy(mean)=%.3f (min = %.3f,max = %.3f)' %
              (i, k_mean, k_min, k_max))
        means.append(k_mean)
        # menyimpan min and max terkait dengan mean
        mins.append(k_mean - k_min)
        maxs.append(k_max - k_mean)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # # enumerate the splits and summarize the distributions
    # for train_ix, test_ix in kfold.split(features, labels):
    #     # select rows
    #     train_X, test_X = features[train_ix], features[test_ix]
    #     train_y, test_y = labels[train_ix], labels[test_ix]
    #     # summarize train and test composition
    #     train_0, train_1 = len(train_y[train_y == 0]), len(
    #         train_y[train_y == 1])
    #     test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
    #     print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' %
    #           (train_0, train_1, test_0, test_1))

    # print('Akurasi: ', akurasi)
    # print("Precision:", metrics.precision_score(ytest, prediksi))
    # print("Recall:", metrics.recall_score(ytest, prediksi))

    for i in range(14):
        print('Prediksi: ', categories[prediksi[i]],
              'Data test:', categories[testData[i]])
        # jamoer = xtest[i].reshape(200, 200)
        # plt.imshow(jamoer, cmap='gray')
        # plt.show()
        # cv.waitKey(0)
        # cv.destroyAllWindows()


# SetupDataset()
SVM()
