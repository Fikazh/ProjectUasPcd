import numpy as np
import cv2 as cv
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from numpy import mean
from skimage import data
from sklearn.svm import SVC
from sklearn import metrics
from skimage import feature
from skimage.color import label2rgb
from skimage.transform import rotate
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage.feature import local_binary_pattern
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict



categories = ['Beracun', 'BisaDimakan']
dataGray, dataLBP, dataLBPHist = list(), list(), list()
dir = os.getcwd()

# settings for LBP
METHOD = 'uniform'
radius = 3
n_points = 8 * radius


def TulisDataset(name, ext):
    pick_in = open(name, 'wb')
    pickle.dump(ext, pick_in)
    pick_in.close()


def BacaDataset(name):
    pick_in = open(name, 'rb')
    ext = pickle.load(pick_in)
    pick_in.close()
    return ext

def DuplicateDataset():
    for category in categories:
        path = os.path.join(dir, category)
        i=24
        j=48
        k=72

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = Image.open(imgpath)

            vertical_img = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            vertical_img.save(f'{path}/{i}.png')

            horizontal_img = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            horizontal_img.save(f'{path}/{j}.png')

            vertical_horizontal_img = horizontal_img.transpose(method=Image.FLIP_TOP_BOTTOM)
            vertical_horizontal_img.save(f'{path}/{k}.png')

            i+=1
            j+=1
            k+=1

def SetupDataset():
    for category in categories:
        path = os.path.join(dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = cv.imread(imgpath, 0)
            image = cv.resize(image, (200, 200))

            # Perubahan image menjadi Tekstur
            lbp = local_binary_pattern(image, n_points, radius, METHOD)
            hist, bins = np.histogram(lbp.ravel(), 256, [0, 256])
            hist_transponse = np.transpose(hist[0:26, np.newaxis]).flatten()
            feature_matrix = np.array(hist_transponse).flatten()
            dataGray.append([image, label])
            dataLBP.append([lbp, label])
            dataLBPHist.append([feature_matrix, label])

    TulisDataset("jamurDatasetGray.pickle", dataGray)
    TulisDataset("jamurDatasetLBP.pickle", dataLBP)
    TulisDataset("jamurDatasetLBPHist.pickle", dataLBPHist)


def getDatasets(datasets):
    data2 = BacaDataset(datasets)
    features = []
    labels = []

    for feature, label in data2:
        features.append(feature)
        labels.append(label)
    return features, labels


def getScores(cv, model, X, y):
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return scores, mean(scores), scores.min(), scores.max()


def SVMandKfold():
    featuresLBP, labelsLBP = getDatasets("jamurDatasetLBP.pickle")
    features, labels = getDatasets("jamurDatasetLBPHist.pickle")

    # Membuat model
    model2 = SVC(C=1, kernel='poly', gamma='auto')
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
        model2.fit(X_train, y_train)

        # cros validation
        scores, k_mean, k_min, k_max = getScores(
            cvK, model2, X_test, y_test)
        print('Isi Score = ', scores)
        print('> Percobaan ke :%d, accuracy(mean)=%.3f (min = %.3f,max = %.3f)' %
              (perIndx, k_mean, k_min, k_max))
        perIndx += 1

        y_pred = cross_val_predict(model2, X_test, y_test, cv=cvK)
        print(y_test)
        print(y_pred)
        # menampilkan benar tidaknya prediksi (satuan) dalam confusion matrix
        print(confusion_matrix(y_pred=y_pred, y_true=y_test))
        # menampilkan benar tidaknya prediksi (presentase) dalam confusion matrix
        matrix = plot_confusion_matrix(model2, X=X_test, y_true=y_test,
                                       display_labels=categories,
                                       cmap=plt.cm.Blues,
                                       normalize='true')
        plt.title('Confusion matrix for our classifier')
        plt.show()


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')


def showHistogram():
    featuresGray, labelsGray = getDatasets("jamurDatasetGray.pickle")
    featuresLBP, labelsLBP = getDatasets("jamurDatasetLBP.pickle")

    jamoer = featuresGray[0].reshape(200, 200)
    plt.imshow(jamoer, cmap='gray')
    plt.show()

    jamoer = featuresLBP[0].reshape(200, 200)
    plt.imshow(jamoer, cmap='gray')
    plt.show()

    plt.style.use("ggplot")
    (fig, ax) = plt.subplots()
    fig.suptitle("Local Binary Patterns")
    plt.ylabel("% of Pixels")
    plt.xlabel("LBP pixel bucket")

    ax.hist(featuresLBP[0].ravel(), density=True, bins=100, range=(0, 256))
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 0.5])
    plt.show()

'''
DupicateDataset() dipakai hanya sekali,
jika dipakai lagi sebelum image no 24-95 di folder Beracun&BisaDimakan
dihapus, maka gambarnya jadi berantakan
'''
#DuplicateDataset()

SetupDataset()
SVMandKfold()
showHistogram()
