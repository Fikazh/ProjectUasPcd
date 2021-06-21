import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

categories = ['Amanita phalloides(Beracun)',
              'Boletus edulis (Bisa dimakan)',
              'Conocybe filaris (Beracun)',
              'Cortinarius caperatus(Bisa Dimakan)',
              'Lactarius Turpis(Beracun)',
              'Suillus bovinus (Bisa Dimakan)']
data = []
dir = os.getcwd()


def GrabCut(source):
    img = cv.imread(source)
    image_resized = cv.resize(img, (200, 200))
    mask = np.zeros(image_resized.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, 195, 195)
    cv.grabCut(image_resized, mask, rect, bgdModel,
               fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image_resized = image_resized*mask2[:, :, np.newaxis]
    imgGray = cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY)
    return imgGray


def TulisDataset(name, ext):
    pick_in = open(name, 'wb')
    pickle.dump(ext, pick_in)
    pick_in.close()


def BacaDataset(name):
    pick_in = open(name, 'rb')
    ext = pickle.load(pick_in)
    pick_in.close()
    return ext


def main():
    for category in categories:
        path = os.path.join(dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = cv.imread(imgpath, 0)
            image = cv.resize(image, (200, 200))
            # GrabCut(imgpath)
            # cv.resize(image, (50, 50))

            image_to_array = np.array(image).flatten()
            data.append([image_to_array, label])
            # print(imgpath)

    TulisDataset("jamurDataset.pickle", data)


def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    return contrast


def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity


def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity


def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return energy


def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return correlation


def asm_feature(matrix_coocurrence):
    asm = greycoprops(matrix_coocurrence, 'ASM')
    return asm


def SVM():
    data1 = BacaDataset("jamurDataset.pickle")
    features = []
    labels = []

    for feature, label in data1:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=0.30)

    # model = SVC(C=1, kernel='poly', gamma='auto')
    # model.fit(xtrain, ytrain)
    # TulisDataset('model1.sav', model)

    model1 = BacaDataset('model1.sav')
    print(features)

    prediksi = model1.predict(xtest)
    akurasi = model1.score(xtest, ytest)

    print('Akurasi: ', akurasi)

    # for i in range(13):
    # print('Prediksi: ', categories[prediksi[i]])
    # jamoer = xtest[i].reshape(50, 50)
    # plt.imshow(jamoer, cmap='gray')
    # plt.show()


# main()
SVM()
cv.waitKey(0)
cv.destroyAllWindows()
