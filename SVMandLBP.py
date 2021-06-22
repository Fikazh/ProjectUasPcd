import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

categories = ['Beracun', 'BisaDimakan']
data = []
dir = os.getcwd()


def get_pixel(img, center, x, y):

    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
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

    # Now, we need to convert binary
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

            # Pengubahan menjadi Tekstur
            height, width = image.shape
            img_lbp = np.zeros((height, width), np.uint8)
            for i in range(0, height):
                for j in range(0, width):
                    img_lbp[i, j] = lbp_calculated_pixel(image, i, j)

            feature_matrix = np.array(img_lbp).flatten()
            data.append([feature_matrix, label])

    TulisDataset("jamurDataset2.pickle", data)


def SVM():
    data1 = BacaDataset("jamurDataset2.pickle")
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
    # TulisDataset('model2.sav', model)

    model2 = BacaDataset('model2.sav')

    prediksi = model2.predict(xtest)
    akurasi = model2.score(xtest, ytest)

    print('Akurasi: ', akurasi)
    print(ytest)

    # for i in range(14):
    #     print('Prediksi: ', categories[prediksi[i]])
    #     jamoer = xtest[i].reshape(200, 200)
    #     plt.imshow(jamoer, cmap='gray')
    #     plt.show()
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()


# SetupDataset()
SVM()
