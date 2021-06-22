import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'''
Membuat Dataset
'''
# categories = ['Agaricus (Bisa Dimakan)', 'Amanita(Beracun)', 'Boletus(Bisa dimakan)', 'Conocybe filaris (Beracun)', 'Cortinarius (Bisa Dimakan)', 'Russula (Beracun)']

# dir = os.getcwd()

# data = []

# for category in categories:
#     path = os.path.join(dir, category)
#     label = categories.index(category)

#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         image = cv.imread(imgpath, 0)
#         image_resized = cv.resize(image,(50,50))
#         image_to_array = np.array(image_resized).flatten()

#         data.append([image_to_array, label])

'''
Menyimpan dataset ke folder directory (dimatikan jika dataset sudah tersimpan)
'''
# pick_in = open('mushroom_dataset.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick.close()

'''
Membaca dataset yang sudah dibuat
'''
pick_in = open('mushroom_dataset.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
print(data)

# random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(
    features, labels, test_size=0.3)

'''
Membuat dan menyimpan model (dimatikan jika model sudah tersimpan)
'''
# model = SVC(C=1, kernel= 'poly', gamma= 'auto')
# model.fit(xtrain,ytrain)

# pick = open('model.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()

'''
Membaca model yang sudah dibuat
'''
pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()
print(model)

prediksi = model.predict(xtest)
akurasi = model.score(xtest, ytest)

categories = ['Agaricus (Bisa Dimakan)', 'Amanita(Beracun)', 'Boletus(Bisa dimakan)',
              'Conocybe filaris (Beracun)', 'Cortinarius (Bisa Dimakan)', 'Russula (Beracun)']


print('Akurasi: ', akurasi)

for i in range(13):
    print('Prediksi: ', categories[prediksi[i]])
    # jamoer = xtest[i].reshape(50, 50)
    # plt.imshow(jamoer, cmap='gray')
    # plt.show()
