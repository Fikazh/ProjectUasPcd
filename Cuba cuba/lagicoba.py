import pandas as pd
import pickle
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 as cv
import os
import re
from sklearn.svm import SVC

# -------------------- Utility function ------------------------


def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("", "", "()"))
    str_ = str_.split("_")
    return ''.join(str_[:2])


def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text


def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
    progr = "#"*round((val)*bar_size/val_len) + " " * \
        round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end="\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" %
              (progr, folder, sub_folder, filename), end="\r")


# -------------------- Load Dataset ------------------------

dataset_dir = "DATASET\\"

imgs = []  # list image matrix
labels = []
descs = []
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder_files = os.listdir(
            os.path.join(dataset_dir, folder, sub_folder))
        len_sub_folder = len(sub_folder_files) - 1
        for i, filename in enumerate(sub_folder_files):
            img = cv.imread(os.path.join(
                dataset_dir, folder, sub_folder, filename))

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # h, w = gray.shape
            # ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
            # crop = gray[ymin:ymax, xmin:xmax]

            resize = cv.resize(gray, (200, 200), fx=0.5, fy=0.5)

            imgs.append(resize)
            labels.append(normalize_label(os.path.splitext(filename)[0]))
            descs.append(normalize_desc(folder, sub_folder))

            print_progress(i, len_sub_folder, folder, sub_folder, filename)


# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------

def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):

    glcm = greycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[
        0]]
    for item in glcm_props:
        feature.append(item)
    feature.append(label)

    return feature


# ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
properties = ['dissimilarity', 'correlation',
              'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []
for img, label in zip(imgs, labels):
    glcm_all_agls.append(
        calc_glcm_all_agls(img,
                           label,
                           props=properties)
    )

columns = []
angles = ['0', '45', '90', '135']
for name in properties:
    for ang in angles:
        columns.append(name + "_" + ang)

columns.append("label")


# Create the pandas DataFrame for GLCM features data
glcm_df = pd.DataFrame(glcm_all_agls,
                       columns=columns)

# save to csv
glcm_df.to_csv("glcm_coffee_dataset.csv")

features = pd.read_csv('glcm_coffee_dataset.csv')
xtrain, xtest, ytrain, ytest = train_test_split(
    features, labels, test_size=0.3, random_state=45)

model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)
filename = 'model.sav'

# tulis sav model
pickle.dump(model, open(filename, 'wb'))

# baca sav model
model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


prediksi = model.predict(xtest)
akurasi = model.score(xtest, ytest)
print(labels)
prediksi = model.predict(xtest)
akurasi = model.score(xtest, ytest)

print('Akurasi: ', akurasi)
cv.imshow("test img", imgs[8])
cv.imshow("test ", imgs[10])

cv.waitKey(0)
cv.destroyAllWindows()
