import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import numpy as np
import xlsxwriter as xls
from skimage.feature import greycomatrix, greycoprops
import math
from scipy import stats
from sklearn.svm import SVC
import pandas as pd

# book = xls.Workbook('dataFeature.xlsx')
# sheet = book.add_worksheet()
# sheet.write(0, 0, 'file')

# column = 1
# # glcm
# glcmFeature = ['correlation', 'homogeneity',
#                'dissimilarity', 'contrast', 'energy', 'ASM']
# angle = ['0', '45', '90', '135']
# for i in glcmFeature:
#     for j in angle:
#         sheet.write(0, column, i+" "+j)
#         column += 1
# sheet.write(0, column, "Label")

# # citra
# statusJamur = ['Beracun', 'BisaDimakan']
# sum_each_type = 23
# row = 1
# for i in statusJamur:
#     for j in range(0, sum_each_type+1):
#         column = 0
#         fileName = 'Datasets/'+i+"/"+str(j)+'.jpg'
#         print(fileName)
#         sheet.write(row, column, fileName)
#         column += 1

#         img = cv.imread(fileName, 1)
#         imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         ret, img1 = cv.threshold(imgGray, 129, 255, cv.THRESH_BINARY_INV)
#         img1 = cv.dilate(img1.copy(), None, iterations=5)
#         img1 = cv.erode(img1.copy(), None, iterations=5)
#         b, g, r = cv.split(img)
#         rgba = [b, g, r, img1]
#         dst = cv.merge(rgba, 4)

#         contours, hierarchy = cv.findContours(
#             img1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#         select = max(contours, key=cv.contourArea)
#         x, y, w, h = cv.boundingRect(select)
#         png = dst[y:y+h:, x:x+w]

#         gray = cv.cvtColor(png, cv.COLOR_BGR2GRAY)
#         gray = cv.resize(gray, (200, 200))

#         # glcm
#         distances = [5]
#         angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#         levels = 256
#         symetric = True
#         normed = True

#         glcm = greycomatrix(gray, distances, angles,
#                             levels, symetric, normed)

#         glcmProps = [propery for name in glcmFeature for propery in greycoprops(glcm, name)[
#             0]]
#         for item in glcmProps:
#             sheet.write(row, column, item)
#             column += 1
#         sheet.write(row, column, i)
#         row += 1
# book.close()

data = pd.read_excel('dataFeature.xlsx', sheet_name='Sheet1')
# data2 = pd.read_excel('dataFeature.xlsx', sheet_name='Sheet2')
enc = LabelEncoder()
data['Label'] = enc.fit_transform(data['Label'].values)

# random.shuffle(data)
atrData = data.drop(columns='file')
atrData = atrData.drop(columns='Label')
clsData = data['Label']

# atrData2 = data2.drop(columns='file')
# atrData2 = atrData2.drop(columns='Label')
# clsData2 = data2['Label']
# print(clsData)

model = SVC(C=1, kernel='linear', gamma='auto')
xtrain, xtest, ytrain, ytest = train_test_split(
    atrData, clsData, test_size=0.2, random_state=42)
# tree_data = SVC(random_state=2)
# tree_data.fit(xtrain, ytrain)
# xtrain = atrData
# ytrain = clsData

# xtest = atrData2
# # ytest = clsData2
# print(xtest)

model = model.fit(xtrain, ytrain)
prediksi = model.predict(xtest)
# akurasi = model.score(xtest, ytest)
prediksiBenar = (prediksi == ytest).sum()
prediksiSalah = (prediksi != ytest).sum()

print(prediksi)
print(prediksiBenar/(prediksiBenar+prediksiSalah)*100, "%")
