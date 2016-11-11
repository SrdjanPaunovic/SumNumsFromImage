#!/usr/bin/python
import numpy as np
from Tools import getTrainRow
import cv2
from matplotlib import pyplot as plt

import sys

train_data = np.array([])
responses = []
n = 0
with open('res.txt') as file:
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id>0):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            n += 1

            train_row = getTrainRow(cols[0])
            if(len(train_data) == 0):
                train_data = train_row.astype(np.float32)
                responses = np.float32(cols[1])
            else:
                train_data = np.vstack([train_data,train_row.astype(np.float32)])
                responses = np.vstack([responses,np.float32(cols[1])])
            continue


#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html
help(cv2.ml)
knn = cv2.ml.KNearest_create()
knn.train(train_data,cv2.ml.ROW_SAMPLE,responses)
for_find = getTrainRow("images/img-0.png").astype(np.float32)
#ret, results,neighbours,dist = knn.findNearest(train_data[3], 1)
cv2.waitKey(0)

plt.show()