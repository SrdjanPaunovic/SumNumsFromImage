#!/usr/bin/python
import numpy as np
from Tools import find_sum
from Tools import train
import cv2
from matplotlib import pyplot as plt


out = {}
n = 0
correct = 0;
#train('digits.png','knn_data.npz')

with np.load('knn_data.npz') as data:
    train = data['train']
    train_labels = data['train_labels']

knn = cv2.KNearest()
knn.train(train, train_labels)

out_file = open('out.txt','w')

with open('res.txt') as file:
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id == 0):
            out_file.write('Srdjan Paunovic\n')
            out_file.write('file\tsum\n')
            continue
        if(id>0):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            SUM = find_sum(cols[0],knn)
            out_file.write('{0}\t{1}\n'.format(cols[0],SUM))
        continue

out_file.close()
cv2.waitKey(0)

plt.show()