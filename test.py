#!/usr/bin/python
import numpy as np
from Tools import getNumsFromImage
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("images//img-8.png");

images = getNumsFromImage(img)

cv2.waitKey(0)

plt.show()