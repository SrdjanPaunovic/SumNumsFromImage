import numpy as np
import cv2
from Rectangle import Rectangle
from Rectangle import intersect_collection

def getNumsFromImage(img):
    cv2.imshow('img', img)
    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    shape_mask = cv2.inRange(img, lower, upper)

    #cv2.imshow('mask', shape_mask)
    cv2.imwrite('mask.png', shape_mask)
    img1 = cv2.imread("mask.png")
    imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('thresh', img1)

    valid_contours = []
    valid_rect = []
    images = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        rect = Rectangle(x, y, width, height)
        if width == 30 and height == 30:
            if not intersect_collection(rect, valid_rect):
                valid_rect.append(rect)
                valid_contours.append(contour)
                crop_img = imgray[y:y + height, x:x + width]
                images.append(crop_img)
            #print rect

    return images
    #print len(valid_rect)

    #cv2.drawContours(img, valid_contours, -1, (0, 255, 0), 3)
    #cv2.imshow('contur', img)