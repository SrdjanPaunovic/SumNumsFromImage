import numpy as np
import cv2
from Rectangle import Rectangle
from Rectangle import intersect_collection


def getNumsFromImage(img):
    #cv2.imshow('img', img)
    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    shape_mask = 255 - cv2.inRange(img, lower, upper)

    #cv2.imshow('mask', shape_mask)
    cv2.imwrite('mask.png', shape_mask)
    img1 = cv2.imread("mask.png")
    imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    imgray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY)
    (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('thresh', img1)

    valid_contours = []
    valid_rect = []
    images = []
    i = 0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        rect = Rectangle(x, y, width, height)
        if width == 30 and height == 30:
            if not intersect_collection(rect, valid_rect):
                i += 1
                valid_rect.append(rect)
                valid_contours.append(contour)
                crop_img = imgray[y:y + height, x:x + width]
               # cv2.imwrite('gray{0}.png '.format(i), crop_img)
                crop_img = cv2.resize(crop_img, (20, 20), interpolation=cv2.INTER_AREA )

              #  cv2.imshow('bla',crop_img)
                images.append(crop_img)
                # print rect

    return np.array(images)


def train(train_image,save_path):

    img = cv2.imread(train_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)
    #cv2.imshow('nummm.png', gray[0:20,0:20])
    # Now we prepare train_data and test_data.
    train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
    #test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    #test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train, train_labels)

    np.savez(save_path, train=train, train_labels=train_labels)
    #ret, result, neighbours, dist = knn.find_nearest(test, k=5)

def find_sum(image_path,knn):
    img = cv2.imread(image_path)
    images = getNumsFromImage(img)
    samples = images[:].reshape(-1, 400).astype(np.float32)
    #print np.shape(samples)


    ret, result, neighbours, dist = knn.find_nearest(samples, k=5)

    return np.sum(result)