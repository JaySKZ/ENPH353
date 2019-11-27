import findPlate
import pickle
import cv2
import numpy as np
import keras
from keras.models import load_model
import pytesseract
PATH = '/home/fizzer/test.png'

if __name__ == "__main__":
    #number_model = pickle.load(open("lettermodel.p", "rb")).pop()
    #number_model = load_model("lettermodel.h5")

    image_orig = cv2.imread(PATH)

    image = findPlate.findCorners(image_orig)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])
    mask1 = cv2.inRange(image, lower_black, upper_black)

    lower_blue = np.array([80,80,15])
    upper_blue = np.array([125,255,255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.imshow('1',mask1)
    cv2.imshow('2',mask2)
    cv2.waitKey(0)
    config = ("-l eng --oem 1 --psm 7")
    text1 = pytesseract.image_to_string(mask1, config=config)
    text2 =pytesseract.image_to_string(mask2, config=config)

    print(text1)
    print(text2)
