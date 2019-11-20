import findPlate
import pickle
import cv2
import numpy as np
import keras
PATH = '/home/fizzer/test.png'

if __name__ == "__main__":
    number_model = pickle.load(open("numbermodel.p", "rb")).pop()

    image = cv2.imread(PATH)
    corners = findPlate.findCorners(image)
    letters = findPlate.findLetters(corners)
    testing_image = letters[4]
    testing_image = np.expand_dims(testing_image, axis=0)
    testing_image = np.expand_dims(testing_image, axis=3)


    print(np.argmax(number_model.predict(testing_image)))
    cv2.imshow('window',letters[4])
    cv2.waitKey(0)