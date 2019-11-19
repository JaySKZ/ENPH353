
import findPlate
import pickle
import cv2
import numpy as np

PATH = '/home/fizzer/Pictures/platetest_cropped.png'
#Setup: load cnn from pickle


#Main loop: continually read camera footage and send data to cnn. If results are unique then send to license server

if __name__ == "__main__":
    model = pickle.load(open("model.p", "rb")).pop()
    raw_image = cv2.imread(PATH)
    coordinates = findPlate.findCorners(raw_image)
    image = findPlate.perspectiveShift(raw_image, coordinates)
    i1, i2, i3 = findPlate.findLetters(image)
    cv2.imshow("yeet", i1)
    cv2.waitKey(0)
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i1 = cv2.resize(i1, (64,64))
    i1 = np.expand_dims(i1,axis=2)
    i1 = np.expand_dims(i1,axis=0)
    
    



    print((model.predict(i1)))

