from __future__ import print_function
import findPlate
import pickle
import cv2
import numpy as np
import roslib
#roslib.load_manifest('enph353_ros_lab')
import sys
import rospy
import cv2
import keras
from numpy import loadtxt
from keras.models import load_model
from keras import models
import numpy as np 

from std_msgs.msg import String
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

PATH = '/home/fizzer/Pictures/platetest_cropped.png'

class license_reader:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.cars_pub = rospy.Publisher('cars', Int16)
        self.license_pub = rospy.Publisher('/license_plate', String)

        self.numbermodel = load_model("numbermodel.h5")
        self.numbermodel._make_predict_function()
        self.lettermodel = load_model("lettermodel.h5")
        self.lettermodel._make_predict_function()
        self.parkingmodel = load_model("parkingmodel.h5")
        self.parkingmodel._make_predict_function()

        self.last_confidence = [0,0,0,0,0]
        self.result = ['a','a','a','a','a']
        self.Flag = False
        self.cars = 0
        self.license_pub.publish('TeamYESSIR,andrew,0,YE37')


    def callback(self, data):
        confidence = self.last_confidence
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        #Search frame for back of car
        corners = findPlate.findCorners(frame)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

        #If ROI found doesnt meet requirements; then reset
        if corners.shape[0] == 0 :
            #print('nothing found boss')
            if self.Flag == True:
              self.Flag = False
              self.cars += 1
              self.cars_pub.publish(self.cars)
              self.license_pub.publish('TeamYESSIR,andrew,{},{}{}{}{}'.format(self.result[0], self.result[1], self.result[2], self.result[3], self.result[4]))
            self.last_confidence = [0,0,0,0,0]
            self.result = ['a','a','a','a','a']

        else:
          self.Flag = True         
          #Find letters
          letters = findPlate.findLetters(corners)
        
          #Transform top number to make nicer
          #pts1 = np.float32([[0,0],[64,0],[0,64],[64,64]])
          #pts2 = np.float32([[15,3],[49,3],[15,61],[49,61]])
          #M = cv2.getPerspectiveTransform(pts1,pts2)  
          #letters[4] = cv2.warpPerspective(letters[4],M,(64,64))
          #letters[4][:,0:15]= 255
          #letters[4][:,49:64]= 255
          #letters[4][59:64,:]= 255
          #letters[4][0:5,:]= 255
          cv2.imshow('corners',corners)
          #cv2.imshow('1', letters[0])
          #cv2.imshow('2', letters[1])
          #cv2.imshow('3', letters[2])
          #cv2.imshow('4', letters[3])
          #cv2.imshow('5', letters[4])    
          cv2.waitKey(1)
          count = 0
          
          #Find max. confidence digit to each letter. The weird count and index stuff is because the order of digits in the array is all whacked up
          for i in [4,0,1,2,3]:
            testing_image = letters[i]
            testing_image = np.expand_dims(testing_image, axis=0)
            testing_image = np.expand_dims(testing_image, axis=3)
            
            #Parking Spot
            if i == 4:
              if np.max(self.parkingmodel.predict(testing_image)) > confidence[count]:
                confidence[count] = np.max(self.parkingmodel.predict(testing_image))
                prediction = np.argmax(self.parkingmodel.predict(testing_image))
                self.result[count] = chr(prediction + 49)
            #Letters of plate
            elif(i == 0 or i == 1):
              if np.max(self.lettermodel.predict(testing_image)) > confidence[count]:
                confidence[count] = np.max(self.lettermodel.predict(testing_image))
                prediction = np.argmax(self.lettermodel.predict(testing_image))
                self.result[count] = chr(prediction + 65)
            #Numbers of plate
            else:
              if np.max(self.lettermodel.predict(testing_image)) > confidence[count]:
                confidence[count] = np.max(self.lettermodel.predict(testing_image))
                prediction = np.argmax(self.numbermodel.predict(testing_image))
                self.result[count] = chr(prediction+48)
            count += 1
          
          self.last_confidence = confidence
          print(confidence)
          print(self.result)
        



def main(args):
  rospy.init_node('license_reader', anonymous=True)
  ic = license_reader()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


