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
import numpy as np 

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

PATH = '/home/fizzer/Pictures/platetest_cropped.png'

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        #self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)
        self.numbermodel = pickle.load(open("numbermodel.p", "rb")).pop()

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow('frame',frame)
        cv2.waitKey()

        corners = findPlate.findCorners(frame)
        if corners.size == 0:
            print('nothing found boss')
        else:

          letters = findPlate.findLetters(corners)
          testing_image = letters[4]
          testing_image = np.expand_dims(testing_image, axis=0)
          testing_image = np.expand_dims(testing_image, axis=3)
          print("Predicted as a: {}".format(np.argmax(self.numbermodel.predict(testing_image))))
        
        



def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


