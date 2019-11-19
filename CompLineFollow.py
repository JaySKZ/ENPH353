#!/usr/bin/env python
from __future__ import print_function

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

# Ideal distance from curb (in pixels)
separation = 400 

# State (need to figure out how to change this)
stage = 2

# last error (need this somewhere outside loop)
last_cX = 900

class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        h = np.size(frame, 0)
        w = np.size(frame, 1)

        # Make feed image HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of white color in HSV
        lower_white = np.array([0, 0, 212])
        upper_white = np.array([0, 0, 255])

        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Remove noise
        kernel_erode = np.ones((4,4), np.uint8)
        eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
        kernel_dilate = np.ones((6,6),np.uint8)
        dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
        
        # This is enough image pre-processing, contour finding happens depending state machine

        # Initial start
        if (stage == 1):
            # Set region of interest to left of screen
            roi = dilated_mask[h-100:h, 0:350]

            # Find the different contours
            im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort by area (keep only the biggest one), which should theoretically be the curb 
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

            M = cv2.moments(contours[0])
            
            # Centroid
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])

            error = abs((w/2 - cX)) - separation


        # Outer ring, go anti-clockwise, track right curb
        elif (stage == 2):
            # Set region of interest to right of screen
            roi = dilated_mask[h-100:h, 750:w]

            M = cv2.moments(roi)

            if (M['m00'] != 0):
                # Centroid
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
            else:
                cX = 400

            error = cX - separation


        # Inner ring, go clockwise, track left curb
        elif (stage == 3):
            # Set region of interest to left of screen
            roi = dilated_mask[h-200:h, 750:w]

            # Find the different contours
            im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort by area (keep only the biggest one), which should theoretically be the curb 
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

            if len(contours) > 0:
                M = cv2.moments(contours[0])

                # Centroid
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
            
                if(abs(last_cX - cX) >= 400):
                    cX = last_cX

            error = abs((w/2 - cX)) - separation 

        velocity = Twist()

        if (abs(error) < 50):
            velocity.angular.z = 0
            velocity.linear.x = 0.2
        elif (error >= 50):
            velocity.linear.x = 0
            if (stage == 2):
                velocity.angular.z = -0.1
            else:
                velocity.angular.z = 0.1
        else:
            velocity.linear.x = 0
            if (stage == 2):
                velocity.angular.z = 0.1
            else:
                velocity.angular.z = -0.1

        self.vel_pub.publish(velocity)

        print(cX)
        cv2.circle(roi, (int(cX), h-100), 20, (0, 0, 255), -1)
        cv2.imshow("Robot Camera", roi)
        cv2.waitKey(1)

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