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

class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)
        self.stage = 1
        self.separation = 400
        self.error = 0

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
        if (self.stage == 1):
            # Set region of interest to left of screen
            roi_center = dilated_mask[h-200:h, (w/2)-50:(w/2)+50]
            roi_right = dilated_mask[h-100:h, 700:w]

            M1 = cv2.moments(roi_center)
            M2 = cv2.moments(roi_right)

            if (M1['m00'] == 0):
                self.error = 0

            elif (M1['m00'] != 0):
                if (M2['m00'] == 0):
                    self.error = 200

                elif (M2['m00'] != 0):
                    self.stage = 2


        # Outer ring, go anti-clockwise, track right curb
        elif (self.stage == 2):
            # Set region of interest to right of screen
            roi = dilated_mask[h-100:h, 750:w]

            M = cv2.moments(roi)

            if (M['m00'] != 0):
                # Centroid
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
            else:
                cX = 400

            self.error = cX - self.separation


        # Inner ring, go clockwise, track left curb
        elif (self.stage == 3):

            self.separation = 420

            # Set region of interest to right of screen
            roi = dilated_mask[h-150:h, 750:w]

            M = cv2.moments(roi)

            if (M['m00'] != 0):
                # Centroid
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
            else:
                cX = 400

            self.error = cX - self.separation


        velocity = Twist()

        if (abs(self.error) < 50):
            velocity.angular.z = 0
            velocity.linear.x = 0.2
        elif (self.error >= 50):
            velocity.linear.x = 0
            if (self.stage == 2 or self.stage == 3):
                velocity.angular.z = -0.1
            else:
                velocity.angular.z = 0.1
        else:
            velocity.linear.x = 0
            if (self.stage == 2 or self.stage == 3):
                velocity.angular.z = 0.1
            else:
                velocity.angular.z = -0.1

        self.vel_pub.publish(velocity)

        #print(cX)
        #cv2.circle(roi, (int(cX), h-100), 20, (0, 0, 255), -1)
        #print(M1['m00'] != 0)
        cv2.imshow("Robot Camera", frame)
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