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
        self.buffer = 50
        self.transition_state = 1 
        self.crosswalk = False
        self.crosscount = 0
        self.pedMove = False
        self.delay = 0

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

        # Detecting red for crosswalks
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])

        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        red_eroded = cv2.erode(red_mask, kernel_erode, iterations=1)
        red_dilated = cv2.dilate(red_eroded, kernel_dilate, iterations=1)

        bigRedM = cv2.moments(red_dilated)
        redM = cv2.moments(red_dilated[h-100:h, (w/3):(2*w/3)])

        # Setup for motion detection
        scan = frame[0:h, 150:(w-150)]
        gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        GBlur = cv2.GaussianBlur(gray, (21, 21), 0)

        first_frame = None
        next_frame = None

        if (redM['m00'] != 0 and bigRedM['m00'] != 0):
            self.crosswalk = True

            self.error = 0

            first_frame = GBlur
            self.delay += 1

            if (self.delay > 10):
                self.delay = 0
                first_frame = next_frame
            
            next_frame = GBlur

            frame_delta = cv2.absdiff(first_frame, next_frame)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations = 2)
            _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:

                # Save the coordinates of all found contours
                (a, b, c, d) = cv2.boundingRect(c)

                # If the contour is too small, ignore it, otherwise, there's transient
                # movement
                if cv2.contourArea(c) > 500:
                    self.pedMove = True
        elif (bigRedM['m00'] == 0):
            self.crosswalk = False
            
            

        # This is enough image pre-processing, contour finding happens depending state machine

        # Initial start
        if (self.stage == 1):
            # Set region of interest to left of screen
            roi_center = dilated_mask[h-200:h, (w/2)-50:(w/2)+50]
            roi_right = dilated_mask[h-100:h, 800:w]

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
        elif (self.stage == 2 and self.crosswalk == False):
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

        # Transitioning to inner ring
        elif (self.stage == 3):
            # Set region of interest to the left of screen
            roi_left = dilated_mask[h-100:h, 0:520]
            roi_center = dilated_mask[h-75:h, (w/2)-50:(w/2)+50]
            roi_right = dilated_mask[h-300:h, 750:w]

            M1 = cv2.moments(roi_left)
            M2 = cv2.moments(roi_center)
            M3 = cv2.moments(roi_right)

            self.separation = 400
            self.buffer = 50

            if (self.transition_state == 1):
                if (M1['m00'] != 0):
                    if (M2['m00'] == 0 and M3['m00'] != 0):
                        # Centroid of left lineMIN_SIZE_FOR_MOVEMENT
                        cX = int(M1['m10']/M1['m00'])

                        # Follow left curb
                        self.error = self.separation - cX
                    elif (M2['m00'] == 0 and M3['m00'] == 0):
                        self.transition_state = 2
                else:
                    self.error = 200
            
            if (self.transition_state == 2):
                roi_right = dilated_mask[h-300:h, 750:w]
                if (M2['m00'] == 0):
                    error = 0
                    print("switching")
                    if(M3['m00'] == 0):
                        self.error = 200
                        if ((int(M3['m10']/M3['m00']) - self.separation) <= 100):
                            self.stage = 4
                            print ("switched")

            # elif (M1['m00'] == 0 and M2['m00'] == 0):
            #     self.error = 0
            #     print("2")

            # elif (M2['m00'] != 0 and M3['m00'] == 0 and M1['m00'] == 0):
            #     self.error = 200
            #     print("3")
            
            # elif (M3['m00'] != 0 and M2['m00'] == 0):
            #     self.stage = 4
            #     print("4")


        # Inner ring, go clockwise, track right
        elif (self.stage == 4):
            
            self.separation = 420
            self.buffer = 50

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

        if (abs(self.error) < self.buffer):
            velocity.angular.z = 0
            velocity.linear.x = 0.2
        elif (self.error >= self.buffer):
            velocity.linear.x = 0
            if (self.stage == 2 or self.stage == 4):
                velocity.angular.z = -0.1
            else:
                velocity.angular.z = 0.1
        else:
            velocity.linear.x = 0
            if (self.stage == 2 or self.stage == 4):
                velocity.angular.z = 0.1
            else:
                velocity.angular.z = -0.1
        # elif (self.crosswalk == True and self.pedMove != True):
        #     velocity.angular.z = 0
        #     velocity.linear.x = 0.2
        # else:
        #     velocity.angular.z = 0
        #     velocity.linear.x = 0

        self.vel_pub.publish(velocity)

        #print(M3['m00'])
        #print(stage)
        #cv2.circle(frame, (int(cX), h-100), 20, (0, 0, 255), -1)
        #print(M1['m00'] != 0)
        #print(self.stage)
        print(self.crosswalk)
        # print(self.crosscount)
        # print(self.pedMove)
        # cv2.imshow("Robot Camera", red_dilated[h-100:h, (w/3):(2*w/3)])
        # cv2.imshow("Other Camera", red_dilated[200:h-300, (w/3):(2*w/3)])
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