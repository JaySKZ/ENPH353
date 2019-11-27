#!/usr/bin/env python

import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw

path = '/home/fizzer/enph353_ws/src/enph353/enph353_gazebo/media/materials/textures/'

NUMBER_OF_PLATES = 5000

for i in range(0, NUMBER_OF_PLATES):

    # Pick two random numbers
    num = randint(1, 8)
    num2 = randint(100,999)

    s = str(num)
    parking_spot = 255 * np.ones(shape=[350, 350, 3], dtype=np.uint8)
    cv2.putText(parking_spot, s, (30, 330), cv2.FONT_HERSHEY_PLAIN, 28,
                (0, 0, 0), 30, cv2.LINE_AA)
    
    # Write license plate to file
    #cv2.imshow('y',parking_spot)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join(path + "dataset2/", 
                                "{}{}.png".format(num,num2)),
                parking_spot)
    
