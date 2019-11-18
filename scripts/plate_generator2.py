#!/usr/bin/env python

import pyqrcode
import string
import random
from random import randint
from string import ascii_uppercase
import cv2
import numpy as np
import csv
import os

path = os.path.dirname(os.path.realpath(__file__)) + "/"
texture_path = '../media/materials/textures/'

with open(path + "plates.csv", 'w') as plates_file:
    csvwriter = csv.writer(plates_file)

    for i in range(0, 16):
        plate_alpha = ""
        for c1 in ascii_uppercase:
            for c2 in ascii_uppercase:
                plate_alpha = c1 + c2
                for plate_num in range(0,99):
                    plate_num = str(plate_num)

                    # Save plate to file
                    csvwriter.writerow([plate_alpha+plate_num])

                    # Write plate to image
                    blank_plate = cv2.imread(path+'blank_plate.png')
                    cv2.putText(blank_plate,
                                plate_alpha + " " + plate_num, (45, 360),
                                cv2.FONT_HERSHEY_PLAIN, 11, (255, 0, 0), 7, cv2.LINE_AA)

                    # Create parking spot label
                    s = "P" + str(i)
                    parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
                    cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                                (0, 0, 0), 30, cv2.LINE_AA)
                    spot_w_plate = np.concatenate((parking_spot, blank_plate), axis=0)

                    # Merge labelled or unlabelled images and save
                    unlabelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                                dtype=np.uint8), spot_w_plate), axis=0)
                    cv2.imwrite(os.path.join(path+texture_path+"unlabelled/",
                                            "plate_" + str(i) + c1 + c2 + plate_num +".png"), unlabelled)

                    # Randomly create distorted images and save
