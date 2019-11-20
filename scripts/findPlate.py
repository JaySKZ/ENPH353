import numpy as np
import cv2


PATH = '/home/fizzer/test.png'
contour_tolerance = 5 #Buffer for detecting contours so that tiny contours dont get registered
MIN_AREA = 10 #Minimum area of the back of a car to be read by the license plate reader: Shoudl operate similar to a min distance
MIN_WIDTH = 5
MAX_WIDTH = 3000
MIN_HEIGHT =10
MAX_LETTER_AREA = 700
MIN_LETTER_AREA = 10

#Returns a array of the four corners of the closest back of a parked car. 
#If there is no car closer than the min area then it returns false
def findCorners(raw_image):
    #Initialize Image
    image = raw_image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_grey = np.array([0,0,93]) 
    upper_grey = np.array([0,0,210])
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #Find two largest contours -> should be back of car
    #TODO find a function that does this better -> boundingrect does it
    largest_area = 0
    top_y = 0
    bottom_y = 0
    left_x = 0
    right_x = 0

    for contour in contours:
        miny = 900
        maxy = 0
        minx = 900
        maxx = 0
        for point in contour:
            if(point[0][0] > maxx):
                maxx = point[0][0]
            if(point[0][0] < minx):
                minx = point[0][0]
            if(point[0][1] > maxy):
                maxy = point[0][1]
            if(point[0][1] < miny):
                miny = point[0][1]
        area = cv2.contourArea(contour)
        if (area > largest_area) and (area > MIN_AREA) and (maxx -minx > MIN_WIDTH) and (maxx -minx < MAX_WIDTH) and (maxy - miny > MIN_HEIGHT):
            largest_area = area
            largest_contour = contour

            right_x = maxx-10
            left_x = minx+10
            top_y = miny
            bottom_y = maxy

    #Coordinates of four corners of back of car
    if(largest_area != 0):
        return raw_image[top_y:int(bottom_y*1.1), left_x:right_x, :]
    else:
        return np.array([])

#Perspective shifts an image according to given coordinates
def perspectiveShift(image, coordinates):
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(coordinates, pts2)   
    image_shifted = cv2.warpPerspective(image, M, (300,300))
    return image_shifted

def findLetters(raw_image):
    image = raw_image.copy()

    #Find license plate letters
    #Apply blue mask to image
    lower_blue = np.array([100,100,30])
    upper_blue = np.array([125,255,255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(image, image, mask=mask)

    
    #Find contours of blue
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=contourArea, reverse=True)
    
    #Find four largest blue contours which should be the digits/letters
    #Flag checks to see if the contour is fully enclosed by another which means that its likely the hole in a 9 or 6 etc
    plate = np.array([0,0,0,0], dtype=object)
    count = 0
    for i in range(len(contours)):
        if (cv2.contourArea(contours[i]) < MAX_LETTER_AREA) and (cv2.contourArea(contours[i]) > MIN_LETTER_AREA) and (count < 4):
            #cv2.drawContours(image, contours, i, (0,255,0), 3)
            flag =  False
            for j in range(len(contours)):
                if i == j: break
                x1,y1,w1,h1 = cv2.boundingRect(contours[i])
                x2,y2,w2,h2 = cv2.boundingRect(contours[j])
                if  ((x1 > x2) and (x1 + w1 < x2 + w2) and (y1 > y2) and (y1 + h1 < y2 + w2)):
                    flag = True

            if not flag:
                plate[count] = contours[i]
                count += 1

    #Create individual digits/letter images
    digitBoxs = []
    for digit in plate:
        #x,y,w,h = cv2.boundingRect(digit)
        #image = cv2.rectangle(image,(x-5, y-5), (x+w+5, y+h+5), (255,0,0), 2)

        digitBox = cv2.boundingRect(digit)
        digitBoxs.append(digitBox)

    digitBoxs = sorted(digitBoxs, key=firstVal)

    #Find parking spot number
    #Apply black mask
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([30, 30, 30])
    mask = cv2.inRange(image, lower_black, upper_black)
    res = cv2.bitwise_and(image, image, mask=mask)

    #Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=contourArea, reverse=True)

    #Create bounding boxes
    parkingLabels = []
    for i in range(len(contours)):
        digitBox =cv2.boundingRect(contours[i])
        parkingLabels.append(digitBox)

    #Find right most
    parkingLabels = sorted(parkingLabels, key=furthestRight, reverse=True)
    
    digitBoxs.append(parkingLabels[0])

    #for i in range(len(digitBoxs)):
    #    x,y,w,h = digitBoxs[i]
    #    image = cv2.rectangle(image,(x-5, y-5), (x+w+5, y+h+5), (255,0,0), 2)

    #Create 64x64 images
    result = []
    for i in range(len(digitBoxs)):
        x,y,w,h = digitBoxs[i]
        digit_image = image[max(y,0):min(y+h,image.shape[0]), max(x,0):min(x+w,image.shape[1]),:]
        digit_image = cv2.resize(digit_image, (64,64))
        if i == 4:
            mask = cv2.inRange(digit_image, lower_black, upper_black)
        else:
            hsv = cv2.cvtColor(digit_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result.append(255-mask)

    return result
    

def contourArea(val):
    return cv2.contourArea(val)

def firstVal(val):
    return val[0]

def furthestRight(val):
    return val[0] + val[2]


if __name__ == "__main__":
    raw_image = cv2.imread(PATH)
    coordinates = findCorners(raw_image)
    #image = perspectiveShift(raw_image, coordinates)
    findLetters(coordinates)