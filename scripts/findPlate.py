import numpy as np 
import cv2


PATH = '/home/fizzer/Pictures/platetest_cropped.png'
contour_tolerance = 5 #Buffer for detecting contours so that tiny contours dont get registered
MIN_AREA = 100 #Minimum area of the back of a car to be read by the license plate reader: Shoudl operate similar to a min distance

#Returns a array of the four corners of the closest back of a parked car. 
#If there is no car closer than the min area then it returns false
def findCorners(raw_image):
    #Initialize Image
    image = raw_image.copy()
    count = 0

    #Convert to back of car to white and everything else to black
    #TODO Replace with a mask
    for i in range(0,900,1):
        for j in range(0,900,1):
            if (image[i,j,0] == 102) and (image[i,j,1] == 102) and (image[i,j,2] == 102):
                count = count + 1
                
                if count > contour_tolerance:
                    image[i,j,0] = 255
                    image[i,j,1] = 255
                    image[i,j,2] = 255

                else: 
                    image[i,j,0] = 0
                    image[i,j,1] = 0
                    image[i,j,2] = 0
            else:
                count = 0
                image[i,j,0] = 0
                image[i,j,1] = 0
                image[i,j,2] = 0

    #Find contours of image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #Find two largest contours -> should be back of car
    #TODO find a function that does this better
    largest_area = 0
    second_largest_area = 0
    second_bottom_y = 0
    second_top_y = 0
    second_left_x = 0
    second_right_x = 0
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
        area = (maxx - minx) * (maxy - miny)
        if area > largest_area:
            second_largest_area = largest_area
            largest_area = area

            second_bottom_y = bottom_y
            second_top_y = top_y
            second_left_x = left_x
            second_right_x = right_x

            right_x = maxx
            left_x = minx
            top_y = miny
            bottom_y = maxy
        elif area > second_largest_area:
            second_largest_area = area
            second_bottom_y = maxy
            second_top_y = miny
            second_left_x = minx
            second_right_x = maxx

    #Coordinates of four corners of back of car
    if(largest_area > MIN_AREA):
        return np.float32([[left_x, top_y], [right_x, top_y], [second_left_x, second_bottom_y], [second_right_x, second_bottom_y]])
    else:
        return False

#Perspective shifts an image according to given coordinates
def perspectiveShift(image, coordinates):
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(coordinates, pts2)   
    image_shifted = cv2.warpPerspective(image, M, (300,300))
    return image_shifted

def findLetters(raw_image):
    parking_spot = raw_image[50:200, :]
    license1 = raw_image[200:300,0:150]
    license2 = raw_image[200:300,150:300]
    return parking_spot, license1, license2 

    


if __name__ == "__main__":
    raw_image = cv2.imread(PATH)
    coordinates = findCorners(raw_image)
    image = perspectiveShift(raw_image, coordinates)
    findLetters(image)