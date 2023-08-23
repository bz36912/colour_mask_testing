import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

GREEN = (0, 255, 0)  # in BGR.
RED = (0, 0, 255)
BLUE= (255, 0, 0)
ORANGE = (0, 128, 255)
PURPLE = (255, 51, 153)
YELLOW = (0, 255, 255)

# the colour green will be marked by red outline. Purple by yellow outline and so on.
complementary = {GREEN:RED, RED:GREEN, PURPLE:YELLOW, YELLOW:PURPLE, BLUE:ORANGE}

# initialising the video camera on computer
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

def drawContour(mask, colour, frame):
    """
    marks the locations of the colour, by tracing an outline around the colour
    mask: contains pixels of the colour in frame
    colour: the name/identifier of the colour
    frame: the original image
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #find contour/outline of a colour
    contourArray = np.zeros((0, 2)) #just an empty array
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 400: #only include large areas of the colour
            contourArray = np.concatenate((contourArray, np.squeeze(contour)))
            cv.drawContours(frame, [contour], 0, complementary[colour], 2)

            if colour == PURPLE: #approximate the location of purple area as a single point marked by a yellow circle
                M = cv.moments(contour)
                finishCentreX = int(M['m10'] / M['m00'])
                finishCentreY = contourArray[np.argmax(contourArray[:, 1])][1]
                cv.circle(frame, (round(finishCentreX), round(finishCentreY)), 10, complementary[colour], -1)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    blurred = cv.blur(frame, (5,5))  #bluring the image helps a lot
    width = int(cap.get(3))
    height = int(cap.get(4))

    #convert to hsv colorspace
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    #lower bound and upper bound for blue
    lowerBlue = np.array([100, 70, 80])
    upperBlue = np.array([130, 255, 255])
    blueMask = cv.inRange(hsv, lowerBlue, upperBlue)

    # lower bound and upper bound for yellow
    lowerYellow = np.array([28, 40, 100])
    upperYellow = np.array([42, 255, 255])
    yellowMask = cv.inRange(hsv, lowerYellow, upperYellow)   #getting a yellow mask     
    
    #lower bound and upper bound for purple color
    lower_purple = np.array([130, 50, 30])
    upper_purple = np.array([170, 255, 255])
    mask_purple = cv.inRange(hsv, lower_purple, upper_purple)

    #lower bound and upper bound for green color
    lower_green = np.array([40, 60, 60])
    upper_green = np.array([80, 255, 255])
    mask_green = cv.inRange(hsv, lower_green, upper_green)

    #lower and upper bound for red color
    lower_red1 = np.array([0, 150, 20])
    lower_red2 = np.array([165, 150, 20])
    upper_red1 = np.array([10, 255, 255])
    upper_red2 = np.array([180, 255,255])
    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask = np.zeros_like(yellowMask)

    allMasks = [mask_purple, mask_red, mask_green, yellowMask, blueMask]
    colours = [PURPLE, RED, GREEN, YELLOW, BLUE]
    for i in range(len(allMasks)):
        mask += allMasks[i]
        drawContour(allMasks[i], colours[i], frame)
    
    cv.imshow('all the masks', mask)
    cv.imshow('frame with contour', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
