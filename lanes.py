import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.maxint = 1e21
print(sys.maxint)
def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    slope = slope
    y1 = image.shape[0]
    y2 = y1*3/5
    x1 = float("{:.20f}".format((y1-intercept)/slope))
    print(x1,"x1")
    x2 = float("{:.20f}".format((y2-intercept)/slope))
    print(x2,"x2")
    print(y1,"y1")
    print(y2,"y2")
    return np.array([x1,y1,x2,y2],np.int32)

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return np.array([])
    for line in lines:
        for x1,y1,x2,y2 in line:
            parameters = np.polyfit((x1,x2),(y1,y2),1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope,intercept))
    if (len(right_fit)==len(left_fit)==0):
        return np.array([], np.int64)
    if(len(left_fit)==0):
        right_fit_average = np.average(right_fit, axis=0)
        right_line=make_coordinates(image, right_fit_average)
        return np.array([right_line],np.int64)
    elif (len(right_fit)==0):
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        return np.array([left_line], np.int64)

    right_fit_average = np.average(right_fit, axis=0)
    right_line = make_coordinates(image, right_fit_average)
    left_fit_average = np.average(left_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)

    return np.array([left_line,right_line], np.int64)


def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([[(300,height-150),(950,height-150),(700,400)]],np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)

cap = cv2.VideoCapture('lane.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)

while(cap.isOpened()):
    ret,frame =  cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,3,(np.pi/180),100,np.array([]),5,5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image,1, 1)
    out.write(combo_image)
    cv2.imshow('result',combo_image)
    #plt.imshow(canny_image)
    #plt.show()
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
