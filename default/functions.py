import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
SUDOKU_SIZE = 9
N_MIN_ACTIVE_PIXELS = 10


# Uzmi ivice slike da bi se remapirala
def getOuterPoints(rcCorners):
    ar = [];
    ar.append(rcCorners[0,0,:]);
    ar.append(rcCorners[1,0,:]);
    ar.append(rcCorners[2,0,:]);
    ar.append(rcCorners[3,0,:]);

    x_sum = sum(rcCorners[x,0,0] for x in range(len(rcCorners))) / len((rcCorners))
    y_sum = sum(rcCorners[y,0,1] for y in range(len(rcCorners))) / len((rcCorners))

    def algo(v):
        return (math.atan2(v[0] - x_sum, v[1] - y_sum)
                + 2 * math.pi) % 2*math.pi
    ar.sort(key=algo)

    return (ar[3],ar[0],ar[1],ar[2])

def extract_number(x, y,warp_gray):
    #square -> position x-y
    im_number = warp_gray[x*IMAGE_HEIGHT:(x+1)*IMAGE_HEIGHT][:, y*IMAGE_WIDTH:(y+1)*IMAGE_WIDTH]

    #threshold
    im_number_thresh = cv2.adaptiveThreshold(im_number,255,1,1,15,9)
    #delete active pixel in a radius (from center)
    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = math.sqrt( (IMAGE_WIDTH/2 - i)**2  + (IMAGE_HEIGHT/2 - j)**2);
            if dist_center > 8:
                im_number_thresh[i,j] = 0;

    n_active_pixels = cv2.countNonZero(im_number_thresh)
    return [im_number, im_number_thresh, n_active_pixels]

def find_biggest_bounding_box(im_number_thresh):
    _,contour,hierarchy = cv2.findContours(im_number_thresh.copy(),
                                         cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)

    biggest_bound_rect = [];
    bound_rect_max_size = 0;
    for i in range(len(contour)):
         bound_rect = cv2.boundingRect(contour[i])
         size_bound_rect = bound_rect[2]*bound_rect[3]
         if  size_bound_rect  > bound_rect_max_size:
             bound_rect_max_size = size_bound_rect
             biggest_bound_rect = bound_rect
    #bounding box a little more bigger
    x_b, y_b, w, h = biggest_bound_rect;
    x_b= x_b-1;
    y_b= y_b-1;
    w = w+2;
    h = h+2;

    return [x_b, y_b, w, h]

#sudoku representation
sudoku = np.zeros(shape=(9*9,IMAGE_WIDTH*IMAGE_HEIGHT))

def Recognize_number( x, y,warp_gray):
    """
    Recognize the number in the rectangle
    """
    #extract the number (small squares)
    [im_number, im_number_thresh, n_active_pixels] = extract_number(x, y,warp_gray)

    if n_active_pixels> N_MIN_ACTIVE_PIXELS:
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        im_t = cv2.adaptiveThreshold(im_number,255,1,1,15,9);
        number = im_t[y_b:y_b+h, x_b:x_b+w]

        if number.shape[0]*number.shape[1]>0:
            number = cv2.resize(number, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            ret,number2 = cv2.threshold(number, 127, 255, 0)
            number = number2.reshape(1, IMAGE_WIDTH*IMAGE_HEIGHT)
            sudoku[x*9+y, :] = number;
            return 1

        else:
            sudoku[x*9+y, :] = np.zeros(shape=(1, IMAGE_WIDTH*IMAGE_HEIGHT));
            return 0

