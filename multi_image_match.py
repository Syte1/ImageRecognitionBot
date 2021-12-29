from time import time

import cv2 as cv
import numpy as np
import os
import pyautogui
import pygetwindow
from windows_capture import WindowCapture

"""------------------------------------- PRELIMINARY PHASE -------------------------------------"""

originals = os.listdir('images/search')
# print(originals)
ORIGINALS_DIR = r'images/Search'

THRESHOLD = .6

# create source image
"""SOURCE_PIC = cv.imread('images/Source/game.png')
SOURCE_PIC_GRAY = cv.cvtColor(SOURCE_PIC, cv.COLOR_BGR2GRAY)"""

SOURCE_CAPTURE = cv.VideoCapture(0)

# create a list of picture directories
read_images = {}
for image_name in originals:
    # print(ORIGINALS_DIR, originals)
    read_images[image_name[:-4]] = os.path.join(ORIGINALS_DIR, image_name)

# store all cv reads to dictionary values
for key, value in read_images.items():
    read_images[key] = cv.imread(value)

# define match shape
FOUND_HEIGHT, FOUND_WIDTH = next(iter(read_images.values())).shape[:2]
# print(FOUND_HEIGHT, FOUND_WIDTH)

# create grayscale dictionary
gray_images = {name: cv.cvtColor(image, cv.COLOR_BGR2GRAY) for name, image in read_images.items()}
# print(gray_images)

"""----------------------------------------- LOOP PHASE -----------------------------------------"""
# def create_image():

# wincap = WindowCapture('')
# print(pygetwindow.getAllTitles())
wincap = WindowCapture('Whack-a-Mole - Free Brain Game — Mozilla Firefox')

while True:
    loop_time = time()
    # take a picture
    SOURCE_PIC = wincap.get_screenshot()
    # convert to BGR for openCV
    # SOURCE_PIC = cv.cvtColor(SOURCE_PIC, cv.COLOR_RGB2BGR)
    # convert to grayscale
    SOURCE_PIC_GRAY = cv.cvtColor(SOURCE_PIC, cv.COLOR_BGR2GRAY)

    # create confidence dictionary
    confidence_dict = {name: cv.matchTemplate(SOURCE_PIC_GRAY, each_gray, cv.TM_CCOEFF_NORMED) for
                       name, each_gray in gray_images.items()}

    # print(confidence_dict)

    # filter confidence dict
    filtered_confidence_dict = {}
    for index, confidence_set in enumerate(confidence_dict.values()):
        filtered_confidence_dict[index] = np.where(confidence_set > THRESHOLD)

    # generate locations array
    locations_array = []
    for confidence_set in filtered_confidence_dict.values():
        locations_array.extend(*[list(zip(*confidence_set[::-1]))])
    # print(locations_array)

    rectangle_list = []
    for each_location in locations_array:
        rectangle_list.append([each_location[0], each_location[1], FOUND_HEIGHT, FOUND_WIDTH])

    rectangles, weights = cv.groupRectangles(rectangle_list, 1, .5)
    # print(rectangles)

    for each in rectangles:
        cv.rectangle(SOURCE_PIC, (each[0], each[1]), (each[0] + each[2], each[1] + each[3]),
                     (0, 255, 0), 3)
        cv.drawMarker(SOURCE_PIC, (each[0] + each[2] // 2, each[1] + each[3] // 2), (0, 0, 255),
                      cv.MARKER_CROSS)
        print(f'click points-----------{(each[0] + each[2] // 2, each[1] + each[3] // 2)}')
        pyautogui.click(x=(each[0] + each[2] // 2), y=(each[1] + each[3] // 2))
    cv.imshow('source', SOURCE_PIC)

    if cv.waitKey(1) == ord('d'):
        break
    # print(read_images)
    # cv.imshow('test', read_images['fullmole1.png'])
    print(f'FPS:{1 / (time() - loop_time)}')
cv.destroyAllWindows()
