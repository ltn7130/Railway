import cv2
import numpy as np
import os
import math
from queue import PriorityQueue

# set the range values for highlight and shadow
SHADOW = [((0, 0, 0), (30, 30, 30)), ((0, 0, 0), (15, 15, 15)),((0, 0, 0), (50, 50, 50)), ((200, 200, 200), (250, 250, 250))]
INVALID_ANGLE = (20, 160)
X_AXIS = (0, 0, 1, 0)
Y_AXIS = (0, 0, 0, 1)
MAX_DISTANCE_OF_TWO_TRACK = 550
MIN_DISTANCE_OF_TWO_TRACK = 300
SCALE = 8
PATH_RESULT = 'result/19/'
PATH_TEST = 'test/19'
TRACK_SIZE = 500000
MAX_LENGTH_OF_VALID_LINE = 550

def adjust_line(track):
    x1, y1, x2, y2 = track
    if y1 > y2:
        track = [x2, y2, x1, y1]
    return track


def line_after_rotation(line, width):
    x1, y1, x2, y2 = line
    y = int((y1+y2)/2)
    return [0, y, width, y]


# find the equation of a line from 2 points and extend the line to the edges of the image
def get_full_line(line, width):
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    y = y2 - y1
    x = x2 - x1
    if x == 0:
        x = 0.001
    start_y = int((-x2 * y / x) + y2)
    end_y = int((y * (width - x2) / x) + y2)
    if start_y < end_y:
        return [0, start_y, width, end_y]
    return [width, end_y, 0, start_y, ]


# calculate distance of 2 points
def length_of_line(line):
    x1, y1, x2, y2 = line
    x = (x2 - x1) * (x2 - x1)
    y = (y2 - y1) * (y2 - y1)
    return int(math.sqrt(x+y))


# calculate slope of the line from 2 points
def angle_between_two_lines(line1, line2):
    vector1 = [line1[0] - line1[2], line1[1] - line1[3]]
    vector2 = [line2[0] - line2[2], line2[1] - line2[3]]
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1,unit_vector2)
    if dot_product > 1 or dot_product < -1:
        dot_product = 1
    angle = np.arccos(dot_product)*180/np.pi
    return angle


# Check if 2 lines are parallel
def is_parallel(line1, line2):
    return angle_between_two_lines(line1, line2) < 5 or angle_between_two_lines(line1, line2) > 175


# check if 2 lines are on a same line
def is_aligned(line1, line2):
    if not is_parallel(line1, line2):
        return False
    x1, y1, x2, y2 = line1
    a1, b1, a2, b2 = line2
    if x1 != a1:
        line2 = [a2, b2, a1, b1]
    distance_two_lines = abs(y1 - line2[1])
    if distance_two_lines > 5:
        return False
    return True


# Detect valid lines
def get_lines(image, mode):
    height, width, color = image.shape
    blank_img = np.zeros((height, width, 3), np.uint8)
    q = PriorityQueue()
    # blurr image
    median = cv2.medianBlur(image, 5)
    # isolate shadow of the tracks from the rest of the image
    lower, upper = mode
    thresh = cv2.inRange(median, lower, upper)
    # apply morphology open and close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # filter contours on area
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    valid_contours = []
    # remove object whose area is larger than track's size
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > TRACK_SIZE:
            valid_contours.append(contours[i])
    cv2.fillPoly(thresh , pts= valid_contours , color=(0, 0, 0))
    # detect lines which has length > 200 and the gap < 10
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength= MAX_LENGTH_OF_VALID_LINE, maxLineGap=10)
    # filter out the invalid lines which have angles larger than 25 degrees or in the 1/4 top and bottom of image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            currLine = [x1, y1, x2, y2]
            currLength = length_of_line(currLine)
            angleOfLine = angle_between_two_lines(currLine, X_AXIS)
            if INVALID_ANGLE[1] > angleOfLine > INVALID_ANGLE[0]:
                continue
            if y1 > 3*height/4 or y1 < 1*height/4 or y2 > 3*height/4 or y2 < 1*height/4:
                continue
            q.put((1/currLength, currLine))
    return q

# determine 2 tracks
def get_tracks(image, q):
    track1 = []
    track2 = []
    if q.empty():
        return track1, track2
    h, w, c = image.shape
    track1 = q.get()[1]
    track1 = get_full_line(list(track1), w)
    track1 = adjust_line(track1)
    while not q.empty():
        currLength, line = q.get()
        currLine = get_full_line(list(line), w)
        x1, y1, x2, y2 = currLine
        if x1 != track1[0]:
            currLine = [x2, y2, x1, y1]
        if is_parallel(track1, currLine) and not is_aligned(track1, currLine) and \
                MAX_DISTANCE_OF_TWO_TRACK > abs(currLine[1] - track1[1]) > MIN_DISTANCE_OF_TWO_TRACK:
            track2 = currLine
            track2 = adjust_line(track2)
            break
    return track1, track2


def main():
    # put all images in folder "test"
    working_directory = PATH_TEST
    fileList = [f for f in os.listdir(working_directory) if f.upper().endswith(".JPG")]
    fileList.sort()
    # read all images in folder
    if not fileList:
        print('Could not find images')
        print('Put images into directory "test"')
    cropped_images = []

    for f in fileList:
        print(f'Processing Image {f} ...')
        file_path = os.path.join(working_directory, f)
        original_image = cv2.imread(file_path)
        # if the image is horizontal, rotate it
        height, width, color = original_image.shape
        if height < width:
            original_image = cv2.rotate(original_image, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite('original_rotated.jpg', original_image)
            image = cv2.imread('original_rotated.jpg')
            os.remove('original_rotated.jpg')
        else:
            image = original_image.copy()
        h, w, c = image.shape

        # get the lines from image using shadow mode
        for mode in SHADOW:
            lines = get_lines(image, mode)
            track1, track2 = get_tracks(image, lines)
            if track1 and track2:
                break
        if not track1 or not track2:
            print("No track found, skipped to next image")
            continue
        # sort points of 2 tracks
        x1, y1, x2, y2 = track2
        if track1[0] != track2[0]:
            track2 = [x2, y2, x1, y1]

        # compute centerLine from 2 tracks
        centerLine = [0, 0, 0, 0]
        for i in range(len(track1)):
            centerLine[i] = int((track1[i] + track2[i]) / 2)
        # draw centerLine to the image
        cv2.line(image, (centerLine[0], centerLine[1]), (centerLine[2], centerLine[3]), (0, 0, 255), 5)

        # Rotate images using centerLine to make the tracks horizontal
        angle = angle_between_two_lines(centerLine, X_AXIS)
        if angle > 90:
            angle = angle - 180
        centerPoint = (int(w / 2), int((centerLine[1] + centerLine[3]) / 2))
        matrix = cv2.getRotationMatrix2D(centerPoint, angle, 1)
        rotated_image = cv2.warpAffine(image, matrix, (w, h))

        # Crop the track part of the image
        centerLine = line_after_rotation(centerLine, width)
        track1 = line_after_rotation(track1, width)
        track2 = line_after_rotation(track2, width)
        # sort the points of track1 and track2
        if track1[1] < track2[1]:
            y1 = track1[1]
            y2 = track2[1]
        else:
            y1 = track2[1]
            y2 = track1[1]

        # calculate the height of the cropped images based on the length of tracks
        width_of_track = abs(y1 - y2)
        distance_track_to_edges = int((1000 - width_of_track) / 2)
        start_point = y1 - distance_track_to_edges
        end_point = y2 + (1000 - distance_track_to_edges - width_of_track)
        crop = rotated_image[start_point: end_point, 200: width - 200]
        cropped_dimensions = (int((crop.shape[1] / SCALE)), int(crop.shape[0]/SCALE))
        crop = cv2.resize(crop, cropped_dimensions)
        cropped_images.append(crop)

        # write the cropped images
        name = 'centerLine' + f
        outputPath = PATH_RESULT + name
        cv2.imwrite(outputPath, crop)

    # create chunked image from the list of cropped images
    if not cropped_images:
        return
    chunked_img = cropped_images[0]
    for i in range(1, len(cropped_images)):
        chunked_img = np.concatenate((chunked_img, cropped_images[i]), axis=1)
    print('Creating chunked image......')
    name = 'chunked.jpg'
    outputPath = PATH_RESULT + name
    cv2.imwrite(outputPath, chunked_img)


if __name__ == "__main__":
    main()
