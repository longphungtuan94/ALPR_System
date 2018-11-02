""" Increasing the threshold value in function segment may result in the need to decrease the condition of the ratio in
function characterRatioCheck """

import cv2
import numpy as np

# Check if a contour has the features of a digit
def isCharacter(plate_image, contour):
    (_, _, contour_width, contour_height) = cv2.boundingRect(contour)
    plate_height, plate_width = plate_image.shape[:2]
    plate_area = plate_height * plate_width
    ratio = plate_area/(contour_width * contour_height)
    ratioCharacter = float(contour_height)/float(plate_height)
    # print ratioCharacter
    # print ratio
    if ((ratio >= 10 and ratio < 43) and (contour_height > contour_width) and (ratioCharacter >= 0.5) and (float(contour_height)/float(contour_width) > 1.2)):
        return True
    else:
        return False

# Sort contours from left to right
def sort_contours_left_to_right(character_contours):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                key=lambda b:b[1][i], reverse=False))
    return character_contours

# Segment the characters in the plate based on the ratio check
def segment_from_plate(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur_plate = cv2.GaussianBlur(gray_plate, (3, 3), 1)
    threshold_plate = cv2.adaptiveThreshold(blur_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # print 'mean', np.mean(plate_image)
    # ret, threshold_plate = cv2.threshold(blur_plate, int(np.mean(plate_image)), 255, cv2.THRESH_BINARY_INV) # threshold value is 120
    # cv2.imshow('threshold plate', threshold_plate)

    _, contours, hier = cv2.findContours(threshold_plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# remove the child contours in characters[]. Get the index value of the contours in contours[] which satisfy
# the isCharacter() condition. After that, check the parent contour of all the contours in characters[] by calling
# hierarchy[0][index][3]. If a contour has a parent contour in characters[] then delete it. This is to eliminate the
# false contours in number 0

    index_of_chosen_contours = []
    characters = []
    for index, cnt in enumerate(contours):
        if isCharacter(plate_image, cnt):
            characters.append(cnt)
            index_of_chosen_contours.append(index)
    
    count = 0
    for i in index_of_chosen_contours:
        if hier[0][i][3] in index_of_chosen_contours:
            characters.pop(count)
            count -= 1
        count += 1

    m = 0
    if (len(characters) >= 6): # eliminate false plates        
        characters = sort_contours_left_to_right(characters)
        characterImageList = []
        for c in characters:
            (x,y,w,h) = cv2.boundingRect(c)
            if (x > 1): # eliminate false characters caused by the boundaries of the plate
                temp = plate_image[(y):(y+h), (x):(x+w)]
                # cv2.imwrite(str(m) + '.jpg', temp)
                # cv2.imshow('a', temp)
                # cv2.waitKey(0)
                characterImageList.append(temp)
                # plate_image = cv2.rectangle(plate_image,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.imshow('Contours', plate_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return characterImageList
    else:
        return None
