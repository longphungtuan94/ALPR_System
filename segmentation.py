""" Increasing the threshold value in function segment may result in the need to decrease the condition of the ratio in
function characterRatioCheck """

import cv2
import numpy as np

import imutils
from skimage.filters import threshold_local
from skimage import measure

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

def segment_characters_from_plate(plate_img, fixed_width):
    # extract the Value component from the HSV color space and apply adaptive thresholding
    # to reveal the characters on the license plate
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method='gaussian')
    thresh = (V > T).astype('uint8') * 255
    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canoncial size
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    convert_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations
    # of the character candidates
    labels = measure.label(thresh, neighbors=8, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    # loof over the unique components
    characters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        # ensure at least one contour was found in the mask
        if len(cnts) > 0:

            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            
            # compute the aspect ratio, solodity, and height ration for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])
            
            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95
            
            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    # cv2.imshow('charC', charCandidates)
    # cv2.waitKey(0)
    _, contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_contours_left_to_right(contours)
        characters = []
        plate_copy = plate_img
        _coordinates = [cv2.boundingRect(c)[1] for c in contours]
        min_index = np.argmin(_coordinates)
        y_min = _coordinates[min_index]

        addPixel = 4 # value to be added to each dimension of the character
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = convert_thresh[y:y+h+(addPixel*2), x:x+w+(addPixel*2)]
            # cv2.imshow('temp', temp)
            # cv2.waitKey(0)
            characters.append(temp)
        return characters
    else:
        return None

    #             temp = plate_img[boxY:boxY+boxH, boxX:boxX+boxW]
    #             # cv2.imshow('temp', temp)
    #             # cv2.waitKey(0)
    #             characters.append((temp, boxX))
    # if len(characters) > 0:
    #     characters = sorted(characters, key=lambda x: x[1]) # sort the character images from left to right based on the coordinates
    #     characters = zip(*characters)
    #     return characters[0]
    # else:
    #     return None
