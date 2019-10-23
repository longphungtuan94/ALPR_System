""" Gets the recognized plate in several frames and calculates the most possible plate value """

import math
from collections import Counter


def getDistance(pointA, pointB):
    """
    calculates the distance between two points in the image
    """
    return math.sqrt(math.pow((pointA[0] - pointB[0]), 2) + math.pow((pointA[1] - pointB[1]), 2))


def tracking(previous_coordinate, current_coordinate):
    distance = getDistance(previous_coordinate, current_coordinate)
    return distance


def get_average_plate_value(plates, plates_length):
    """
    inputs an array of plates and returns the most possible value (average value) of the array
    """
    # plates_length is an array containing the number of characters detected on each plate in plate array
    plates_to_be_considered = []
    number_char_on_plate = Counter(plates_length).most_common(1)[0][0]
    for plate in plates:
        if (len(plate) == number_char_on_plate):
            plates_to_be_considered.append(plate)

    temp = ''
    for plate in plates_to_be_considered:
        temp = temp + plate
    
    counter = 0
    final_plate = ''
    for i in range(number_char_on_plate):
        my_list = []
        for i in range(len(plates_to_be_considered)):
            my_list.append(temp[i*number_char_on_plate + counter])
        final_plate = final_plate + str(Counter(my_list).most_common(1)[0][0])
        counter += 1
    return final_plate