import numpy as np
import cv2
import tensorflow
import time
import argparse
import glob
import math
import threading
from collections import Counter
import multiprocessing as mp

from class_CNN import NeuralNetwork
from class_PlateDetection import PlateDetector

plateDetector = PlateDetector(type_of_plate='RECT_PLATE', minPlateArea=3000, maxPlateArea=30000) # Initialize the plate detector
myNetwork = NeuralNetwork(modelFile="model/binary_128_0.50_ver3.pb", labelFile="model/binary_128_0.50_labels_ver2.txt") # Initialize the Neural Network

# calculates the distance between two points in the image
def getDistance(pointA, pointB):
     return math.sqrt(math.pow((pointA[0] - pointB[0]), 2) + math.pow((pointA[1] - pointB[1]), 2))

def tracking(previous_coordinate, current_coordinate):
    distance = getDistance(previous_coordinate, current_coordinate)
    return distance

def get_average_plate_value(plates, plates_length):
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


def recognized_plate(input_segmments, size):
    t0 = time.time()
    
    plates_value = []
    plates_length = []
    for segmented_characters in input_segmments:
        plate, len_plate = myNetwork.label_image_list(segmented_characters, size)
        plates_value.append(plate)
        plates_length.append(len_plate)
        
    final_plate = get_average_plate_value(plates_value, plates_length)   
    print("recognized plate: " + final_plate)
    
    print("threading time: " + str(time.time() - t0))

cap = cv2.VideoCapture('test_videos/test.MOV')
# cap = cv2.VideoCapture(0)
coordinates = (0, 0)
plates_value = []
plates_length = []
processes = []
input_segmments = []
i = 0 

if __name__=="__main__":
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if (frame is None):
            print("[INFO] End of Video")
            break
    
        # frame = cv2.resize(frame, (1024, 768))
        frame_height, frame_width = frame.shape[:2]
        cropped_frame = frame[int(frame_height*0.3):frame_height, 0:int(frame_width*0.8)]
        cv2.rectangle(frame, (0, int(frame_height*0.3)), (int(frame_width*0.8), frame_height), (255, 0, 0), 2)
        # cropped_frame = frame[int(frame_height/4):int(frame_height*0.92), int(frame_width*0.2):(frame_width-int(frame_width*0.2))]
        # cv2.rectangle(_frame,(int(frame_width*0.2),int(frame_height/4)),((frame_width-int(frame_width*0.2)),int(frame_height*0.92)),(255,0,0),2)
        cv2.imshow('video', frame)
        
        possible_plates = plateDetector.find_possible_plates(cropped_frame)
        if possible_plates is not None:
            distance = tracking(coordinates, plateDetector.corresponding_area[0])
            coordinates = plateDetector.corresponding_area[0]
            if (distance < 40):
                if(i < 5):
                    for plates in possible_plates:
                        cv2.imshow('Plate', plates)
                        # for c in plateDetector.char_on_plate[0]:
                        #     cv2.imshow('c', c)
                        #     cv2.waitKey(0)
                        # myNetwork.label_image_list(plateDetector.char_on_plate[0], 128)
                        input_segmments.append(plateDetector.char_on_plate[0])
                        i = i+1
                elif(i==5):
                    threading.Thread(target=recognized_plate, args=(input_segmments, 128)).start()
                    
                    i = i+1
            else:
                # if (i < 5 and i > 0):
                #     recognized_plate(input_segmments, 128)
                i = 0
                input_segmments = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
