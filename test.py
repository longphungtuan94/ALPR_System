import numpy as np
import cv2
from class_CNN import NeuralNetwork
import tensorflow
import time
from class_PlateDetection import PlateDetector
import argparse
from segmentation import segment_from_plate
import glob
import math
import threading
from collections import Counter
import multiprocessing as mp

plateDetector = PlateDetector(type_of_plate='RECT_PLATE') # Initialize the plate detector
myNetwork = NeuralNetwork("model/128_0.50_ver2.pb", "model/128_0.50_labels_ver2.txt") # Initialize the Neural Network

# calculates the distance between two points in the image
def getDistance(pointA, pointB):
     return math.sqrt(math.pow((pointA[0] - pointB[0]), 2) + math.pow((pointA[1] - pointB[1]), 2))

def tracking(previous_coordinate, current_coordinate):
    distance = getDistance(previous_coordinate, current_coordinate)
    return distance

# Get the plate values in several frames and calculates the most possible plate value
def get_average_plate_value(plates, plates_length):
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

cap = cv2.VideoCapture('plate_3.MOV')
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
    
        frame = cv2.resize(frame, (1024, 768))
        
        
        frame_height, frame_width = frame.shape[:2]
        cropped_frame = frame[int(frame_height/4):int(frame_height*0.92), int(frame_width*0.2):(frame_width-int(frame_width*0.2))]
        cv2.rectangle(frame,(int(frame_width*0.2),int(frame_height/4)),((frame_width-int(frame_width*0.2)),int(frame_height*0.92)),(255,0,0),2)
        cv2.imshow('video', frame)
        
        possible_plates = plateDetector.find_possible_plates(cropped_frame)
        if possible_plates:
            distance = tracking(coordinates, plateDetector.corresponding_area[0])
            # print 'distance = ', distance
            coordinates = plateDetector.corresponding_area[0]
            if (distance < 40):
            # if (distance < 40 and len(plates_value) < 6):
                if(i<5):
                    for plates in possible_plates:
                        segmented_characters = segment_from_plate(plates)
                        
                        if segmented_characters:
                            cv2.imshow('Plate', plates)
                            input_segmments.append(segmented_characters)
                            #threading.Thread(target=myNetwork.label_image_list(segmented_characters, 128)).start()
                            #processes.append(mp.Process(target=recognized_plate, args=(segmented_characters, 128)))
                            i = i+1
                elif(i==5):
                    threading.Thread(target=recognized_plate, args=(input_segmments, 128)).start()
                    
                    i = i+1
            else:
                i = 0
                input_segmments = []

            #             plate_value, plate_length = myNetwork.label_image_list(segmented_characters, 128)
            #             plates_value.append(plate_value)
            #             plates_length.append(plate_length)
            #             if (len(plates_value) == 5):
            #                 final_plate = get_average_plate_value(plates_value, plates_length)
            #                 print final_plate
            # elif (distance > 40):
            #     plates_value = []
            #     plates_length = []

                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
