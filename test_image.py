import cv2
from class_CNN import NeuralNetwork
from class_PlateDetection import PlateDetector


########### INIT ###########
# Initialize the plate detector
plateDetector = PlateDetector(type_of_plate='RECT_PLATE',
                                        minPlateArea=4500,
                                        maxPlateArea=30000)

# Initialize the Neural Network
myNetwork = NeuralNetwork(modelFile="model/binary_128_0.50_ver3.pb",
                            labelFile="model/binary_128_0.50_labels_ver2.txt")

img = cv2.imread('test_videos/test.jpg')
cv2.imshow('original image', img)
cv2.waitKey(0)
possible_plates = plateDetector.find_possible_plates(img)
if possible_plates is not None:
    for i, p in enumerate(possible_plates):
        chars_on_plate = plateDetector.char_on_plate[i]
        recognized_plate, _ = myNetwork.label_image_list(chars_on_plate, imageSizeOuput=128)
        print(recognized_plate)
        cv2.imshow('plate', p)
        cv2.waitKey(0)