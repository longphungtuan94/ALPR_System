# Vietnamese ALPR_System
_An Automatic License Plate Recognition System for Vietnamese Plates_

This system can detect and recognize 2 types of license plate in Vietnam, rectangle plates and square plates.

# Abstract
- This system can detect and recognize license plates from images, videos and webcams.
- Within this project, the camera's position is fixed and only one car at a time can drive through the gate. Therefore, the system is only able to detect 1 plate at a time although there may be two or more plate in the current frame.
# Method
1. Plate detection
   - Sobel X for detecting vertical edges followed by a morphological transformation
   - Finding contours which satisfy the ratio of the plate to get the possible plates
   - Checking for characters on the possible plates found to assure it is a license plate
2. Plate recognition
   - For character recognition tested several Convolutional Neural Networks. MobileNet_v1_0.5_128 was our final choice as it was lightweight and suitable for real-time recognition.

# Requirements
- Python 2.7
- OpenCV 3.4.2, imutils, Tensorflow 1.11.0, scikit-image

# Implementation
- run `test.py` for testing

# Result
- ![Demo](https://github.com/longphungtuan94/ALPR_System/blob/master/test_videos/screenshot_1.png)
- ![Demo2](https://github.com/longphungtuan94/ALPR_System/blob/master/test_videos/screenshot_2.png)


# References
- Short link [paper](https://google.com)
- Or full link https://google.com

# TODO
- Upgrade the tracking algorithm to be able to track 2 or more plates at a time
- Retraining the Neural Network for higher recognition accuracy
