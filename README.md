# ALPR_System
_An Automatic License Plate Recognition System for Vietnamese Plates_

This system can detect 2 types of license plate in Vietnam, rectangle plates and square plates. Currently this version supports only character recognition for rectangle plates.

# Abstract
- Write something here
# Method
1. Plate detection
   - Sobel X for detecting vertical edges followed by a morphological transformation
   - Finding contours which satisfy the ratio of the plate to get the possible plates
   - Checking for characters on the possible plates found to assure it is a license plate
2. Plate recognition
   - For character recognition tested several Convolutional Neural Networks. MobileNet_v1_0.5_128 was our final choice as it was lightweight and suitable for real-time recognition.

# Requirements
- Python 2.7
- OpenCV 3.4.2, Tensorflow 1.11.0

# Implementation
- run `test.py` for testing

# Result
- some images here
# References
- Short link [paper](https://google.com)
- Or full link https://google.com

# TODO
