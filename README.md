# Vietnamese ALPR_System
_An Automatic License Plate Recognition System for Vietnamese Plates_

This system can detect and recognize 2 types of license plate in Vietnam, rectangle plates and square plates.

## Abstract
- This system can detect and recognize license plates from images, videos and webcams.
- Within this project, the camera's position is fixed and only one car at a time can drive through the gate. Therefore, the system is only able to detect 1 plate per frame.
## Method
1. Plate detection
   - Sobel X for detecting vertical edges followed by a morphological transformation.
   - Finding contours which satisfy the ratio of the plate to get the possible plates
   - Checking for characters on the possible plates found to assure it is a license plate.
2. Plate recognition
   - For character recognition, I used MobileNet_v1_0.5_128 as it was lightweight and suitable for real-time recognition.

## Requirements
- python 3.6
- run `pip install -r requirements.txt`

## Implementation
- run `test.py` for testing on a video.
- run `test_image.py` for testing on an image.

## Result
- ![Demo](https://github.com/longphungtuan94/ALPR_System/blob/master/test_videos/screenshot_1.png)
- ![Demo2](https://github.com/longphungtuan94/ALPR_System/blob/master/test_videos/screenshot_2.png)

## Note
- You should play with these parameters: `minPlateArea`, `maxPlateArea` and `ksize` in `cv2.getStructuringElement` to implement successfully on your own case.