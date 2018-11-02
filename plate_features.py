import cv2
import numpy as np

# Check if a contour has the features of a plate
def ratioCheck(type_of_plate, area, width, height):
	if (type_of_plate == 0):
		aspect = 4.272727 # ratio of the rectangle Vietnamese plate.
		min = 3000  # minimum area of the plate
		max = 30000 # maximum area of the plate

		rmin = 3
		rmax = 7
	else:
		aspect = 1
		min = 4000
		max = 30000

		rmin = 0.5
		rmax = 2

	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	# aspect = 4.2727 # ratio of the long Vietnamese plate. 4.2727 = 47cm/11cm
	# min = 30*aspect*30  # minimum area of the plate
	# # min = 100 
	# max = 125*aspect*125  # maximum area of the plate

	# rmin = 3
	# rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True

	# def _ratioCheck(type_of_plate, width, height):
	# 	if (type_of_plate == 0):
	# 		rmin = 

# check the if the detected contour satisfies the white pixels by black pixels condition
def isMaxWhite(plate):
	avg = np.mean(plate)
	if(avg >= 40): # old value was 115, second chosen value was 90
		return True
	else:
 		return False

def validateRotationAndRatio(type_of_plate, rect):
	plate_type = type_of_plate
	(x, y), (width, height), rect_angle = rect

	if(width>height):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle

	if angle > 15:
	 	return False

	if height == 0 or width == 0:
		return False

	area = height*width
	if not ratioCheck(plate_type, area,width,height):
		return False
	else:
		return True