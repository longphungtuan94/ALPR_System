import numpy as np
import cv2
from utils.segmentation import segment_characters_from_plate


class PlateDetector():
    def __init__(self, type_of_plate, minPlateArea, maxPlateArea):
        self.minPlateArea = minPlateArea # minimum area of the plate
        self.maxPlateArea = maxPlateArea # maximum area of the plate

        if (type_of_plate == 'RECT_PLATE'):
            self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))
            self.type_of_plate = 0
        if (type_of_plate== 'SQUARE_PLATE'):
            self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(26, 5))
            self.type_of_plate = 1

        
    def find_possible_plates(self, input_img):
        """
        Find plates candidates
        """
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates =  self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        if(len(plates) > 0):
            return plates
        else:
            return None


    def find_characters_on_plate(self, plate):
        if (self.type_of_plate == 0): # rectangle plate
            charactersFound = segment_characters_from_plate(plate, 400)
            if charactersFound:
                return charactersFound
        elif (self.type_of_plate == 1): # square plate
            # divide the square plate into half to get a one line plate
            plate_upper = plate[0:plate.shape[0]/2, 0:plate.shape[1]]
            plate_lower = plate[plate.shape[0]/2: plate.shape[0], 0:plate.shape[1]]

            # get the characters of the upper plate and the lower plate
            upper_charactersFound = segment_characters_from_plate(plate_upper, 200)
            lower_charactersFound = segment_characters_from_plate(plate_lower, 200)
            if (upper_charactersFound and lower_charactersFound):
                charactersFound = upper_charactersFound + lower_charactersFound
                return charactersFound
                

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0) # old window was (5,5)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY) # convert to gray
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) # sobelX to get the vertical edges
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        element = self.element_structure
        morph_img_threshold = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
        return morph_img_threshold

        
    def extract_contours(self, after_preprocess):
        _, extracted_contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        return extracted_contours
        

    def crop_rotated_contour(self, plate, rect):
        """
        Rotate the plate and crop the plate with its rotation
        """
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        W = rect[1][0]
        H = rect[1][1]
        
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        
        angle = rect[2]
        if angle < (-45):
            angle += 90
            
        # Center of rectangle in source image
        center = ((x1 + x2)/2,(y1 + y2)/2)

        # Size of the upright rectangle bounding the rotated rectangle
        size = (x2-x1, y2-y1)
        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

        # Cropped upright rectangle
        cropped = cv2.getRectSubPix(plate, size, center)
        cropped = cv2.warpAffine(cropped, M, size)
        croppedW = H if H > W else W
        croppedH = H if H < W else W

        # Final cropped & rotated rectangle
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
        return croppedRotated

        
    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas) # index of the largest contour in the area array
            
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x,y,w,h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = self.crop_rotated_contour(plate, rect)
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None


    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRotationAndRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y+h, x:x+w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if (characters_on_plate is not None and len(characters_on_plate) > 5):
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1+x, y1+y
                    after_check_plate_img = after_clean_plate_img
                    return after_check_plate_img, characters_on_plate, coordinates
        return None, None, None


#################### PLATE FEATURES ####################
    def ratioCheck(self, area, width, height):
        min = self.minPlateArea
        max = self.maxPlateArea
        if (self.type_of_plate == 0):
            ratioMin = 3
            ratioMax = 6
        else:
            ratioMin = 1
            ratioMax = 2
        ratio = float(width)/float(height)
        if ratio < 1:
            ratio = 1/ratio
        
        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    
    def preRatioCheck(self, area, width, height):
        min = self.minPlateArea
        max = self.maxPlateArea
        if (self.type_of_plate == 0):
            ratioMin = 2.5
            ratioMax = 7
        else:
            ratioMin = 0.8
            ratioMax = 2.5
        ratio = float(width)/float(height)
        if ratio < 1:
            ratio = 1/ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True


    def validateRotationAndRatio(self, rect):
        (x, y), (width, height), rect_angle = rect

        if (width > height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
        
        if angle > 15:
            return False
        if (height == 0 or width == 0):
            return False

        area = width*height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True