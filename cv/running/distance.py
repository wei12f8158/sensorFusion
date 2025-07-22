#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Calculate the distance between two items
#
###

#TODO: improve data structure

import math
import torch

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Distance")

class distanceCalculator:
    def __init__(self, trainImgSize, config) -> None:
        ##Configs
        self.modelImgSize = trainImgSize #What is the pxl count of the image directly to inferance
        self.pxPer_mm  = config['imagePxlPer_mm'] #How many pixels per mm
        self.hThresh = config['handThreshold']
        self.oThresh = config['objectThreshold']
        self.handClassNum = config['handClass']
        self.classMap = config['classMap']

        #logger.info(f"pxPer_mm: {self.pxPer_mm}")

        self.zeroData()

    def zeroData(self):
        self.nHands = 0
        self.nNonHand = 0
        self.grabObject = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) # Init to UL
        self.handObject = None

        self.bestCenter = (0,0)
        self.handCenter = (self.modelImgSize[0], self.modelImgSize[1])            # Init to LR
        self.handConf = 0.0

        self.bestDist = self.calcDist(self.grabObject)
        #logger.info(f"zeroData, Max dist: {self.bestDist}")

    def loadData(self, data ):
        '''
        Return True iff there is one and only one hand
        Loads the data used
        Converts to actual image size
        Args:
            (tenor data): x, y, w, h, probability, class for each item
            (list of classes seen): 
        Returns:
            (Bool): Is or is not valid
        Raises:
        '''
        classField = 5
        confField = 4

        self.zeroData()
        self.time_ms = 0
        self.data = data
        
        logger.info(f"Processing {len(data)} raw detections")
        logger.info(f"Hand threshold: {self.hThresh}, Object threshold: {self.oThresh}")
        logger.info(f"Hand class number: {self.handClassNum}")
        logger.info(f"Class mapping: {self.classMap}")

        for i, object in enumerate(data):
            # E.x. if it reports a bottle, claim appleA
            objClass = int(object[classField])
            originalClass = objClass
            if len(self.classMap) >= objClass:
                object[classField] = self.classMap[objClass]
                logger.info(f"Detection {i}: class {originalClass} -> {object[classField]} (mapped)")

            logger.info(f"Detection {i}: class={object[classField]}, conf={object[confField]:.3f}, bbox=[{object[0]:.1f}, {object[1]:.1f}, {object[2]:.1f}, {object[3]:.1f}]")
            
            if object[classField] == self.handClassNum and object[confField] >= self.hThresh:
                self.nHands += 1
                logger.info(f"  -> Hand detected (confidence: {object[confField]:.3f})")
                #self.hand = object
                self.handCenter = self.findCenter(object)
                self.handObject = object
            elif object[confField] >= self.oThresh: 
                self.nNonHand += 1
                logger.info(f"  -> Object detected (confidence: {object[confField]:.3f})")
                ## we still want to be able to display if we don't have a hand
                # use the best confidence untill we can be bothered to show multiple objectes
                if object[confField] > self.grabObject[confField]:
                    # Convert numpy values to float for PyTorch tensor
                    self.grabObject[confField] = float(object[confField])
                    # Convert the entire object to a PyTorch tensor
                    self.grabObject = torch.tensor([float(x) for x in object])
                    self.bestCenter = self.findCenter(object)
                    logger.info(f"  -> New best object (confidence: {object[confField]:.3f})")
            else:
                logger.info(f"  -> Below threshold (hand: {object[confField] < self.hThresh}, object: {object[confField] < self.oThresh})")


        # If we have multiple hands use the one with the highest confidence
        # Its own loop to leave room for multi gloves
        if self.nHands >= 1:
            logger.info(f"Processing {self.nHands} hands for best confidence...")
            for object in data:
                if object[classField] == self.handClassNum and object[confField] >= self.hThresh and object[confField] > self.handConf:
                    self.handConf = object[confField]
                    self.handCenter = self.findCenter(object)
                    self.handObject = object
                    logger.info(f"New best hand (confidence: {object[confField]:.3f})")


        if self.nNonHand == 0 or self.nHands == 0:
            logger.info(f"loadData: we need at least one hand:{self.nHands} and one target:{self.nNonHand}")
            return False
        
        # Once we have the hand object, get the closest distance
        logger.info("Calculating distances to find closest object...")
        
        # FIX: Prioritize plate class (class 6) over distance
        plate_found = False
        for object in data:
            if object[classField] != self.handClassNum and object[confField] >= self.oThresh: 
                # If this is a plate (class 6), prioritize it
                if object[classField] == 6:  # Plate class
                    logger.info(f"Found plate (class 6) with confidence {object[confField]:.3f}")
                    self.grabObject = torch.tensor([float(x) for x in object])
                    self.bestDist = self.calcDist(object)
                    self.bestCenter = self.findCenter(object)
                    plate_found = True
                    logger.info(f"Selected plate as target object")
                    break
        
        # If no plate found, fall back to closest object
        if not plate_found:
            for object in data:
                if object[classField] != self.handClassNum and object[confField] >= self.oThresh: 
                    thisDist = self.calcDist(object)
                    logger.info(f"Distance to object (class {object[classField]}): {thisDist:.1f}mm")
                    if thisDist < self.bestDist: 
                        # Convert the entire object to a PyTorch tensor
                        self.grabObject = torch.tensor([float(x) for x in object])
                        self.bestDist = thisDist
                        self.bestCenter = self.findCenter(object)
                        logger.info(f"New closest object: {thisDist:.1f}mm")

        logger.info(f"N objects detected: hands = {self.nHands}, non hands = {self.nNonHand},  Distance = {self.bestDist:.0f}mm")
        return True

    def calcDist(self, object):
        '''
        Calculates the distance between the hand and the object
        Args:
        Returns:
            (float): The distance in mm
        Raises:
        '''
        center = self.findCenter(object)
        yDist = (self.handCenter[0]- center[0])
        xDist = (self.handCenter[1]- center[1])

        pxlDist = math.sqrt(pow(yDist,2) + pow(xDist,2))
        return pxlDist/self.pxPer_mm

    def getBox(self, object):
        '''
        Gets the object box
        Args:
            (tenor data): x1, y1, x2, y2, probability, class for each item
        Returns:
            (Upper left and Lower Right Corners of the box)

        '''
        x1, y1, x2, y2 = self.getXY(object)
        UL = [int(x1), int(y1)]
        LR = [int(x2), int(y2)]

        return UL, LR

    def findCenter(self, object):
        '''
        Gets the center of an object
        Args:
            (tenor data): x, y, w, h, probability, class for each item
        Returns:
            (int x, int y): the center of the object in pxles
        Raises:
        '''

        '''
        # corner/size
        #from https://github.com/MIC-Laboratory/Prosthetic_hand_control_MQTT_SSDMobileNet/blob/master/openMV/distance_sender_auto_exposure.py
        center_x = math.floor(x + (w / 2))
        center_y = math.floor(y + (h / 2))
        '''
        #Corner/corner
        x1, y1, x2, y2 = self.getXY(object)
        center_x = math.floor(x1 + (x2 - x1)/2)
        center_y = math.floor(y1 + (y2 - y1)/2)

        return center_x, center_y
    
    def getXY(self, object):
        # Handle both numpy arrays and PyTorch tensors
        if hasattr(object[0], 'item'):
            # PyTorch tensor
            x1 = object[0].item()
            y1 = object[1].item()
            x2 = object[2].item()
            y2 = object[3].item()
        else:
            # Numpy array
            x1 = float(object[0])
            y1 = float(object[1])
            x2 = float(object[2])
            y2 = float(object[3])

        return x1, y1, x2, y2
