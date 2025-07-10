#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Run YOLO on a called image
#
###

import platform
import os, sys, glob
import cv2
import time

from threading import Thread

# From MICLab
## Configuration
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
configs = config.get_config()

## Logging
import logging
debug = configs['debugs']['debug']
logging.basicConfig(filename=configs['debugs']['logFile'], level=logging.INFO)
#logging.basicConfig(filename='log.txt', level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True

## What platform are we running on
machine = platform.machine()
logger.info(f"machine: {machine}")

# Check if we're on Raspberry Pi
def is_raspberry_pi():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except:
        return False

# Add IMX500 imports
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Platform detection
if machine == "aarch64":
    if is_raspberry_pi():
        # Check for IMX500 config
        if configs['runTime'].get('use_imx500', False):
            device = "imx500"
            logger.info("Detected Raspberry Pi 5 with IMX500")
        else:
            device = "rpi"
            logger.info("Detected Raspberry Pi 5")
    else:
        device = "tpu"
        logger.info("Detected Coral Dev Board (TPU)")
else:
    import torch
    device = "cpu" 
    if torch.cuda.is_available(): device = "cuda" 
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"

#Define the cameras
from camera import camera
inputCam_1 = camera(configs, configs['runTime']['camId'])
if(configs['runTime']['nCameras'] == 2):
    inputCam_2 = camera(configs, configs['runTime']['camId_2'])

from serialComms import commsClass
serialPort = commsClass(configs['comms'])

def openPin():
    pin = GPIO(f"/dev/gpiochip{gpioChip}", gpioPin, "in")
    #timeTrigerGPIO = GPIO(f"/dev/gpiochip{gpioChip}", gpioPin, "in", bias="pull_up") # pull up not avail
    pin.edge = "both" #“none”, “rising”, “falling”, or “both”
    return pin

if device == "tpu":
    runTimeCheckThread = True
    from periphery import GPIO  #pip install python-periphery
    gpioChip = configs['timeSync']['gpio_chip']
    gpioPin = configs['timeSync']['gpio_pin']
    logger.info(f"GPIO chip:pin {gpioChip}:{gpioPin}")


    #logger.info(f"GPIO chip:pin {gpioChip}:{gpioPin} interupt support = {timeTrigerGPIO.supports_interrupts}")

def checkClockReset_thread():
    timeTrigerGPIO = openPin()
    logger.info(f"Starting clock reset thread: GPIO {timeTrigerGPIO}")

    while runTimeCheckThread:
        #pinStatus = 
        if timeTrigerGPIO.poll(0.25): #Wait for the edge. Pure interup is not imprelented
            pinStatus = timeTrigerGPIO.read()
            logger.info(f"Checking clock reset pin: {pinStatus}")
            if pinStatus: 
                inputCam_1.setZeroTime()
                logger.info(f"Zero Clock: {pinStatus}")
                if(configs['runTime']['nCameras'] == 2): inputCam_2.setZeroTime()
            # There is a bug and the interup is not clearing. So:
            timeTrigerGPIO.close()
            time.sleep(0.05) # wait for it to close
            timeTrigerGPIO = openPin()



## Import our items after we set the log leve
from modelRunTime import modelRunTime
import distance
import display


runCam = [True]* configs['runTime']['nCameras'] 
def get_cam1Image(cam):
    logger.info(f"Init Camera Thread")
    while runCam[0]:
        cam.capImage()
def get_cam2Image(cam):
    logger.info(f"Init Camera 2 Thread")
    while runCam[1]:
        cam.capImage()




def sanitizeStr(str):
    import re
    str = re.sub(r"[^\w\s]", "-", str) # Remove special chars
    str = re.sub(r"\s+", "-", str) #remove white space
    return str


def handleImage(image, imgCapTime, dCalc, objDisp:display.displayHandObject, camId = 1 ):
    ##
    # Immediatly make a copy
    ##
    imageCopy = image.copy()
    logger.info(f"------------------Camera {camId}---------------------------")
    #logger.info(f"Image Capture Time: {imgCapTime}") # THisis in each cameras log
    #logger.info(f"size: {image_1.shape}")

    if(configs['debugs']['runInfer']):
        logger.info(f"Run inference on: Image size: {imageCopy.shape}")
        results, imageCopy = infer.runInference(imageCopy)
        logger.info(f"Infer done")
        if isinstance(results, int): #Did we get a bad inference
            validRes = False
            logger.error(f"!!!Inference Failed!!!!")
        else:
            validRes = dCalc.loadData(results )

            # send over serial: timeMS= 0, handConf= 0, object=None, objectConf=0, distance=0
            # Grab object from inference: 4=Confidence, 5 = class
            serialPort.sendString(timeMS=imgCapTime, handConf=distCalc.handConf, 
                              object=distCalc.grabObject[5], objectConf=distCalc.grabObject[4], distance=distCalc.bestDist)

        # Send the results over serial
        # make object from serialComms.py
        # $24, "CV", imgCapTime (uint_32), handConf (uint8), object class (uint8), object conf (uint8), Distance (uint16), <LF><CR>
    else: 
        validRes = False

    if configs['debugs']['dispResults']:
        # Show the image
        exitStatus = objDisp.draw(imageCopy, dCalc, validRes, imageFile=imageFile)
        #exitStatus = True

        if exitStatus == ord('q'):  # q = 113
            return False
            logger.info(f"********   quit now ***********")
        
    return True

def change_log_file(logger_ptr, fileName):
    for handeler in logger_ptr.handlers[:]:
        if isinstance(handeler, logging.FileHandler):
            logger_ptr.removeHandler(handeler)
            handeler.close()
    new_fHandle = logging.FileHandler(fileName)
    logger_ptr.addHandler(new_fHandle)

import numpy as np

def parse_detections(metadata, intrinsics, imx500, picam2, threshold, iou, max_detections):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return []
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)
    class Detection:
        def __init__(self, box, category, conf):
            self.box = box
            self.category = category
            self.conf = conf
    detections = [Detection(box, category, score) for box, score, category in zip(boxes, scores, classes) if score > threshold]
    return detections

if __name__ == "__main__":
    ## Set up the file saves
    imageFile = ""
    if(configs['debugs']['saveImages']):
        subject = sanitizeStr(input("Enter the subject ID: \n> "))
        object  = sanitizeStr(input("Enter the object: \n> "))
        run     = sanitizeStr(input("Enter the run (The run will start on <enter>): \n> "))

        logger.info(f"subject: {subject}")
        logger.info(f"object: {object}")
        logger.info(f"run: {run}")
        imageFile = f"{subject}_{object}_{run}"
        # Set the log and video file to this run
        change_log_file(logger, f'{imageFile}.txt')
        if configs['debugs']['videoFile'] != "":
            configs['debugs']['videoFile'] = f"{imageFile}.avi"
            

    ## set the model information
    if(configs['debugs']['runInfer']):
        infer = modelRunTime(configs, device)
    
    distCalc = distance.distanceCalculator(configs['training']['imageSize'], 
                                           configs['runTime']['distSettings'])

    handObjDisp = display.displayHandObject(configs)

    if(configs['runTime']['nCameras'] == 2):
        distCalc_2 = distance.distanceCalculator(configs['training']['imageSize'], 
                                           configs['runTime']['distSettings'])
        handObjDisp_2 = display.displayHandObject(configs, camNum=2)
    
    # Add IMX500 initialization if needed
    if device == "imx500":
        imx500_model = configs['runTime']['imx500_model']
        imx500_labels = configs['runTime'].get('imx500_labels', None)
        imx500_threshold = configs['runTime'].get('imx500_threshold', 0.5)
        imx500_iou = configs['runTime'].get('imx500_iou', 0.5)
        imx500_max_detections = configs['runTime'].get('imx500_max_detections', 10)

        imx500 = IMX500(imx500_model)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        if imx500_labels:
            with open(imx500_labels, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        intrinsics.update_with_defaults()
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
        picam2.start(config, show_preview=False)

    ## Get image
    if configs['runTime']['imgSrc'] == 'camera':
        ## Load the camera
        camThread_1 = Thread(target=get_cam1Image, args=(inputCam_1, ))
        camThread_1.start()
        if(configs['runTime']['nCameras'] == 2):
            camThread_2 = Thread(target=get_cam2Image, args=(inputCam_2, ))
            camThread_2.start()

        startTime = [time.time()] * configs['runTime']['nCameras'] 
        dataRateTime = [0] * configs['runTime']['nCameras']  
        frameTime = 1/configs['runTime']['camRateHz']

        if device == "tpu":
            getTimeSetThread = Thread(target=checkClockReset_thread)
            getTimeSetThread.start()


        # Get the image
        while all(runCam):
            thisTime = time.time()
            camStat = [False, False]
            #camStat = [False]*configs['runTime']['nCameras'] 
            dataRateTime[0] = (thisTime-startTime[0])
            if(dataRateTime[0]) >= frameTime: 
                #logger.info(f"Get next image")
                if device == "imx500":
                    # Get detections from IMX500
                    metadata = picam2.capture_metadata()
                    detections = parse_detections(metadata, intrinsics, imx500, picam2, imx500_threshold, imx500_iou, imx500_max_detections)
                    # Convert detections to your expected format for downstream processing
                    # For example, create a numpy array: [x, y, w, h, conf, class]
                    results = []
                    for det in detections:
                        x, y, w, h = det.box
                        conf = det.conf
                        cls = det.category
                        results.append([x, y, x+w, y+h, conf, cls])
                    results = np.array(results)
                    image_1 = picam2.capture_array()
                    camTime_1 = int((time.time() - startTime[0]) * 1000)
                    camStat[0] = True
                else:
                    camStat[0], image_1, camTime_1 = inputCam_1.getImage()

            if(configs['runTime']['nCameras'] == 2):
                dataRateTime[1] = (thisTime-startTime[1])
                if(dataRateTime[1]) >= frameTime: 
                    camStat[1], image_2, camTime_2 = inputCam_2.getImage()

            if camStat[0]:
                if device == "imx500":
                    # Use IMX500 results directly
                    validRes = distCalc.loadData(results)
                    serialPort.sendString(timeMS=camTime_1, handConf=distCalc.handConf, 
                        object=distCalc.grabObject[5], objectConf=distCalc.grabObject[4], distance=distCalc.bestDist)
                    if configs['debugs']['dispResults']:
                        exitStatus = handObjDisp.draw(image_1, distCalc, validRes, imageFile=imageFile)
                        if exitStatus == ord('q'):
                            runCam[0] = False
                else:
                    runCam[0] = handleImage(image_1, camTime_1, distCalc, handObjDisp, camId=1)
                logger.info(f"Total cam 1 time: {dataRateTime[0]*1000:.2f}ms, " \
                            f"{1/dataRateTime[0]:.1f}Hz, " \
                            f"Cap Time: {camTime_1}ms")
                startTime[0] = time.time()

            if camStat[1]:
                runCam[1] = handleImage(image_2, camTime_2, distCalc_2, handObjDisp_2, camId=2)
                logger.info(f"Total cam 2 time: {dataRateTime[1]*1000:.2f}ms, " \
                            f"{1/dataRateTime[1]:.1f}Hz, " \
                            f"Cap Time: {camTime_2}ms")

                startTime[1] = time.time()


            if(configs['runTime']['displaySettings']['runCamOnce']): 
                logger.info(f"Exit after one shot")

            
        # Destructor
        infer.exit() # We must exit for the TPU
        runCam = [False, False] # Kill both cameras
        camThread_1.join() # join the thread back to main
        del inputCam_1 
        if(configs['runTime']['nCameras'] == 2):
            camThread_2.join() # join the thread back to main
            del inputCam_2 
        
        if device == "tpu":
            runTimeCheckThread = False
            getTimeSetThread.join()
            #timeTrigerGPIO.close()


    else: # image file(s)
        imagePath = configs['runTime']['imageDir'] + '/' + configs['runTime']['imgSrc']
        imageFiles = glob.glob(imagePath)  # Find all matching images
        imageFiles = sorted(imageFiles)  # Alphabetical sorting should work for timestamps

        for i,image in enumerate(imageFiles, start=1): #Start the index at 1
            logger.info("---------------------------------------------")
            logger.info(f"File {i}/{len(imageFiles)}: {image}")
            thisImg = cv2.imread(image)  # Read the image
            #thisImg = cv2.rotate(thisImg, cv2.ROTATE_180)

            ## TODO: Move to handleImg
            #    runCam[1] = handleImage(image_2, camTime_2, distCalc_2, handObjDisp_2, camId=2)

            # Run inference
            results, image = infer.runInference(thisImg)
            # get the distance
            validRes = distCalc.loadData(results)
            # send over serial: timeMS= 0, handConf= 0, object=None, objectConf=0, distance=0
            # Grab object from inference: 4=Confidence, 5 = class
            serialPort.sendString(timeMS=0, handConf=distCalc.handConf, 
                                  object=distCalc.grabObject[5], objectConf=distCalc.grabObject[4], distance=distCalc.bestDist)

            # Draw it
            if configs['debugs']['dispResults']:
                exitStatus = handObjDisp.draw(image, distCalc, validRes, asFile=True)
                if exitStatus == ord('q'):  # q = 113
                    logger.info(f"********   quit now ***********")
                    exit()
    

    serialPort.close()      # close the serial port
