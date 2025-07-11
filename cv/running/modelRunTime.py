#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# sorts out if we are on a tpu, or not
# Sending camera data or saved image
# Sends the image to the appropriate inference model
#
###
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modelRunTime")

class modelRunTime:
    def __init__(self, configs, device) -> None:
        self.device = device 
        self.configs = configs
        self.debug = configs['debugs']['showInfResults']
        tpuTimeout = configs['debugs']['tpuThreadTimeout']

        if device == "tpu":
            from edgetpumodel import EdgeTPUModel
            import numpy as np
            weightsFile = configs['training']['weightsFile_tpu']
        elif device == "rpi":
            from rpiModel import RaspberryPiModel
            import numpy as np
            # Use TFLite or ONNX model for Raspberry Pi
            weightsFile = configs['training'].get('weightsFile_rpi', configs['training']['weightsFile_tpu'])
        else:
            from ultralytics import YOLO
            weightsFile = configs['training']['weightsFile']

        modelPath = Path(configs['training']['weightsDir'])
        modelFile = modelPath/weightsFile
        logger.info(f"model: {modelFile}")

        # Load the state dict
        if device == "tpu":
            thresh = min(configs['runTime']['distSettings']['handThreshold'],
                         configs['runTime']['distSettings']['objectThreshold'])
            dataSetFile = configs['training']['dataSetDir'] + '/' +configs['training']['dataSet']
            self.model = EdgeTPUModel(modelFile, dataSetFile,
                                      conf_thresh=thresh, #only over this
                                      iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
                                      filter_classes=None,  # Not implemented
                                      agnostic_nms=False,
                                      max_det=100,
                                      v8=True, timeOut=tpuTimeout)
            # Prime with the image size
            self.input_size = self.model.get_image_size()
            print(f"self.input_size: {type(self.input_size)}, {self.input_size}")
            x = (255*np.random.random((3,*self.input_size))).astype(np.int8)
            self.model.forward(x) 
        elif device == "rpi":
            thresh = min(configs['runTime']['distSettings']['handThreshold'],
                         configs['runTime']['distSettings']['objectThreshold'])
            dataSetFile = configs['training']['dataSetDir'] + '/' +configs['training']['dataSet']
            
            # Get Raspberry Pi specific settings
            use_gpu = configs['runTime'].get('rpi_use_gpu', False)
            num_threads = configs['runTime'].get('rpi_num_threads', 4)
            
            self.model = RaspberryPiModel(modelFile, dataSetFile,
                                         conf_thresh=thresh,
                                         iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
                                         filter_classes=None,
                                         agnostic_nms=False,
                                         max_det=100,
                                         v8=True,
                                         use_gpu=use_gpu,
                                         num_threads=num_threads)
            # Prime with the image size
            self.input_size = self.model.get_image_size()
            print(f"self.input_size: {type(self.input_size)}, {self.input_size}")
            x = (255*np.random.random((3,*self.input_size))).astype(np.float32)
            self.model.forward(x)
        else:
            self.model = YOLO(modelFile)  #

    def exit(self):
        if self.device == "tpu":
            print("exit inference")
            self.model.exit()
        

    def runInference(self, image):
        #logger.info(f"image type: {type(image)}")

        if self.device == 'tpu':
            if isinstance(image, str):
                yoloResults =self.runInferenceTPUFile(image)
            else:
                yoloResults =self.runInferenceTPUWebCam(image)

            #inference time, nms time
            logger.info(f"TPU Inference, nms time: {self.model.get_last_inference_time()}") 
            #logger.info(f"TPU Results: {type(yoloResults)}, {yoloResults}")
            return yoloResults, image
        elif self.device == 'rpi':
            if isinstance(image, str):
                yoloResults = self.runInferenceRPiFile(image)
            else:
                yoloResults = self.runInferenceRPiWebCam(image)

            #inference time, nms time
            logger.info(f"Raspberry Pi Inference, nms time: {self.model.get_last_inference_time()}") 
            return yoloResults, image
        else:
            yoloResults = self.model.predict(image) # Returns a dict
            #logger.info(yoloResults[0].speed)

            #logger.info(f"Results.boxes: {type(yoloResults[0].boxes.data)}, {yoloResults[0].boxes.data}")
            return yoloResults[0].boxes.data, image

    def runInferenceTPUFile(self, image):
            #logger.info(f"Running TPU file infernece")
            # Returns a numpy array: x1, x2, y1, y2, conf, class
            results = self.model.predict(image, save_img=self.debug, save_txt=self.debug)

            return results, image

    def runInferenceTPUWebCam(self, image):
            from utils import get_image_tensor
            #logger.info(f"Running TPU webcam infernece")
            full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
            #logger.info(f"Done padding")
            pred = self.model.forward(net_image)
            if isinstance(pred, int): return 0 # inference failed
            #logger.info(f"Done forward path")
            results = self.model.process_predictions(det=pred[0], 
                                                     output_image=full_image, 
                                                     pad=pad,
                                                     save_img=False,
                                                     save_txt=False,
                                                     hide_labels=True,
                                                     hide_conf=True)
                        
            tinference, tnms = self.model.get_last_inference_time()
            #logger.info("Frame done in {}".format(tinference+tnms))
            return results

    def runInferenceRPiFile(self, image):
        #logger.info(f"Running Raspberry Pi file inference")
        # Returns a numpy array: x1, x2, y1, y2, conf, class
        results = self.model.predict(image, save_img=self.debug, save_txt=self.debug)
        return results

    def runInferenceRPiWebCam(self, image):
        from utils import get_image_tensor
        #logger.info(f"Running Raspberry Pi webcam inference")
        full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
        #logger.info(f"Done padding")
        pred = self.model.forward(net_image)
        if isinstance(pred, int): return 0 # inference failed
        #logger.info(f"Done forward path")
        results = self.model.process_predictions(det=pred[0], 
                                                 output_image=full_image, 
                                                 pad=pad,
                                                 save_img=False,
                                                 save_txt=False,
                                                 hide_labels=True,
                                                 hide_conf=True)
                        
        tinference, tnms = self.model.get_last_inference_time()
        #logger.info("Frame done in {}".format(tinference+tnms))
        return results
