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
import numpy as np

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
        elif device == "imx500":
            # IMX500 AI camera mode - no model loading needed
            logger.info("IMX500 AI camera mode - inference handled by IMX500")
            self.model = None
            self.input_size = (640, 640)  # Default size for IMX500
            return
        else:
            from ultralytics import YOLO
            weightsFile = configs['training']['weightsFile']

        modelPath = Path(configs['training']['weightsDir'])
        modelFile = modelPath/weightsFile
        print(f"Loading model from: {modelFile}")
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
            self.model = YOLO(modelFile)  #Try this with the new model

    def exit(self):
        if self.device == "tpu":
            print("exit inference")
            self.model.exit()
        elif self.device == "imx500":
            print("exit IMX500 inference")
            # No cleanup needed for IMX500 AI camera mode
        

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
        elif self.device == 'imx500':
            # IMX500 AI camera mode - inference is handled by IMX500, not here
            logger.warning("IMX500 AI camera mode: inference should be handled by IMX500, not modelRunTime")
            return np.empty((0, 6)), image  # Return empty results
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
        logger.info(f"Running Raspberry Pi webcam inference")
        logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
        logger.info(f"Preprocessed image shape: {net_image.shape}, pad: {pad}")
        
        logger.info("Running model forward pass...")
        pred = self.model.forward(net_image)
        logger.info(f"Model forward returned: {type(pred)}, value: {pred}")
        if isinstance(pred, int): 
            logger.error(f"Inference failed, returned: {pred}")
            return 0 # inference failed
        
        # Handle the case where pred might be empty or not a list
        if hasattr(pred, 'shape'):
            # pred is a numpy array
            logger.info(f"Model output shape: {pred.shape}")
            logger.info(f"Model output type: {type(pred)}")
            if len(pred) > 0:
                logger.info(f"Raw predictions: {len(pred)} detections")
                logger.info(f"First raw prediction: {pred[0]}")
            else:
                logger.info("No raw predictions found")
        else:
            # pred might be a list or other type
            logger.info(f"Model output type: {type(pred)}")
            if isinstance(pred, list) and len(pred) > 0 and hasattr(pred[0], 'shape'):
                logger.info(f"Model output shape: {pred[0].shape}")
                if len(pred[0]) > 0:
                    logger.info(f"Raw predictions: {len(pred[0])} detections")
                    logger.info(f"First raw prediction: {pred[0][0]}")
                else:
                    logger.info("No raw predictions in list")
            else:
                logger.info("No predictions found or invalid format")
        
        logger.info("Processing predictions...")
        # Handle different prediction formats
        if hasattr(pred, 'shape'):
            # pred is a numpy array
            det = pred
        elif isinstance(pred, list) and len(pred) > 0:
            # pred is a list, take first element
            det = pred[0]
        else:
            # Empty or invalid prediction
            det = np.empty((0, 6))
        
        results = self.model.process_predictions(det=det, 
                                                 output_image=full_image, 
                                                 pad=pad,
                                                 save_img=False,
                                                 save_txt=False,
                                                 hide_labels=True,
                                                 hide_conf=True)
        
        logger.info(f"Final results shape: {results.shape if hasattr(results, 'shape') else 'No shape'}")
        if len(results) > 0:
            logger.info(f"Final detections after NMS: {len(results)}")
            for i, det in enumerate(results):
                logger.info(f"Final detection {i}: {det}")
        else:
            logger.info("No final detections after NMS")
                        
        tinference, tnms = self.model.get_last_inference_time()
        logger.info(f"Timing - Inference: {tinference:.3f}s, NMS: {tnms:.3f}s, Total: {tinference+tnms:.3f}s")
        return results
