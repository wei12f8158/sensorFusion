import time
import os
import sys
import logging

import yaml
import numpy as np
import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common
from nms import non_max_suppression, non_max_suppresion_v8
import cv2
import json

from utils import plot_one_box, Colors, get_image_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")

# MJB (from chatbot) we need to be able to restart
from tpuWorker import TPUWorker
from tflite_runtime.interpreter import Interpreter


class EdgeTPUModel:

    def __init__(self, model_file, names_file, conf_thresh=0.25, iou_thresh=0.45, filter_classes=None, agnostic_nms=False, max_det=1000, v8=False, timeOut=1.0):
        """
        Creates an object for running a Yolov5 model on an EdgeTPU
        
        Inputs:
          - model_file: path to edgetpu-compiled tflite file
          - names_file: yaml names file (yolov5 format)
          - conf_thresh: detection threshold
          - iou_thresh: NMS threshold
          - filter_classes: only output certain classes
          - agnostic_nms: use class-agnostic NMS
          - max_det: max number of detections
        """
    
        model_file = os.path.abspath(model_file)
    
        if not model_file.endswith('tflite'):
            model_file += ".tflite"
            
        self.model_file = model_file
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.filter_classes = filter_classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.v8 = v8
        
        logger.info("Confidence threshold: {}".format(conf_thresh))
        logger.info("IOU threshold: {}".format(iou_thresh))
        
        self.inference_time = None
        self.nms_time = None
        self.interpreter = None
        self.colors = Colors()  # create instance for 'from utils.plots import colors'
        
        self.get_names(names_file)
        self.make_interpreter(timeOut)
        
    def get_names(self, path):
        """
        Load a names file
        
        Inputs:
          - path: path to names file in yaml format
        """

        with open(path, 'r') as f:
          cfg = yaml.load(f, Loader=yaml.SafeLoader)
          
        names = cfg['names']
        logger.info("Loaded {} classes".format(len(names)))
        
        self.names = names
    
    def make_interpreter(self, timeout):
        """
        Internal function that loads the tflite file and creates
        the interpreter that deals with the EdgetPU hardware.
        """
        # Load the model and allocate
        # new way
        self.tpu = TPUWorker(self.model_file, timeout=timeout) # MJB Create the TPU worker
        self.input_zero = self.tpu.input_zero
        self.input_scale = self.tpu.input_scale
        self.output_zero = self.tpu.output_zero
        self.output_scale = self.tpu.output_scale
        self.input_size = self.tpu.input_size
    
        logger.info("Input scale: {}".format(self.input_scale))
        logger.info("Input zero: {}".format(self.input_zero))
        logger.info("Output scale: {}".format(self.output_scale))
        logger.info("Output zero: {}".format(self.output_zero))
        logger.info("edgetpumodel: input_size: {self.input_size}")
        
        logger.info("Successfully loaded {}".format(self.model_file))
    
    def get_image_size(self):
        if self.input_size is not None:
            return self.input_size
        else:
            logger.warning("Interpreter is not yet loaded")
            print("Interp has not been loaded yet")

    def predict(self, image_path, save_img=True, save_txt=True):
        logger.info("Attempting to load {}".format(image_path))
    
        full_image, net_image, pad = get_image_tensor(image_path, self.input_size[0])
        pred = self.forward(net_image)
        
        base, ext = os.path.splitext(image_path)


        ### Josh sez: writing to the image path makes chaos
        #output_path = base + "_detect" + ext
        imgName = os.path.split(base)
        #print(f"base: {base}, {imgName[1]}")
        output_path = "out/" + imgName[1] + "_detect" + ext 
        det = self.process_predictions(pred[0], full_image, pad, output_path, save_img=save_img, save_txt=save_txt)
        
        return det

    def forward(self, x:np.ndarray, with_nms=True) -> np.ndarray:
        """
        Predict function using the EdgeTPU

        Inputs:
            x: (C, H, W) image tensor
            with_nms: apply NMS on output

        Returns:
            prediction array (with or without NMS applied)

        """
        tstart = time.time()
        # Transpose if C, H, W
        if x.shape[0] == 3:
          x = x.transpose((1, 2, 0))
        
        x = x.astype('float32')
        
        # Scale input, conversion is: real = (int_8 - zero)*scale
        x = (x/self.input_scale) + self.input_zero
        if self.v8:
            x = x[np.newaxis].astype(np.int8)
        else:
            x = x[np.newaxis].astype(np.uint8)
        #print(f"v8: {self.v8}, image: {x.shape}") 


        #MJB: Beable to detect the TPU crash and restart
        status, raw_output = self.tpu.infer(x)
        if status == "ok":
            pass
            #print("Output shape:", raw_output.shape)
        elif status == "timeout":
            print("Timeout occurred — restarting")
            self.tpu.restart()
            return 0
        else:
            print("Status:", status)
            return 0
        
        # Scale output
        #print(f"scale output")
        #raw_output = common.output_tensor(self.interpreter, 0).copy() #MJB, use a copy so we con't hold on error
        result = (raw_output.astype('float32') - self.output_zero) * self.output_scale
        if self.v8:
            result = np.transpose(result, [0, 2, 1])  # tranpose for yolov8 models
        
        self.inference_time = time.time() - tstart
        logger.info(f"inference time {self.inference_time}")
        
        if with_nms:
        
            tstart = time.time()
            if self.v8:
                nms_result = non_max_suppresion_v8(result, self.conf_thresh, self.iou_thresh, self.filter_classes,
                                                   self.agnostic_nms, max_det=self.max_det)
            else:
                nms_result = non_max_suppression(result, self.conf_thresh, self.iou_thresh, self.filter_classes,
                                                 self.agnostic_nms, max_det=self.max_det)
            self.nms_time = time.time() - tstart
            
            return nms_result
            
        else:    
          return result
          
    def get_last_inference_time(self, with_nms=True):
        """
        Returns a tuple containing most recent inference and NMS time
        """
        res = [self.inference_time]
        
        if with_nms:
          res.append(self.nms_time)
          
        return res
    
    def get_scaled_coords(self, xyxy, output_image, pad):
        """
        Converts raw prediction bounding box to orginal
        image coordinates.
        
        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        pad_w, pad_h = pad
        in_h, in_w = self.input_size
        out_h, out_w, _ = output_image.shape
                
        ratio_w = out_w/(in_w - pad_w)
        ratio_h = out_h/(in_h - pad_h) 
        
        out = []
        for coord in xyxy:

            x1, y1, x2, y2 = coord
                        
            x1 *= in_w*ratio_w
            x2 *= in_w*ratio_w
            y1 *= in_h*ratio_h
            y2 *= in_h*ratio_h
            
            x1 = max(0, x1)
            x2 = min(out_w, x2)
            
            y1 = max(0, y1)
            y2 = min(out_h, y2)
            
            out.append((x1, y1, x2, y2))
        
        return np.array(out).astype(int)

    def process_predictions(self, det, output_image, pad, output_path="detection.jpg", save_img=True, save_txt=True, hide_labels=False, hide_conf=False):
        """
        Process predictions and optionally output an image with annotations
        """
        if len(det):
            # Rescale boxes from img_size to im0 size
            # x1, y1, x2, y2=
            det[:, :4] = self.get_scaled_coords(det[:, :4], output_image, pad)
            output = {}
            base, ext = os.path.splitext(output_path)
            
            s = ""
            
            # Print results
            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            if s != "":
                s = s.strip()
                s = s[:-1]
            
            logger.info("Detected: {}".format(s))
            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                    output_image = plot_one_box(xyxy, output_image, label=label, color=self.colors(c, True))
                if save_txt:
                    output[base] = {}
                    output[base]['box'] = xyxy
                    output[base]['conf'] = conf
                    output[base]['cls'] = cls
                    output[base]['cls_name'] = self.names[c]
                    
            if save_txt:
                output_txt = base+".txt"
                print(f"Output file: {output_txt}")
                with open(output_txt, 'w') as f:
                   json.dump(output, f, indent=1)
            if save_img:
              print(f"Output image: {output_path}")
              cv2.imwrite(output_path, output_image)
            
        return det

    def exit(self):
        logger.info(f"Exiting TPU Inference")
        self.tpu.stop()
