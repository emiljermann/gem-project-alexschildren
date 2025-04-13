#!/usr/bin/env python3

#================================================================
# File name: lane_detection.py                                                                  
# Description: learning-based lane detection module                                                            
# Author: Siddharth Anand
# Email: sanand12@illinois.edu                                                                 
# Date created: 08/02/2021                                                                
# Date last modified: 03/14/2025
# Version: 1.0                                                                   
# Usage: python lane_detection.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import cv2
import csv
import math
import time
import torch
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
from cv_bridge import CvBridge, CvBridgeError

from filters import OnlineFilter


# ROS Headers
import rospy
from nav_msgs.msg import Path

# GEM Sensor Headers
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, Bool, Float32, Float64

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped


###############################################################################
# Lane Detection Node
# 
# This module implements deep learning-based lane detection using YOLOPv2.
# It processes images from a camera, identifies lane markings, and publishes
# waypoints for autonomous navigation.
###############################################################################

class PedestrianDetector:
    """
    Main class for lane detection using YOLOPv2 neural network.
    
    This class handles:
    1. Image preprocessing and enhancement
    2. Deep learning model inference
    3. Lane detection and boundary identification
    4. Waypoint generation for vehicle navigation
    5. Visual feedback through annotated images
    """
    
    def __init__(self):
        """
        Initialize the lane detection node with model, parameters and ROS connections.
        
        Sets up:
        - Frame buffering for stable detection
        - Deep learning model (YOLOPv2)
        - ROS publishers and subscribers
        - Image processing parameters
        """

        # Frame buffer for batch processing to increase efficiency
        self.frame_buffer = []
        self.buffer_size = 4  # Process 4 frames at once for better throughput
        self.batch = False # Flag to choose batch or not @TODO: Implement batch, do we want that?
        
        # Initialize ROS node
        rospy.init_node('lane_detection_node', anonymous=True)
        
        # Image processing utilities and state variables
        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images
        self.prev_left_boundary = None  # Store previous lane boundary for smoothing
        self.estimated_lane_width_pixels = 200  # Approximate lane width in image pixels
        self.prev_waypoints = None  # Previous waypoints for temporal consistency
        self.endgoal = None  # Target point for navigation
        
        ###############################################################################
        # Deep Learning Model Setup
        ###############################################################################
        
        # Set up compute device (GPU if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Commented out stop sign detection model code
        self.pedestrian_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
        self.pedestrian_model.to(self.device).eval()
        
        ###############################################################################
        # Camera and Control Parameters
        ###############################################################################
        
        self.Focal_Length = 800  # Camera focal length in pixels
        self.Real_Height_SS = .75  # Height of stop sign in meters (not used currently)
        self.Brake_Distance = 5  # Distance at which to apply brakes (not used currently)
        self.Brake_Duration = 3  # Duration to hold brakes (not used currently)
        self.Conf_Threshold = 0.8 # Confidence threshold to keep person prediction
        
        ###############################################################################
        # ROS Communication Setup
        ###############################################################################
        
        # Subscribe to camera feed
        self.sub_image = rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.img_callback, queue_size=1)
        # note that oak/rgb/image_raw is the topic name for the GEM E4. If you run this on the E2, you will need to change the topic name to e.g., /zed2/zed_node/left/image_rect_color
        
        # Publishers for visualization and control
        self.pub_bounding_box = rospy.Publisher("pedestrian_detection/bounding_box", list, queue_size=1)

    def letterbox(self, img, new_shape=(384, 640), color=(114, 114, 114)):
        """
        Resize image with letterboxing to maintain aspect ratio.
        
        This is important for neural network input to prevent distortion.
        
        Args:
            img: Input image
            new_shape: Target shape (height, width)
            color: Padding color
            
        Returns:
            resized_img: Resized and padded image
            ratio: Scale ratio (used for inverse mapping)
            padding: Padding values (dw, dh)
        """
        # Original shape
        shape = img.shape[:2]
        
        # Handle single dimension input
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Calculate scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        
        # Calculate new unpadded dimensions
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Calculate padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize image
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, ratio, (dw, dh)
    
    
    def get_pedestrian_box_multi(self, model, frames):
        """
        Gets pose points from a given YOLO model and frame.
        Args:
            model: YOLO model.
            frame: Frame from the camera.
        Returns:
            bounding_box: bounding box of person with highest confidence, None otherwise
            in format (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            
            confidence: confidence score that said pedestrian exists
        """
        with torch.no_grad():
            results = model(frames).pandas().xyxy

        if len(results) == 0:
            return None, 0
        
        all_box_coords, all_highest_conf = [], []

        for result in results:
            box_coords = None
            highest_conf = 0
            for obj in result.itertuples():
                if obj.name == "person":
                    boxes = (obj.xmin, obj.ymin, obj.xmax, obj.ymax)
                    confidence = obj.confidence
                    if confidence > highest_conf:
                        highest_conf = confidence
                        box_coords = boxes
                        highest_conf = confidence
                    break
            all_box_coords.append(box_coords)
            all_highest_conf.append(highest_conf)

        return all_box_coords, all_highest_conf 
    
    def get_pedestrian_box_single(self, model, frame):
        """
        Gets pose points from a given YOLO model and frame.
        Args:
            model: YOLO model.
            frame: Frame from the camera.
        Returns:
            bounding_box: bounding box of person with highest confidence, None otherwise
            in format (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            
            confidence: confidence score that said pedestrian exists
        """
        with torch.no_grad():
            result = model(frame).pandas().xyxy

        if len(result) == 0:
            return None, 0
        
        box_coords = None
        highest_conf = 0
        for obj in result.itertuples():
            if obj.name == "person":
                boxes = (obj.xmin, obj.ymin, obj.xmax, obj.ymax)
                confidence = obj.confidence
                if confidence > highest_conf:
                    highest_conf = confidence
                    box_coords = boxes
                    highest_conf = confidence
                break

        return box_coords, highest_conf 
    
    def img_callback(self, img):
        """
        Process incoming camera images to detect  pedestrians and generate bounding boxes.
        
        This function:
        1. Converts ROS image to OpenCV format
        2. Enhances image using color filtering and contrast
        3. Preprocesses image for neural network
        4. Adds image to buffer for batch processing
        5. Performs inference when buffer is full
        6. Generates and publishes waypoints and visualizations
        
        Args:
            img: ROS Image message from camera
        """
        try:
            # Convert ROS Image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            
            ###############################################################################
            # Model Inference Pipeline
            ###############################################################################
            
            resized_img = self.letterbox(img)[0] # ATTENTION: This changes the coordinates of the bounding box, so use of the outputted bounding box must be taken into account

            if self.batch: # @TODO: Currently this makes no sense bc the queue is size 1, but I kept it unless we want it later
                # Add to buffer for batch processing
                self.frame_buffer.append(resized_img)
                
                # When buffer is full, process batch for efficiency
                if len(self.frame_buffer) >= self.buffer_size:

                    # Run inference on batch
                    with torch.no_grad():
                        all_box_coords, all_conf = self.get_pedestrian_box_multi(self.pedestrian_model, self.frame_buffer)

                    self.frame_buffer.clear()

                    for i in range(self.buffer_size): 
                        conf = all_conf[i]
                        box_coords = all_box_coords[i]
                        if conf > self.Conf_Threshold:
                            self.pub_bounding_box.publish(box_coords)

            else:
                # Run inference on single image
                with torch.no_grad():
                    box_coords, conf = self.get_pedestrian_box_single(self.pedestrian_model, resized_img)

                if conf > self.Conf_Threshold:
                    self.pub_bounding_box.publish(box_coords)


        except CvBridgeError as e:
            print(e)
    
    def detect_pedestrians(self, img):
        # @TODO: Maybe post process bounding box info in some way?
        return None


###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        # Create detector instance
        detector = PedestrianDetector()
        
        # Keep node running until shutdown
        rate = rospy.spin() #r ospy.Rate(10)  # 10 Hz control loop
    except rospy.ROSInterruptException:
        pass
