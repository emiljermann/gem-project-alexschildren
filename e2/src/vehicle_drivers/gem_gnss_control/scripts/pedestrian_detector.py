#!/usr/bin/env python3

#================================================================
# File name: pedestrian_detector.py                                                                  
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
import message_filters

# GEM Sensor Headers
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
        rospy.init_node('pedestrian_detector_node', anonymous=True)
        
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
        self.Conf_Threshold = 0.7 # Confidence threshold to keep person prediction
        
        ###############################################################################
        # ROS Communication Setup
        ###############################################################################
        
        # Subscribe to camera feed
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.img_callback, queue_size=1)
        self.sub_rgb_img = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
        self.sub_depth_img = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb_img, self.sub_depth_img], queue_size=2, slop=0.1)
        self.ts.registerCallback(self.img_callback)
        
        # Publishers for visualization and control
        self.pub_bounding_box = rospy.Publisher("pedestrian_detection/bounding_box", Float32MultiArray, queue_size=1)
        self.pub_rgb_pedestrian_image = rospy.Publisher("pedestrian_detection/rgb/pedestrian_image", Image, queue_size=1)
        self.pub_depth = rospy.Publisher("pedestrian_detection/avg_depth", Float32MultiArray, queue_size=1)

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
    
    def get_pedestrian_box(self, model, frame):
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
            result = model(frame).pandas().xyxy[0]

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
    
    def process_pedestrian_box(self, box_coords, conf, pad, ratio, rgb_img, depth_img):

        avg_depth, med_depth, sd_depth, center_depth, first_quartile_depth = None, None, None, None, None

        if box_coords and conf > self.Conf_Threshold:

            # Draw the bounding box on the image (adjust for padding and ratio)
            x1, y1, x2, y2 = box_coords
            # Convert to int and adjust for letterbox padding
            x1 = int((x1 - pad[0]) / ratio[0])
            y1 = int((y1 - pad[1]) / ratio[1])
            x2 = int((x2 - pad[0]) / ratio[0])
            y2 = int((y2 - pad[1]) / ratio[1])

            # Publish box coordinate data
            box_coords_msg = Float32MultiArray()
            box_coords_msg.data = list(box_coords)
            self.pub_bounding_box.publish(box_coords_msg)

            # Extract region of interest from the depth image
            depth_roi = depth_img[y1:y2, x1:x2]
            valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
            if valid_depths.size > 0:
                # Remove outliers: keep values within 2 standard deviations
                mean_depth = np.mean(valid_depths)
                std_depth = np.std(valid_depths)
                filtered_depths = valid_depths[(valid_depths > (mean_depth - 2 * std_depth)) & (valid_depths < (mean_depth + 2 * std_depth))]

                if filtered_depths.size > 0:
                    avg_depth = np.mean(filtered_depths)
                    med_depth = np.median(filtered_depths)
                    sd_depth = np.std(filtered_depths)
                    first_quartile_depth = np.percentile(filtered_depths, 25)

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    if 0 <= center_y < depth_img.shape[0] and 0 <= center_x < depth_img.shape[1]:
                        center_value = depth_img[center_y, center_x]
                        if np.isfinite(center_value) and center_value > 0:
                            center_depth = float(center_value)
                else:
                    avg_depth = med_depth = sd_depth = center_depth = first_quartile_depth = None
            else:
                avg_depth = med_depth = sd_depth = center_depth = first_quartile_depth = None
            
            # Add rectangle to rgb image
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.circle(rgb_img, (center_x, center_y), 4, (0, 0, 255), -1)  # red dot
            if center_depth is not None:
                cv2.putText(rgb_img, f"{center_depth:.2f}m", (center_x + 5, center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(rgb_img, f"Pedestrian: {avg_depth:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return rgb_img, avg_depth, med_depth, sd_depth, center_depth, first_quartile_depth

    
    def img_callback(self, rgb_img, depth_img):
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")

            resized_img, ratio, pad = self.letterbox(rgb_img)  # unpack ratio and padding

            box_coords, conf = self.get_pedestrian_box(self.pedestrian_model, resized_img) # Get bounding box of detected pedestrian

            rgb_img, avg_depth, med_depth, sd_depth, center_depth, first_quartile_depth = self.process_pedestrian_box(box_coords, conf, pad, ratio, rgb_img, depth_img)

            # Publish RGB image
            ros_rgb_img = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
            self.pub_rgb_pedestrian_image.publish(ros_rgb_img)

            # Publish Depth Data:
            depth_values = [avg_depth, med_depth, sd_depth, center_depth, first_quartile_depth]
            safe_values = [float(v) if v is not None else float('nan') for v in depth_values]

            depth_data = Float32MultiArray()
            depth_data.data = safe_values
            self.pub_depth.publish(depth_data)

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

