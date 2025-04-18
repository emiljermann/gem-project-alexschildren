#!/usr/bin/env python3

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
import matplotlib.pyplot as plt
import datetime

from filters import OnlineFilter

# import alvinxy.alvinxy as axy # Import AlvinXY transformation module

# ROS Headers
import rospy
from nav_msgs.msg import Path
import message_filters

# GEM Sensor Headers
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray, Float64MultiArray
# from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva @TODO: Uncomment

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped


###############################################################################
# Pedestrian Detector Node
# 
# This module implements deep learning-based pedestrian detection using YOLOv5n.
#
# python3 -W ignore pedestrian_detector.py
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
        self.pedestrian_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        self.pedestrian_model.to(self.device).eval()
        
        ###############################################################################
        # Camera and Control Parameters
        ###############################################################################
        
        self.Focal_Length = 800  # Camera focal length in pixels
        self.Real_Height_SS = .75  # Height of stop sign in meters (not used currently)
        self.Brake_Distance = 5  # Distance at which to apply brakes (not used currently)
        self.Brake_Duration = 3  # Duration to hold brakes (not used currently)
        self.Conf_Threshold = 0.7 # Confidence threshold to keep person prediction
        self.wheelbase = 1.75 # meters
        self.offset = 0.46 # meters

        self.olat = 40.0928563
        self.olon = -88.2359994

        self.lon = None
        self.lat = None
        
        ###############################################################################
        # ROS Communication Setup
        ###############################################################################
        
        # Subscribe to camera feed
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.img_callback, queue_size=1)
        self.sub_rgb_img = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
        self.sub_depth_img = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb_img, self.sub_depth_img], queue_size=2, slop=0.1)
        self.ts.registerCallback(self.img_callback)

        # Subscribe to GNSS data
        # self.gnss_sub = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback) @TODO: Uncomment
        
        # Publishers for visualization and control
        self.pub_bounding_box = rospy.Publisher("pedestrian_detection/bounding_box", Float32MultiArray, queue_size=1)
        self.pub_rgb_pedestrian_image = rospy.Publisher("pedestrian_detection/rgb/pedestrian_image", Image, queue_size=1)
        self.pub_depth = rospy.Publisher("pedestrian_detection/avg_depth", Float32MultiArray, queue_size=1)
        self.pub_pedestrian_gnss = rospy.Publisher("pedestrian_detector/gnss", Float64MultiArray)


        # ###############################################################################
        # # Controller Initialization
        # ###############################################################################
        self.pub_speed_command = rospy.Publisher("/pacmod/as_rx/accel_cmd", Float64, queue_size=1)
        self.pub_brake_command = rospy.Publisher("/pacmod/as_rx/brake_cmd", Float64, queue_size=1)
        # self.pedestrian_proximity_threshold = 5.0  # meters
        # self.slowing_threshold = 10.0  # meters
        # self.normal_speed = 0.3  # normal throttle value
        # self.is_slowing_for_pedestrian = False
        # self.min_stop_duration = 3.0  # seconds
        # self.stop_timer = None

        self.last_save_time = rospy.Time.now()

    ###############################################################################
    # Pedestrian GNSS Localization
    ###############################################################################

    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y   

    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)
    
    def local_xy_to_wps(self, local_x, local_y):
        lat, lon = axy.xy2ll(local_x, local_y, self.olat, self.olon)
        return lat, lon

    def process_pedestrian_gnss(self, x_ped_cam, y_ped_cam):
        if not self.lon or not self.lat:
            print("No gem gnss info yet")
            return
        
        x_gem, y_gem, yaw_gem = self.get_gem_state()

        x_ped = x_ped_cam * np.cos(yaw_gem) - y_ped_cam * np.sin(yaw_gem)
        y_ped = x_ped_cam * np.sin(yaw_gem) + y_ped_cam * np.sin(yaw_gem)

        x_ped += x_gem
        y_ped += y_gem

        lat_ped, lon_ped = self.local_xy_to_wps(x_ped, y_ped)

        pedestrian_gnss_msg = Float64MultiArray()
        pedestrian_gnss_msg.data = [lat_ped, lon_ped]

        self.pub_pedestrian_gnss.publish(pedestrian_gnss_msg)

    ###############################################################################
    # Vehicle Control Functions
    ###############################################################################

    def control_vehicle_for_pedestrian(self, pedestrian_distance):
        """
        If the pedestrian is within 5 m, apply a full brake. Otherwise, do nothing.
        
        Args:
            pedestrian_distance: Mean distance to pedestrian in meters
        """
        # Only act when we have a valid measurement
        if pedestrian_distance is None or not np.isfinite(pedestrian_distance):
            return

        # Hard brake if within 5 m
        if pedestrian_distance <= 5.0:
            rospy.loginfo(f"Pedestrian within 5m ({pedestrian_distance:.2f} m) – applying hard brake")
            self.pub_speed_command.publish(Float64(0.0))
            self.pub_brake_command.publish(Float64(1.0))  # Max braking
        # else: do nothing, other controllers remain in charge


    ###############################################################################
    # Pedestrian Detection and Depth Perception Helper Functions
    ###############################################################################

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
    
    def get_pedestrian_box(self, model, frame):
        """
        Gets pose points from a given YOLO model and image frame.
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

        mean_depth, med_depth, std_depth = None, None, None

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
            box_coords_msg.data = [x1, y1, x2, y2]
            self.pub_bounding_box.publish(box_coords_msg)

            # Get bounding box pixel points
            chest_y = (3*y1 + y2) // 4
            chest_delta = 10
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            # Extract region of interest from the depth image
            depth_roi = depth_img[y1:y2, x1:x2]
            valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]

            # Plot depth values to show distribution
            # self.plot_depths_distribution(valid_depths.flatten(), rgb_img)

            if valid_depths.size > 0:

                # Calculate distribution parameter estimates
                mean_depth = np.mean(valid_depths)
                std_depth = np.std(valid_depths)

                # Keep values in the first quartile
                flattened_valid_depths = valid_depths.flatten()
                first_quartile_val = np.quantile(flattened_valid_depths, 0.25)
                filtered_depths = flattened_valid_depths[flattened_valid_depths <= first_quartile_val]

                if filtered_depths.size > 0:
                    mean_depth = np.mean(filtered_depths)
                    med_depth = np.median(filtered_depths)
                    std_depth = np.std(filtered_depths)

                    # Get x position of pedestrian
                    cx = rgb_img.shape[1] / 2
                    x_cam = (u - cx) * mean_depth / self.Focal_Length
                    z_cam = math.sqrt(mean_depth**2 - x_cam**2)
                    
                    # Control vehicle speed based on mean distance to pedestrian
                    if mean_depth is not None:
                        self.control_vehicle_for_pedestrian(mean_depth) 
                    
                    # self.process_pedestrian_gnss(x_cam, z_cam) @TODO: Uncomment 
                    # print(f"Pedestrian at x: {x_cam:.2f}m, mean depth: {mean_depth:.2f}m")

                else:
                    mean_depth = med_depth = std_depth = None
            else:
                mean_depth = med_depth = std_depth = None

            # Add rectangle to rgb image
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_img, f"Mean dist: {mean_depth:.2f}m | Med dist: {med_depth:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(rgb_img, (u, chest_y), 4, (0,0,255), -1)
        
        return rgb_img, mean_depth, med_depth, std_depth

    ###############################################################################
    # Debugging Helper Functions
    ###############################################################################    
    
    def plot_depths_distribution(self, depth_vals, rgb_img):
        now = rospy.Time.now()

        if (now - self.last_save_time).to_sec() >= 1.0:
            dt = datetime.datetime.fromtimestamp(now.to_sec())
            timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")

            fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

            # Plot depth on the left
            axs[0].hist(depth_vals, bins=40)
            axs[0].set_title("Depth Histogram")

            # Show rgb_img on the right
            axs[1].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Camera Image")
            axs[1].axis('off')  # Hide axis for image

            fig.savefig(f"{timestamp}_combined.png")
            plt.close(fig)

            self.last_save_time = now
        

    ###############################################################################
    # Pedestrian Perception Callback
    ###############################################################################

    def img_callback(self, rgb_img, depth_img):
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")

            resized_img, ratio, pad = self.letterbox(rgb_img)  # unpack ratio and padding

            box_coords, conf = self.get_pedestrian_box(self.pedestrian_model, resized_img) # Get bounding box of detected pedestrian

            rgb_img, avg_depth, med_depth, std_depth = self.process_pedestrian_box(box_coords, conf, pad, ratio, rgb_img, depth_img)

            # Publish RGB image
            ros_rgb_img = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
            self.pub_rgb_pedestrian_image.publish(ros_rgb_img)

            # Publish Depth Data: (publishes [none, none, none] if no pedestrian is detected)
            depth_data = Float32MultiArray()
            depth_data.data = [avg_depth, med_depth, std_depth]
            self.pub_depth.publish(depth_data)

        except CvBridgeError as e:
            print(e)


###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        # Create detector instance
        detector = PedestrianDetector()
        
        # Keep node running until shutdown
        rate = rospy.spin() # rospy.Rate(10)  # 10 Hz control loop
    except rospy.ROSInterruptException:
        pass

