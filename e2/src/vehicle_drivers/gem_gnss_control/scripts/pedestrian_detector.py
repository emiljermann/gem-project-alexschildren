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
import mediapipe as mp

from filters import OnlineFilter

import alvinxy as axy # Import AlvinXY transformation module

# ROS Headers
import rospy
from nav_msgs.msg import Path
import message_filters

# GEM Sensor Headers
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray, Float64MultiArray

from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


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


        # State control
        self.pub_transition = rospy.Publisher("state_manager_node/transition", String, queue_size = 1)
        self.state_sub = rospy.Subscriber("/state_manager_node/state", String, self.set_state)
        self.state = ""
        
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

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.last_pose_time = 0  # Throttle timer for pose estimation
        self.pose_throttle_interval = 0.0  # Minimum time (s) between pose checks

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

        # Subscribe to GNSS data (All my homies hate Inspva)
        self.gnss_sub   = message_filters.Subscriber("/septentrio_gnss/navsatfix", NavSatFix)
        self.ins_sub    = message_filters.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod)

        # Gets pacmod enable 
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.gs = message_filters.ApproximateTimeSynchronizer([self.gnss_sub, self.ins_sub], queue_size=10, slop=0.1)
        self.gs.registerCallback(self.gnss_callback)
        
        
        
        # Publishers for visualization and control
        self.pub_bounding_box = rospy.Publisher("pedestrian_detection/bounding_box", Float32MultiArray, queue_size=1)
        self.pub_rgb_pedestrian_image = rospy.Publisher("pedestrian_detection/rgb/pedestrian_image", Image, queue_size=1)
        self.pub_depth = rospy.Publisher("pedestrian_detection/avg_depth", Float32MultiArray, queue_size=1)
        self.pub_pedestrian_gnss = rospy.Publisher("pedestrian_detector/gnss", Float64MultiArray)

        self.last_save_time = rospy.Time.now()

    def set_state(self, msg):
        self.state = msg.data
        
    ###############################################################################
    # Pedestrian GNSS Localization
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

    def gnss_callback(self, gnss_msg, ins_msg):
        self.lat     = gnss_msg.latitude  # latitude
        self.lon     = gnss_msg.longitude # longitude
        self.heading = ins_msg.heading   # heading in degrees

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
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
    
    def is_hand_raised(self, cropped_img, return_debug_img=False):
        now = time.time()
        if now - self.last_pose_time < self.pose_throttle_interval:
            rospy.loginfo("Skipping pose estimation due to throttle")
            return False, None

        debug_img = cropped_img.copy() if return_debug_img else None
        cropped_height, cropped_width = cropped_img.shape[:2]
        rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return False, debug_img

        landmarks = results.pose_landmarks.landmark

        # Get key landmarks
        left_wrist = landmarks[15]
        left_shoulder = landmarks[11]
        left_hip = landmarks[23]
        right_wrist = landmarks[16]
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]
        nose = landmarks[0]

        # Visibility check
        left_valid = left_wrist.visibility > 0.5 and left_shoulder.visibility > 0.5
        right_valid = right_wrist.visibility > 0.5 and right_shoulder.visibility > 0.5

        # Compute hand raise flags
        left_hand_raised = left_valid and left_wrist.y > nose.y and left_wrist.y < (nose.y + abs(nose.y - left_hip.y) / 2.0) and (left_wrist.x < 0.25 or left_wrist.x > 0.75)
        right_hand_raised = right_valid and right_wrist.y > nose.y and right_wrist.y < (nose.y + abs(nose.y - right_hip.y) / 2.0) and (right_wrist.x < 0.25 or right_wrist.x > 0.75)

        # === Debug overlay if requested ===
        if debug_img is not None:
            for lm in landmarks:
                cx = int(lm.x * cropped_width)
                cy = int(lm.y * cropped_height)
                cv2.circle(debug_img, (cx, cy), 2, (0, 255, 255), -1)

            # Draw valid regions (blue rectangles)
            y_min = int(nose.y * cropped_height)
            y_max_left = int((nose.y + abs(nose.y - left_hip.y) / 2.0) * cropped_height)
            y_max_right = int((nose.y + abs(nose.y - right_hip.y) / 2.0) * cropped_height)
            x_left_min, x_left_max = 0, int(0.25 * cropped_width)
            x_right_min, x_right_max = int(0.75 * cropped_width), cropped_width

            cv2.rectangle(debug_img, (x_left_min, y_min), (x_left_max, y_max_left), (255, 0, 0), 1)
            cv2.rectangle(debug_img, (x_right_min, y_min), (x_right_max, y_max_right), (255, 0, 0), 1)
            cv2.putText(debug_img, "POSE DEBUG", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return left_hand_raised or right_hand_raised, debug_img
    
    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def process_pedestrian_gnss(self, x_ped_cam, y_ped_cam):
        if not self.lon or not self.lat:
            print("No gem gnss info yet")
            return
        
        x_gem, y_gem, yaw_gem = self.get_gem_state()

        x_ped = x_ped_cam * np.cos(yaw_gem) - y_ped_cam * np.sin(yaw_gem)
        y_ped = x_ped_cam * np.sin(yaw_gem) + y_ped_cam * np.cos(yaw_gem)

        x_ped += x_gem
        y_ped += y_gem

        lat_ped, lon_ped = self.local_xy_to_wps(x_ped, y_ped)

        pedestrian_gnss_msg = Float64MultiArray()
        pedestrian_gnss_msg.data = [lat_ped, lon_ped]
        # @TODO: do we want to use lat lon synchronizer here to do this or just publish local coords and let 
        self.pub_pedestrian_gnss.publish(pedestrian_gnss_msg)

    ###############################################################################
    # Vehicle Control Functions
    ###############################################################################

    def transition_vehicle_for_pedestrian(self, pedestrian_distance):
        """
        If the pedestrian is within 5 m, apply a full brake. Otherwise, do nothing.
        
        Args:
            pedestrian_distance: Mean distance to pedestrian in meters
        """
        # Only act when we have a valid measurement
        if pedestrian_distance is None or not np.isfinite(pedestrian_distance):
            return

        # Request stop if pedestrian detected within 5m 
        if pedestrian_distance <= 5.0:
            self.pub_transition.publish(String(data = "PICKING_UP"))
    

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
    
    def get_pedestrian_boxes(self, model, frame):
        with torch.no_grad():
            result = model(frame).pandas().xyxy[0]

        boxes = []
        for obj in result.itertuples():
            if obj.name == "person":
                boxes.append(((obj.xmin, obj.ymin, obj.xmax, obj.ymax), obj.confidence))
        return boxes

    def process_all_pedestrian_boxes(self, boxes, pad, ratio, rgb_img, depth_img):
        best = None  # (box, mean_depth, hand_raised, x_cam, z_cam)

        for box_coords, conf in boxes:
            if conf < self.Conf_Threshold:
                continue

            # Unpack and unletterbox
            x1, y1, x2, y2 = box_coords
            x1 = int((x1 - pad[0]) / ratio[0])
            y1 = int((y1 - pad[1]) / ratio[1])
            x2 = int((x2 - pad[0]) / ratio[0])
            y2 = int((y2 - pad[1]) / ratio[1])

            cropped = rgb_img[y1:y2, x1:x2]
            hand_raised, pose_debug_img = self.is_hand_raised(cropped, return_debug_img=False)

            # Depth filtering
            depth_roi = depth_img[y1:y2, x1:x2]
            valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
            mean, med, std = None, None, None
            x_cam, z_cam = None, None

            if valid_depths.size > 0:
                q1 = np.quantile(valid_depths, 0.25)
                filtered = valid_depths[valid_depths <= q1]
                if filtered.size > 0:
                    mean = np.mean(filtered)
                    med = np.median(filtered)
                    std = np.std(filtered)

                    u = (x1 + x2) // 2
                    cx = rgb_img.shape[1] / 2
                    x_cam = (u - cx) * mean / self.Focal_Length
                    z_cam = math.sqrt(mean**2 - x_cam**2)

            # Color logic
            color = (0, 255, 0) if hand_raised and (best is None or (mean and mean < best[1])) else \
                    (255, 0, 0) if hand_raised else (0, 0, 255)

            if mean and hand_raised and (best is None or mean < best[1]):
                best = ((x1, y1, x2, y2), mean, hand_raised, x_cam, z_cam)

            label = "HAND RAISED" if hand_raised else "Person"
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rgb_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return rgb_img, best

    ###############################################################################
    # Pedestrian Perception Callback
    ###############################################################################

    def img_callback(self, rgb_img_msg, depth_img_msg):
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_img_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "32FC1")

            resized_img, ratio, pad = self.letterbox(rgb_img)
            boxes = self.get_pedestrian_boxes(self.pedestrian_model, resized_img)

            rgb_img, best_pedestrian = self.process_all_pedestrian_boxes(boxes, pad, ratio, rgb_img, depth_img)

            if best_pedestrian:
                (x1, y1, x2, y2), mean, hand_raised, x_cam, z_cam = best_pedestrian

                # Publish data for the selected pedestrian
                self.pub_bounding_box.publish(Float32MultiArray(data=[x1, y1, x2, y2]))
                self.transition_vehicle_for_pedestrian(mean)
                self.process_pedestrian_gnss(x_cam, z_cam)
                self.pub_depth.publish(Float32MultiArray(data=[mean, mean, 0.0]))  # std not tracked here
            else:
                self.pub_depth.publish(Float32MultiArray(data=[None, None, None]))

            ros_rgb = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
            self.pub_rgb_pedestrian_image.publish(ros_rgb)

        except CvBridgeError as e:
            rospy.logerr(e)


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

