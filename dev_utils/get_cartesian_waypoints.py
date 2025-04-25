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

# ROS Headers
import rospy
from nav_msgs.msg import Path

# GEM Sensor Headers
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped

import alvinxy.alvinxy as axy # Import AlvinXY transformation module

class Writer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('get_cartesian_waypoints', anonymous=True)

        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images

        # Rostopic Subscriptions
        # self.sub_pedestrian_bounding_box = rospy.Subscriber("pedestrian_detection/bounding_box", Float32MultiArray, queue_size=1)
        # self.sub_rgb_pedestrian_image = rospy.Subscriber("pedestrian_detection/rgb/pedestrian_image", Image, self.printer, queue_size=1)
        self.gnss_sub  = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.time_stamp = list()
        
        self.offset     = 0.46 # meters
        self.olat       = 40.0928563
        self.olon       = -88.2359994
        
        self.filename   = "xyhead_custom_pp.csv"
    
    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y   
    
    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
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

    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees
        
        cx, cy, yaw = self.get_gem_state() # convert to local cartesian coordinates
        
        print("Cartesian Position: ", cx, cy, yaw)
        
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../e2/src/vehicles_drivers/gem_gnss_control/waypoints', self.filename)
        
        with open(filename,"a") as f:
            f.write(f"{cx},{cy},{yaw}\n")
        exit(0) # exit after first message
        
        
        
        


###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        # Create detector instance
        writer = Writer()
        
        # Keep node running until shutdown
        rate = rospy.spin() #r ospy.Rate(10)  # 10 Hz control loop
    except rospy.ROSInterruptException:
        pass
