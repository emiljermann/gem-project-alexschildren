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
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped


class Printer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('image_printer_node', anonymous=True)

        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images

        # Rostopic Subscriptions
        # self.sub_pedestrian_bounding_box = rospy.Subscriber("pedestrian_detection/bounding_box", Float32MultiArray, queue_size=1)
        self.sub_rgb_pedestrian_image = rospy.Subscriber("pedestrian_detection/rgb/pedestrian_image", Image, self.printer, queue_size=1)
    
    def printer(self, ros_img):
        try:
            # Show the image
            img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")

            # Show the image
            cv2.imshow("Pedestrian Detection", img)
            cv2.waitKey(1)  # 1 ms wait so it updates the window without blocking

        except CvBridgeError as e:
            print(e)


###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        # Create detector instance
        printer = Printer()
        
        # Keep node running until shutdown
        rate = rospy.spin() #r ospy.Rate(10)  # 10 Hz control loop
    except rospy.ROSInterruptException:
        pass
