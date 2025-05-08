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
# State Management Node
# 
# This module implements a DFA state machine for managing pedestrian pickup.abs
# It provides a way to poll the current state and transition between states
# python3 -W ignore state_manager.py
###############################################################################
"""
States:
    1. SEARCHING
    2. PICKING_UP
    3. DROPPING_OFF
    4. DROPOFF_END
    
Transitions:
    1. SEARCHING -> PICKING_UP
    2. PICKING_UP -> DROPPING_OFF
    4. DROPPING_OFF -> DROPOFF_END
    5. DROPOFF_END -> SEARCHING
    
    @TODO: Add more transitions as needed
    
"""
class StateManager:
    def __init__(self):
        self.rate       = rospy.Rate(10)
        
        rospy.init_node('state_manager_node', anonymous=True)
        
        self.transition_sub = message_filters.Subscriber("/state_manager_node/transition", String, self.transition_callback)
        self.pub_state = rospy.Publisher("state_manager_node/state", String, queue_size=1)
        self.state = "SEARCHING"
        self.transition = ""
        
    def start_manager(self):
        self.publish_state()
           
    def publish_state(self):
        self.pub_state.publish(String(data = self.state))
    
    def transition_callback(self, msg):
        prev_state = self.state
        self.transition = msg.data
        rospy.loginfo("Received transition: %s", self.transition)
        self.run()
        self.transition = ""
        if self.state != prev_state:
            self.publish_state()
            rospy.loginfo("State changed from %s to %s", prev_state, self.state)
        
    def run(self):
        # Runs the state machine and updates current state
        if self.state == "SEARCHING":
            if self.transition == "PICKING_UP":
                self.state = "PICKING_UP"
        elif self.state == "PICKING_UP":
            if self.transition == "DROPPING_OFF":
                self.state = "DROPPING_OFF"
        elif self.state == "DROPPING_OFF":
            if self.transition == "DROPOFF_END":
                self.state = "DROPOFF_END"
        elif self.state == "DROPOFF_END":
            if self.transition == "SEARCHING":
                self.state = "SEARCHING"
        else:
            rospy.logwarn("Unknown transition, maintaining state: %s", self.state)
        
        

###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        state_manager = StateManager()
        print("State Manager Node Initialized, Starting...")
        state_manager.start_manager()
    except rospy.ROSInterruptException:
        pass

