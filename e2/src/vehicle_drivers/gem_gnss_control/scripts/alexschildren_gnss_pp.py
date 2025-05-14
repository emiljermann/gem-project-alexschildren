#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 03/14/2025                                                
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import subprocess

from filters import OnlineFilter
from pid_controllers import PID
from joystick_reader import RawJoystickReader

# ROS Headers
import alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64, Float64MultiArray
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

OUT_DIR = "map_videos"

class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(10)

        self.look_ahead = 7
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters


        # state management
        self.state = ""
        self.sub_state = rospy.Subscriber("/state_manager_node/state", String, self.set_state)
        self.pub_transition = rospy.Publisher("state_manager_node/transition", String, queue_size=1)
        
        
        # we replaced novatel hardware with septentrio hardware on e2
        self.gnss_sub   = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub    = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.olat       = 40.0928563
        self.olon       = -88.2359994
        
        self.sub_pedestrian_gnss = rospy.Subscriber("pedestrian_detector/gnss", Float64MultiArray, self.pedestrian_gnss_callback)
        self.pedestrian_lat = None
        self.pedestrian_lon = None

        
        # read waypoints into the system 
        self.goal       = 0  
        self.poly = None
        self.polyfit_yaw = None          
        self.read_waypoints() 

        self.desired_speed = 1.0  # m/s, reference speed
        self.max_accel     = 0.45 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)
        self.controller = RawJoystickReader()
        self.prev_axes = None
        self.prev_buttons = None
        # @TODO: this should work? though per ros docs get_param is for static params
        self.stop_wp_index = rospy.get_param("~stop_waypoint_index", None)  
        self.stop_dist     = rospy.get_param("~stop_distance_thresh", 1.0)
        self.stopped       = False
        self._last_logged_dist_to_stop = None
        self.closest_wp_index = None
        self._log_dist_threshold = 0.5  # meters
        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second

        # Pedestrian pickup time subscriber
        self.pub_pickup_time = rospy.Publisher('pedestrian_detector/pickup_time', Float32, queue_size = 1)

        # OUR LOCATION V.S. GOAL POINT ON PURE PURSUIT POINT MAP
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.path_plot, = self.ax.plot([], [], 'k--', label='Path')
        self.curr_pos_plot, = self.ax.plot([], [], 'bo', label='Current Position')
        self.goal_plot, = self.ax.plot([], [], 'ro', label='Goal Wayoint')
        self.dest_plot, = self.ax.plot([], [], 'o', label='Destination Point', color='purple')
        self.pedestrian_plot, = self.ax.plot([], [], 'go', label='Pedestrian Point')
        # self.polyfit_plot, = self.ax.plot([], [], 'r-', label='Polyfit Path')
        self.heading_arrows = self.ax.quiver([0,0], [0,0], [0,0], [0,0], angles='xy', scale_units='xy', scale=1, color='g')
        self.polyfit_arrow = self.ax.quiver([0], [0], [0], [0], angles='xy', scale_units='xy', scale=1, color='r')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Pure Pursuit Live Map")
        self.ax.legend()
        self.time_stamps = list()
        self.start_time = time.time()
        self.frame_count = 0
        self.save_vid = True

        if self.save_vid:
            os.makedirs("frames", exist_ok=True)
    
    def update_plot(self, curr_x, curr_y, curr_h, goal_x, goal_y, goal_h):

        arrow_len = 2
        self.path_plot.set_data(self.path_points_x, self.path_points_y)
        self.curr_pos_plot.set_data([curr_x], [curr_y])
        self.goal_plot.set_data([goal_x], [goal_y])

        if self.stop_wp_index is not None:
            stop_x = self.path_points_x[self.stop_wp_index]
            stop_y = self.path_points_y[self.stop_wp_index]
            self.dest_plot.set_data([stop_x], [stop_y])
            
        self.heading_arrows.set_UVC([arrow_len*np.cos(curr_h), arrow_len*np.cos(goal_h)], 
                                     [arrow_len*np.sin(curr_h), arrow_len*np.sin(goal_h)])
        self.heading_arrows.set_offsets(np.array([[curr_x, curr_y], [goal_x, goal_y]]))
        if self.pedestrian_lat and self.pedestrian_lon:
            local_x, local_y = self.wps_to_local_xy(self.pedestrian_lat, self.pedestrian_lon)
            self.pedestrian_plot.set_data([local_x], [local_y])

        # if self.poly is not None:
        #     x_polyfit = np.linspace(curr_x - 5, curr_x + 5, 100)
        #     y_polyfit = self.poly(x_polyfit)
        #     self.polyfit_plot.set_data(x_polyfit, y_polyfit)
        #     self.polyfit_arrow.set_UVC([arrow_len*np.cos(self.polyfit_yaw)], 
        #                              [arrow_len*np.sin(self.polyfit_yaw)])
        #     self.polyfit_arrow.set_offsets(np.array([[curr_x, curr_y]]))

        self.ax.set_xlim(curr_x - 10, curr_x + 10)
        self.ax.set_ylim(curr_y - 10, curr_y + 10)
        self.fig.canvas.draw()

        if self.save_vid:
            t = time.time() - self.start_time
            fname = f"frames/frame_{self.frame_count:04d}.png"
            self.fig.savefig(fname)
            self.time_stamps.append(t)

        self.fig.canvas.flush_events()
        self.frame_count += 1
        

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)

    def pedestrian_gnss_callback(self, msg):
        self.pedestrian_lat = round(msg.data[0], 6)
        self.pedestrian_lon = round(msg.data[1], 6)
    
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)
        
        
    def set_state(self, msg):
        self.state = msg.data

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle

    def _apply_brakes(self):
        self.accel_cmd.enable = False
        self.accel_cmd.f64_cmd = 0.0
        self.brake_cmd.enable = True
        self.brake_cmd.f64_cmd = 0.5
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.turn_cmd.ui16_cmd = 1
        self.turn_pub.publish(self.turn_cmd)
        
    def get_destination_input(self):
        fig, ax = plt.subplots()
        path_plot, = ax.plot([], [], 'k--', label='Path')
        curr_pos_plot, = ax.plot([], [], 'bo', label='Current Position')
        path_plot.set_data(self.path_points_x, self.path_points_y)
        curr_x, curr_y, curr_yaw = self.get_gem_state()
        curr_pos_plot.set_data([curr_x], [curr_y])
        ax.set_xlim(np.min(self.path_points_x) - 10, np.max(self.path_points_x) + 10)
        ax.set_ylim(np.min(self.path_points_y) - 10, np.max(self.path_points_y) + 10)
        plt.show(block=False)
        clicks = plt.ginput(n=1)[0]
        
        distances = (self.path_points_x - clicks[0])**2 + (self.path_points_y - clicks[1])**2
        closest_index = np.argmin(distances)
        plt.close(fig)
        return closest_index
        
        
        
    def resume_motion(self):
        # Clear brake command
        self.brake_cmd.enable = False
        self.brake_cmd.f64_cmd = 0.0
        self.brake_pub.publish(self.brake_cmd)

        # Re-enable acceleration (set f64_cmd elsewhere based on control)
        self.accel_cmd.enable = True
        self.accel_cmd.ignore = False
        self.accel_cmd.clear = False
        self.accel_pub.publish(self.accel_cmd)

        # Neutral turn signal (unless you're setting this elsewhere)
        self.turn_cmd.ui16_cmd = 1
        self.turn_pub.publish(self.turn_cmd)

        # self.stopped = False  # optional: reset if you want to re-enter motion state


    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/final_waypoints.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        self.path_points_heading = [float(point[2]) for point in path_points] # heading
        self.wp_size             = len(self.path_points_lon_x)
        self.dist_arr            = np.zeros(self.wp_size)

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

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)
    
    def estimate_drive_time_to_stop_point_arc(self):
        if self.closest_wp_index < 0 or self.stop_wp_index < 0 or self.stop_wp_index >= self.wp_size or self.closest_wp_index >= self.wp_size:
            return float('inf')  # invalid indices
        
        if self.closest_wp_index == self.stop_wp_index:
            return 0.0

        total_distance = 0.0
        i = self.closest_wp_index
        while i != self.stop_wp_index:
            x1 = self.path_points_x[i]
            y1 = self.path_points_y[i]
            i_next = (i + 1) % self.wp_size  # wrap around
            x2 = self.path_points_x[i_next]
            y2 = self.path_points_y[i_next]
            total_distance += self.dist((x1, y1), (x2, y2))
            i = i_next
            if i == self.closest_wp_index:
                # Full loop completed, stop_wp_index must not be reachable
                return float('inf')

        # Avoid divide by zero
        effective_speed = max(self.speed, 0.1)
        return total_distance / effective_speed
    
    def estimate_drive_time_to_stop_point_birds_eye(self):
        if self.stop_wp_index < 0 or self.stop_wp_index >= self.wp_size:
            return float('inf')  # invalid index

        curr_x, curr_y, _ = self.get_gem_state()

        stop_x = self.path_points_x[self.stop_wp_index]
        stop_y = self.path_points_y[self.stop_wp_index]

        dist_to_stop = self.dist((curr_x, curr_y), (stop_x, stop_y))

        effective_speed = max(self.speed, 0.1)

        return dist_to_stop / effective_speed
    
    def handle_pickup(self):
        self._apply_brakes()

        axes, buttons = self.controller.get_state()
        if axes != self.prev_axes or buttons != self.prev_buttons:
            rospy.loginfo(f"Joystick Axes: {axes}")
            rospy.loginfo(f"Joystick Buttons: {buttons}")
        self.prev_axes = axes
        self.prev_buttons = buttons

        # Dropoff selection

        print("Select a dropoff location:")

    
        self.stop_wp_index = self.get_destination_input() # returns waypoint idx of where we want to go

        self.closest_wp_index = int(np.argmin(self.dist_arr))
        estimated_time_to_stop_point = self.estimate_drive_time_to_stop_point_arc()
        # estimated_time_to_stop_point = self.estimate_drive_time_to_stop_point_birds_eye()
        rospy.loginfo(f"Estimated time to dropoff pedestrian: {estimated_time_to_stop_point}")
        self.pub_pickup_time.publish(Float32(estimated_time_to_stop_point))
        
                

    def handle_dropoff(self):
        self._apply_brakes()
        # would be cool if it could conclude pedestrian has left the vehicle autonomously but we'll just wait 20 seconds for now
        for _ in range(10):
            if rospy.is_shutdown():
                return
            rospy.sleep(1)
        self.stop_wp_index = None
        self._last_logged_dist_to_stop = None
        self.closest_wp_index = None
        
    def pp_iter(self):
        curr_x, curr_y, curr_yaw = self.get_gem_state()
        #Polyfit to the next few points and the original
        num_points = 20
        jump = 1
        degree = 2
        point_idx = np.arange(self.goal, self.goal+num_points*jump, jump)%self.wp_size
        points_x = self.path_points_x[point_idx]
        points_y = self.path_points_y[point_idx]
        points_x = np.insert(points_x, 0, curr_x)
        points_y = np.insert(points_y, 0, curr_y)
        coeffs = np.polyfit(points_x, points_y, degree)
        poly = np.poly1d(coeffs)
        px = (curr_x+self.path_points_x[self.goal])/2.0
        self.polyfit_yaw = math.atan(poly.deriv()(px))
        
        v1 = [math.cos(curr_yaw), math.sin(curr_yaw)]
        v2 = [math.cos(self.polyfit_yaw), math.sin(self.polyfit_yaw)]
        if np.dot(v1, v2) < 0:
            self.polyfit_yaw += np.pi
        self.poly = poly
        
        L = self.dist_arr[self.goal]
        L = max(L, 0.1)
        
        #@TODO: may need to be tuned to work with orientation
        # self.polyfit_yaw = self.heading_to_yaw(np.degrees(self.polyfit_yaw))
        # alpha = self.polyfit_yaw - curr_yaw
        dx = self.path_points_x[self.goal] - curr_x
        dy = self.path_points_y[self.goal] - curr_y
        alpha = math.atan2(dy, dx) - curr_yaw
        
        # ----------------- tuning this part as needed -----------------
        k = 1.0
        angle_i = (k * 2.0 * math.sin(alpha)) / L
        # ----------------- tuning this part as needed -----------------
        f_delta = math.atan(self.wheelbase * angle_i)
        f_delta = round(np.clip(f_delta, -0.61, 0.61), 3)

        f_delta_deg = np.degrees(f_delta)

        # steering_angle in degrees
        steering_angle = self.front2steer(f_delta_deg)
        
        if(self.gem_enable == True):
            print("Current index: " + str(self.goal))
            print("Forward velocity: " + str(self.speed))
            ct_error = round(np.sin(alpha) * L, 3)
            print("Crosstrack Error: " + str(ct_error))
            print("Front steering angle: " + str(np.degrees(f_delta)) + " degrees")
            print("Steering wheel angle: " + str(steering_angle) + " degrees" )
            print(f"Ignored: {self.brake_cmd.ignore}, {self.accel_cmd.ignore}")
            print("\n")
        
        # if abs(np.degrees(f_delta)) > 20:
        #     self.desired_speed = 1.1
        # else:
        #     self.desired_speed = 1.5

        current_time = rospy.get_time()
        filt_vel     = self.speed_filter.get_data(self.speed)
        output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

        if output_accel > self.max_accel:
            output_accel = self.max_accel

        if output_accel < 0.3:
            output_accel = 0.3

        if (f_delta_deg <= 30 and f_delta_deg >= -30):
            self.turn_cmd.ui16_cmd = 1
        elif(f_delta_deg > 30):
            self.turn_cmd.ui16_cmd = 2 # turn left
        else:
            self.turn_cmd.ui16_cmd = 0 # turn right

        self.accel_cmd.f64_cmd = output_accel
        self.steer_cmd.angular_position = np.radians(steering_angle)
        self.accel_pub.publish(self.accel_cmd)
        self.steer_pub.publish(self.steer_cmd)
        self.turn_pub.publish(self.turn_cmd)
        
        
    def start_pp(self):
        self.pub_transition.publish(String(data = "BOOT"))
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                if (self.pacmod_enable == True):
                        # ---------- enable PACMod ----------
                        print("RENABLED PACMod!")
                        # enable forward gear
                        self.gear_cmd.ui16_cmd = 3

                        # enable brake
                        self.brake_cmd.enable  = True
                        self.brake_cmd.clear   = False
                        self.brake_cmd.ignore  = False
                        self.brake_cmd.f64_cmd = 0.0

                        # enable gas 
                        self.accel_cmd.enable  = True
                        self.accel_cmd.clear   = False
                        self.accel_cmd.ignore  = False
                        self.accel_cmd.f64_cmd = 0.0

                        self.gear_pub.publish(self.gear_cmd)
                        print("PP Foward Engaged!")

                        self.turn_pub.publish(self.turn_cmd)
                        print("PP Turn Signal Ready!")
                        
                        self.brake_pub.publish(self.brake_cmd)
                        print("PP Brake Engaged!")

                        self.accel_pub.publish(self.accel_cmd)
                        print("PP Gas Engaged!")

                        self.gem_enable = True

            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3) )[0]

            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x, self.path_points_y[idx]-curr_y]
                v2 = [math.cos(curr_yaw), math.sin(curr_yaw)]
                if np.dot(v1, v2) > 0:
                    #@TODO: check if the goal point is in front of the vehicle
                    v3 = [math.cos(self.path_points_heading[idx]), math.sin(self.path_points_heading[idx])]
                    if np.dot(v2, v3) > 0:
                        self.goal = idx
                        # break
        
            if self.stop_wp_index is not None:
                stop_x = self.path_points_x[self.stop_wp_index]
                stop_y = self.path_points_y[self.stop_wp_index]
                dist_to_stop = self.dist((curr_x, curr_y), (stop_x, stop_y))
            # -----------------------------------STATE CODE BEGINS------------------------------------------------------------------
            if self.state=="PICKING_UP" and self.gem_enable:
                self.handle_pickup() # brakes, waits for user input, unbrakes
                # request state change to DROPPING_OFF (note since the pickup process is asynchronous we want to transition internally to avoid race condition execution repeats)
                self.resume_motion()
                # self.state = "DROPPING_OFF"
                self.pub_transition.publish(String(data="DROPPING_OFF"))
            elif self.state == "DROPPING_OFF" and self.gem_enable:
                self.pp_iter()
                stop_x = self.path_points_x[self.stop_wp_index]
                stop_y = self.path_points_y[self.stop_wp_index]
                dist_to_stop = self.dist((curr_x, curr_y), (stop_x, stop_y))

                if dist_to_stop <= self.stop_dist:
                    # transition to let ped out state if close enough to desired waypoint
                    self.pub_transition.publish(String(data="DROPOFF_END"))
            elif self.state == "DROPOFF_END" and self.gem_enable:
                if self.stop_wp_index is not None:
                    rospy.loginfo("Reached stop waypoint %d  (%.2f m away) - braking", self.stop_wp_index, dist_to_stop)
                self.handle_dropoff() # brakes, waits 20 seconds, unbrakes
                self.resume_motion()
                # self.state = "SEARCHING" #Similar deal here to DROPPING_OFF, anytime we wait need to run only once
                self.pub_transition.publish(String(data="SEARCHING"))  # Release override
            elif self.state == "SEARCHING" and self.gem_enable:
                self.pp_iter()
                if self.stop_wp_index is not None:
                    if (self._last_logged_dist_to_stop is None or
                        abs(dist_to_stop - self._last_logged_dist_to_stop) >= self._log_dist_threshold):
                        rospy.loginfo("Not at stop waypoint %d (still %.2f m away) - continuing", 
                                    self.stop_wp_index, dist_to_stop)
                        self._last_logged_dist_to_stop = dist_to_stop
            else:
                rospy.loginfo("Shoot, something broke!!")
            # -----------------------------------STATE CODE ENDS------------------------------------------------------------------
           
            goal_x = self.path_points_x[self.goal]
            goal_y = self.path_points_y[self.goal]
            goal_heading = self.path_points_heading[self.goal]
            self.update_plot(curr_x, curr_y, curr_yaw, goal_x, goal_y, goal_heading)

            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass
    finally:
        if pp.save_vid:
            ts = time.strftime("%m%d_%H%M")
            with open(os.path.join(OUT_DIR, f"timestamps_{ts}.txt"), "w") as f:
                for t in pp.time_stamps:
                    f.write(f"{t}\n")
            # Get delays (in 1/100s for GIF)
            delays = [int(100 * (pp.time_stamps[i+1] - pp.time_stamps[i])) for i in range(len(pp.time_stamps)-1)]
            delays.append(delays[-1])  # Repeat last delay

            from PIL import Image
            frames = [Image.open(f"frames/frame_{i:04d}.png") for i in range(len(pp.time_stamps))]
            gif_path = os.path.join(OUT_DIR, f"map_gif_{ts}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                        duration=delays, loop=0)

            # Estimate FPS from timestamps
            if len(pp.time_stamps) >= 2:
                avg_fps = 1.0 / np.mean(np.diff(pp.time_stamps))
            else:
                avg_fps = 10  # fallback

            video_path = os.path.join(OUT_DIR, f"map_video_{ts}.mp4")
            create_mp4_from_frames(folder="frames", output=video_path,
                       fps=round(avg_fps))
            delete_frames(folder = "frames")

def create_mp4_from_frames(folder="frames", output="map_video.mp4", fps=10):
    command = [
        "ffmpeg",
        "-y",  # overwrite
        "-framerate", str(fps),
        "-i", f"{folder}/frame_%04d.png",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output
    ]
    try:
        subprocess.run(command, check=True)
        print(f"MP4 video saved as {output}")
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:", e)
def delete_frames(folder="frames"):
    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))
    print(f"Deleted all frames in {folder}")

if __name__ == '__main__':
    pure_pursuit()