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

from filters import OnlineFilter
from pid_controllers import PID


# ROS Headers
import alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(10)

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters


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

        self.pedestrian_state = None
        self.sub_pedestrian_state = rospy.Subscriber("pedestrian_detector/state", String, self.state_callback)

        # read waypoints into the system 
        self.goal       = 0            
        self.read_waypoints() 

        self.desired_speed = 1.5  # m/s, reference speed
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)
        # @TODO: this should work? though per ros docs get_param is for static params
        self.stop_wp_index = rospy.get_param("~stop_waypoint_index", 120)  
        self.stop_dist     = rospy.get_param("~stop_distance_thresh", 1.0)
        self.stopped       = False
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
        self.goal_plot, = self.ax.plot([], [], 'ro', label='Goal Point')
        self.heading_arrows = self.ax.quiver([0,0], [0,0], [0,0], [0,0], angles='xy', scale_units='xy', scale=1, color='g')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Pure Pursuit Live Map")
        self.ax.legend()
    
    def update_plot(self, curr_x, curr_y, curr_h, goal_x, goal_y, goal_h):
        arrow_len = 2
        self.path_plot.set_data(self.path_points_x, self.path_points_y)
        self.curr_pos_plot.set_data([curr_x], [curr_y])
        self.goal_plot.set_data([goal_x], [goal_y])
        self.heading_arrows.set_UVC([arrow_len*np.cos(curr_h), arrow_len*np.cos(goal_h)], [arrow_len*np.sin(curr_h), arrow_len*np.sin(goal_h)])
        self.heading_arrows.set_offsets(np.array([[curr_x, curr_y], [goal_x, goal_y]]))
        self.ax.set_xlim(curr_x - 10, curr_x + 10)
        self.ax.set_ylim(curr_y - 10, curr_y + 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
    
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def state_callback(self, msg):
        self.pedestrian_state = msg.data

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


    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/test2waypoints.csv')
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
    
    def estimate_drive_time_to_stop_point(self):
        if self.closest_wp_index < 0 or self.stop_wp_index < 0:
            return float('inf')  # invalid indices

        if self.closest_wp_index >= self.stop_wp_index:
            return 0.0  # already past or at the stop point

        total_distance = 0.0
        for i in range(self.closest_wp_index, self.stop_wp_index):
            x1 = self.path_points_x[i]
            y1 = self.path_points_y[i]
            x2 = self.path_points_x[i+1]
            y2 = self.path_points_y[i+1]
            total_distance += self.dist((x1, y1), (x2, y2))

        if self.speed <= 0.1:
            return total_distance / 0.1  # avoid divide by zero / unrealistic speed
        else:
            return total_distance / self.speed  # estimated time in seconds

    def start_pp(self):
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                if (self.pacmod_enable == True):
                        # ---------- enable PACMod ----------

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
            
            # State tracking, ignores command if state is PICKING_UP
            if (self.pedestrian_state == "SEARCHING" and self.gem_enable):
                # ---------- dont ignore PACMod ----------
                # dont ignore brake
                self.brake_cmd.ignore  = True
                # dont ignore gas 
                self.accel_cmd.ignore  = True

                self.brake_pub.publish(self.brake_cmd)
                print("PP Brake Not Ignored!")
                self.accel_pub.publish(self.accel_cmd)
                print("PP Gas Ignored!")
            elif (self.pedestrian_state == "PICKING_UP"):
                # ---------- ignore PACMod ----------
                # ignore brake
                self.brake_cmd.ignore  = True
                # ignore gas 
                self.accel_cmd.ignore  = True

                self.brake_pub.publish(self.brake_cmd)
                print("PP Brake Ignored!")
                self.accel_pub.publish(self.accel_cmd)
                print("PP Gas Ignored!")

            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            # index of the closest point to current position
            self.closest_wp_index = int(np.argmin(self.dist_arr))
            if self.stop_wp_index > 0:
                estimated_time_to_stop_point = self.estimate_drive_time_to_stop_point()
                rospy.loginfo(f"Estimated time to stop for pedestrian: {estimated_time_to_stop_point}")
                self.pub_pickup_time.publish(Float32(estimated_time_to_stop_point))

            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3) )[0]

            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1,v2)
                # find correct look-ahead point by using heading information
                diff = abs(curr_yaw-self.path_points_heading[idx])
                if min(diff, np.pi*2 - diff) < np.pi/2:
                    # make sure the look-ahead point is in front of the vehicle
                    dx = self.path_points_x[idx] - curr_x
                    dy = self.path_points_y[idx] - curr_y
                    alpha_vector = math.atan2(dy, dx) - curr_yaw
                    alpha_vector = math.atan2(math.sin(alpha_vector), math.cos(alpha_vector))
                    if abs(alpha_vector) < np.pi/2:
                        print(f"Difference: {curr_yaw-self.path_points_heading[idx]}")
                        print(f"Alpha_vector: {alpha_vector}")
                        self.goal = idx

            # === DEBUG BLOCK START ===
            rospy.loginfo("----- GNSS / INS Debug -----")
            rospy.loginfo(f"Raw GNSS lat: {self.lat}, lon: {self.lon}")
            rospy.loginfo(f"INS heading (degrees): {self.heading}")
            rospy.loginfo(f"Converted yaw (radians): {curr_yaw}")
            rospy.loginfo(f"Local vehicle position x: {curr_x}, y: {curr_y}")
            rospy.loginfo(f"Goal index: {self.goal}")
            if 0 <= self.goal < self.wp_size:
                wp_x = self.path_points_x[self.goal]
                wp_y = self.path_points_y[self.goal]
                wp_heading = self.path_points_heading[self.goal]
                rospy.loginfo(f"Waypoint at index: ({wp_x}, {wp_y}), heading: {wp_heading}")
                rospy.loginfo(f"Distance to waypoint: {self.dist((wp_x, wp_y), (curr_x, curr_y))} m")
            else:
                rospy.logwarn("Closest waypoint index out of bounds!")
            # === DEBUG BLOCK END ===

            # # finding the distance between the goal point and the vehicle
            # # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]
            
            # if (not self.stopped and self.stop_wp_index >= 0 and self.goal >= self.stop_wp_index and L <= self.stop_dist):                 
            #     rospy.loginfo("Reached stop waypoint %d  (d=%.2fm) - braking", self.stop_wp_index, L)
            #     self._apply_brakes()
            #     self.stopped = True

            # if self.stopped:
            #     self.rate.sleep()
            #     continue

            

            
            # find the curvature and the angle 
            # w is the amount we prioritize the distance to goal heading over matching goal x, y
            w = 0.2
            
            alpha_heading = self.path_points_heading[self.goal] - curr_yaw
            
            dx = self.path_points_x[self.goal] - curr_x
            dy = self.path_points_y[self.goal] - curr_y
            alpha_vector = math.atan2(dy, dx) - curr_yaw
            alpha_vector = math.atan2(math.sin(alpha_vector), math.cos(alpha_vector))

            alpha = alpha_heading * w + alpha_vector * (1-w)
            # ----------------- tuning this part as needed -----------------
            k       = 0.41 
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i #*2
            # ----------------- tuning this part as needed -----------------

            f_delta = round(np.clip(angle, -0.61, 0.61), 3)

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
                print("\n")

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

            goal_x = self.path_points_x[self.goal]
            goal_y = self.path_points_y[self.goal]
            goal_heading = self.path_points_heading[self.goal]
            self.update_plot(curr_x, curr_y, curr_yaw, goal_x, goal_y, goal_heading)

            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

