import os 
import cv2
import csv
import math
import time
import torch
import numpy as np
from numpy import linalg as la
import alvinxy.alvinxy as axy


# created using `rosbag echo /septentrio_gnss/insnavgeod -b <bag_file> > <output_file>`
gnss_input_filename = "dev_utils/coord_extraction/gnss_demo_path.txt" 
# created using `rosbag echo /septentrio_gnss/navsatfix -b <bag_file> > <output_file>`
ins_input_filename = "dev_utils/coord_extraction/ins_demo_path.txt" 

output_filename   = "e2/src/vehicles_drivers/gem_gnss_control/waypoints/xyhead_custom_pp.csv"

# given constants from pedestrian detector (used to create waypoints in a cartesian frame at origin olat,olon)
offset     = 0.46 # meters
olat       = 40.0928563
olon       = -88.2359994

    
def wps_to_local_xy(lon_wp, lat_wp):
    lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, olat, olon)
    return lon_wp_x, lat_wp_y   

def heading_to_yaw(heading_curr):
    if (heading_curr >= 270 and heading_curr < 360):
        yaw_curr = np.radians(450 - heading_curr)
    else:
        yaw_curr = np.radians(90 - heading_curr)
    return yaw_curr

def get_gem_state(lat, lon, heading):
    local_x_curr, local_y_curr = wps_to_local_xy(lon, lat)
    curr_yaw = heading_to_yaw(heading) 
    
    curr_x = local_x_curr - offset * np.cos(curr_yaw)
    curr_y = local_y_curr - offset * np.sin(curr_yaw)
    
    return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)
    
def get_waypoints(gnss_filename, ins_filename):
    gnss_points = []
    ins_points = []
    with open(gnss_filename, 'r') as gnss_file:
        lines = gnss_file.readlines()
        i = 0
        while i < len(lines):
            line1 = lines[i].strip()
            if line1.startswith("latitude"):
                line2 = lines[i+1].strip()
                if line2.startswith("longitude"):
                    gnss_points.append([float(line1[10:]),float(line2[11:])])
                    i += 1
            i += 1
    with open(ins_filename, 'r') as ins_file:
        lines = ins_file.readlines()
        i = 0
        while i < len(lines):
            line1 = lines[i].strip()
            if line1.startswith("heading"): 
                    ins_points.append(float(line1[7:]))
            i += 1
    
    gnss_points = np.array(gnss_points)
    ins_points = np.array(ins_points)
    waypoints = np.hstack((gnss_points, ins_points), axis=1)
    
    waypoints = np.array([get_gem_state(lat, lon, heading) for (lat, lon, heading) in waypoints])
    return waypoints
    
def write_to_file(filename, waypoints):
    with open(filename,"w") as f:
        for (cx,cy, yaw) in waypoints:
            f.write(f"{cx},{cy},{yaw}\n")
            
if __name__ == "__main__":
    
    waypoints = get_waypoints(gnss_input_filename, ins_input_filename)
    write_to_file(output_filename, waypoints)
        
        