import os 
import numpy as np
import alvinxy as axy
import sys
"""
--------------------------------
This script extracts waypoints from a rosbag file containing GNSS and INS data.

To run the script::
python3 dev_utils/coord_extraction <rosbag_file>

To record bag::
rosbag record -O <bag_file> /septentrio_gnss/navsatfix /septentrio_gnss/insnavgeod
---------------------------------
"""

# Change this! ->
output_filename   = "e2/src/vehicle_drivers/gem_gnss_control/waypoints/xyhead_new_track.csv"

# Typical paths: changing these will not do anything unless you want to break the script apart and run it manually
# created using `rostopic echo /septentrio_gnss/navsatfix -b <bag_file> > <output_file>`
gnss_input_filename = "dev_utils/coord_extraction/temp_gnss_path.txt" 
# created using `rostopic echo /septentrio_gnss/insnavgeod -b <bag_file> > <output_file>`
ins_input_filename = "dev_utils/coord_extraction/temp_ins_path.txt" 


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
    
def create_txt_file(rosbag_file, gnss_filename, ins_filename):
    # Create a temporary file to store the GNSS and INS data
    status = os.system(f"rostopic echo /septentrio_gnss/navsatfix -b {rosbag_file} > {gnss_filename}")
    if status:
        print("Error creating GNSS file")
        return 1
    
    status = os.system(f"rostopic echo /septentrio_gnss/insnavgeod -b {rosbag_file} > {ins_filename}")
    if status:
        print("Error creating INS file")
        return 1
    return 0
    
def delete_txt_file(gnss_filename, ins_filename):
    # Delete the temporary files
    os.remove(gnss_filename)
    os.remove(ins_filename)
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
            if line1.startswith("heading:"): 
                    ins_points.append(float(line1[8:]))
            i += 1
    
    gnss_points = np.array(gnss_points)
    ins_points = np.array(ins_points)[:gnss_points.shape[0]]
    print(gnss_points.shape, ins_points.shape)
    waypoints = np.hstack((gnss_points, ins_points.reshape(-1,1)))
    
    waypoints = np.array([get_gem_state(lat, lon, heading) for (lat, lon, heading) in waypoints])
    #20 appears to match sampling rate of given files, can be changed
    return waypoints[::7]
    
def write_to_file(filename, waypoints):
    with open(filename,"w") as f:
        for (cx,cy, yaw) in waypoints:
            f.write(f"{cx},{cy},{yaw}\n")
            
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dev_utils/coord_extraction <rosbag_file> <optional! 1 to keep temp files>")
        return
    rosbag_file = sys.argv[1]
    keep_temp_files = False
    if len(sys.argv) > 2:
        keep_temp_files = int(sys.argv[2])
    print("Creating temp rosbag output files...")
    if(create_txt_file(rosbag_file, gnss_input_filename, ins_input_filename)):
        return
    print("Extracting waypoints...")
    waypoints = get_waypoints(gnss_input_filename, ins_input_filename)
    print("writing waypoints to file")
    write_to_file(output_filename, waypoints)
    if not keep_temp_files:
        print("Cleaning up temp files...")
        delete_txt_file(gnss_input_filename, ins_input_filename)
    print(f"Wrote {waypoints.shape[0]} waypoints to {output_filename}")
        
        