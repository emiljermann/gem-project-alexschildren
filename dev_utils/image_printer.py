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
from datetime import datetime

# ROS Headers
import rospy
from nav_msgs.msg import Path

# GEM Sensor Headers
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped

SAVE_IMAGES   = False          # write individual JPGs
SAVE_VIDEO    = True           # write an .mp4
OUT_DIR       = "../e2/src/vehicle_drivers/gem_gnss_control/scripts/image_printer_videos"  # can be relative (.. takes you up a level)

class Printer:
    def __init__(self):

        # Set up compute device (GPU if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize ROS node
        rospy.init_node('image_printer_node', anonymous=True)

        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images

        # timestamped folder (e.g., 20250510_203855)
        ts = datetime.now().strftime("%m%d_%H%M")

        # resolve OUT_DIR relative to this .py file, then add timestamp
        self.out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), OUT_DIR, ts)
        )
        os.makedirs(self.out_dir, exist_ok=True)

        # file templates
        self.jpg_tmpl   = os.path.join(self.out_dir, "frame_%05d.jpg")
        self.video_path = os.path.join(self.out_dir, f"image_printer_vid_{ts}.mp4")

        # toggles from the global settings
        self.save_images = SAVE_IMAGES
        self.store_video = SAVE_VIDEO

        self.vw = None         # cv2.VideoWriter set lazily

        # Rostopic Subscriptions
        self.sub_rgb_pedestrian_image = rospy.Subscriber("pedestrian_detection/rgb/pedestrian_image", Image, self.printer, queue_size=1)
        self.sub_state = rospy.Subscriber("/state_manager_node/state", String, self.state_callback)
       
        self.state = "NULL_STATE"

        self.frame_count = 0
    
    def state_callback(self, msg):
        self.state = msg.data
    
    def pickup_time_callback(self, msg):
        self.estimated_pickup_time = msg.data

    def printer(self, ros_img):
        try:
            img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0) 
            thickness = 2

            state_text = f"State: {self.state}"

            cv2.putText(img, state_text, (10, 30), font, font_scale, color, thickness)
            
            # save JPG
            if self.save_images:
                cv2.imwrite(self.jpg_tmpl % self.frame_count, img)

            # save MP4 (open writer on first frame)
            if self.store_video:
                if self.vw is None:
                    h, w, _ = img.shape
                    self.vw = cv2.VideoWriter(
                        self.video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        10,           # fps
                        (w, h)
                    )
                self.vw.write(img)

            self.frame_count += 1

        except CvBridgeError as e:
            print(e)

    def save_video(self):
        if self.vw is not None:
            self.vw.release()
            print(f"Saved video to {self.video_path}")


###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        printer = Printer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        printer.save_video()  # Save video on shutdown
        cv2.destroyAllWindows()
