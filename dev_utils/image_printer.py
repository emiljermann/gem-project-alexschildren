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
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, Bool, Float32, Float64, Float32MultiArray

# GEM PACMod Headers
from geometry_msgs.msg import PoseStamped


class Printer:
    def __init__(self):

        # Set up compute device (GPU if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Commented out stop sign detection model code
        self.pedestrian_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        self.pedestrian_model.to(self.device).eval()

        # Initialize ROS node
        rospy.init_node('image_printer_node', anonymous=True)

        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images

        # Rostopic Subscriptions
        self.sub_rgb_pedestrian_image = rospy.Subscriber("pedestrian_detection/rgb/pedestrian_image", Image, self.printer, queue_size=1)
        self.sub_pedestrian_state = rospy.Subscriber("pedestrian_detection/state", String, self.state_callback)
        self.sub_pickup_time = rospy.Subscriber('pedestrian_detector/pickup_time', Float32, self.pickup_time_callback)

       
        self.pedestrian_state = "UNINITIALIZED"
        self.estimated_pickup_time = float('inf')

        self.frame_count = 0
        self.save_images = True  # Toggle for saving
        self.image_dir = "saved_images"
        os.makedirs(self.image_dir, exist_ok=True)

        # Optional: Store frames for video export
        self.video_frames = []
        self.store_video = True
    
    def state_callback(self, msg):
        self.pedestrian_state = msg.data
    
    def pickup_time_callback(self, msg):
        self.estimated_pickup_time = msg.data

    def printer(self, ros_img):
        try:
            img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)
            thickness = 2

            state_text = f"State: {self.pedestrian_state}"
            time_text = f"Pickup Time: {self.estimated_pickup_time:.2f}s"

            cv2.putText(img, state_text, (10, 30), font, font_scale, color, thickness)
            cv2.putText(img, time_text, (10, 60), font, font_scale, color, thickness)

            # # Display the image
            # cv2.imshow("Pedestrian Detection", img)
            # cv2.waitKey(1)

            # Save image if enabled
            if self.save_images:
                img_path = os.path.join(self.image_dir, f"frame_{self.frame_count:05d}.jpg")
                cv2.imwrite(img_path, img)
            
            # Store for video
            if self.store_video:
                self.video_frames.append(img.copy())

            self.frame_count += 1

        except CvBridgeError as e:
            print(e)

    def save_video(self, filename="output_video.avi", fps=10):
        if not self.video_frames:
            print("No frames to save.")
            return

        height, width, _ = self.video_frames[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        for frame in self.video_frames:
            out.write(frame)

        out.release()
        print(f"Saved video to {filename}")


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
