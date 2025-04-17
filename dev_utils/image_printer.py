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
        # self.sub_pedestrian_bounding_box = rospy.Subscriber("pedestrian_detection/bounding_box", Float32MultiArray, queue_size=1)
        # self.sub_rgb_pedestrian_image = rospy.Subscriber("pedestrian_detection/rgb/pedestrian_image", Image, self.printer, queue_size=1)
        self.sub_rgb_img = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.yolo_callback, queue_size = 1)

        self.time_stamp = list()
    
    def printer(self, ros_img):
        try:
            # Show the image
            img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")

            # Show the image
            cv2.imshow("Pedestrian Detection", img)
            cv2.waitKey(1)  # 1 ms wait so it updates the window without blocking

        except CvBridgeError as e:
            print(e)

    def get_pedestrian_box(self, model, frame):
        """
        Gets pose points from a given YOLO model and image frame.
        Returns:
            bounding_box: bounding box of person with highest confidence, None otherwise
                          in format (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            confidence: confidence score that said pedestrian exists
        """
        with torch.no_grad():
            t = time.time()
            result = model(frame).pandas().xyxy[0]
            self.time_stamp.append(time.time() - t)
            if len(self.time_stamp) == 1000:
                print(f"Average Model Time: {np.mean(self.time_stamp)}")
                # Average Model Time: 0.0124795982837677 [YoloV5S]
                # Average Model Time: 0.011895380735397339 [YoloV5n]


        if len(result) == 0:
            return None, 0
        
        all_box_coords = []
        all_confs = []

        box_coords = None
        highest_conf = 0
        for obj in result.itertuples():
            if obj.name == "person":
                box = (obj.xmin, obj.ymin, obj.xmax, obj.ymax)
                confidence = obj.confidence
                if confidence > 0.7:
                    all_box_coords.append(box)
                    all_confs.append(confidence)

        return all_box_coords, all_confs 
    
    def yolo_callback(self, ros_img):
        try:
            img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
            all_box_coords, all_confs = self.get_pedestrian_box(self.pedestrian_model, img)

            # No img was found
            if all_box_coords is None:
                return
            for i, box in enumerate(all_box_coords):
                confidence = all_confs[i]

                x1, y1, x2, y2 = box
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Image with Boxes", img)
            cv2.waitKey(1)
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
