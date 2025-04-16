#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray
import math

class DistanceEvaluator:
    def __init__(self):
        rospy.Subscriber("/pedestrian_detection/avg_depth", Float32MultiArray, self.callback)
        self.errors = []
        self.timestamps = []
        self.last_time = None
        self.shutdown_triggered = False

    def callback(self, msg):
        if len(msg.data) < 5:
            return

        estimated = msg.data[4]
        actual = msg.data[3]
        now = rospy.Time.now().to_sec()

        # ✅ Detect rosbag loop once and stop
        if self.last_time is not None and now < self.last_time and not self.shutdown_triggered:
            rospy.logwarn("ROS time reset detected — plotting once and shutting down.")
            self.plot_error()
            self.shutdown_triggered = True
            rospy.signal_shutdown("Finished 1 bag loop")
            return

        # Record latest time
        self.last_time = now

        if not math.isnan(estimated) and not math.isnan(actual):
            error = estimated - actual
            percent_error = (error / actual) * 100 if actual != 0 else float('nan')
            self.errors.append(percent_error)
            # self.errors.append(error)
            self.timestamps.append(now)
            rospy.loginfo(f"[{now:.2f}] Estimated: {estimated:.2f} | Actual: {actual:.2f} | Percent Error: {percent_error:.2f}")
        else:
            rospy.loginfo(f"[{now:.2f}] Received NaN — skipping")

    def plot_error(self):
        if not self.errors:
            print("No valid data collected.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.timestamps, self.errors, marker='o')
        plt.xlabel("ROS Time (s)")
        plt.ylabel("Percent Depth Error (%)")
        plt.title("Pedestrian Depth Estimation Percent Error")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot instead of displaying it
        import time
        filename = f"depth_error_plot_{int(time.time())}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")

if __name__ == "__main__":
    rospy.init_node("distance_error_evaluator")
    evaluator = DistanceEvaluator()
    rospy.spin()
