import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo


class ZED2DepthListener:
    def __init__(self):
        # ── Subscribers ────────────────────────────────────────────────
        self.depth_sub = rospy.Subscriber(
            "/zed2/zed_node/depth/depth_registered",
            Image,
            self.depth_callback,
            queue_size=2
        )

        # Grab intrinsics once (optional – comment out if you don’t need them)
        self.info_sub = rospy.Subscriber(
            "/zed2/zed_node/depth/camera_info",
            CameraInfo,
            self.info_callback,
            queue_size=1
        )

        # ── Helpers ────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.K = None                # 3 × 3 camera matrix

        rospy.loginfo("ZED2DepthListener initialised – waiting for depth frames…")

    # ──────────────────────────────────────────────────────────────────
    #  CameraInfo callback (runs once, then we unsubscribe)
    # ──────────────────────────────────────────────────────────────────
    def info_callback(self, msg: CameraInfo):
        self.K = np.array(msg.K).reshape(3, 3)
        rospy.loginfo(f"Camera intrinsics received:\n{self.K}")
        self.info_sub.unregister()   # no need to keep receiving

    # ──────────────────────────────────────────────────────────────────
    #  Depth Image callback
    # ──────────────────────────────────────────────────────────────────
    def depth_callback(self, msg: Image):
        """
        Convert the incoming depth image to a NumPy array.
        Pixel units:
            • 16UC1 → millimetres  → converted to metres (float32)
            • 32FC1 → already metres (float32)
        """
        try:
            # 'passthrough' preserves original encoding (16UC1 or 32FC1)
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge conversion failed: {e}")
            return

        depth_np = np.asarray(depth_cv)

        # Convert 16‑bit mm → float32 m
        if msg.encoding == "16UC1":
            depth_np = depth_np.astype(np.float32) * 0.001

        # Optional: mask out invalid zeros or NaNs before stats
        valid = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
        if valid.size == 0:
            rospy.logwarn_throttle(5, "All depth values are invalid in this frame.")
            return

        # Print a short summary once per second
        rospy.loginfo_throttle(
            1.0,
            f"Depth frame {depth_np.shape}: "
            f"min={valid.min():.3f} m, mean={valid.mean():.3f} m, "
            f"max={valid.max():.3f} m"
        )
        
        # ── Now `depth_np` is yours to use! ───────────────────────────
        # For example, pass it to OpenCV, SciPy, PyTorch, etc.
        # -------------------------------------------------------------


def main():
    rospy.init_node("zed2_depth_listener", anonymous=True)
    listener = ZED2DepthListener()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()