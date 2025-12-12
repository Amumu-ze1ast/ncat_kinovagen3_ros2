#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys # For sys.exit for clean shutdown

# --- Configuration ---
CAMERA_IMAGE_TOPIC = "/wrist_mounted_camera/image"
# ---------------------

class HSVTunerNode(Node):
    
    def __init__(self):
        super().__init__("hsv_tuner")
        self.get_logger().info("HSV tuner node starting...")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.is_running = True

        # 1. Subscriber to the Kinova Camera Topic
        self.sub = self.create_subscription(
            Image, 
            CAMERA_IMAGE_TOPIC, 
            self.image_callback, 
            1
        )
        self.get_logger().info(f"Subscribing to {CAMERA_IMAGE_TOPIC}")

        # 2. OpenCV Window and Trackbars Setup
        cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tuner", 1200, 500)

        for name, val, mx in [
            ("H_low", 0, 180), ("S_low", 0, 255), ("V_low", 0, 255),
            ("H_high", 180, 180), ("S_high", 255, 255), ("V_high", 255, 255)
        ]:
            cv2.createTrackbar(name, "tuner", val, mx, lambda x: None)

        cv2.createTrackbar("MinArea", "tuner", 800, 20000, lambda x: None)
        
        # 3. ROS 2 Timer for Continuous OpenCV Display/Input (Must run outside the callback)
        # This timer calls the 'process_and_display' function every 30ms (approx 33 FPS)
        self.timer = self.create_timer(0.03, self.process_and_display)

        self.get_logger().info("HSV tuner running. Use sliders to adjust thresholds.")
        self.get_logger().info("Press 's' to print thresholds. Press 'q' or ESC to quit.")

    def image_callback(self, msg):
        """Stores the latest image frame from the camera topic."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to CV2: {e}")
            self.latest_frame = None

    def process_and_display(self):
        """Reads sliders, processes the image, and handles key presses."""
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()

        # --- read sliders ---
        hL = cv2.getTrackbarPos("H_low", "tuner")
        sL = cv2.getTrackbarPos("S_low", "tuner")
        vL = cv2.getTrackbarPos("V_low", "tuner")
        hH = cv2.getTrackbarPos("H_high", "tuner")
        sH = cv2.getTrackbarPos("S_high", "tuner")
        vH = cv2.getTrackbarPos("V_high", "tuner")
        min_area = cv2.getTrackbarPos("MinArea", "tuner")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Ensure correct lower and upper bounds order
        lower_bound = np.array([hL, sL, vL])
        upper_bound = np.array([hH, sH, vH])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # --- Contour Detection and Visualization ---
        # Note: cv2.findContours signature changed in OpenCV 4.x
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()
        
        for idx, c in enumerate(cnts, start=1):
            if cv2.contourArea(c) < min_area: continue
            
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
            cx, cy = x+w//2, y+h//2
            cv2.circle(vis, (cx,cy), 4, (0,0,255), -1)
            
            cv2.putText(vis, f"obj {idx}", (x, y-8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # --- Display ---
        stacked = np.hstack((vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("tuner", stacked)

        # --- Handle Key Press ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.get_logger().info(
                f"lower: [{hL},{sL},{vL}]  upper: [{hH},{sH},{vH}]  min_area: {min_area}"
            )
        elif key in (27, ord('q')):  # ESC or q
            self.get_logger().info("User quit. Shutting down...")
            self.is_running = False # Flag to stop the main loop
            

def main(args=None):
    rclpy.init(args=args)
    
    node = HSVTunerNode()
    
    # Run the node's loop until the 'is_running' flag is set to False
    try:
        while rclpy.ok() and node.is_running:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
        
    # Clean up and shut down
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()