#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# --- Configuration ---
CAMERA_IMAGE_TOPIC = "/wrist_mounted_camera/image"
WINDOW_NAME = "tuner"
# ---------------------

class HSVTunerNode(Node):
    
    def __init__(self):
        super().__init__("hsv_tuner")
        self.bridge = CvBridge()
        self.gui_ready = False
        self.latest_frame = None
        self.is_running = True

        # 1. Subscriber to the Kinova Camera Topic
        self.sub = self.create_subscription(
            Image, 
            CAMERA_IMAGE_TOPIC, 
            self.image_callback, 
            1
        )

        # 2. ROS 2 Timer for Continuous GUI/Key Input (Critical for cv2 responsiveness)
        self.timer = self.create_timer(0.03, self.process_and_display)

        self.get_logger().info("HSV tuner node initialized.")
        self.get_logger().info(f"Subscribing to {CAMERA_IMAGE_TOPIC}")

    def _setup_gui_once(self):
        """Initializes OpenCV window and trackbars."""
        if self.gui_ready:
            return
            
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 500)

        # show a tiny dummy frame once so Qt creates the window
        dummy = np.zeros((10, 10, 3), np.uint8)
        cv2.imshow(WINDOW_NAME, dummy)
        cv2.waitKey(1)

        # Create trackbars
        for name, val, mx in [
            ("H_low", 0, 180), ("S_low", 0, 255), ("V_low", 0, 255),
            ("H_high", 180, 180), ("S_high", 255, 255), ("V_high", 255, 255)
        ]:
            cv2.createTrackbar(name, WINDOW_NAME, val, mx, lambda x: None)

        cv2.createTrackbar("MinArea", WINDOW_NAME, 800, 20000, lambda x: None)
        self.gui_ready = True
        self.get_logger().info("GUI ready. Use sliders to adjust thresholds.")

    def _get_tb(self, name, default):
        """Safely read a trackbar position, handling potential errors if the window isn't ready."""
        if not self.gui_ready:
            return default
        try:
            return cv2.getTrackbarPos(name, WINDOW_NAME)
        except cv2.error:
            return default

    def image_callback(self, msg: Image):
        """Stores the latest image frame from the camera topic."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to CV2: {e}")
            self.latest_frame = None

    def process_and_display(self):
        """
        ROS 2 Timer Callback: Handles all GUI interaction and image processing.
        This runs independently of the image stream frequency.
        """
        # Ensure GUI is created
        self._setup_gui_once()
        
        if self.latest_frame is None or not self.gui_ready:
            # We still need to call waitKey even if we have no frame, to process GUI events
            cv2.waitKey(1) 
            return

        # Make a copy for processing
        frame = self.latest_frame.copy()

        # Read sliders (using safe method)
        hL = self._get_tb("H_low", 0)
        sL = self._get_tb("S_low", 0)
        vL = self._get_tb("V_low", 0)
        hH = self._get_tb("H_high", 180)
        sH = self._get_tb("S_high", 255)
        vH = self._get_tb("V_high", 255)
        min_area = self._get_tb("MinArea", 800)

        # --- Image Processing ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([hL, sL, vL])
        upper_bound = np.array([hH, sH, vH])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()
        
        idx = 1
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"obj {idx}", (x, y - 8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            idx += 1

        # --- Display ---
        stacked = np.hstack((vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow(WINDOW_NAME, stacked)

        # --- Handle Key Press ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.get_logger().info(
                f"lower: [{hL},{sL},{vL}]  upper: [{hH},{sH},{vH}]  min_area: {min_area}"
            )
        elif key in (27, ord('q')):  # ESC or q
            self.get_logger().info("User quit. Shutting down...")
            self.is_running = False
            

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