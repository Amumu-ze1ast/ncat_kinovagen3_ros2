#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# --- Configuration ---
CAMERA_IMAGE_TOPIC = "/wrist_mounted_camera/image"
DEBUG_IMAGE_TOPIC = "/step3_detect_part/debug_image"

# Default ranges defined as a single, FLAT list of doubles (Hlo, Slo, Vlo, Hhi, Shi, Vhi)
DEFAULT_RANGES_FLAT = [0.0, 0.0, 0.0, 180.0, 255.0, 0.0]
# ---------------------

class HSVRangesDetector(Node):
    
    def __init__(self):
        super().__init__("step3_hsv_ranges_detector")
        self.bridge = CvBridge()
        
        # 1. Declare and Get ROS 2 Parameters
        
        # FIX APPLIED: Access the integer value of the Parameter Type enum member using .value
        range_descriptor = ParameterDescriptor(
            type=rclpy.Parameter.Type.DOUBLE_ARRAY.value
        )
        
        # Pass the descriptor to declare_parameter using the 'descriptor' keyword
        self.declare_parameter('ranges', DEFAULT_RANGES_FLAT, 
                               descriptor=range_descriptor)
        
        self.declare_parameter('min_area', 800)
        self.declare_parameter('rgb_topic', CAMERA_IMAGE_TOPIC)
        
        # Retrieve the parameter values
        raw_ranges = self.get_parameter('ranges').value
        self.min_area = self.get_parameter('min_area').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        
        # 2. Reshape the flat list back into a list of 6-element tuples
        if len(raw_ranges) % 6 != 0:
            self.get_logger().error("Parameter 'ranges' must contain groups of 6 values (Hlo, Slo, Vlo, Hhi, Shi, Vhi).")
            self.ranges = []
        else:
            # Reshape the flat array into N rows of 6 columns
            self.ranges = [
                tuple(map(int, raw_ranges[i:i + 6])) 
                for i in range(0, len(raw_ranges), 6)
            ]

        # 3. Setup Subscriber and Publisher
        self.sub = self.create_subscription(
            Image, 
            self.rgb_topic, 
            self.cb, 
            1
        )
        self.pub = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 1)

        self.get_logger().info(f"HSVRangesDetector listening on {self.rgb_topic}")
        self.get_logger().info(f"HSV ranges: {self.ranges}")

    def cb(self, msg: Image):
        """Processes the image, applies all HSV ranges, and publishes the result."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Build mask = OR of all provided ranges
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for r in self.ranges:
            hL,sL,vL,hH,sH,vH = r
            
            lower_bound = np.array([hL, sL, vL])
            upper_bound = np.array([hH, sH, vH])
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask_total = cv2.bitwise_or(mask_total, mask)

        # Clean up mask using morphological operations
        mask_total = cv2.medianBlur(mask_total, 5)
        mask_total = cv2.morphologyEx(
            mask_total, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        )

        # Draw green box For multi objects (your existing logic)
        cnts, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        kept = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
                
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cx, cy = x+w//2, y+h//2
            cv2.circle(img, (cx,cy), 4, (0,0,255), -1)
            
            kept += 1
            cv2.putText(img, f"obj {kept}", (x, y-8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Stack the original image and the mask (in BGR format)
        stacked = np.hstack((img, cv2.cvtColor(mask_total, cv2.COLOR_GRAY2BGR)))
        
        # Publish the final stacked image
        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(stacked, "bgr8"))
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HSVRangesDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()