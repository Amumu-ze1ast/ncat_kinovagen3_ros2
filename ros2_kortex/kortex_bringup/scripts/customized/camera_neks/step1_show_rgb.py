#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# --- Configuration ---
# Your Kinova camera topics as identified from 'ros2 topic list | grep camera'
CAMERA_IMAGE_TOPIC = "/wrist_mounted_camera/image"
DEBUG_IMAGE_TOPIC = "/step1_show_rgb/debug_image"
# ---------------------

class ShowRGB(Node):
    
    def __init__(self):
        # Initialize the ROS 2 node
        super().__init__("step1_show_rgb")
        
        # Initialize the CV Bridge
        self.bridge = CvBridge()
        
        # 1. Create Publisher (for the processed image)
        self.pub = self.create_publisher(
            Image, 
            DEBUG_IMAGE_TOPIC, 
            1 # queue_size becomes 'qos_profile=1'
        )
        
        # 2. Create Subscriber (to the raw camera image)
        self.sub = self.create_subscription(
            Image, 
            CAMERA_IMAGE_TOPIC, 
            self.image_callback, # ROS 2 callback name convention
            1
        )
        self.get_logger().info(f"Subscribing to {CAMERA_IMAGE_TOPIC} and publishing to {DEBUG_IMAGE_TOPIC}")

        # Note: The subscriber prevents garbage collection by existing
        self.sub
        
    def image_callback(self, msg: Image):
        """Processes the incoming ROS Image message."""
        
        try:
            # Convert ROS Image -> OpenCV (BGR)
            # We use "bgr8" which is standard for color images in OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to CV2: {e}")
            return
            
        # --- Image Processing ---
        
        # (Optional) draw a tiny green crosshair in the center
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w//2, h//2), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        
        # Display the image in a local OpenCV window (optional, but helpful for debugging)
        cv2.imshow("ROS 2 Camera Viewer", frame)
        cv2.waitKey(1) # Necessary to refresh the window

        # --- End Image Processing ---
        
        # Publish the processed image back to a ROS topic
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert CV2 to ROS Image: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    node = ShowRGB()
    
    try:
        # Keep the node alive to process incoming messages
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    # Clean up and shut down
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()