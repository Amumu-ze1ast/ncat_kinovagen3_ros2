#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor 
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from image_geometry import PinholeCameraModel
import cv2
import numpy as np
import time
import sys

# --- Configuration (UPDATED TOPICS) ---
# Topics based on your ROS 2 environment: /wrist_mounted_camera
RGB_TOPIC_DEFAULT = "/wrist_mounted_camera/image"
DEPTH_TOPIC_DEFAULT = "/wrist_mounted_camera/depth_image"
CAMINFO_TOPIC_DEFAULT = "/wrist_mounted_camera/camera_info"
DEBUG_IMAGE_TOPIC = "/step5_hsv_depth_xyz/debug_image"

# Default HSV ranges (List of strings, currently hardcoded)
DEFAULT_RANGES_STRING = [
    # 1. Dark/Gray objects
    "0, 0, 0, 180, 255, 0", 
    
    # 2. Example: Blue objects
    # "100,150,0,140,255,255", 
    
    # 3. Example: Yellow objects
    # "20,100,100,30,255,255", 
]
# ----------------------------------------

class HSVWithDepthXYZ(Node):
    
    def __init__(self):
        super().__init__("step4_object_xyz")
        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.have_cam_info = False
        self.last_log_time = time.time()
        
        # 1. Declare and Get ROS 2 Parameters
        
        # Ranges (STRING_ARRAY)
        range_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.STRING_ARRAY.value)
        self.declare_parameter('ranges', DEFAULT_RANGES_STRING, descriptor=range_descriptor)
        
        # Min Area (INT) - FIX APPLIED: Using ParameterDescriptor
        min_area_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.INTEGER.value)
        self.declare_parameter('min_area', 800, descriptor=min_area_descriptor)
        
        # Depth Window (INT) - FIX APPLIED: Using ParameterDescriptor
        depth_win_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.INTEGER.value)
        self.declare_parameter('depth_window', 5, descriptor=depth_win_descriptor)
        
        # Topic Names (STRING) - FIX APPLIED: Using ParameterDescriptor
        string_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.STRING.value)
        self.declare_parameter('rgb_topic', RGB_TOPIC_DEFAULT, descriptor=string_descriptor)
        self.declare_parameter('depth_topic', DEPTH_TOPIC_DEFAULT, descriptor=string_descriptor)
        self.declare_parameter('camera_info', CAMINFO_TOPIC_DEFAULT, descriptor=string_descriptor)
        
        # Retrieve values
        raw_ranges_str = self.get_parameter('ranges').value
        self.min_area = self.get_parameter('min_area').value
        self.depth_win = self.get_parameter('depth_window').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.caminfo_topic = self.get_parameter('camera_info').value
        
        # Parse ranges from string array
        self.ranges = []
        for range_str in raw_ranges_str:
            try:
                parts = [int(p.strip()) for p in range_str.split(',')]
                if len(parts) == 6:
                    self.ranges.append(tuple(parts))
            except ValueError:
                self.get_logger().error(f"Invalid range format: {range_str}")
        
        if not self.ranges:
             self.get_logger().fatal("No valid HSV ranges defined. Shutting down.")
             sys.exit(1)

        # 2. Setup Subscribers, Synchronizer, and Publisher
        
        # Camera Info subscriber (no synchronization needed)
        self.create_subscription(
            CameraInfo, self.caminfo_topic, self.cb_cam_info, 1
        )

        # RGB and Depth subscribers for synchronization
        sub_rgb = message_filters.Subscriber(self, Image, self.rgb_topic)
        sub_depth = message_filters.Subscriber(self, Image, self.depth_topic)
        
        # ROS 2 ApproximateTimeSynchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=20, slop=0.2
        )
        self.ts.registerCallback(self.cb_rgbd)

        # Publisher
        self.pub_dbg = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 1)

        self.get_logger().info(
            f"HSV+Depth+XYZ:\n  RGB:  {self.rgb_topic}\n  Depth: {self.depth_topic}\n  Info:  {self.caminfo_topic}\n  Ranges: {self.ranges}"
        )

    def cb_cam_info(self, msg):
        """Loads camera intrinsic parameters."""
        if not self.have_cam_info:
            self.cam_model.fromCameraInfo(msg)
            self.have_cam_info = True
            self.get_logger().info(
                "Camera intrinsics loaded: fx=%.3f fy=%.3f cx=%.3f cy=%.3f" % (
                    self.cam_model.fx(), self.cam_model.fy(), 
                    self.cam_model.cx(), self.cam_model.cy()
                )
            )

    def _depth_to_meters(self, depth_msg):
        """Converts ROS Depth Image message to CV2 numpy array in meters."""
        try:
            if depth_msg.encoding == "32FC1":
                d = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1").astype(np.float32)
            elif depth_msg.encoding == "16UC1":
                d16 = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                d = d16.astype(np.float32) / 1000.0
            else:
                d = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding).astype(np.float32)
                if np.nanmax(d) > 20.0:
                    d = d / 1000.0
            
            d[d <= 0.0] = np.nan
            return d
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error during depth conversion: {e}")
            return None

    def _throttled_log(self, log_str, throttle_rate=0.5):
        """Custom simple throttled logging (replaces rospy.loginfo_throttle)."""
        if time.time() - self.last_log_time > throttle_rate:
            self.get_logger().info(log_str)
            self.last_log_time = time.time()


    def cb_rgbd(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth messages."""
        try:
            img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError:
            return
            
        depth = self._depth_to_meters(depth_msg)
        if depth is None:
            return

        # 1. HSV Masking (OR all ranges)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hL,sL,vL,hH,sH,vH in self.ranges:
            mask = cv2.inRange(hsv, (hL,sL,vL), (hH,sH,vH))
            mask_total = cv2.bitwise_or(mask_total, mask)

        # 2. Mask Clean-up
        mask_total = cv2.medianBlur(mask_total, 5)
        mask_total = cv2.morphologyEx(
            mask_total, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        )

        # 3. Contour Detection and 3D Calculation
        vis = img.copy()
        cnts, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        win = self.depth_win if self.depth_win % 2 == 1 else self.depth_win + 1
        r = win // 2 # radius
        H, W = mask_total.shape[:2]

        for i, c in enumerate(cnts, 1):
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            
            x,y,w,h = cv2.boundingRect(c)
            cx_pix, cy_pix = x + w//2, y + h//2

            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(vis, (cx_pix,cy_pix), 4, (0,0,255), -1)

            x0 = max(0, cx_pix - r); x1 = min(W, cx_pix + r + 1)
            y0 = max(0, cy_pix - r); y1 = min(H, cy_pix + r + 1)
            z = np.nanmedian(depth[y0:y1, x0:x1]) if (x1>x0 and y1>y0) else np.nan

            label = f"obj {i}"
            if np.isfinite(z) and self.have_cam_info:
                X = (cx_pix - self.cam_model.cx()) * z / self.cam_model.fx()
                Y = (cy_pix - self.cam_model.cy()) * z / self.cam_model.fy() 
                
                label += f" X={X:.3f} Y={Y:.3f} Z={z:.3f} m"
                
                # self._throttled_log(f"obj {i} @ camera_optical: X={X:.3f} Y={Y:.3f} Z={z:.3f} m")
                self._throttled_log(f"obj {i} @ camera_optical: {X:.3f}, {Y:.3f}, {z:.3f} m")
            
            elif np.isfinite(z):
                label += f" Z={z:.3f} m (no cam_info)"
            else:
                label += " Z=nan"

            cv2.putText(vis, label, (x, max(0,y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 4. Depth Colormap Visualization
        depth_vis = depth.copy()
        zmin, zmax = np.nanpercentile(depth_vis, 2), np.nanpercentile(depth_vis, 98)
        zmin = max(0.1, float(zmin)); zmax = max(zmin+1e-3, float(zmax))
        
        depth_norm = (np.nan_to_num(depth_vis, nan=0.0) - zmin) / (zmax - zmin)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        depth_color = cv2.applyColorMap((depth_norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        
        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(depth_color, (x,y), (x+w,y+h), (0,255,0), 2)

        # 5. Publish Debug Image (Stacked)
        mask_bgr = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((vis, depth_color, mask_bgr))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(stacked, "bgr8"))
        except CvBridgeError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = HSVWithDepthXYZ()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()