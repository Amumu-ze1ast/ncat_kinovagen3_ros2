#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor 
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped 
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from image_geometry import PinholeCameraModel
import tf2_ros 
from tf2_ros import TransformException
from rclpy.duration import Duration
import cv2
import numpy as np
import time
import sys
# FIX 1: Import the necessary conversions for geometry_msgs to work with TF2
import tf2_geometry_msgs 

# --- Configuration ---
RGB_TOPIC_DEFAULT = "/wrist_mounted_camera/image"
DEPTH_TOPIC_DEFAULT = "/wrist_mounted_camera/depth_image"
CAMINFO_TOPIC_DEFAULT = "/wrist_mounted_camera/camera_info"
DEBUG_IMAGE_TOPIC = "/step5_hsv_depth_xyz_tf/debug_image"
BASE_FRAME_DEFAULT = "base_link" 

DEFAULT_RANGES_STRING = [
    # Dark/Gray objects
    "0, 0, 0, 180, 255, 0", 
]
# ---------------------

class HSVDepthTF(Node):
    
    def __init__(self):
        super().__init__("step5_hsv_depth_xyz_tf")
        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.have_cam_info = False
        
        # FIX 2: Dictionary to track last log time for custom throttled logging
        self.last_log_times = {} 
        
        # 1. Declare and Get ROS 2 Parameters
        string_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.STRING.value)
        int_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.INTEGER.value)
        range_descriptor = ParameterDescriptor(type=rclpy.Parameter.Type.STRING_ARRAY.value)
        
        self.declare_parameter('base_frame', BASE_FRAME_DEFAULT, descriptor=string_descriptor)
        self.declare_parameter('rgb_topic', RGB_TOPIC_DEFAULT, descriptor=string_descriptor)
        self.declare_parameter('depth_topic', DEPTH_TOPIC_DEFAULT, descriptor=string_descriptor)
        self.declare_parameter('camera_info', CAMINFO_TOPIC_DEFAULT, descriptor=string_descriptor)
        self.declare_parameter('ranges', DEFAULT_RANGES_STRING, descriptor=range_descriptor)
        self.declare_parameter('min_area', 800, descriptor=int_descriptor)
        self.declare_parameter('depth_window', 5, descriptor=int_descriptor)
        
        # Retrieve values
        self.base_frame = self.get_parameter('base_frame').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.caminfo_topic = self.get_parameter('camera_info').value
        self.min_area = self.get_parameter('min_area').value
        self.depth_win = self.get_parameter('depth_window').value
        raw_ranges_str = self.get_parameter('ranges').value
        
        # Parse ranges
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

        # 2. Setup TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 3. Setup Subscribers, Synchronizer, and Publisher
        self.create_subscription(CameraInfo, self.caminfo_topic, self.cb_cam_info, 1)

        sub_rgb = message_filters.Subscriber(self, Image, self.rgb_topic)
        sub_depth = message_filters.Subscriber(self, Image, self.depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=20, slop=0.2
        )
        self.ts.registerCallback(self.cb_rgbd)

        self.pub_dbg = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 1)
        self.pub_cam_pt = self.create_publisher(PointStamped, "~/point_camera", 10)
        self.pub_base_pt = self.create_publisher(PointStamped, "~/point_base", 10)

        self.get_logger().info(
            f"HSV+Depth+TF:\n  Base: {self.base_frame}\n  RGB: {self.rgb_topic}\n  Depth: {self.depth_topic}\n  Info: {self.caminfo_topic}"
        )

    def cb_cam_info(self, msg):
        """Loads camera intrinsic parameters."""
        if not self.have_cam_info:
            self.cam_model.fromCameraInfo(msg)
            self.have_cam_info = True
            self.get_logger().info("Camera intrinsics loaded.")

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

    def _throttled_warn(self, log_str, key, throttle_rate=1.0):
        """Custom throttled warning using internal timer logic."""
        current_time = time.time()
        last_time = self.last_log_times.get(key, 0.0)
        
        if current_time - last_time > throttle_rate:
            self.get_logger().warn(log_str)
            self.last_log_times[key] = current_time

    def _throttled_info(self, log_str, throttle_rate=0.5):
        """Custom simple throttled logging (used for successful log)."""
        current_time = time.time()
        # Use a generic key for info messages if they happen frequently
        last_time = self.last_log_times.get('INFO_LOG', 0.0) 
        if current_time - last_time > throttle_rate:
            self.get_logger().info(log_str)
            self.last_log_times['INFO_LOG'] = current_time


    def cb_rgbd(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth messages."""
        if not self.have_cam_info:
            self._throttled_warn("Waiting for CameraInfo...", 'CAM_INFO_WAIT', 1.0)
            return

        try:
            img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError:
            return
            
        depth = self._depth_to_meters(depth_msg)
        if depth is None:
            return

        # --- Object Detection ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hL,sL,vL,hH,sH,vH in self.ranges:
            mask = cv2.inRange(hsv, (hL,sL,vL), (hH,sH,vH))
            mask_total = cv2.bitwise_or(mask_total, mask)
        mask_total = cv2.medianBlur(mask_total, 5)
        mask_total = cv2.morphologyEx(
            mask_total, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        )

        vis = img.copy()
        cnts, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        win = self.depth_win if self.depth_win % 2 == 1 else self.depth_win + 1
        r = win // 2 
        H, W = mask_total.shape[:2]
        
        # Depth Colormap Visualization setup
        depth_vis = depth.copy()
        zmin, zmax = np.nanpercentile(depth_vis, 2), np.nanpercentile(depth_vis, 98)
        zmin = max(0.1, float(zmin)); zmax = max(zmin+1e-3, float(zmax))
        depth_norm = (np.nan_to_num(depth_vis, nan=0.0) - zmin) / (zmax - zmin)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        depth_color = cv2.applyColorMap((depth_norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        
        # --- Transformation Loop ---
        for i, c in enumerate(cnts, 1):
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            
            x,y,w,h = cv2.boundingRect(c)
            cx_pix, cy_pix = x + w//2, y + h//2

            # 3D Calculation
            x0 = max(0, cx_pix - r); x1 = min(W, cx_pix + r + 1)
            y0 = max(0, cy_pix - r); y1 = min(H, cy_pix + r + 1)
            z = np.nanmedian(depth[y0:y1, x0:x1]) if (x1>x0 and y1>y0) else np.nan
            
            # Draw on debug images
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(vis, (cx_pix,cy_pix), 4, (0,0,255), -1)
            cv2.rectangle(depth_color, (x,y), (x+w,y+h), (0,255,0), 2)

            label = f"obj {i}"
            if np.isfinite(z):
                # Convert (u,v,Z) to (X,Y,Z) in camera frame (using optical math)
                X = (cx_pix - self.cam_model.cx()) * z / self.cam_model.fx()
                Y = (cy_pix - self.cam_model.cy()) * z / self.cam_model.fy() 
                
                # Create PointStamped message 
                pt_cam = PointStamped()
                pt_cam.header.stamp = rgb_msg.header.stamp
                
                # FIX 3: Use "camera_link" as the source frame, as manually verified to work 
                # and to avoid the non-existent optical frame ID error.
                pt_cam.header.frame_id = "camera_link" 
                
                # Explicitly cast to float()
                pt_cam.point.x = float(X)
                pt_cam.point.y = float(Y)
                pt_cam.point.z = float(z)

                self.pub_cam_pt.publish(pt_cam)

                # Transform to base frame using TF2
                base_str = ""
                try:
                    # FIX 4: Correct use of Duration for timeout
                    timeout = Duration(seconds=0.1)
                    # FIX 5: Correct transform call (no 'time=' keyword)
                    pt_base = self.tf_buffer.transform(pt_cam, self.base_frame, timeout) 
                    
                    bx, by, bz = pt_base.point.x, pt_base.point.y, pt_base.point.z
                    base_str = f" ({self.base_frame}) X={bx:.3f} Y={by:.3f} Z={bz:.3f}"
                    self.pub_base_pt.publish(pt_base)
                    
                    self._throttled_info(
                        f"obj {i} cam:[{X:.3f} {Y:.3f} {z:.3f}] base:[{bx:.3f} {by:.3f} {bz:.3f}]", 0.5
                    )
                except TransformException as e:
                    self._throttled_warn(f"TF failed: {e}", 'TF_FAIL', 1.0)
                    
                # Update label
                label += f" X={X:.3f} Y={Y:.3f} Z={z:.3f} m"
                if base_str:
                    cv2.putText(vis, base_str, (x, min(H-5, y+h+15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            else:
                label += " Z=nan"

            cv2.putText(vis, label, (x, max(0,y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


        # 5. Publish Debug Image (Stacked)
        mask_bgr = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((vis, depth_color, mask_bgr))
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(stacked, "bgr8"))
        except CvBridgeError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = HSVDepthTF()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()