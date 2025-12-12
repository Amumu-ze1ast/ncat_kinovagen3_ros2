#!/usr/bin/env python3
# Simple KDL FK test for ROS2 + Gazebo (no IK, only FK, joint-name safe)

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
import xml.etree.ElementTree as ET

# Define a minimal Node class just to access ROS services
class AxisExtractor(Node):
    def __init__(self):
        super().__init__("axis_extractor")
        self.get_axes()

    def get_robot_description(self):
        """Connects to robot_state_publisher and retrieves the URDF XML."""
        cli = self.create_client(
            GetParameters, "/robot_state_publisher/get_parameters"
        )
        if not cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                "Service /robot_state_publisher/get_parameters not available."
            )
            return None

        req = GetParameters.Request()
        req.names = ["robot_description"]

        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result() or not future.result().values:
            self.get_logger().error("No robot_description returned.")
            return None

        return future.result().values[0].string_value

    def get_axes(self):
        """Parses the URDF and prints the axis for key joints."""
        urdf_xml = self.get_robot_description()
        if urdf_xml is None:
            return

        try:
            root = ET.fromstring(urdf_xml)
        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF XML: {e}")
            return

        critical_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        
        print("\n=== CRITICAL JOINT AXIS DEFINITIONS (xyz) ===")
        for j_name in critical_joints:
            j = root.find(f"joint[@name='{j_name}']")
            
            if j is not None and j.get("type") in ("revolute", "continuous"):
                axis_elem = j.find("axis")
                
                if axis_elem is not None and "xyz" in axis_elem.attrib:
                    axis_str = axis_elem.get("xyz")
                    print(f"  {j_name}: Axis=[{axis_str}]")
                else:
                    print(f"  {j_name}: Axis not explicitly defined or is default [0 0 1].")
            else:
                # Catch fixed joints or those not found
                print(f"  {j_name}: Joint not found or is not a moving joint.")
        print("============================================\n")


def main(args=None):
    rclpy.init(args=args)
    # The node runs its __init__ method which calls get_axes()
    node = AxisExtractor()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

    