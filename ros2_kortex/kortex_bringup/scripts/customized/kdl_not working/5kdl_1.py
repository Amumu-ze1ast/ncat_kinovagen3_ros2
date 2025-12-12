#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class ReadURDF(Node):
    def __init__(self):
        super().__init__('read_urdf')

        # PARAMETER: robot_description from robot_state_publisher
        full_param = '/robot_state_publisher.robot_description'

        urdf = self.get_parameter_or(full_param, None)

        if urdf is None or not urdf:
            self.get_logger().error("❌ Could not read robot_description")
        else:
            self.get_logger().info("✅ Successfully read robot_description")
            print("\n================ URDF XML BEGIN ================\n")
            print(urdf.value)
            print("\n================= URDF XML END =================\n")


def main():
    rclpy.init()
    node = ReadURDF()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
