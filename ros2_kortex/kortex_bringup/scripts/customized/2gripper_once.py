#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
import time


class GripperSimple(Node):

    def __init__(self):
        super().__init__('gripper_simple')

        # Robotiq gripper action server
        self.client = ActionClient(
            self,
            GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd'
        )

    def send(self, position, effort=100.0):
        """Send a gripper position command."""
        if not self.client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("❌ Gripper action server NOT available!")
            return

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(effort)

        self.get_logger().info(f"Sending gripper → position={position:.2f}")
        self.client.send_goal_async(goal)

    def open(self):
        self.send(0.1)

    def close(self):
        self.send(0.7)


def main(args=None):
    rclpy.init(args=args)
    node = GripperSimple()

    # node.get_logger().info("Opening gripper...")
    # node.open()
    # time.sleep(2)

    node.get_logger().info("Closing gripper...")
    node.close()
    time.sleep(2)

    node.get_logger().info("Done.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
