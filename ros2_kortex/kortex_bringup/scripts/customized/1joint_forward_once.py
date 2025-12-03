#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np


# Joint order MUST match controller
JOINT_NAMES = [
    'joint_1', 'joint_2', 'joint_3',
    'joint_4', 'joint_5', 'joint_6', 'joint_7'
]


class SimpleGen3Command(Node):

    def __init__(self):
        super().__init__('simple_gen3_command')

        # Publisher for joint trajectory
        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscribe to joint states
        self.latest_joint_states = None
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Desired joint targets
        self.desired_positions = np.array([
            0.0,     # joint_1
            0.0,    # joint_2
            0.0,     # joint_3
            0.0,     # joint_4
            0.0,     # joint_5
            0.0,     # joint_6
            0.0      # joint_7
        ], dtype=float)

        # Flags
        self.sent = False
        self.done = False

        # Send command once using timer
        self.send_timer = self.create_timer(1.0, self.send_once)

        # Timer for checking completion
        self.check_timer = None

    # ----------------------------------------------------

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_states = msg

    # ----------------------------------------------------

    def send_once(self):
        """Send one joint trajectory command, then begin checking for completion."""
        if self.sent:
            return

        self.sent = True

        self.get_logger().info("Sending joint trajectory command...")

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = self.desired_positions.tolist()
        point.time_from_start.sec = 3
        traj.points = [point]

        self.pub.publish(traj)
        self.get_logger().info("Command sent. Waiting for robot to reach target...")

        # Begin polling every 0.1s to check when robot arrives
        self.check_timer = self.create_timer(0.1, self.check_if_finished)

    # ----------------------------------------------------

    def get_actual_positions(self):
        """Return numpy array of actual joint positions; None if not ready."""
        if self.latest_joint_states is None:
            return None

        name_to_pos = dict(zip(
            self.latest_joint_states.name,
            self.latest_joint_states.position
        ))

        actual = []
        for j in JOINT_NAMES:
            if j not in name_to_pos:
                return None
            actual.append(name_to_pos[j])

        return np.array(actual, dtype=float)

    # ----------------------------------------------------

    def check_if_finished(self):
        """Check if robot has reached target within tolerance."""
        actual = self.get_actual_positions()
        if actual is None:
            return

        error = actual - self.desired_positions

        # Threshold for considering the command finished
        if np.all(np.abs(error) < 0.01):
            self.get_logger().info("Target reached. Comparing results...")
            self.check_timer.cancel()
            self.compare_once()

    # ----------------------------------------------------

    def compare_once(self):
        """Print desired vs actual vs error, then exit."""
        actual = self.get_actual_positions()
        if actual is None:
            self.get_logger().warn("No joint states available for comparison.")
            self.done = True
            return

        desired_r = np.round(self.desired_positions, 4)
        actual_r = np.round(actual, 4)
        error_r = np.round(actual_r - desired_r, 4)

        print("\n============================")
        print(" Joint Comparison Results")
        print("============================\n")
        print("Desired:", desired_r.tolist())
        print("Actual : ", actual_r.tolist())
        print("Error  : ", error_r.tolist())
        print("\n============================\n")

        self.done = True


# ----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SimpleGen3Command()

    try:
        # Custom spin loop so we can exit cleanly when done = True
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
