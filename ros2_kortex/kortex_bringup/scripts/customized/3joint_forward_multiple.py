#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

# JOINT ORDER EXACTLY AS CONTROLLER EXPECTS
JOINT_NAMES = [
    'joint_1', 'joint_2', 'joint_3',
    'joint_4', 'joint_5', 'joint_6', 'joint_7'
]

# 🌟 SPEED SCALING (0.1 = slow, 1.0 = max allowed)
SPEED_SCALE = 0.3   # 30% of robot speed


# ===== Example Waypoints (Random for Now) =====
WAYPOINTS = [
    np.array([0.0, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.3, -0.3, 0.6, 0.2, 0.0, -0.1, 0.0]),
    np.array([-0.2, -0.4, 0.2, -0.1, 0.2, 0.0, 0.0]),
]
# ==============================================


class MultiWaypointGen3(Node):

    def __init__(self):
        super().__init__('multi_waypoint_gen3')

        # Publisher
        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber
        self.latest_js = None
        self.create_subscription(
            JointState,
            '/joint_states',
            self.js_callback,
            10
        )

        # Internal state
        self.current_index = 0
        self.done = False

        # Start after 1 sec
        self.start_timer = self.create_timer(1.0, self.start_motion)
        self.check_timer = None

    # ------------------------------------------------------

    def js_callback(self, msg):
        self.latest_js = msg

    # ------------------------------------------------------

    def get_actual(self):
        """Return joint positions in correct order."""
        if self.latest_js is None:
            return None
        mapping = dict(zip(self.latest_js.name, self.latest_js.position))
        if not all(j in mapping for j in JOINT_NAMES):
            return None
        return np.array([mapping[j] for j in JOINT_NAMES], dtype=float)

    # ------------------------------------------------------

    def compute_time_for_waypoint(self, start, target):
        """Compute time_from_start based on speed scaling."""
        dist = np.max(np.abs(target - start))
        base_vel = 0.8  # rad/s (approx robot safe limit)
        scaled_vel = base_vel * SPEED_SCALE
        t = dist / scaled_vel
        return max(t, 0.5)  # never below 0.5 sec

    # ------------------------------------------------------

    def start_motion(self):
        """Send first waypoint."""
        self.start_timer.cancel()
        self.send_next_waypoint()

    # ------------------------------------------------------

    def send_next_waypoint(self):

        if self.current_index >= len(WAYPOINTS):
            self.get_logger().info("All waypoints completed.")
            self.done = True
            return

        target = WAYPOINTS[self.current_index]
        self.get_logger().info(f"Sending waypoint {self.current_index+1}/{len(WAYPOINTS)}")

        # Get current state for timing calculation
        actual = self.get_actual()
        if actual is None:
            self.get_logger().warn("Waiting for joint states...")
            self.create_timer(0.5, self.send_next_waypoint)
            return

        duration = self.compute_time_for_waypoint(actual, target)

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions = target.tolist()
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1.0) * 1e9)

        traj.points = [pt]
        self.pub.publish(traj)

        # Start monitoring until reached
        self.check_timer = self.create_timer(0.1, self.wait_until_reached)

    # ------------------------------------------------------

    def wait_until_reached(self):
        """Check if robot arrived at current waypoint."""
        actual = self.get_actual()
        if actual is None:
            return

        target = WAYPOINTS[self.current_index]
        error = np.abs(actual - target)

        if np.all(error < 0.015):  # tolerance
            # Arrived → Compare and move on
            self.check_timer.cancel()
            self.compare_current()

            self.current_index += 1
            self.send_next_waypoint()

    # ------------------------------------------------------

    def compare_current(self):
        actual = self.get_actual()
        target = WAYPOINTS[self.current_index]

        desired_r = np.round(target, 4)
        actual_r = np.round(actual, 4)
        error_r = np.round(actual_r - desired_r, 4)

        print("\n===============================")
        print(f" Waypoint {self.current_index+1} Results")
        print("===============================\n")
        print("Desired:", desired_r.tolist())
        print("Actual : ", actual_r.tolist())
        print("Error  : ", error_r.tolist())
        print("\n===============================\n")


# ------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MultiWaypointGen3()

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
