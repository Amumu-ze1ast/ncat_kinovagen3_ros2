#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import threading
import time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive


GEN3_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

END_EFFECTOR_LINK = "end_effector_link"   # change if needed


class IntegratedIK(Node):
    def __init__(self):
        super().__init__("integrated_ik")

        self.current_joints = None

        # Subscribe to joint states
        self.create_subscription(
            JointState, "/joint_states", self.joint_cb, 10
        )

        # Planning service
        self.cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /plan_kinematic_path...")

        self.get_logger().info("plan_kinematic_path service is ready.")

        self.timer = self.create_timer(2.0, self.run_once)
        self.did_run = False

    # ----------------------------------------------------------
    # Filter only the 7 Kinova arm joints
    # ----------------------------------------------------------
    def joint_cb(self, msg):
        filt = JointState()
        for n, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort):
            if n in GEN3_JOINTS:
                filt.name.append(n)
                filt.position.append(p)
                filt.velocity.append(v)
                filt.effort.append(e)

        if len(filt.name) == 7:
            self.current_joints = filt

    # ----------------------------------------------------------
    # Main IK (using planning)
    # ----------------------------------------------------------
    def run_once(self):
        if self.did_run:
            return

        if self.current_joints is None:
            self.get_logger().info("Waiting for Gen3 joint state...")
            return

        self.did_run = True
        self.get_logger().info("Calling planner as IK...")

        req = GetMotionPlan.Request()
        mpr: MotionPlanRequest = req.motion_plan_request

        mpr.group_name = "manipulator"
        mpr.allowed_planning_time = 5.0

        # Start state (seed)
        ordered = JointState()
        ordered.name = GEN3_JOINTS
        ordered.position = [
            self.current_joints.position[self.current_joints.name.index(j)]
            for j in GEN3_JOINTS
        ]
        ordered.velocity = [0.0] * 7
        ordered.effort = [0.0] * 7

        mpr.start_state.joint_state = ordered

        # Target pose
        target_pose = Pose()


        target_pose_list = [0.121, -0.692, 0.585]

        target_pose.position.x = target_pose_list[0]
        target_pose.position.y = target_pose_list[1]
        target_pose.position.z = target_pose_list[2]
        target_pose.orientation.w = 1.0

        # ---------------- POSITION CONSTRAINT -------------------
        pos_con = PositionConstraint()
        pos_con.header.frame_id = "base_link"
        pos_con.link_name = END_EFFECTOR_LINK

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]

        region_pose = Pose()
        region_pose.position = target_pose.position
        region_pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(region_pose)

        pos_con.constraint_region = bv
        pos_con.weight = 1.0

        # ---------------- ORIENTATION CONSTRAINT ----------------
        ori_con = OrientationConstraint()
        ori_con.header.frame_id = "base_link"
        ori_con.link_name = END_EFFECTOR_LINK
        ori_con.orientation = target_pose.orientation
        ori_con.absolute_x_axis_tolerance = 0.2
        ori_con.absolute_y_axis_tolerance = 0.2
        ori_con.absolute_z_axis_tolerance = 0.2
        ori_con.weight = 1.0

        constr = Constraints()
        constr.position_constraints.append(pos_con)
        constr.orientation_constraints.append(ori_con)
        mpr.goal_constraints.append(constr)

        # Call planner
        future = self.cli.call_async(req)

        # ---------------- Extract without waiting forever ----------------
        def extract_result():
            time.sleep(2.0)
            res = future.result()

            if not res or not hasattr(res, "motion_plan_response"):
                self.get_logger().error("Planner returned no valid response.")
                rclpy.shutdown()
                return

            mp = res.motion_plan_response

            if not mp.trajectory.joint_trajectory.points:
                self.get_logger().error("Empty trajectory from planner.")
                rclpy.shutdown()
                return

            # Final IK joint angles
            final_joints = mp.trajectory.joint_trajectory.points[-1].positions

            # Format values to 4 significant digits
            formatted = [float(f"{v:.4f}") for v in final_joints]

            self.get_logger().info("IK Solution (4 significant digits):")
            self.get_logger().info(str(formatted))

            # Kill the node automatically
            self.get_logger().info("IK extraction complete. Exiting node.")
            rclpy.shutdown()

        threading.Thread(target=extract_result).start()


def main():
    rclpy.init()
    node = IntegratedIK()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
