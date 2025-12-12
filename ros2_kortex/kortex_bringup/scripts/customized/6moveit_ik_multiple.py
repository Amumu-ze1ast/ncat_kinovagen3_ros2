#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import time
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from tf_transformations import quaternion_from_euler

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

END_EFFECTOR_LINK = "end_effector_link"
PLANNING_GROUP = "manipulator"

# --- Define the sequence of Cartesian points ---
# Adjust these coordinates if they are outside your robot's workspace.
WAYPOINTS = [
    (-0.60, 0.40, 0.60),  # Target 1
    (-0.55, 0.50, 0.50),  # Target 2
    (-0.45, 0.60, 0.65),  # Target 3
]

# Fixed orientation (e.g., facing downward, RPY: 0, 180, 0)
ROLL, PITCH, YAW = 0, 3.14159, 0
Q_X, Q_Y, Q_Z, Q_W = quaternion_from_euler(ROLL, PITCH, YAW)


class IntegratedIK(Node):
    def __init__(self):
        super().__init__("integrated_ik")

        self.current_joints = None
        self.waypoints_to_process = WAYPOINTS
        self.target_index = 0
        self.is_planning = False # State flag to prevent simultaneous calls

        # Subscribe to joint states
        self.create_subscription(
            JointState, "/joint_states", self.joint_cb, 10
        )

        # Planning service client
        self.cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /plan_kinematic_path...")

        self.get_logger().info("plan_kinematic_path service is ready.")

        # Timer runs the main sequential planning logic in the main ROS thread
        self.timer = self.create_timer(0.1, self.process_next_waypoint)


    # ----------------------------------------------------------
    # Joint State Callback (Used for initial seed)
    # ----------------------------------------------------------
    def joint_cb(self, msg):
        filt = JointState()
        for n, p in zip(msg.name, msg.position):
            if n in GEN3_JOINTS:
                filt.name.append(n)
                filt.position.append(p)
        
        if len(filt.name) == 7:
            if self.current_joints is None:
                self.current_joints = JointState()
                self.current_joints.name = GEN3_JOINTS
            
            # Map positions based on the defined order (GEN3_JOINTS)
            self.current_joints.position = [
                filt.position[filt.name.index(j)] for j in GEN3_JOINTS
            ]


    # ----------------------------------------------------------
    # Main Loop (Executed by ROS 2 Timer in the main thread)
    # ----------------------------------------------------------
    def process_next_waypoint(self):
        # 1. Guard against simultaneous planning and check for completion
        if self.is_planning:
            return
            
        if self.target_index >= len(self.waypoints_to_process):
            self.get_logger().info("All IK waypoints processed. Shutting down in 5 seconds...")
            self.timer.cancel()
            time.sleep(5) # Pause to allow final logs to print
            rclpy.shutdown()
            return
            
        # 2. Wait for initial joint states
        if self.current_joints is None:
            self.get_logger().info("Waiting for initial joint state...")
            return

        # 3. Start Planning for the current waypoint
        self.is_planning = True # Lock the timer
        
        target_xyz = self.waypoints_to_process[self.target_index]
        self.get_logger().info(f"--- Solving IK for Target {self.target_index + 1}: {target_xyz} ---")

        future = self._call_planner_for_ik_async(target_xyz)
        
        # --- DEBUG LOG 1: Before Blocking Call ---
        self.get_logger().warn(f"DEBUG: Sending request to /plan_kinematic_path...")

        # THIS IS THE BLOCKING CALL - MUST BE IN THE MAIN THREAD
        rclpy.spin_until_future_complete(self, future)
        
        # --- DEBUG LOG 2: After Blocking Call ---
        self.get_logger().warn("DEBUG: Service call returned.")
        
        solution_joints = self._process_planner_result(future.result())

        # 4. Process Result and Update Seed
        if solution_joints:
            self.get_logger().warn(f"DEBUG: IK Solution found for Target {self.target_index + 1}. Proceeding.")
            # **CRUCIAL: SEEDING**
            self.current_joints.position = solution_joints
            self.target_index += 1
        else:
            self.get_logger().error(f"DEBUG: IK failed or returned empty response for Target {self.target_index + 1}. Stopping.")
            self.timer.cancel()
            rclpy.shutdown()
        
        # 5. Release Lock
        self.is_planning = False
        time.sleep(0.5) # Small pause before the timer fires again

    # ----------------------------------------------------------
    # Helper function to call the planning service (returns a future)
    # ----------------------------------------------------------
    def _call_planner_for_ik_async(self, target_xyz):
        req = GetMotionPlan.Request()
        mpr: MotionPlanRequest = req.motion_plan_request

        mpr.group_name = PLANNING_GROUP
        mpr.allowed_planning_time = 15.0 # Increased planning time to avoid timeouts
        mpr.start_state.joint_state = self.current_joints

        # Target Pose assembly
        target_pose = Pose()
        target_pose.position.x = target_xyz[0]
        target_pose.position.y = target_xyz[1]
        target_pose.position.z = target_xyz[2]
        target_pose.orientation.x = Q_X
        target_pose.orientation.y = Q_Y
        target_pose.orientation.z = Q_Z
        target_pose.orientation.w = Q_W
        
        # Constraints assembly
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

        return self.cli.call_async(req)


    # ----------------------------------------------------------
    # Helper function to process the result
    # ----------------------------------------------------------
    def _process_planner_result(self, res):
        if not res or not hasattr(res, "motion_plan_response"):
            self.get_logger().error("ERROR: Planner returned no valid response (res is None or missing mp).")
            return None

        mp = res.motion_plan_response
        
        # Check MoveIt's planning error code
        if mp.error_code.val != 1: # MoveItErrorCode.SUCCESS == 1
             self.get_logger().error(f"ERROR: MoveIt Planning failed with code: {mp.error_code.val}")
             target_xyz = self.waypoints_to_process[self.target_index]
             self.get_logger().error(f"FAILED POSE: x={target_xyz[0]}, y={target_xyz[1]}, z={target_xyz[2]}")
             return None


        if not mp.trajectory.joint_trajectory.points:
            self.get_logger().error("ERROR: Empty trajectory from planner. IK failed.")
            return None
        
        # Final IK joint angles (the end point of the calculated path)
        final_joints = list(mp.trajectory.joint_trajectory.points[-1].positions)
        
        # Format and log the result
        formatted = [float(f"{v:.4f}") for v in final_joints]
        
        # *** LOGGED AS FATAL FOR MAXIMUM VISIBILITY IN THE TERMINAL ***
        self.get_logger().fatal(f"IK SOLUTION FOUND (Target {self.target_index + 1}): {formatted}")
        
        return final_joints 


def main():
    rclpy.init()
    node = IntegratedIK()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()