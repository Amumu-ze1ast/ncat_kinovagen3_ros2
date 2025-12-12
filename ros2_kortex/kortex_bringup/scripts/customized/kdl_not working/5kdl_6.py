#!/usr/bin/env python3
# Simple KDL FK test for ROS2 + Gazebo (PyKDL, manual joint input)

import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters

import PyKDL as kdl
import xml.etree.ElementTree as ET


class SimpleKDLFK(Node):
    def __init__(self):
        super().__init__("simple_kdl_fk")

        # ---- 1) Connect to robot_state_publisher ----
        self.cli = self.create_client(
            GetParameters, "/robot_state_publisher/get_parameters"
        )
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Service /robot_state_publisher/get_parameters not available"
            )
            rclpy.shutdown()
            return

        # ---- 2) Get URDF string ----
        urdf_xml = self.get_robot_description()
        if urdf_xml is None:
            rclpy.shutdown()
            return

        # ---- 3) Build KDL Tree from URDF ----
        tree, root_link = self.build_tree_from_urdf(urdf_xml)
        if tree is None:
            rclpy.shutdown()
            return

        # ---- 4) Build KDL Chain (set base & tip here) ----
        base_link = "base_link"
        tip_link = "end_effector_link"   # change if needed

        print(f"\nTrying to build chain: {base_link} -> {tip_link}")
        try:
            chain = tree.getChain(base_link, tip_link)
        except RuntimeError:
            self.get_logger().error(
                "Could not build KDL chain. Check base/tip link names."
            )
            rclpy.shutdown()
            return

        nj = chain.getNrOfJoints()
        print("Number of joints in chain:", nj)

        # ---- 5) FK solver ----
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        # ---- 6) MANUAL JOINT INPUT (EDIT THIS LIST) ----
        # Joint values in *radians*, in the SAME ORDER as the chain joints.
        # For Kinova Gen3, usually: [joint_1, joint_2, ..., joint_7]
        joint_positions_rad = [
            # 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            # 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57    
            # 0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785
            0.3925, 0.3925, 0.3925, 0.3925, 0.3925, 0.3925, 0.3925 
        ]

        if len(joint_positions_rad) != nj:
            self.get_logger().error(
                f"Length of joint_positions_rad ({len(joint_positions_rad)}) "
                f"does not match number of joints in chain ({nj})."
            )
            rclpy.shutdown()
            return

        # ---- 7) Fill KDL JntArray with your joint values ----
        q = kdl.JntArray(nj)
        for i, val in enumerate(joint_positions_rad):
            q[i] = val

        # ---- 8) FK: compute end-effector pose ----
        fk_frame = kdl.Frame()
        fk_solver.JntToCart(q, fk_frame)

        p = fk_frame.p
        R = fk_frame.M
        roll, pitch, yaw = R.GetRPY()

        print("\n=== FK RESULT for your joint input ===")
        print(f"Joint input (rad): {joint_positions_rad}")
        print(f"Joint input (deg): {[math.degrees(v) for v in joint_positions_rad]}")
        print(f"\nPosition [m]:  [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
        print(f"RPY [rad]:     [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]")
        print(f"RPY [deg]:     [{math.degrees(roll):.2f}, "
              f"{math.degrees(pitch):.2f}, {math.degrees(yaw):.2f}]")

        # done
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Get /robot_state_publisher robot_description
    # ------------------------------------------------------------------
    def get_robot_description(self):
        req = GetParameters.Request()
        req.names = ["robot_description"]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result() or not future.result().values:
            self.get_logger().error("No robot_description returned.")
            return None

        urdf_xml = future.result().values[0].string_value
        if not urdf_xml:
            self.get_logger().error("robot_description is empty.")
            return None

        print("\n=== FIRST 200 CHARS OF URDF ===")
        print(urdf_xml[:200])
        print("================================\n")
        return urdf_xml

    # ------------------------------------------------------------------
    # Very simple URDF -> KDL.Tree builder using PyKDL
    # ------------------------------------------------------------------
    def build_tree_from_urdf(self, urdf_xml):
        try:
            root = ET.fromstring(urdf_xml)
        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF XML: {e}")
            return None, None

        links = {l.get("name"): l for l in root.findall("link")}
        joints = root.findall("joint")
        print(f"Detected joints in URDF: {len(joints)}")

        # root link = link that is never a child
        child_links = {j.find("child").get("link") for j in joints}
        all_links = set(links.keys())
        root_candidates = list(all_links - child_links)
        root_link = root_candidates[0] if root_candidates else "world"
        print("Detected root link:", root_link)

        tree = kdl.Tree(root_link)

        # parent -> list of joints
        parent_map = {}
        for j in joints:
            parent = j.find("parent").get("link")
            parent_map.setdefault(parent, []).append(j)

        added = {root_link}
        queue = [root_link]
        seg_count = 0

        while queue:
            parent = queue.pop(0)
            for j in parent_map.get(parent, []):
                name = j.get("name")
                child = j.find("child").get("link")

                # origin
                origin = j.find("origin")
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if origin is not None:
                    if "xyz" in origin.attrib:
                        xyz = list(map(float, origin.get("xyz").split()))
                    if "rpy" in origin.attrib:
                        rpy = list(map(float, origin.get("rpy").split()))
                frame = kdl.Frame(
                    kdl.Rotation.RPY(*rpy),
                    kdl.Vector(*xyz),
                )

                # joint type
                jtype = j.get("type")
                axis_elem = j.find("axis")
                axis = [0.0, 0.0, 1.0]
                if axis_elem is not None and "xyz" in axis_elem.attrib:
                    axis = list(map(float, axis_elem.get("xyz").split()))

                if jtype in ("revolute", "continuous"):
                    k_joint = kdl.Joint(
                        name,
                        kdl.Vector(0, 0, 0),
                        kdl.Vector(*axis),
                        kdl.Joint.RotAxis,
                    )
                elif jtype == "prismatic":
                    k_joint = kdl.Joint(
                        name,
                        kdl.Vector(0, 0, 0),
                        kdl.Vector(*axis),
                        kdl.Joint.TransAxis,
                    )
                else:
                    k_joint = kdl.Joint(name, kdl.Joint.Fixed)

                segment = kdl.Segment(child, k_joint, frame)
                if tree.addSegment(segment, parent):
                    seg_count += 1
                    added.add(child)
                    queue.append(child)

        print("Segments successfully added:", seg_count)
        print("tree.getNrOfSegments():", tree.getNrOfSegments())
        return tree, root_link


# ----------------------------------------------------------------------
def main():
    rclpy.init()
    SimpleKDLFK()


if __name__ == "__main__":
    main()
