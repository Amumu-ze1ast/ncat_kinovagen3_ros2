#!/usr/bin/env python3
# Simple KDL FK test for ROS2 + Gazebo (with Joint Axis Patch)

import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters

import PyKDL as kdl
import xml.etree.ElementTree as ET


class SimpleKDLFK(Node):
    # --- AXIS CORRECTION MAPPING (The Fix) ---
    # Based on diagnosis that the URDF incorrectly sets all joints to Z-axis.
    # We patch the bending joints (2, 4, 6) to rotate around the Y-axis (0 1 0),
    # which is common for the Kinova Gen3 structure after fixed link rotations.
    AXIS_CORRECTION = {
        "joint_2": [0.0, -1.0, 0.0],  # Corrected to Y-axis for elbow bend
        "joint_4": [0.0, -1.0, 0.0],  # Corrected to Y-axis for wrist bend
        "joint_6": [0.0, -1.0, 0.0],  # Corrected to Y-axis for wrist bend
    }
    # ----------------------------------------
    
    def __init__(self):
        super().__init__("simple_kdl_fk")

        # 1) Connect to robot_state_publisher
        self.cli = self.create_client(
            GetParameters, "/robot_state_publisher/get_parameters"
        )
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Service /robot_state_publisher/get_parameters not available"
            )
            rclpy.shutdown()
            return

        # 2) Get URDF string
        urdf_xml = self.get_robot_description()
        if urdf_xml is None:
            rclpy.shutdown()
            return

        # 3) Build KDL Tree from URDF with Axis Correction
        tree, root_link = self.build_tree_from_urdf(urdf_xml)
        if tree is None:
            rclpy.shutdown()
            return

        # 4) Build KDL Chain (base_link -> bracelet_link for accurate comparison)
        base_link = "base_link"
        tip_link = "bracelet_link"

        print(f"\nTrying to build chain: {base_link} -> {tip_link}")
        try:
            chain = tree.getChain(base_link, tip_link)
        except RuntimeError:
            self.get_logger().error(
                "Could not build KDL chain. Check base/tip link names."
            )
            rclpy.shutdown()
            return

        n_joints = chain.getNrOfJoints()
        print("Number of joints in chain:", n_joints)

        # 5) Extract chain joint names in correct order
        chain_joint_names = []
        for i in range(chain.getNrOfSegments()):
            seg = chain.getSegment(i)
            j = seg.getJoint()
            if j.getType() != kdl.Joint.Fixed:
                chain_joint_names.append(j.getName())

        # 6) Define your joint positions here (0.5 rad for all)
        user_joints_rad = {
            "joint_1": 0.5, "joint_2": 0.5, "joint_3": 0.5,
            "joint_4": 0.5, "joint_5": 0.5, "joint_6": 0.5,
            "joint_7": 0.5,
        }

        # 7) Fill KDL JntArray in *chain order* from the dict
        q = kdl.JntArray(len(chain_joint_names))

        print("\n=== USING JOINT VALUES (radians) ===")
        for idx, name in enumerate(chain_joint_names):
            angle = float(user_joints_rad.get(name, 0.0))
            q[idx] = angle
            print(f"{name}: {angle:.4f} rad ({math.degrees(angle):.2f} deg)")

        # 8) FK solver
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)
        fk_frame = kdl.Frame()
        fk_solver.JntToCart(q, fk_frame)

        p = fk_frame.p
        x = p[0]  # Corrected KDL mapping: p[0]=X, p[1]=Y, p[2]=Z
        y = p[1]
        z = p[2]

        R = fk_frame.M
        roll, pitch, yaw = R.GetRPY()

        print("\n=== FK RESULT (KDL - Corrected Axes) ===")
        print(f"Position [m]:  [{x:.4f}, {y:.4f}, {z:.4f}]")
        print(f"RPY [rad]:     [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]")
        print(
            f"RPY [deg]:     [{math.degrees(roll):.2f}, "
            f"{math.degrees(pitch):.2f}, {math.degrees(yaw):.2f}]"
        )
        
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Get /robot_state_publisher robot_description
    # ------------------------------------------------------------------
    def get_robot_description(self):
        req = GetParameters.Request()
        req.names = ["robot_description"]
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if not future.result() or not future.result().values: return None
        urdf_xml = future.result().values[0].string_value
        if not urdf_xml: return None
        print(f"\n=== FIRST 200 CHARS OF URDF ===\n{urdf_xml[:200]}\n================================")
        return urdf_xml

    # ------------------------------------------------------------------
    # URDF -> KDL.Tree builder (MODIFIED to apply AXIS_CORRECTION)
    # ------------------------------------------------------------------
    def build_tree_from_urdf(self, urdf_xml):
        try:
            root = ET.fromstring(urdf_xml)
        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF XML: {e}")
            return None, None
        
        # --- Axis check and root finding logic (omitted for brevity) ---
        
        links = {l.get("name"): l for l in root.findall("link")}
        joints = root.findall("joint")
        child_links = {j.find("child").get("link") for j in joints}
        all_links = set(links.keys())
        root_candidates = list(all_links - child_links)
        root_link = root_candidates[0] if root_candidates else "world"

        tree = kdl.Tree(root_link)
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

                # Origin and Frame creation
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

                # Joint Type and Axis
                jtype = j.get("type")
                axis_elem = j.find("axis")
                axis = [0.0, 0.0, 1.0] # Default Z-axis

                if axis_elem is not None and "xyz" in axis_elem.attrib:
                    axis = list(map(float, axis_elem.get("xyz").split()))
                
                # --- APPLY AXIS CORRECTION PATCH ---
                if name in self.AXIS_CORRECTION:
                    axis = self.AXIS_CORRECTION[name]
                    self.get_logger().info(f"[PATCH] Corrected joint axis for {name} to: {axis}")
                # -----------------------------------
                
                if jtype in ("revolute", "continuous"):
                    k_joint = kdl.Joint(
                        name,
                        kdl.Vector(0, 0, 0),
                        kdl.Vector(*axis), # Use the corrected axis
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

        print(f"Segments successfully added: {seg_count}")
        return tree, root_link


# ----------------------------------------------------------------------
def main():
    rclpy.init()
    SimpleKDLFK()


if __name__ == "__main__":
    main()