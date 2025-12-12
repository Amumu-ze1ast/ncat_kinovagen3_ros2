import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
import PyKDL as kdl
import xml.etree.ElementTree as ET

# --- Configuration ---
AXIS_CORRECTION = {
    "joint_2": [0.0, -1.0, 0.0],  
    "joint_4": [0.0, -1.0, 0.0],  
    "joint_6": [0.0, -1.0, 0.0], 
}
# ---------------------

class SimpleKDLFK(Node):
    
    def __init__(self):
        super().__init__("simple_kdl_fk")
        self.get_logger().info("Starting KDL Forward Kinematics setup...")

        # --- FIX: Client Initialization moved here ---
        self.cli = self.create_client(
            GetParameters, "/robot_state_publisher/get_parameters"
        )
        if not self.cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                "Service /robot_state_publisher/get_parameters not available."
            )
            # Cannot proceed without the service
            rclpy.shutdown()
            return
        # --------------------------------------------

        # 1. Retrieve URDF XML from ROS parameter server
        urdf_xml = self.get_robot_description()
        if urdf_xml is None:
            self.get_logger().fatal("Failed to retrieve robot_description. Shutting down.")
            rclpy.shutdown()
            return
        
        # 2. Build KDL Tree (with axis correction patch)
        tree, root_link = self.build_tree_from_urdf(urdf_xml)
        if tree is None:
            rclpy.shutdown()
            return

        # 3. Define and build the KDL Chain
        base_link = "base_link"
        tip_link = "bracelet_link"

        self.get_logger().info(f"Building chain: {base_link} -> {tip_link}")
        try:
            chain = tree.getChain(base_link, tip_link)
        except RuntimeError:
            self.get_logger().error("Could not build KDL chain. Check base/tip link names.")
            rclpy.shutdown()
            return

        n_joints = chain.getNrOfJoints()
        
        # 4. Define and load joint positions (0.5 rad for all)
        chain_joint_names = self._get_chain_joint_names(chain)
        user_joints_rad = {name: 0.5 for name in chain_joint_names}

        q = kdl.JntArray(len(chain_joint_names))

        print("\n=== USING JOINT VALUES (radians) ===")
        for idx, name in enumerate(chain_joint_names):
            angle = float(user_joints_rad.get(name, 0.0))
            q[idx] = angle
            print(f"{name}: {angle:.4f} rad ({math.degrees(angle):.2f} deg)")

        # 5. FK solver and output
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)
        fk_frame = kdl.Frame()
        fk_solver.JntToCart(q, fk_frame)

        p = fk_frame.p
        x, y, z = p[0], p[1], p[2]

        R = fk_frame.M
        roll, pitch, yaw = R.GetRPY()

        print("\n=== FINAL FK RESULT (KDL - Patched Axes) ===")
        print(f"Position [m]:  [{x:.4f}, {y:.4f}, {z:.4f}]")
        print(f"RPY [rad]:     [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]")
        print(
            f"RPY [deg]:     [{math.degrees(roll):.2f}, "
            f"{math.degrees(pitch):.2f}, {math.degrees(yaw):.2f}]"
        )
        print("==============================================")
        
        rclpy.shutdown()

# ------------------------------------------------------------------
# Data Retrieval Function (Now uses self.cli)
# ------------------------------------------------------------------
    def get_robot_description(self):
        """Retrieves the URDF XML using the client initialized in __init__."""
        
        req = GetParameters.Request()
        req.names = ["robot_description"]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future) # Use spin with future

        if not future.result() or not future.result().values:
            self.get_logger().error("No robot_description returned.")
            return None

        urdf_xml = future.result().values[0].string_value
        if not urdf_xml:
            self.get_logger().error("robot_description is empty.")
            return None
        return urdf_xml
    
# ------------------------------------------------------------------
# Helper to get joint names (for loading the JntArray)
# ------------------------------------------------------------------
    def _get_chain_joint_names(self, chain: kdl.Chain) -> list[str]:
        chain_joint_names = []
        for i in range(chain.getNrOfSegments()):
            seg = chain.getSegment(i)
            j = seg.getJoint()
            if j.getType() != kdl.Joint.Fixed:
                chain_joint_names.append(j.getName())
        return chain_joint_names


# ------------------------------------------------------------------
# KDL Tree Builder with Axis Correction Patch
# ------------------------------------------------------------------
    def build_tree_from_urdf(self, urdf_xml):
        try:
            root = ET.fromstring(urdf_xml)
        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF XML: {e}")
            return None, None
        
        # Simplified root link finding
        joints = root.findall("joint")
        child_links = {j.find("child").get("link") for j in joints}
        
        # Guard against empty link set
        all_links = set(l.get("name") for l in root.findall("link"))
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

                # 1. Read Origin and create Fixed Frame
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

                # 2. Read Joint Type and Axis
                jtype = j.get("type")
                axis_elem = j.find("axis")
                axis = [0.0, 0.0, 1.0] # Default Z-axis
                if axis_elem is not None and "xyz" in axis_elem.attrib:
                    axis = list(map(float, axis_elem.get("xyz").split()))
                
                # --- APPLY THE DEFINITIVE AXIS CORRECTION PATCH ---
                if name in AXIS_CORRECTION:
                    axis = AXIS_CORRECTION[name]
                    self.get_logger().warn(f"[PATCH] Corrected joint axis for {name} to: {axis}")
                # --------------------------------------------------
                
                # 3. Create KDL Joint and Segment
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

        self.get_logger().info(f"Segments successfully added: {seg_count}")
        return tree, root_link


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    SimpleKDLFK()


if __name__ == "__main__":
    main()