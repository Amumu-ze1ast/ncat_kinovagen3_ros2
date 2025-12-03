import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
import PyKDL as kdl
import xml.etree.ElementTree as ET


class KDLBuilder(Node):

    def __init__(self):
        super().__init__("simple_kdl_builder")

        # Connect to /robot_state_publisher
        self.cli = self.create_client(GetParameters,
                                      "/robot_state_publisher/get_parameters")

        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service not available!")
            return

        self.build_tree()

    # ----------------------------------------------------------------------
    # STEP 1 — Get robot_description from ROS2 param
    # ----------------------------------------------------------------------
    def get_robot_description(self):
        req = GetParameters.Request()
        req.names = ["robot_description"]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result():
            self.get_logger().error("Empty result from /get_parameters")
            return None

        xml = future.result().values[0].string_value

        if len(xml) < 50:
            self.get_logger().error("Received empty URDF!")
            return None

        return xml

    # ----------------------------------------------------------------------
    # STEP 2 — Build KDL tree manually from URDF
    # ----------------------------------------------------------------------
    def build_tree(self):
        urdf_xml = self.get_robot_description()
        if urdf_xml is None:
            return

        print("\n===== FIRST 300 CHARACTERS OF URDF =====")
        print(urdf_xml[:300], "\n")

        # Parse URDF XML
        root = ET.fromstring(urdf_xml)

        # KDL Tree creation
        root_link = root.find("link").get("name")
        print("Detected root link:", root_link)

        tree = kdl.Tree(root_link)

        joints = root.findall("joint")
        print("Detected joints in URDF:", len(joints))

        added_segments = 0

        # ------------------------------------------------------------------
        # STEP 3 — Parse joints
        # ------------------------------------------------------------------
        for joint in joints:
            jname = joint.get("name")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            jtype = joint.get("type")

            # ---- origin ----
            origin = joint.find("origin")
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]

            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = list(map(float, origin.get("xyz").split()))
                if "rpy" in origin.attrib:
                    rpy = list(map(float, origin.get("rpy").split()))

            frame = kdl.Frame(
                kdl.Rotation.RPY(*rpy),
                kdl.Vector(*xyz)
            )

            # ---- axis ----
            axis_tag = joint.find("axis")
            if axis_tag is not None:
                axis_xyz = list(map(float, axis_tag.get("xyz").split()))
            else:
                axis_xyz = [0, 0, 1]

            axis = kdl.Vector(*axis_xyz)

            # ------------------------------------------------------------------
            # STEP 4 — Make PyKDL Joint (CORRECT SYNTAX)
            # ------------------------------------------------------------------
            if jtype == "fixed":
                k_joint = kdl.Joint(jname, kdl.Joint.Fixed)

            elif jtype in ["revolute", "continuous"]:
                k_joint = kdl.Joint(
                    jname,
                    kdl.Vector(*xyz),
                    axis,
                    kdl.Joint.RotAxis
                )

            elif jtype == "prismatic":
                k_joint = kdl.Joint(
                    jname,
                    kdl.Vector(*xyz),
                    axis,
                    kdl.Joint.TransAxis
                )

            else:
                print(f"[WARN] Unknown joint type {jtype} → treating as fixed")
                k_joint = kdl.Joint(jname, kdl.Joint.Fixed)

            # Create segment and add to tree
            segment = kdl.Segment(child, k_joint, frame)

            try:
                tree.addSegment(segment, parent)
                added_segments += 1
            except RuntimeError:
                print(f"[SKIP] Couldn't add {jname} (parent missing: {parent})")

        # ------------------------------------------------------------------
        # STEP 5 — Print results
        # ------------------------------------------------------------------
        print("\n===== KDL TREE BUILT SUCCESSFULLY =====")
        print("Segments successfully added:", added_segments)
        print("tree.getNrOfSegments():", tree.getNrOfSegments())

        # ------------------------------------------------------------------
        # STEP 6 — Try building a chain: base_link → end_effector_link
        # ------------------------------------------------------------------
        base = "base_link"
        tip = "end_effector_link"

        print(f"\nTrying to build chain: {base} → {tip}")

        try:
            chain = tree.getChain(base, tip)
            print("Number of joints in chain:", chain.getNrOfJoints())
        except:
            print("[ERROR] Could not build chain. Check base/tip names.")

        # ----------------------------------------------------------------------
        # STEP 7 — Print joint names and segment names in the chain
        # ----------------------------------------------------------------------

        print("\n===== JOINT NAMES IN CHAIN =====")
        for i in range(chain.getNrOfSegments()):
            joint = chain.getSegment(i).getJoint()
            name = joint.getName()
            if name != "None":      # skip fixed/no-joint segments
                print(f"{i}: {name}")

        print("\n===== SEGMENT (LINK) NAMES IN CHAIN =====")
        for i in range(chain.getNrOfSegments()):
            seg = chain.getSegment(i)
            print(f"{i}: {seg.getName()}")

        # ------------------------------------------------------------------
        # STEP 8B - Test Forward Kinematics (FK)
        # ------------------------------------------------------------------
        print("\n=== TEST FORWARD KINEMATICS ===")

        # Set all joint angles = 0.0 (example)
        q_test = [0.0] * chain.getNrOfJoints()

        fk_frame = self.compute_fk(chain, q_test)

        pos = fk_frame.p
        rpy = fk_frame.M.GetRPY()

        print(f"FK Position (XYZ): {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}")
        print(f"FK Orientation (RPY): {rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}")

    # ------------------------------------------------------------------
    # STEP 8A - Forward Kinematics function
    # ------------------------------------------------------------------
    def compute_fk(self, chain, joint_positions):
        """Compute FK using KDL ChainFkSolverPos_recursive"""
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        # Create JntArray with given joint values
        q = kdl.JntArray(chain.getNrOfJoints())
        for i, pos in enumerate(joint_positions):
            q[i] = pos

        frame = kdl.Frame()
        fk_solver.JntToCart(q, frame)

        return frame





# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    rclpy.init()
    node = KDLBuilder()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
