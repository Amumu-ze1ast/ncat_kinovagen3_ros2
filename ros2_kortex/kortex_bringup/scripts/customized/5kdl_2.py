#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
import PyKDL as kdl
import xml.etree.ElementTree as ET


class KDLBuilder(Node):
    def __init__(self):
        super().__init__("simple_kdl_builder")

        self.cli = self.create_client(GetParameters,
                                      "/robot_state_publisher/get_parameters")

        # Wait for service
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /robot_state_publisher/get_parameters not available")
            return

        # Build KDL tree
        self.build_tree()

    def strip_namespace(self, root):
        """Convert tags like {ns}joint → 'joint'."""
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

    def get_robot_description(self):
        req = GetParameters.Request()
        req.names = ["robot_description"]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result():
            self.get_logger().error("Empty result from service")
            return None

        urdf_xml = future.result().values[0].string_value
        if len(urdf_xml) < 10:
            self.get_logger().error("robot_description looks empty")
            return None

        return urdf_xml

    def build_tree(self):
        urdf_xml = self.get_robot_description()
        if urdf_xml is None:
            return

        print("\n=== FIRST 200 CHARACTERS OF URDF ===")
        print(urdf_xml[:200], "\n")

        # Parse XML
        try:
            root = ET.fromstring(urdf_xml)
        except Exception as e:
            self.get_logger().error(f"Failed to parse URDF XML: {e}")
            return

        # Remove namespaces
        self.strip_namespace(root)

        # ---- FIRST PASS: find root link name ----
        parents = set()
        children = set()

        for joint in root.findall("joint"):
            parent_link = joint.find("parent").get("link")
            child_link  = joint.find("child").get("link")
            parents.add(parent_link)
            children.add(child_link)

        if not parents:
            print("No joints found in URDF!")
            return

        root_candidates = parents - children
        if not root_candidates:
            root_link_name = list(parents)[0]  # fallback
        else:
            root_link_name = list(root_candidates)[0]

        print(f"Detected root link: {root_link_name}")

        # Create tree with correct root link
        tree = kdl.Tree(root_link_name)

        # ---- SECOND PASS: add segments ----
        added = 0
        for joint in root.findall("joint"):
            name = joint.get("name")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")

            origin = joint.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]

            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = list(map(float, origin.get("xyz").split()))
                if "rpy" in origin.attrib:
                    rpy = list(map(float, origin.get("rpy").split()))

            frame = kdl.Frame(
                kdl.Rotation.RPY(*rpy),
                kdl.Vector(*xyz)
            )

            # For now treat all joints as fixed; we’ll refine later
            k_joint = kdl.Joint(name, kdl.Joint.Fixed)

            segment = kdl.Segment(child, k_joint, frame)

            ok = tree.addSegment(segment, parent)
            if not ok:
                print(f"  [WARN] Failed to add segment {child} with parent {parent}")
            else:
                added += 1

        print("=== KDL TREE BUILT ===")
        print("Detected joints in URDF :", len(parents | children))
        print("Segments successfully added:", added)
        print("tree.getNrOfSegments()   :", tree.getNrOfSegments())
        # store if you want to use later
        self.tree = tree


def main():
    rclpy.init()
    node = KDLBuilder()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
