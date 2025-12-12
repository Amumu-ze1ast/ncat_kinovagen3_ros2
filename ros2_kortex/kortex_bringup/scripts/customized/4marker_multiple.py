#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class RedPointsPublisher(Node):

    def __init__(self, points_xyz):
        super().__init__('two_red_points_marker')

        # Publisher
        self.pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )

        # Save points
        self.points_xyz = points_xyz

        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Build the marker message (static part)
        self.marker = Marker()
        self.marker.header.frame_id = 'base_link'
        self.marker.ns = "multi_point"
        self.marker.id = 0
        self.marker.type = Marker.SPHERE_LIST
        self.marker.action = Marker.ADD

        self.marker.scale.x = 0.05
        self.marker.scale.y = 0.05
        self.marker.scale.z = 0.05

        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0

        # Add the points
        for (x, y, z) in self.points_xyz:
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = z
            self.marker.points.append(pt)

        self.get_logger().info(f"Loaded {len(self.points_xyz)} red points")

    def timer_callback(self):
        """Publish marker repeatedly so RViz2 keeps it alive."""
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.marker)


def main(args=None):
    rclpy.init(args=args)

    # ---- Your list of points (unchanged) ----
    two_points = [
        (-0.024, -0.408, 0.887),
        (0.121, -0.692, 0.585),  
        (0.098, -0.685, 0.605),      
    ]

    node = RedPointsPublisher(two_points)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
