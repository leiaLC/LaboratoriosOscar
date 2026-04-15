#!/home/zuriel_tov/yolo_env/bin/python
import rclpy
from rclpy.node import Node 
from rclpy.time import Time


from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2 
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # --- Parameter for point cloud topic ---
        self.declare_parameter('cloud_topic', '/kinect/points')
        cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.get_logger().info(f'[vision] Using cloud_topic: {cloud_topic}')

        # --- Subscribe to PointCloud2 ---
        self.subscription = self.create_subscription(
            PointCloud2,
            cloud_topic,
            self.cloud_callback,
            10
        )

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')

        self.get_logger().info("[vision] Waiting for point cloud...")

    # ======================================
    # POINT CLOUD CALLBACK
    # ======================================
    def cloud_callback(self, msg: PointCloud2):
        # Read points (x, y, z, rgb)
        points = list(point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "rgb"),
            reshape_organized_cloud=True,
#           skip_nans=True
        ))

        # Convert to numpy
        points = np.array(points)

        # Extract RGB and create image
        if points.shape[2] == 4:  # x,y,z,rgb
            rgb_values = points[:, :, 3].astype(np.uint32)
            height, width = rgb_values.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    rgb = rgb_values[i, j]
                    r = (rgb >> 16) & 0xFF
                    g = (rgb >> 8) & 0xFF
                    b = rgb & 0xFF
                    rgb_image[i, j] = [r, g, b]

            # Run YOLO detection
            results = self.model(rgb_image)
            self.get_logger().info(f"[vision] YOLO detections: {len(results[0].boxes)} objects")

        # Just inspect
        self.get_logger().info(f"[vision] num points: {points.shape}")

        if len(points) > 0:
            self.get_logger().info(f"[vision] first point: {points[0,0]}")
            self.get_logger().info("[vision] PointCloud2 message received")


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


