import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from robotino_interfaces.msg import PersonTracking

from cv_bridge import CvBridge
from ultralytics import YOLO


class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')

        self.bridge = CvBridge()

        # Subs
        self.rgb_sub = self.create_subscription(
            Image,
            '/kinect_sim/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/kinect_sim/depth/image_raw',
            self.depth_callback,
            10
        )

        # Pub
        self.person_pub = self.create_publisher(
            PersonTracking,
            '/person_tracking',
            10
        )

        # Modelo YOLO
        self.model = YOLO('yolov8n.pt')

        # Última depth
        self.depth_image = None

        # Control de concurrencia
        self.processing = False

        self.get_logger().info('YOLO Person Detector + Depth iniciado.')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
        except Exception as e:
            self.get_logger().error(f'Error depth: {e}')

    def rgb_callback(self, msg):
        if self.processing:
            return

        if self.depth_image is None:
            return

        self.processing = True

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            h, w = frame.shape[:2]

            results = self.model(frame, verbose=False)

            best_conf = 0.0
            best_cx = 0.0
            best_cy = 0.0
            detected = False

            # ======================
            # YOLO detección
            # ======================
            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # clase 0 = person
                    if cls_id != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    if conf > best_conf:
                        best_conf = conf
                        best_cx = cx
                        best_cy = cy
                        detected = True

            # ======================
            # Depth
            # ======================
            distance = -1.0

            if detected:
                px = int(best_cx)
                py = int(best_cy)

                if 0 <= py < self.depth_image.shape[0] and 0 <= px < self.depth_image.shape[1]:
                    depth_value = self.depth_image[py, px]

                    if math.isfinite(depth_value) and depth_value > 0:
                        distance = float(depth_value)

            # ======================
            # Publicar
            # ======================
            msg_out = PersonTracking()

            msg_out.detected = detected
            msg_out.x = float(best_cx)
            msg_out.y = float(best_cy)
            msg_out.width = float(w)
            msg_out.height = float(h)
            msg_out.confidence = float(best_conf)

            # float32 distance
            msg_out.distance = float(distance)

            self.person_pub.publish(msg_out)

            self.get_logger().info(
                f'det={detected} x={best_cx:.1f} dist={distance:.2f}',
                throttle_duration_sec=1.0
            )

        except Exception as e:
            self.get_logger().error(f'Error RGB: {e}')

        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()