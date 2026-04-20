#!/usr/bin/env python3
"""
person_follower_node.py

Subscribes:
  /person_tracking   (robotino_interfaces/msg/PersonTracking)   — from person_tracker_node
  /scan              (sensor_msgs/msg/LaserScan)                 — lidar safety

Publishes:
  /cmd_vel           (geometry_msgs/msg/Twist)

Optional service client:
  /face_recog        (robotino_interfaces/srv/FaceRecog)         — identify target person

Behavior:
  - Proportional control: angular to center person horizontally, linear to hold target distance.
  - Safety layer: if lidar front-cone reads closer than safety_distance, linear motion is clamped to 0.
  - If no detection for lost_timeout seconds -> stop.
  - If enable_face_id=true -> requires periodic confirmation of target_name via /face_recog
    before allowing motion. Between confirmations, continues following visually.

Parameters:
  target_distance        [m]    desired following distance                  default 1.2
  distance_deadband      [m]    don't move linearly if |error| < deadband   default 0.15
  max_linear_speed       [m/s]                                               default 0.25
  max_angular_speed      [rad/s]                                             default 0.8
  kp_linear              linear gain                                         default 0.6
  kp_angular             angular gain                                        default 1.5
  safety_distance        [m]   obstacle threshold (front cone +/-20 deg)     default 0.40
  lost_timeout           [s]   stop after this long without detection        default 1.5
  enable_face_id         bool  gate motion on face identity                  default False
  target_name            str   name registered via /face_train               default ""
  face_check_period      [s]   how often to re-check identity                default 2.0
  face_confidence_min    float minimum confidence for identity match         default 0.5
  allow_backward         bool  if true, back up when person is too close     default False
"""

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from robotino_interfaces.msg import PersonTracking
from robotino_interfaces.srv import FaceRecog


class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower_node')

        # --- Parameters ---
        self.declare_parameter('target_distance', 1.2)
        self.declare_parameter('distance_deadband', 0.15)
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('kp_linear', 0.6)
        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('safety_distance', 0.40)
        self.declare_parameter('lost_timeout', 1.5)
        self.declare_parameter('enable_face_id', False)
        self.declare_parameter('target_name', '')
        self.declare_parameter('face_check_period', 2.0)
        self.declare_parameter('face_confidence_min', 0.5)
        self.declare_parameter('allow_backward', False)

        self.target_distance = float(self.get_parameter('target_distance').value)
        self.distance_deadband = float(self.get_parameter('distance_deadband').value)
        self.max_v = float(self.get_parameter('max_linear_speed').value)
        self.max_w = float(self.get_parameter('max_angular_speed').value)
        self.kp_v = float(self.get_parameter('kp_linear').value)
        self.kp_w = float(self.get_parameter('kp_angular').value)
        self.safety_distance = float(self.get_parameter('safety_distance').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        self.enable_face_id = bool(self.get_parameter('enable_face_id').value)
        self.target_name = str(self.get_parameter('target_name').value)
        self.face_check_period = float(self.get_parameter('face_check_period').value)
        self.face_conf_min = float(self.get_parameter('face_confidence_min').value)
        self.allow_backward = bool(self.get_parameter('allow_backward').value)

        # --- State ---
        self.last_detection = None
        self.last_detection_time = None
        self.front_min_distance = float('inf')
        # If face_id disabled, treat the person as confirmed target from the start.
        self.target_confirmed = not self.enable_face_id
        self.last_face_check_time = None
        self.face_call_inflight = False

        # --- ROS interfaces ---
        self.create_subscription(
            PersonTracking, '/person_tracking', self._person_cb, 10
        )
        self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        if self.enable_face_id:
            if not self.target_name:
                self.get_logger().error(
                    '[FOLLOW] enable_face_id=True but target_name is empty. '
                    'Pass -p target_name:="name_registered_via_face_train"'
                )
            self.face_client = self.create_client(FaceRecog, '/face_recog')
            self.get_logger().info(
                f'[FOLLOW] Face ID enabled. Target name: "{self.target_name}"'
            )
        else:
            self.face_client = None

        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'[FOLLOW] Ready. target_dist={self.target_distance:.2f}m '
            f'safety={self.safety_distance:.2f}m '
            f'lost_timeout={self.lost_timeout:.1f}s'
        )

    # ----- Callbacks -----
    def _person_cb(self, msg: PersonTracking):
        self.last_detection = msg
        if msg.detected:
            self.last_detection_time = self.get_clock().now()

    def _scan_cb(self, msg: LaserScan):
        """Compute min distance in the front cone (+/- 20 deg)."""
        front_ranges = []
        for i, d in enumerate(msg.ranges):
            if not math.isfinite(d) or d <= 0.0:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if -0.35 <= angle <= 0.35:  # ~+/- 20 deg
                front_ranges.append(d)
        self.front_min_distance = (
            min(front_ranges) if front_ranges else float('inf')
        )

    # ----- Control loop -----
    def _control_loop(self):
        cmd = Twist()  # zero by default

        if self.last_detection is None or self.last_detection_time is None:
            self.cmd_pub.publish(cmd)
            return

        # "Lost": no detected=True message for lost_timeout seconds
        dt = (self.get_clock().now() - self.last_detection_time).nanoseconds * 1e-9
        if dt > self.lost_timeout or not self.last_detection.detected:
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                'Person lost, stopping.', throttle_duration_sec=1.0
            )
            return

        # Identity gate
        if self.enable_face_id:
            self._maybe_check_identity()
            if not self.target_confirmed:
                self.cmd_pub.publish(cmd)
                self.get_logger().info(
                    f'Target "{self.target_name}" not confirmed, stopping.',
                    throttle_duration_sec=1.0
                )
                return

        d = self.last_detection
        image_cx = d.width / 2.0 if d.width > 0 else 320.0

        # --- Angular control: center person in image ---
        # err_x > 0 when person is to the right of center -> need to turn right -> w < 0
        err_x = (d.x - image_cx) / max(1.0, image_cx)  # normalized [-1, 1]
        w = -self.kp_w * err_x
        w = max(-self.max_w, min(self.max_w, w))

        # --- Linear control: hold target distance ---
        v = 0.0
        if d.distance > 0.0:  # valid depth
            err_d = d.distance - self.target_distance
            if abs(err_d) > self.distance_deadband:
                v = self.kp_v * err_d
                v = max(-self.max_v, min(self.max_v, v))
                if v < 0 and not self.allow_backward:
                    v = 0.0

        # --- Safety: lidar front-cone obstacle blocks linear motion ---
        if v > 0.0 and self.front_min_distance < self.safety_distance:
            v = 0.0
            self.get_logger().warn(
                f'Obstacle at {self.front_min_distance:.2f}m, blocking linear motion.',
                throttle_duration_sec=1.0
            )

        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'det x={d.x:.0f}/{image_cx:.0f} dist={d.distance:.2f}m '
            f'-> v={v:.2f} w={w:.2f}',
            throttle_duration_sec=0.5
        )

    # ----- Face identity (async, non-blocking) -----
    def _maybe_check_identity(self):
        if self.face_client is None or self.face_call_inflight:
            return
        if not self.face_client.service_is_ready():
            return
        now = self.get_clock().now()
        if (self.last_face_check_time is not None and
                (now - self.last_face_check_time).nanoseconds * 1e-9
                < self.face_check_period):
            return

        self.last_face_check_time = now
        self.face_call_inflight = True

        req = FaceRecog.Request()
        req.name_request = []          # recognize anyone present
        req.min_confidence = float(self.face_conf_min)
        future = self.face_client.call_async(req)
        future.add_done_callback(self._face_response_cb)

    def _face_response_cb(self, future):
        self.face_call_inflight = False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().warn(f'[FOLLOW] face_recog call failed: {e}')
            return
        if res is None:
            self.target_confirmed = False
            return

        found = any(
            name == self.target_name and conf >= self.face_conf_min
            for name, conf in zip(res.name_response, res.confidence)
        )
        if found and not self.target_confirmed:
            self.get_logger().info(
                f'[FOLLOW] Target "{self.target_name}" CONFIRMED.'
            )
        elif not found and self.target_confirmed:
            self.get_logger().info(
                f'[FOLLOW] Target "{self.target_name}" LOST in last face check.'
            )
        self.target_confirmed = found


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure robot stops on shutdown
        try:
            stop = Twist()
            node.cmd_pub.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
