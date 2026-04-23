import math
from enum import Enum

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from robotino_interfaces.msg import PersonTracking


def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class State(Enum):
    INIT_SCAN = 1
    EXPLORE = 2
    AVOID_OBSTACLE = 3
    GO_TO_HUMAN = 4


class PersonSeekerNode(Node):
    def __init__(self):
        super().__init__('person_seeker_node')

        # -----------------------------------------------------------
        # Parameter declarations
        # -----------------------------------------------------------
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('camera_fov_deg', 60.0)

        # Forward motion
        self.declare_parameter('forward_speed', 0.22)        # m/s in EXPLORE
        self.declare_parameter('approach_speed', 0.20)       # m/s in GO_TO_HUMAN (aligned)
        self.declare_parameter('approach_speed_turning', 0.10)  # while still turning
        self.declare_parameter('max_angular_speed', 0.8)

        # INIT_SCAN
        self.declare_parameter('scan_angular_speed', 0.5)
        self.declare_parameter('scan_timeout', 20.0)

        # Obstacle detection / avoidance
        self.declare_parameter('front_cone_deg', 25.0)
        self.declare_parameter('side_cone_deg', 60.0)
        self.declare_parameter('obstacle_threshold', 0.70)   # trigger AVOID
        self.declare_parameter('clear_threshold', 1.00)      # exit AVOID
        self.declare_parameter('turn_speed', 0.6)            # rad/s while avoiding

        # GO_TO_HUMAN
        self.declare_parameter('arrival_distance', 0.30)
        self.declare_parameter('heading_tolerance_deg', 15.0)
        self.declare_parameter('heading_curve_deg', 60.0)    # above this, rotate in place
        self.declare_parameter('kw_go', 1.2)
        self.declare_parameter('lost_timeout', 3.0)

        # -----------------------------------------------------------
        # Load parameters
        # -----------------------------------------------------------
        def p(n): return self.get_parameter(n).value

        self.rate = float(p('control_rate'))
        self.fov = math.radians(float(p('camera_fov_deg')))
        self.forward_speed = float(p('forward_speed'))
        self.approach_speed = float(p('approach_speed'))
        self.approach_speed_turning = float(p('approach_speed_turning'))
        self.max_w = float(p('max_angular_speed'))
        self.scan_w = float(p('scan_angular_speed'))
        self.scan_timeout = float(p('scan_timeout'))
        self.front_cone = math.radians(float(p('front_cone_deg')))
        self.side_cone = math.radians(float(p('side_cone_deg')))
        self.obstacle_th = float(p('obstacle_threshold'))
        self.clear_th = float(p('clear_threshold'))
        self.turn_speed = float(p('turn_speed'))
        self.arrival_distance = float(p('arrival_distance'))
        self.heading_tol = math.radians(float(p('heading_tolerance_deg')))
        self.heading_curve = math.radians(float(p('heading_curve_deg')))
        self.kw_go = float(p('kw_go'))
        self.lost_timeout = float(p('lost_timeout'))

        # -----------------------------------------------------------
        # Runtime state
        # -----------------------------------------------------------
        self.state = State.INIT_SCAN
        self.state_entry_time = self.get_clock().now()

        # Scan rotation tracking
        self.scan_prev_yaw = 0.0
        self.scan_total_rotation = 0.0
        self.scan_initialized = False

        # AVOID direction for current episode (+1 left, -1 right)
        self.avoid_direction = None

        # Sensors
        self.last_detection = None
        self.last_detection_time = None
        self.scan_msg = None

        # Odom
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_ready = False

        # Target memory (odom frame)
        self.last_known_x = None
        self.last_known_y = None

        # -----------------------------------------------------------
        # ROS interfaces
        # -----------------------------------------------------------
        self.create_subscription(
            PersonTracking, '/person_tracking', self._person_cb, 10
        )
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 20)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.rate, self._loop)

        self.get_logger().info(
            f'[SEEKER] Ready in INIT_SCAN. '
            f'fwd={self.forward_speed:.2f} m/s | obs_th={self.obstacle_th:.2f} m | '
            f'arrival={self.arrival_distance:.2f} m | lost_timeout={self.lost_timeout:.1f} s'
        )

    # =========================================================
    # Callbacks
    # =========================================================
    def _person_cb(self, msg: PersonTracking):
        self.last_detection = msg
        if msg.detected:
            self.last_detection_time = self.get_clock().now()
            if self.odom_ready and msg.distance > 0.0:
                self._update_last_known(msg)

    def _scan_cb(self, msg: LaserScan):
        self.scan_msg = msg

    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.odom_ready = True

    # =========================================================
    # Helpers
    # =========================================================
    def _update_last_known(self, d: PersonTracking):
        image_cx = d.width / 2.0 if d.width > 0 else 320.0
        angle_rel = ((image_cx - d.x) / image_cx) * (self.fov / 2.0)
        x_rel = d.distance * math.cos(angle_rel)
        y_rel = d.distance * math.sin(angle_rel)
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)
        self.last_known_x = self.robot_x + x_rel * cos_y - y_rel * sin_y
        self.last_known_y = self.robot_y + x_rel * sin_y + y_rel * cos_y

    def _target_in_robot_frame(self):
        """Return (angle_rel [rad], distance [m]) to last_known. (None, None) if no target."""
        if self.last_known_x is None or not self.odom_ready:
            return None, None
        dx = self.last_known_x - self.robot_x
        dy = self.last_known_y - self.robot_y
        distance = math.hypot(dx, dy)
        angle_rel = normalize_angle(math.atan2(dy, dx) - self.robot_yaw)
        return angle_rel, distance

    def _detection_fresh(self) -> bool:
        if self.last_detection is None or self.last_detection_time is None:
            return False
        if not self.last_detection.detected:
            return False
        dt = (self.get_clock().now() - self.last_detection_time).nanoseconds * 1e-9
        return dt < self.lost_timeout

    def _sector_min(self, center_angle: float, half_angle: float) -> float:
        if self.scan_msg is None:
            return float('inf')
        m = float('inf')
        scan = self.scan_msg
        low = center_angle - half_angle
        high = center_angle + half_angle
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if low <= a <= high:
                if r < m:
                    m = r
        return m

    def _front_distance(self):
        return self._sector_min(0.0, self.front_cone)

    def _left_distance(self):
        center = self.front_cone + self.side_cone / 2.0
        return self._sector_min(center, self.side_cone / 2.0)

    def _right_distance(self):
        center = -(self.front_cone + self.side_cone / 2.0)
        return self._sector_min(center, self.side_cone / 2.0)

    def _set_state(self, new_state: State):
        if new_state != self.state:
            self.get_logger().info(f'[SEEKER] {self.state.name} -> {new_state.name}')
            self.state = new_state
            self.state_entry_time = self.get_clock().now()
            if new_state != State.AVOID_OBSTACLE:
                self.avoid_direction = None
            if new_state == State.INIT_SCAN:
                self.scan_initialized = False
                self.scan_total_rotation = 0.0

    def _time_in_state(self):
        return (self.get_clock().now() - self.state_entry_time).nanoseconds * 1e-9

    def _have_target(self):
        return self.last_known_x is not None

    def _after_avoid_state(self):
        return State.GO_TO_HUMAN if (self._have_target() or self._detection_fresh()) else State.EXPLORE

    # =========================================================
    # State handlers
    # =========================================================
    def _handle_init_scan(self):
        if self._detection_fresh():
            self._set_state(State.GO_TO_HUMAN)
            return None
        if not self.odom_ready:
            return Twist()  # wait silently for odom to publish
        if not self.scan_initialized:
            self.scan_prev_yaw = self.robot_yaw
            self.scan_total_rotation = 0.0
            self.scan_initialized = True
        dy = normalize_angle(self.robot_yaw - self.scan_prev_yaw)
        self.scan_total_rotation += abs(dy)
        self.scan_prev_yaw = self.robot_yaw
        if self.scan_total_rotation >= 2.0 * math.pi:
            self.get_logger().info('[SEEKER] Full 360° scan complete, no person found.')
            self._set_state(State.EXPLORE)
            return None
        if self._time_in_state() > self.scan_timeout:
            self.get_logger().warn('[SEEKER] INIT_SCAN timeout, forcing EXPLORE.')
            self._set_state(State.EXPLORE)
            return None
        cmd = Twist()
        cmd.angular.z = self.scan_w
        return cmd

    def _handle_explore(self):
        if self._detection_fresh() or self._have_target():
            self._set_state(State.GO_TO_HUMAN)
            return None
        if self._front_distance() < self.obstacle_th:
            self._set_state(State.AVOID_OBSTACLE)
            return None
        cmd = Twist()
        cmd.linear.x = self.forward_speed
        return cmd

    def _handle_avoid_obstacle(self):
        front = self._front_distance()
        if front >= self.clear_th:
            self._set_state(self._after_avoid_state())
            return None
        if self.avoid_direction is None:
            left = self._left_distance()
            right = self._right_distance()
            target_angle, _ = self._target_in_robot_frame()
            # Decide direction
            if left > right * 1.3:
                self.avoid_direction = +1
            elif right > left * 1.3:
                self.avoid_direction = -1
            elif target_angle is not None:
                self.avoid_direction = +1 if target_angle > 0 else -1
            else:
                self.avoid_direction = +1 if left >= right else -1
            self.get_logger().info(
                f'[AVOID] front={front:.2f} left={left:.2f} right={right:.2f} -> '
                f'turn {"LEFT" if self.avoid_direction > 0 else "RIGHT"}'
            )
        cmd = Twist()
        cmd.angular.z = self.avoid_direction * self.turn_speed
        return cmd

    def _handle_go_to_human(self):
        if not self._have_target():
            self._set_state(State.EXPLORE)
            return None

        # Obstacle check first
        if self._front_distance() < self.obstacle_th:
            self._set_state(State.AVOID_OBSTACLE)
            return None

        target_angle, target_dist = self._target_in_robot_frame()
        if target_dist is None:
            self._set_state(State.EXPLORE)
            return None

        # Arrived?
        if target_dist < self.arrival_distance:
            if self._detection_fresh():
                # Person visible and close — hold position
                return Twist()
            else:
                # Person not visible and we reached the remembered spot — clear and resume EXPLORE
                self.get_logger().info(
                    f'[SEEKER] Arrived at last known (d={target_dist:.2f}m) without '
                    f'fresh detection. Clearing target and resuming EXPLORE.'
                )
                self.last_known_x = None
                self.last_known_y = None
                self._set_state(State.EXPLORE)
                return None

        # Drive toward target with smooth curve
        cmd = Twist()
        cmd.angular.z = max(-self.max_w, min(self.max_w, self.kw_go * target_angle))
        abs_err = abs(target_angle)
        if abs_err < self.heading_tol:
            cmd.linear.x = self.approach_speed
        elif abs_err < self.heading_curve:
            cmd.linear.x = self.approach_speed_turning
        else:
            cmd.linear.x = 0.0  # too off-axis, rotate in place first
        return cmd

    # =========================================================
    # Main loop
    # =========================================================
    def _loop(self):
        cmd = None
        for _ in range(4):
            if self.state == State.INIT_SCAN:
                cmd = self._handle_init_scan()
            elif self.state == State.EXPLORE:
                cmd = self._handle_explore()
            elif self.state == State.AVOID_OBSTACLE:
                cmd = self._handle_avoid_obstacle()
            elif self.state == State.GO_TO_HUMAN:
                cmd = self._handle_go_to_human()
            else:
                cmd = Twist()
            if cmd is not None:
                break
        if cmd is None:
            cmd = Twist()
        self.cmd_pub.publish(cmd)

        front = self._front_distance()
        d = self.last_detection
        person_str = (
            f'{d.distance:.2f}m' if (d is not None and d.detected and d.distance > 0) else '--'
        )
        if self.last_known_x is not None and self.odom_ready:
            _, lkd = self._target_in_robot_frame()
            lk_str = f' lk_d={lkd:.2f}m' if lkd is not None else ''
        else:
            lk_str = ''
        self.get_logger().info(
            f'[{self.state.name}] v={cmd.linear.x:+.2f} w={cmd.angular.z:+.2f} '
            f'front={front:.2f}m person={person_str}{lk_str}',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = PersonSeekerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()