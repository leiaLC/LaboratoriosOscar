# reactive_navigation/reactive_navigation/reactive_navigation_node.py

import math
from enum import Enum

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class NavigationState(Enum):
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4


class ReactiveNavigationNode(Node):
    def __init__(self):
        super().__init__('reactive_navigation_node')

        # Subscriber a LaserScan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publisher a cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer de control
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Último scan recibido
        self.latest_scan = None

        # Estado actual
        self.state = NavigationState.FORWARD

        # Parámetros de seguridad
        self.obstacle_threshold = 0.8
        self.stop_threshold = 0.35

        # Velocidades
        self.forward_speed = 0.2
        self.turn_speed = 0.6

        self.get_logger().info('Reactive Navigation Node started.')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def get_sector_min_distance(self, ranges, start_idx, end_idx):
        left = []
        front = []
        right = []

        for i, d in enumerate(self.latest_scan.ranges):

            if not math.isfinite(d):
                continue

            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment

            # DERECHA (-120° a -20°)
            if -2.094 < angle < -0.35:
                right.append(d)

            # FRENTE (-20° a 20°)
            elif -0.35 <= angle <= 0.35:
                front.append(d)

            # IZQUIERDA (20° a 120°)
            elif 0.35 < angle < 2.094:
                left.append(d)

        left_min = min(left) if left else float('inf')
        front_min = min(front) if front else float('inf')
        right_min = min(right) if right else float('inf')

        return left_min, front_min, right_min
    
    def evaluate_state(self):
        if self.latest_scan is None:
            return NavigationState.STOP

        ranges = self.latest_scan.ranges
        total_ranges = len(ranges)

        if total_ranges == 0:
            return NavigationState.STOP

        # Dividir scan en sectores simples
        # Ajuste inicial genérico:
        # izquierda: primer tercio
        # frente: tercio central
        # derecha: último tercio
        left_start = 0
        left_end = total_ranges // 3

        left_min, front_min, right_min = self.get_sector_min_distance(ranges, left_start, left_end)

        self.get_logger().info(
            f'left: {left_min:.2f}, front: {front_min:.2f}, right: {right_min:.2f}',
            throttle_duration_sec=1.0
        )

        # Lógica de estados
        if front_min < self.stop_threshold:
            return NavigationState.STOP

        if front_min < self.obstacle_threshold:
            if left_min > right_min:
                return NavigationState.TURN_LEFT
            else:
                return NavigationState.TURN_RIGHT

        return NavigationState.FORWARD

    def control_loop(self):
        self.state = self.evaluate_state()

        cmd = Twist()

        if self.state == NavigationState.FORWARD:
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0

        elif self.state == NavigationState.TURN_LEFT:
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_speed

        elif self.state == NavigationState.TURN_RIGHT:
            cmd.linear.x = 0.0
            cmd.angular.z = -self.turn_speed

        elif self.state == NavigationState.STOP:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info(
            f'Current state: {self.state.name}',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
        