#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reactive_navigation_node.py

Máquina de estados con ESCALADO DE ESCAPE cuando el robot se atora
repetidamente en la misma zona.

Idea central:
    Un contador cuenta los intentos de "unstuck" dentro de una ventana de
    tiempo. Si el robot consigue moverse libremente por >30s, el contador
    se resetea. Cada nuevo atoramiento dentro de esa ventana sube el nivel:

        Nivel 1  UNSTUCK_SOFT     giro random 2.5 s
        Nivel 2  UNSTUCK_HARD     retroceder + giro 180°
        Nivel 3  UNSTUCK_WANDER   retroceder + giro random fijo + caminar ciego

Inspirado en:
    Arkin, R. C. (1998). Behavior-Based Robotics. MIT Press. (Cap. 4,
    "escape behaviors" con coordinación jerárquica).
"""

import math
import random
from enum import Enum

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class State(Enum):
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4
    UNSTUCK_SOFT = 5
    UNSTUCK_HARD = 6
    UNSTUCK_WANDER = 7


class ReactiveNavigationNode(Node):

    def __init__(self):
        super().__init__('reactive_navigation_node')

        # ---------- velocidades ----------
        self.forward_speed = 0.2
        self.reverse_speed = 0.12
        self.turn_speed = 0.6

        # ---------- umbrales de obstáculo ----------
        self.stop_threshold = 0.35
        self.obstacle_threshold = 0.8
        self.clear_threshold = 1.1          # histéresis: sigue girando hasta esta dist

        # ---------- detección de atoramiento ----------
        self.stuck_distance = 0.15
        self.stuck_time_limit = 6.0

        # ---------- duraciones de escape ----------
        self.soft_duration = 2.5
        self.hard_back_duration = 1.5
        self.hard_turn_duration = math.pi / self.turn_speed  # ~180°
        self.wander_back_duration = 2.0
        self.wander_turn_duration = 3.0
        self.wander_walk_duration = 4.0

        # ---------- ventana para resetear escalado ----------
        self.reset_counter_after = 30.0

        # ---------- estado ----------
        self.state = State.FORWARD
        self.latest_scan = None

        # tracking de movimiento
        self.last_position = None
        self.last_move_time = self.get_clock().now()

        # contador de escapes
        self.unstuck_counter = 0
        self.last_unstuck_time = None
        self.unstuck_phase = 0
        self.unstuck_start_time = None
        self.unstuck_direction = 1.0

        # bias de dirección (para no oscilar izquierda↔derecha)
        self.turn_bias = 0
        self.turn_bias_until = self.get_clock().now()

        # pub/sub
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Reactive navigation with escalating unstuck started.')

    # ------------------------------------------------------------------
    def scan_cb(self, msg):
        self.latest_scan = msg

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        now = self.get_clock().now()

        if self.last_position is None:
            self.last_position = (pos.x, pos.y)
            self.last_move_time = now
            return

        dx = pos.x - self.last_position[0]
        dy = pos.y - self.last_position[1]
        if math.hypot(dx, dy) > self.stuck_distance:
            self.last_position = (pos.x, pos.y)
            self.last_move_time = now

            # ¿Pasó mucho tiempo libre desde el último unstuck? resetear contador
            if self.last_unstuck_time is not None:
                elapsed = (now - self.last_unstuck_time).nanoseconds / 1e9
                if elapsed > self.reset_counter_after and self.unstuck_counter > 0:
                    self.get_logger().info(
                        f'Libre por {elapsed:.0f}s → reset unstuck_counter'
                    )
                    self.unstuck_counter = 0

    # ------------------------------------------------------------------
    def is_stuck(self):
        elapsed = (self.get_clock().now() - self.last_move_time).nanoseconds / 1e9
        return elapsed > self.stuck_time_limit

    def get_sector_distances(self):
        left, front, right = [], [], []
        for i, d in enumerate(self.latest_scan.ranges):
            if not math.isfinite(d):
                continue
            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
            if -2.094 < angle < -0.44:
                right.append(d)
            elif -0.44 <= angle <= 0.44:
                front.append(d)
            elif 0.44 < angle < 2.094:
                left.append(d)
        return (
            min(left) if left else float('inf'),
            min(front) if front else float('inf'),
            min(right) if right else float('inf'),
        )

    def elapsed_in_unstuck(self):
        return (self.get_clock().now() - self.unstuck_start_time).nanoseconds / 1e9

    # ------------------------------------------------------------------
    def trigger_unstuck(self):
        """Dispara un unstuck; el nivel escala con el contador."""
        now = self.get_clock().now()
        self.unstuck_counter += 1
        self.last_unstuck_time = now
        self.unstuck_start_time = now
        self.unstuck_phase = 0
        self.unstuck_direction = random.choice([-1.0, 1.0])

        self.get_logger().warn(f'ATORADO (intento #{self.unstuck_counter})')

        if self.unstuck_counter == 1:
            return State.UNSTUCK_SOFT
        elif self.unstuck_counter == 2:
            self.get_logger().warn('Escalando → UNSTUCK_HARD')
            return State.UNSTUCK_HARD
        else:
            self.get_logger().warn('Escalando → UNSTUCK_WANDER')
            # Mantener en 2 para no crecer infinito
            self.unstuck_counter = 2
            return State.UNSTUCK_WANDER

    # ------------------------------------------------------------------
    def decide_next_state(self):
        if self.latest_scan is None:
            return State.STOP

        # ---- estados de unstuck: continuar hasta completar ----
        if self.state == State.UNSTUCK_SOFT:
            if self.elapsed_in_unstuck() < self.soft_duration:
                return State.UNSTUCK_SOFT
            self.last_move_time = self.get_clock().now()
            return State.FORWARD

        if self.state == State.UNSTUCK_HARD:
            t = self.elapsed_in_unstuck()
            if self.unstuck_phase == 0 and t > self.hard_back_duration:
                self.unstuck_phase = 1
                self.unstuck_start_time = self.get_clock().now()
                return State.UNSTUCK_HARD
            if self.unstuck_phase == 1 and t > self.hard_turn_duration:
                self.last_move_time = self.get_clock().now()
                return State.FORWARD
            return State.UNSTUCK_HARD

        if self.state == State.UNSTUCK_WANDER:
            t = self.elapsed_in_unstuck()
            if self.unstuck_phase == 0 and t > self.wander_back_duration:
                self.unstuck_phase = 1
                self.unstuck_start_time = self.get_clock().now()
                return State.UNSTUCK_WANDER
            if self.unstuck_phase == 1 and t > self.wander_turn_duration:
                self.unstuck_phase = 2
                self.unstuck_start_time = self.get_clock().now()
                return State.UNSTUCK_WANDER
            if self.unstuck_phase == 2 and t > self.wander_walk_duration:
                self.last_move_time = self.get_clock().now()
                return State.FORWARD
            return State.UNSTUCK_WANDER

        # ---- ¿nuevo atoramiento? ----
        if self.is_stuck():
            return self.trigger_unstuck()

        # ---- navegación normal ----
        left_d, front_d, right_d = self.get_sector_distances()

        self.get_logger().info(
            f'L={left_d:.2f} F={front_d:.2f} R={right_d:.2f} '
            f'| {self.state.name} | cnt={self.unstuck_counter}',
            throttle_duration_sec=1.0
        )

        if front_d < self.stop_threshold:
            return State.STOP

        # Histéresis: si ya estamos girando, mantener el giro hasta estar claramente libre
        if self.state in (State.TURN_LEFT, State.TURN_RIGHT):
            if front_d < self.clear_threshold:
                return self.state
            return State.FORWARD

        if front_d < self.obstacle_threshold:
            now = self.get_clock().now()
            # Respetar bias de dirección si sigue vigente
            if now < self.turn_bias_until and self.turn_bias != 0:
                return State.TURN_LEFT if self.turn_bias > 0 else State.TURN_RIGHT

            # Elegir lado más despejado y fijar bias
            if left_d > right_d:
                self.turn_bias = 1
            else:
                self.turn_bias = -1
            self.turn_bias_until = now + rclpy.duration.Duration(seconds=2.0)
            return State.TURN_LEFT if self.turn_bias > 0 else State.TURN_RIGHT

        self.turn_bias = 0
        return State.FORWARD

    # ------------------------------------------------------------------
    def control_loop(self):
        self.state = self.decide_next_state()

        cmd = Twist()

        if self.state == State.FORWARD:
            cmd.linear.x = self.forward_speed
        elif self.state == State.TURN_LEFT:
            cmd.angular.z = self.turn_speed
        elif self.state == State.TURN_RIGHT:
            cmd.angular.z = -self.turn_speed
        elif self.state == State.STOP:
            pass  # ceros

        elif self.state == State.UNSTUCK_SOFT:
            cmd.angular.z = self.turn_speed * self.unstuck_direction

        elif self.state == State.UNSTUCK_HARD:
            if self.unstuck_phase == 0:
                cmd.linear.x = -self.reverse_speed
            else:
                cmd.angular.z = self.turn_speed * self.unstuck_direction

        elif self.state == State.UNSTUCK_WANDER:
            if self.unstuck_phase == 0:
                cmd.linear.x = -self.reverse_speed
            elif self.unstuck_phase == 1:
                cmd.angular.z = self.turn_speed * self.unstuck_direction
            else:
                cmd.linear.x = self.forward_speed

        self.cmd_vel_pub.publish(cmd)


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