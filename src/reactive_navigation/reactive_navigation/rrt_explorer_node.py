#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rrt_explorer_node.py  —  versión corregida

FIXES respecto a la versión anterior:
    1. Filtra fronteras cercanas al robot (min_frontier_dist). Antes el robot
       se quedaba oscilando entre fronteras a <1 m que apenas eran el borde
       del mapa ya conocido.
    2. Detección de llegada más robusta: usa distancia real + un cooldown
       para que no replanee un path nuevo en el mismo tick que llega al goal.
    3. No replanea inmediatamente cuando un path chico termina: espera a que
       pot_fields haya terminado su maniobra (hay una pausa pequeña).

FSM:
    WAIT_MAP ──▶ PLAN ──▶ FOLLOW ──▶ (waypoint OK) ──▶ FOLLOW siguiente
                  │                        │
                  │                        └──▶ (atorado) ──▶ RECOVERY
                  │
                  └──▶ (sin fronteras) ──▶ DONE
"""

import math
import random
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import tf2_ros


class State(Enum):
    WAIT_MAP = 0
    PLAN = 1
    FOLLOW = 2
    RECOVERY = 3
    DONE = 4


# ============================================================================
# RRT
# ============================================================================
class RRTNode:
    __slots__ = ('x', 'y', 'parent')
    def __init__(self, x, y, parent=None):
        self.x = x; self.y = y; self.parent = parent


class RRT:
    def __init__(self, grid, resolution, origin,
                 free_thr=50, step=0.5, goal_bias=0.15,
                 max_iter=3000, goal_tol=0.5, robot_radius=0.25):
        self.grid = grid
        self.res = resolution
        self.ox, self.oy = origin
        self.H, self.W = grid.shape
        self.free_thr = free_thr
        self.step = step
        self.bias = goal_bias
        self.max_iter = max_iter
        self.goal_tol = goal_tol
        self.inflate = max(1, int(math.ceil(robot_radius / resolution)))

    def w2g(self, x, y):
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.W and 0 <= gy < self.H

    def is_free(self, x, y):
        gx, gy = self.w2g(x, y)
        if not self.in_bounds(gx, gy):
            return False
        r = self.inflate
        x0, x1 = max(0, gx - r), min(self.W, gx + r + 1)
        y0, y1 = max(0, gy - r), min(self.H, gy + r + 1)
        return not np.any(self.grid[y0:y1, x0:x1] > self.free_thr)

    def segment_free(self, x1, y1, x2, y2, n=10):
        for t in np.linspace(0.0, 1.0, n):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self.is_free(x, y):
                return False
        return True

    def sample(self, goal):
        if random.random() < self.bias:
            return goal
        x = self.ox + random.random() * self.W * self.res
        y = self.oy + random.random() * self.H * self.res
        return x, y

    @staticmethod
    def nearest(nodes, x, y):
        best_i, best_d = 0, float('inf')
        for i, n in enumerate(nodes):
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def steer(self, frm, tx, ty):
        dx, dy = tx - frm.x, ty - frm.y
        d = math.hypot(dx, dy)
        if d <= self.step:
            return tx, ty
        k = self.step / d
        return frm.x + dx * k, frm.y + dy * k

    def plan(self, start, goal):
        nodes = [RRTNode(start[0], start[1])]
        for _ in range(self.max_iter):
            sx, sy = self.sample(goal)
            i = self.nearest(nodes, sx, sy)
            near = nodes[i]
            nx, ny = self.steer(near, sx, sy)
            if not self.segment_free(near.x, near.y, nx, ny):
                continue
            nodes.append(RRTNode(nx, ny, i))
            if math.hypot(nx - goal[0], ny - goal[1]) < self.goal_tol:
                path, k = [], len(nodes) - 1
                while k is not None:
                    path.append((nodes[k].x, nodes[k].y))
                    k = nodes[k].parent
                path.reverse()
                return self.shortcut(path)
        return None

    def shortcut(self, path):
        if len(path) < 3:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.segment_free(path[i][0], path[i][1],
                                     path[j][0], path[j][1], n=15):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out


# ============================================================================
# Frontier detection
# ============================================================================
def find_frontiers(grid, free_thr=50, min_cluster=8):
    H, W = grid.shape
    free = (grid >= 0) & (grid <= free_thr)
    unknown = (grid < 0)
    up = np.zeros_like(unknown); up[:-1, :] = unknown[1:, :]
    dn = np.zeros_like(unknown); dn[1:, :] = unknown[:-1, :]
    lf = np.zeros_like(unknown); lf[:, :-1] = unknown[:, 1:]
    rt = np.zeros_like(unknown); rt[:, 1:] = unknown[:, :-1]
    mask = free & (up | dn | lf | rt)

    visited = np.zeros_like(mask, dtype=bool)
    clusters = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        cells = []
        while stack:
            y, x = stack.pop()
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            cells.append((y, x))
            if y + 1 < H: stack.append((y + 1, x))
            if y - 1 >= 0: stack.append((y - 1, x))
            if x + 1 < W: stack.append((y, x + 1))
            if x - 1 >= 0: stack.append((y, x - 1))
        if len(cells) >= min_cluster:
            cy = sum(c[0] for c in cells) / len(cells)
            cx = sum(c[1] for c in cells) / len(cells)
            clusters.append((int(cx), int(cy), len(cells)))
    return clusters


# ============================================================================
# Nodo
# ============================================================================
class RRTExplorer(Node):

    def __init__(self):
        super().__init__('rrt_explorer')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('free_threshold', 50)
        self.declare_parameter('min_cluster', 8)
        self.declare_parameter('rrt_step', 0.5)
        self.declare_parameter('rrt_max_iter', 3000)
        self.declare_parameter('rrt_goal_tol', 0.5)
        self.declare_parameter('robot_radius', 0.25)
        # IMPORTANTE: pot_fields se detiene a d<1.0, entonces esto debe ser > 1.0
        self.declare_parameter('waypoint_reached_dist', 1.2)
        # NUEVO: no perseguir fronteras demasiado cercanas (evita oscilación)
        self.declare_parameter('min_frontier_dist', 2.0)
        self.declare_parameter('stuck_time', 18.0)
        self.declare_parameter('stuck_distance', 0.20)
        self.declare_parameter('republish_period', 1.0)
        # NUEVO: pausa entre planes para que pot_fields se "calme" al llegar
        self.declare_parameter('replan_cooldown', 2.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.state = State.WAIT_MAP
        self.occ: Optional[OccupancyGrid] = None
        self.path: List[Tuple[float, float]] = []
        self.path_idx = 0
        self.plan_fails = 0

        # tracking de atoramiento
        self.last_pose = None
        self.last_move_time = self.get_clock().now()

        # cooldown después de llegar al último waypoint del path
        self.cooldown_until = self.get_clock().now()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.goal_pub = self.create_publisher(PointStamped, '/clicked_point', goal_qos)
        self.path_pub = self.create_publisher(Marker, '/rrt_explorer/path', 10)
        self.front_pub = self.create_publisher(Marker, '/rrt_explorer/frontiers', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/rrt_explorer/goal', 10)

        self.create_timer(0.5, self.fsm_tick)
        self.create_timer(self.get_parameter('republish_period').value,
                          self.republish_goal)

        self.get_logger().info(
            f'rrt_explorer started (map_frame={self.map_frame}, '
            f'base_frame={self.base_frame}).'
        )

    # ------------------------------------------------------------------
    def map_cb(self, msg):
        if self.occ is None:
            self.get_logger().info(
                f'First /map received ({msg.info.width}x{msg.info.height} '
                f'@ {msg.info.resolution:.3f} m/cell).'
            )
        self.occ = msg

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        now = self.get_clock().now()
        if self.last_pose is None:
            self.last_pose = (pos.x, pos.y)
            self.last_move_time = now
            return
        dx = pos.x - self.last_pose[0]
        dy = pos.y - self.last_pose[1]
        if math.hypot(dx, dy) > self.get_parameter('stuck_distance').value:
            self.last_pose = (pos.x, pos.y)
            self.last_move_time = now

    # ------------------------------------------------------------------
    def get_robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.3)
            )
            return t.transform.translation.x, t.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}',
                                   throttle_duration_sec=5.0)
            return None

    def grid_to_world(self, gx, gy):
        info = self.occ.info
        wx = info.origin.position.x + (gx + 0.5) * info.resolution
        wy = info.origin.position.y + (gy + 0.5) * info.resolution
        return wx, wy

    def is_stuck(self):
        elapsed = (self.get_clock().now() - self.last_move_time).nanoseconds / 1e9
        return elapsed > self.get_parameter('stuck_time').value

    def publish_goal(self, x, y):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.point.x = float(x); msg.point.y = float(y); msg.point.z = 0.0
        self.goal_pub.publish(msg)

        m = Marker()
        m.header = msg.header
        m.ns = 'goal'; m.id = 0; m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        self.goal_marker_pub.publish(m)

    def republish_goal(self):
        if self.state != State.FOLLOW or not self.path:
            return
        if self.path_idx >= len(self.path):
            return
        wx, wy = self.path[self.path_idx]
        self.publish_goal(wx, wy)

    # ------------------------------------------------------------------
    def pick_frontier(self, robot_xy):
        info = self.occ.info
        grid = np.array(self.occ.data, dtype=np.int16
                        ).reshape(info.height, info.width)
        clusters = find_frontiers(
            grid,
            free_thr=self.get_parameter('free_threshold').value,
            min_cluster=self.get_parameter('min_cluster').value,
        )
        self.publish_frontier_markers(clusters)
        if not clusters:
            return None

        rx, ry = robot_xy
        min_d = self.get_parameter('min_frontier_dist').value

        # FIX: filtrar fronteras demasiado cerca (causan oscilación)
        valid = []
        for cx, cy, size in clusters:
            wx, wy = self.grid_to_world(cx, cy)
            d = math.hypot(wx - rx, wy - ry)
            if d < min_d:
                continue  # frontera muy cerca, la ignoramos
            valid.append((wx, wy, size, d))

        if not valid:
            self.get_logger().info(
                f'Todas las fronteras ({len(clusters)}) están a <{min_d:.1f}m. '
                'Explorando localmente...'
            )
            # Fallback: si todas están cerca, tomar la más lejana de las cercanas
            # (mejor que decir "DONE" prematuramente)
            best_d = 0
            best = None
            for cx, cy, size in clusters:
                wx, wy = self.grid_to_world(cx, cy)
                d = math.hypot(wx - rx, wy - ry)
                if d > best_d:
                    best_d, best = d, (wx, wy)
            return best

        # Del conjunto válido, preferir cercana y grande
        best, best_score = None, float('inf')
        for wx, wy, size, d in valid:
            score = d - 0.3 * math.log1p(size)
            if score < best_score:
                best_score, best = score, (wx, wy)

        self.get_logger().info(
            f'{len(valid)}/{len(clusters)} fronteras válidas, '
            f'best goal ({best[0]:.2f}, {best[1]:.2f})'
        )
        return best

    # ------------------------------------------------------------------
    def fsm_tick(self):
        self.get_logger().info(
            f'FSM={self.state.name} | path_len={len(self.path)} idx={self.path_idx}',
            throttle_duration_sec=3.0
        )

        if self.state == State.WAIT_MAP:
            if self.occ is None:
                return
            if self.get_robot_xy() is None:
                return
            self.get_logger().info('Map + TF ready → PLAN')
            self.state = State.PLAN

        elif self.state == State.PLAN:
            self.do_plan()

        elif self.state == State.FOLLOW:
            self.do_follow()

        elif self.state == State.RECOVERY:
            self.do_recovery()

    # ----- PLAN -----
    def do_plan(self):
        # FIX: respetar cooldown tras llegar a un goal anterior
        now = self.get_clock().now()
        if now < self.cooldown_until:
            return

        robot = self.get_robot_xy()
        if robot is None:
            return
        goal = self.pick_frontier(robot)
        if goal is None:
            self.get_logger().info('Sin fronteras → exploración completa')
            self.state = State.DONE
            return

        info = self.occ.info
        grid = np.array(self.occ.data, dtype=np.int16
                        ).reshape(info.height, info.width)
        rrt = RRT(
            grid,
            resolution=info.resolution,
            origin=(info.origin.position.x, info.origin.position.y),
            free_thr=self.get_parameter('free_threshold').value,
            step=self.get_parameter('rrt_step').value,
            max_iter=self.get_parameter('rrt_max_iter').value,
            goal_tol=self.get_parameter('rrt_goal_tol').value,
            robot_radius=self.get_parameter('robot_radius').value,
        )
        path = rrt.plan(robot, goal)

        if path is None or len(path) < 2:
            self.plan_fails += 1
            self.get_logger().warn(f'RRT falló (#{self.plan_fails})')
            if self.plan_fails >= 3:
                self.get_logger().warn('Fallback: goal directo (sin RRT)')
                self.path = [robot, goal]
                self.path_idx = 1
                self.plan_fails = 0
                self.state = State.FOLLOW
                self.publish_goal(*goal)
                self.publish_path_marker(self.path)
                self.last_move_time = self.get_clock().now()
            return

        self.plan_fails = 0
        self.path = path
        self.path_idx = 1
        self.publish_path_marker(path)
        self.publish_goal(*self.path[self.path_idx])
        self.last_move_time = self.get_clock().now()
        self.state = State.FOLLOW
        self.get_logger().info(
            f'Path: {len(path)} waypoints. Yendo a '
            f'({self.path[self.path_idx][0]:.2f}, {self.path[self.path_idx][1]:.2f})'
        )

    # ----- FOLLOW -----
    def do_follow(self):
        if self.is_stuck():
            self.get_logger().warn('Atorado → RECOVERY')
            self.state = State.RECOVERY
            return
        robot = self.get_robot_xy()
        if robot is None:
            return
        if self.path_idx >= len(self.path):
            self.state = State.PLAN
            return
        wx, wy = self.path[self.path_idx]
        d = math.hypot(wx - robot[0], wy - robot[1])
        if d < self.get_parameter('waypoint_reached_dist').value:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self.get_logger().info('Path completado → cooldown + PLAN')
                # FIX: poner cooldown para que pot_fields se asiente antes
                # de mandarle otro goal que quizá esté cerca del actual.
                cooldown = self.get_parameter('replan_cooldown').value
                self.cooldown_until = (
                    self.get_clock().now() + Duration(seconds=cooldown)
                )
                self.state = State.PLAN
                return
            nwx, nwy = self.path[self.path_idx]
            self.publish_goal(nwx, nwy)
            self.last_move_time = self.get_clock().now()
            self.get_logger().info(
                f'→ waypoint {self.path_idx}/{len(self.path)-1}: '
                f'({nwx:.2f}, {nwy:.2f})'
            )

    # ----- RECOVERY -----
    def do_recovery(self):
        self.get_logger().info('Recovery: descartando path y replanteando')
        self.path = []
        self.path_idx = 0
        self.last_move_time = self.get_clock().now()
        # cooldown también en recovery para que el robot se asiente
        cooldown = self.get_parameter('replan_cooldown').value
        self.cooldown_until = self.get_clock().now() + Duration(seconds=cooldown)
        self.state = State.PLAN

    # ------------------------------------------------------------------
    def publish_path_marker(self, path):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'rrt_path'; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.05
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        for x, y in path:
            p = Point(); p.x = x; p.y = y; p.z = 0.05
            m.points.append(p)
        self.path_pub.publish(m)

    def publish_frontier_markers(self, clusters):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'frontiers'; m.id = 0
        m.type = Marker.POINTS; m.action = Marker.ADD
        m.scale.x = 0.15; m.scale.y = 0.15
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        for cx, cy, _ in clusters:
            wx, wy = self.grid_to_world(cx, cy)
            p = Point(); p.x = wx; p.y = wy; p.z = 0.1
            m.points.append(p)
        self.front_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = RRTExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()