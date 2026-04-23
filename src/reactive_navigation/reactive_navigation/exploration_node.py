import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


# ----------------------------------------------------------------------------
# Estados de la FSM
# ----------------------------------------------------------------------------
class ExplorationState(Enum):
    IDLE = 0          # Esperando primer scan/map
    ROTATE_SCAN = 1   # Girando 360° para mapear inicialmente
    EXPLORE = 2       # Persiguiendo un goal de frontera
    AVOID = 3         # Obstáculo enfrente → desviar
    RECOVERY = 4      # Atorado → retroceder + girar
    DONE = 5          # Ya no quedan fronteras


# ----------------------------------------------------------------------------
# Utilidad: clustering de fronteras con BFS (sin scipy)
# ----------------------------------------------------------------------------
def label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    num = 0
    mask_flat = mask.ravel()

    for start in range(h * w):
        if not mask_flat[start] or labels.flat[start] != 0:
            continue
        num += 1
        # BFS con cola (lista como deque manual)
        stack = [start]
        while stack:
            idx = stack.pop()
            if labels.flat[idx] != 0 or not mask_flat[idx]:
                continue
            labels.flat[idx] = num
            y, x = divmod(idx, w)
            # 4-vecinos
            if x > 0:
                stack.append(idx - 1)
            if x < w - 1:
                stack.append(idx + 1)
            if y > 0:
                stack.append(idx - w)
            if y < h - 1:
                stack.append(idx + w)
    return labels, num


# ----------------------------------------------------------------------------
# Nodo principal
# ----------------------------------------------------------------------------
class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        # ---------------- Parámetros (declarados para que sean tuneables por CLI / launch)
        self.declare_parameter('forward_speed', 0.25)
        self.declare_parameter('turn_speed', 0.6)
        self.declare_parameter('stop_threshold', 0.40)     # pelota de seguridad frontal (m)
        self.declare_parameter('obstacle_threshold', 0.70) # distancia de "cuidado" (m)
        self.declare_parameter('goal_reached_dist', 0.35)  # radio de llegada al goal (m)
        self.declare_parameter('stuck_dist', 0.15)         # menos de X m ...
        self.declare_parameter('stuck_time', 8.0)          # ... en Y s => atorado
        self.declare_parameter('min_frontier_size', 10)    # celdas mínimas por cluster
        self.declare_parameter('free_threshold', 25)       # [0,100] valor máx. libre
        self.declare_parameter('revisit_grid_cell', 0.5)   # discretización para anti-revisita (m)
        self.declare_parameter('revisit_penalty', 1.5)     # peso del castigo por revisita (m equiv)
        self.declare_parameter('planning_period', 1.0)     # periodo de re-planeación (s)
        self.declare_parameter('control_period', 0.1)      # periodo de control (s)
        self.declare_parameter('goal_timeout', 20.0)       # si no llega al goal en Ts → replan

        # Cache de parámetros
        self._reload_params()

        # ---------------- Estado interno
        self.state: ExplorationState = ExplorationState.IDLE
        self.latest_scan: Optional[LaserScan] = None
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_odom: Optional[Odometry] = None

        self.current_goal: Optional[Tuple[float, float]] = None  # en frame map
        self.goal_set_time = self.get_clock().now()

        # Detección de atoramiento
        self.last_pose = None
        self.last_move_time = self.get_clock().now()

        # Anti-revisita (conteo por celda discretizada)
        self.visit_counts: dict = {}

        # Control del RECOVERY
        self.recovery_start_time = None
        self.recovery_phase = 0  # 0 = retroceder, 1 = girar

        # Control del ROTATE_SCAN
        self.rotate_scan_start_time = None
        self.rotate_scan_duration = 2 * math.pi / 0.6 + 1.0  # tiempo para dar una vuelta

        # ---------------- QoS para /map (SLAM Toolbox publica TRANSIENT_LOCAL)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ---------------- Suscripciones
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # ---------------- Publicaciones
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PointStamped, '/exploration/goal', 10)
        self.frontier_pub = self.create_publisher(Marker, '/exploration/frontiers', 10)

        # ---------------- TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Timers
        self.create_timer(self.control_period, self.control_loop)
        self.create_timer(self.planning_period, self.planning_loop)

        self.get_logger().info('Exploration node started.')

  
    def _reload_params(self):
        g = lambda n: self.get_parameter(n).value
        self.forward_speed      = g('forward_speed')
        self.turn_speed         = g('turn_speed')
        self.stop_threshold     = g('stop_threshold')
        self.obstacle_threshold = g('obstacle_threshold')
        self.goal_reached_dist  = g('goal_reached_dist')
        self.stuck_dist         = g('stuck_dist')
        self.stuck_time         = g('stuck_time')
        self.min_frontier_size  = g('min_frontier_size')
        self.free_threshold     = g('free_threshold')
        self.revisit_grid_cell  = g('revisit_grid_cell')
        self.revisit_penalty    = g('revisit_penalty')
        self.planning_period    = g('planning_period')
        self.control_period     = g('control_period')
        self.goal_timeout       = g('goal_timeout')

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg

    def odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    #Lecutra de sensores
    def sector_min_distances(self) -> dict:
        """
        Divide el scan en 5 sectores y devuelve la distancia mínima en cada uno.
        Asume FOV ~240° (Robotino).
        """
        scan = self.latest_scan
        sectors = {'right': [], 'fright': [], 'front': [], 'fleft': [], 'left': []}

        for i, d in enumerate(scan.ranges):
            if not math.isfinite(d) or d < scan.range_min:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            # Rangos en radianes
            if -2.09 < angle <= -1.05:      sectors['right'].append(d)   # [-120°,-60°]
            elif -1.05 < angle <= -0.35:    sectors['fright'].append(d)  # [-60°,-20°]
            elif -0.35 < angle <= 0.35:     sectors['front'].append(d)   # [-20°, 20°]
            elif 0.35 < angle <= 1.05:      sectors['fleft'].append(d)   # [ 20°, 60°]
            elif 1.05 < angle <= 2.09:      sectors['left'].append(d)    # [ 60°,120°]

        return {k: (min(v) if v else float('inf')) for k, v in sectors.items()}

    def get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        """Devuelve (x, y, yaw) del robot en el frame 'map', o None si TF aún no está lista."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except (LookupException, ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        # yaw desde quaternion (sólo rotación alrededor de z)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return t.x, t.y, yaw

    # ========================================================================
    # CAPA DELIBERATIVA: DETECCIÓN DE FRONTERAS
    # ========================================================================
    def compute_frontiers(self) -> List[Tuple[float, float, int]]:
        """
        Devuelve una lista de centroides de clusters de frontera:
        [(world_x, world_y, size_in_cells), ...] en el frame map.
        """
        if self.latest_map is None:
            return []

        m = self.latest_map
        w, h = m.info.width, m.info.height
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y

        grid = np.array(m.data, dtype=np.int16).reshape(h, w)

        free_mask = (grid >= 0) & (grid <= self.free_threshold)
        unknown_mask = (grid < 0)

        # Una celda es frontera si es libre y tiene al menos un vecino desconocido
        frontier = np.zeros_like(free_mask)
        # Shifts 4-conexos — np.roll envuelve pero el borde desconocido raramente importa
        frontier |= free_mask & np.roll(unknown_mask,  1, axis=0)
        frontier |= free_mask & np.roll(unknown_mask, -1, axis=0)
        frontier |= free_mask & np.roll(unknown_mask,  1, axis=1)
        frontier |= free_mask & np.roll(unknown_mask, -1, axis=1)

        if not frontier.any():
            return []

        labels, num = label_connected_components(frontier)

        clusters = []
        for k in range(1, num + 1):
            ys, xs = np.where(labels == k)
            if ys.size < self.min_frontier_size:
                continue
            cy = ys.mean()
            cx = xs.mean()
            wx = ox + (cx + 0.5) * res
            wy = oy + (cy + 0.5) * res
            clusters.append((wx, wy, int(ys.size)))
        return clusters

    def pick_best_frontier(
        self,
        clusters: List[Tuple[float, float, int]],
        robot_xy: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Heurística de Yamauchi extendida:
        score = distancia - bonus_por_tamaño + castigo_por_revisita
        (menor = mejor)
        """
        if not clusters:
            return None

        rx, ry = robot_xy
        best_score = float('inf')
        best_xy = None

        for wx, wy, size in clusters:
            d = math.hypot(wx - rx, wy - ry)
            # Normalización suave del tamaño: log para no sobre-pesar megaclusters
            size_bonus = 0.3 * math.log1p(size)
            revisit = self.revisit_penalty * self.get_visit_count((wx, wy))
            score = d - size_bonus + revisit
            if score < best_score:
                best_score = score
                best_xy = (wx, wy)
        return best_xy

    # ========================================================================
    # ANTI-REVISITA (grid discretizado de celdas visitadas)
    # ========================================================================
    def _discretize(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        cell = self.revisit_grid_cell
        return (int(math.floor(xy[0] / cell)), int(math.floor(xy[1] / cell)))

    def mark_visit(self, xy: Tuple[float, float]):
        key = self._discretize(xy)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

    def get_visit_count(self, xy: Tuple[float, float]) -> int:
        return self.visit_counts.get(self._discretize(xy), 0)

    # ========================================================================
    # LOOP DELIBERATIVO (1 Hz)
    # ========================================================================
    def planning_loop(self):
        if self.latest_map is None:
            return
        pose = self.get_robot_pose_in_map()
        if pose is None:
            return
        rx, ry, _ = pose

        # Marcar celda actual como visitada
        self.mark_visit((rx, ry))

        clusters = self.compute_frontiers()
        self.publish_frontier_marker(clusters)

        if not clusters:
            if self.state != ExplorationState.IDLE:
                self.get_logger().info('No remaining frontiers → DONE')
                self.state = ExplorationState.DONE
            return

        # ¿Necesitamos (re)planear?
        need_new_goal = (
            self.current_goal is None or
            (self.get_clock().now() - self.goal_set_time) >
                Duration(seconds=self.goal_timeout)
        )

        if need_new_goal:
            goal = self.pick_best_frontier(clusters, (rx, ry))
            if goal is not None:
                self.current_goal = goal
                self.goal_set_time = self.get_clock().now()
                self.get_logger().info(
                    f'New goal: ({goal[0]:.2f}, {goal[1]:.2f})  '
                    f'frontiers={len(clusters)}'
                )
                self.publish_goal_marker(goal)

    # ========================================================================
    # LOOP REACTIVO (10 Hz)
    # ========================================================================
    def control_loop(self):
        if self.latest_scan is None or self.latest_odom is None:
            self.publish_twist(0.0, 0.0)
            return

        sectors = self.sector_min_distances()

        # --- Actualizar detección de atoramiento (usa odom, no TF, por robustez)
        pos = self.latest_odom.pose.pose.position
        now = self.get_clock().now()
        if self.last_pose is None:
            self.last_pose = (pos.x, pos.y)
            self.last_move_time = now
        else:
            if math.hypot(pos.x - self.last_pose[0], pos.y - self.last_pose[1]) > self.stuck_dist:
                self.last_pose = (pos.x, pos.y)
                self.last_move_time = now

        stuck = (
            self.state in (ExplorationState.EXPLORE, ExplorationState.AVOID) and
            (now - self.last_move_time) > Duration(seconds=self.stuck_time)
        )

        # --- Transiciones globales (tienen prioridad)
        if stuck and self.state != ExplorationState.RECOVERY:
            self.get_logger().warn('Robot stuck — entering RECOVERY')
            self.state = ExplorationState.RECOVERY
            self.recovery_start_time = now
            self.recovery_phase = 0
            # invalidar goal viejo para forzar replan después
            self.current_goal = None

        # --- Despacho por estado
        if self.state == ExplorationState.IDLE:
            if self.latest_map is not None:
                self.state = ExplorationState.ROTATE_SCAN
                self.rotate_scan_start_time = now
                self.get_logger().info('Initial ROTATE_SCAN to populate map.')
            self.publish_twist(0.0, 0.0)

        elif self.state == ExplorationState.ROTATE_SCAN:
            self.handle_rotate_scan(now)

        elif self.state == ExplorationState.EXPLORE:
            self.handle_explore(sectors)

        elif self.state == ExplorationState.AVOID:
            self.handle_avoid(sectors)

        elif self.state == ExplorationState.RECOVERY:
            self.handle_recovery(now, sectors)

        elif self.state == ExplorationState.DONE:
            self.publish_twist(0.0, 0.0)

        # Log throttled
        self.get_logger().info(
            f'[{self.state.name}] front={sectors["front"]:.2f} '
            f'goal={self.current_goal}',
            throttle_duration_sec=1.0
        )

    # ========================================================================
    # HANDLERS POR ESTADO
    # ========================================================================
    def handle_rotate_scan(self, now):
        elapsed = (now - self.rotate_scan_start_time).nanoseconds / 1e9
        if elapsed > self.rotate_scan_duration:
            self.state = ExplorationState.EXPLORE
            self.get_logger().info('Initial scan complete → EXPLORE')
            return
        self.publish_twist(0.0, self.turn_speed)

    def handle_explore(self, sectors):
        # 1) Obstáculo crítico al frente
        if sectors['front'] < self.stop_threshold:
            self.state = ExplorationState.AVOID
            return

        # 2) ¿Tenemos goal?
        if self.current_goal is None:
            # Sin goal pero sin obstáculo: girar lentamente para encontrar más fronteras
            self.publish_twist(0.0, self.turn_speed * 0.5)
            return

        # 3) ¿Llegamos al goal?
        pose = self.get_robot_pose_in_map()
        if pose is None:
            self.publish_twist(0.0, 0.0)
            return
        rx, ry, yaw = pose
        gx, gy = self.current_goal
        dist = math.hypot(gx - rx, gy - ry)

        if dist < self.goal_reached_dist:
            self.get_logger().info('Goal reached → re-planning')
            self.current_goal = None
            self.publish_twist(0.0, 0.0)
            return

        # 4) Perseguir el goal: controlador proporcional de yaw + avance
        desired_yaw = math.atan2(gy - ry, gx - rx)
        yaw_error = self._wrap_angle(desired_yaw - yaw)

        # Modulación de velocidad por proximidad a obstáculos (suave)
        front_d = sectors['front']
        if front_d < self.obstacle_threshold:
            # Bias para desviar antes de llegar al umbral duro
            if sectors['fleft'] > sectors['fright']:
                yaw_error += 0.6
            else:
                yaw_error -= 0.6

        # Ganancia proporcional de yaw, saturada
        k_yaw = 1.2
        w = max(-self.turn_speed, min(self.turn_speed, k_yaw * yaw_error))

        # Avanzar sólo si el yaw está razonablemente alineado
        if abs(yaw_error) > 0.6:
            v = 0.0  # girar en sitio
        else:
            # Reducir velocidad cerca del goal o de obstáculos
            v = self.forward_speed * min(1.0, dist / 0.8)
            v *= min(1.0, front_d / self.obstacle_threshold)

        self.publish_twist(v, w)

    def handle_avoid(self, sectors):
        # Salir del estado cuando el frente esté libre
        if sectors['front'] > self.obstacle_threshold:
            self.state = ExplorationState.EXPLORE
            return
        # Elegir el lado más despejado combinando ambos sectores laterales
        left_clear  = min(sectors['fleft'],  sectors['left'])
        right_clear = min(sectors['fright'], sectors['right'])
        if left_clear > right_clear:
            self.publish_twist(0.0, self.turn_speed)
        else:
            self.publish_twist(0.0, -self.turn_speed)

    def handle_recovery(self, now, sectors):
        # Fase 0: retroceder 1 s (si hay espacio atrás — aquí asumimos sí, el FOV es 240°)
        # Fase 1: girar ~180°
        elapsed = (now - self.recovery_start_time).nanoseconds / 1e9
        if self.recovery_phase == 0:
            self.publish_twist(-self.forward_speed * 0.6, 0.0)
            if elapsed > 1.2:
                self.recovery_phase = 1
                self.recovery_start_time = now
        else:
            self.publish_twist(0.0, self.turn_speed)
            rot_time = math.pi / self.turn_speed  # tiempo para ~180°
            if elapsed > rot_time:
                # Volver a explorar y forzar replan
                self.state = ExplorationState.EXPLORE
                self.last_move_time = now       # reinicia detección de atoramiento
                self.current_goal = None

    # ========================================================================
    # UTILIDADES
    # ========================================================================
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Envuelve un ángulo al rango [-pi, pi]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def publish_twist(self, v: float, w: float):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    def publish_goal_marker(self, goal: Tuple[float, float]):
        ps = PointStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.point.x, ps.point.y = goal
        self.goal_pub.publish(ps)

    def publish_frontier_marker(self, clusters):
        """Publica un Marker tipo POINTS con todos los centroides — útil en RViz."""
        mk = Marker()
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.header.frame_id = 'map'
        mk.ns = 'frontiers'
        mk.id = 0
        mk.type = Marker.POINTS
        mk.action = Marker.ADD
        mk.scale.x = 0.15
        mk.scale.y = 0.15
        mk.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        for wx, wy, _ in clusters:
            p = Point()
            p.x, p.y, p.z = wx, wy, 0.0
            mk.points.append(p)
        self.frontier_pub.publish(mk)


# ----------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_twist(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()