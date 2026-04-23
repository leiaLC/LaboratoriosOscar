# Mapping & Autonomous Exploration — Robotino

Este paquete (`reactive_navigation`) contiene dos enfoques para que el Robotino
explore un entorno desconocido mientras genera un mapa de ocupación. Ambos se
apoyan en el launch `slam_mapping.launch.py` del paquete `robotino_webots`, que
ya venía en el repo y levanta todo el stack base en un solo comando (Webots +
controlador del Robotino + SLAM Toolbox + RViz).

## Requisitos

- Paquete `robotino_webots` compilado (ya incluye el launch de SLAM con su
  configuración y RViz).
- Paquete `robot_movement` compilado (proporciona el nodo `pot_fields`, usado
  por el Approach 2).

## Levantar el stack base

Una sola terminal arranca Webots, el controlador del Robotino, SLAM Toolbox y
RViz ya configurado:

```bash
cd ~/laboratorios_ws
source install/setup.bash
ros2 launch robotino_webots slam_mapping.launch.py
```

A partir de aquí, en otra terminal se corre cualquiera de los dos approaches
descritos abajo.

---

## Approach 1 — Máquina de estados con escalado de escape

Archivo: `reactive_navigation/reactive_navigation_node.py`

### Qué se modificó

Partimos de un nodo reactivo básico que ya existía en el paquete y le agregamos
un **mecanismo de recuperación** para que no se quedara atorado
indefinidamente en esquinas con muebles. Los cambios principales fueron:

- Se agregó un suscriptor a `/odom` para medir si el robot realmente se está
  moviendo (antes solo miraba el `/scan`).
- Se agregó un **contador de intentos de desatascamiento** (`unstuck_counter`)
  que crece cada vez que el robot detecta que lleva varios segundos sin
  moverse.
- Se dividió el comportamiento de "salir del atasco" en **tres niveles de
  agresividad** que se van activando según el valor del contador.
- Se agregó **histéresis** en la decisión de girar y un **bias de dirección**
  para que el robot no oscile entre TURN_LEFT ↔ TURN_RIGHT frente a una misma
  esquina.

### Cómo funciona a grandes rasgos

La máquina de estados sigue siendo la misma idea simple (avanzar, girar cuando
hay obstáculo, parar si está muy cerca), pero ahora cuando el robot lleva más
de 6 segundos sin desplazarse al menos 15 cm, entra a un estado de
desatascamiento cuyo nivel depende del contador:

1. **Primer atasco → giro aleatorio corto** (2.5 s en una dirección al azar).
2. **Segundo atasco consecutivo → retrocede y da media vuelta** (útil cuando
   está metido en una esquina real).
3. **Tercero o más → retrocede, gira mucho tiempo y camina ciego 4 s** para
   salir completamente de la zona problemática.

Si el robot logra moverse libremente durante 30 segundos seguidos, el contador
se resetea a cero. Así, atoramientos aislados no escalan hasta los niveles
agresivos; solo lo hacen cuando son **consecutivos** (señal clara de que
estamos bloqueados en la misma zona).

### Cómo ejecutar

Con el stack base ya corriendo, en otra terminal:

```bash
source ~/laboratorios_ws/install/setup.bash
ros2 run reactive_navigation reactive_navigation_node
```

### Comportamiento observado

Para este **no se modificó el mundo de Webots**, el robot se prueba
con el apartamento completo, mesa central incluida.

El robot recorre la mayor parte del apartamento correctamente. Hay una esquina
específica donde no siempre llega bien, no porque la máquina de estados falle,
sino porque **el robot no consigue posicionarse lo suficientemente cerca de esa
esquina** para que el LIDAR capture los bordes con detalle. El mapa generado
en esa zona queda menos definido que en el resto.

---

## Approach 2 — RRT + Potential Fields

Archivos:
- `reactive_navigation/rrt_explorer_node.py` (nuevo, la parte de planeación).
- `robot_movement/pot_fields.py` (ya existía, se reutiliza para el control).

### Qué se modificó

Aquí se creó un nodo completamente nuevo (`rrt_explorer_node`) que planea
trayectorias, y se decidió **reutilizar sin modificar** el nodo `pot_fields` que
ya traía el repo para mover al robot hacia los waypoints.

Los cambios principales:

- Se añadió el nodo `rrt_explorer_node` que se suscribe al `/map` de SLAM
  Toolbox para detectar fronteras, planear con RRT y publicar waypoints.
- Se aprovechó el hecho de que `pot_fields` ya escuchaba `/clicked_point` como
  su atractor (originalmente se usaba clickeando en RViz) — ahora el RRT
  publica ahí programáticamente y `pot_fields` ni se entera de que el goal
  viene de un planner.
- Para este approach **se quitó la mesa central** del mundo de Webots
  (`robotino_apartment.wbt`). Las patas delgadas de esa mesa causaban dos
  problemas: SLAM las veía de forma intermitente (no las marcaba bien como
  pared) y además RRT planeaba caminos que pasaban "por dentro" de la mesa
  porque el mapa decía que esa zona estaba libre. Quitándola se pudo validar
  el funcionamiento del planner sin ese ruido.

### Cómo funciona a grandes rasgos

El flujo es: el nodo lee el mapa de SLAM, encuentra las **zonas frontera**
(lugares donde el mapa ya conocido termina y empieza lo desconocido), elige una
frontera como objetivo, corre **RRT** para planear un camino hasta ahí, y le va
pasando los waypoints del camino a `pot_fields` uno por uno. `pot_fields` es el
que realmente mueve al robot (potential fields: atracción al waypoint, repulsión
de obstáculos cercanos).

La máquina de estados del explorer tiene cinco estados: espera mapa, planea,
sigue el camino, recupera si se atora, y termina cuando ya no hay fronteras.

Se agregaron también algunos mecanismos de robustez: un **cooldown** de 2 s
entre planes para que `pot_fields` se asiente al llegar, un **filtro de
fronteras muy cercanas** (<2 m del robot) para evitar oscilación, detección de
"sin progreso" al waypoint actual, y una **blacklist temporal** para goals que
resultaron inalcanzables.

### Cómo ejecutar

Con el stack base ya corriendo, en dos terminales adicionales:

**Terminal — Potential fields:**
```bash
source ~/laboratorios_ws/install/setup.bash
ros2 run robot_movement pot_fields
```

**Terminal — RRT explorer:**
```bash
source ~/laboratorios_ws/install/setup.bash
ros2 run reactive_navigation rrt_explorer_node
```

### Visualización en RViz

La ventana de RViz que abre `slam_mapping.launch.py` ya muestra el mapa y el
LaserScan. Para ver lo que hace el explorer hay que agregar manualmente estos
displays (todos con frame fijo en `map`).

### Comportamiento observado

Con la mesa central quitada, el robot explora mucho más eficientemente y genera
caminos coherentes hacia cada zona sin explorar. Sin embargo, queda una esquina
que sigue sin mapearse bien. Al igual que en el Approach 1, la causa no es el
algoritmo sino que **el robot no alcanza a llegar físicamente lo
suficientemente cerca** de esa esquina para que el LIDAR la detalle — la
frontera queda identificada pero los waypoints hacia ella no consiguen
aproximarlo lo necesario antes de que el planner cambie de objetivo.

---

## Guardar el mapa generado

Mientras cualquiera de los dos approaches está corriendo, desde otra terminal
se puede guardar el mapa actual:

```bash
cd ~/laboratorios_ws
mkdir -p maps
ros2 run nav2_map_server map_saver_cli -f maps/mi_mapa
```

Esto genera dos archivos: `maps/mi_mapa.pgm` (imagen del mapa) y
`maps/mi_mapa.yaml` (metadatos con resolución y origen). Ambos son el formato
estándar de ROS y se pueden cargar después en Nav2 o volver a abrir en RViz.

---

## Demo

Video del robot explorando:

[`../demos/exploration_demo.webm`](../demos/exploration_demo.webm)
