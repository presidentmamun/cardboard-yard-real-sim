"""CardboardYard: 3D MuJoCo wheel-loader environment.

A wheel loader works in a recycling warehouse: drive to the cardboard pile,
scoop cardboard with the bucket, carry it to the hopper and dump it there.

Coordinates (metres, z up): back wall at y=+14, pile against the wall around
(6, 11), hopper (open steel container) at (-8, 2), loader starts near (6, 1)
facing +y. A worker walks a fixed beat on the right side; touching him ends
the episode.

Loader model: the rear frame moves on planar joints (x, y, yaw) driven by
velocity actuators that follow articulated-vehicle kinematics. The front
frame is joined by a real articulation hinge, and the boom and bucket are
real hinges with position servos. Cardboard sheets are free rigid bodies, so
scooping, carrying and dumping are real contact physics.

Action (4,), each in [-1, 1]:
    0 throttle  -> target speed (x MAX_SPEED, negative = reverse)
    1 steer     -> target articulation angle (x MAX_STEER)
    2 boom      -> boom target rate (positive = raise)
    3 bucket    -> bucket target rate (positive = curl back / hold load)
"""

from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------- geometry
PILE_CENTER = np.array([6.0, 11.4])
PILE_HALF = np.array([1.9, 1.6])  # half extent of the pile footprint
HOPPER_CENTER = np.array([-8.0, 2.0])
HOPPER_HALF = np.array([1.8, 1.7])  # inner half extent
HOPPER_WALL_H = 1.3
BACK_WALL_Y = 13.6
COLUMN_X = (-14.0, -6.0, 2.0, 10.0, 18.0)
COLUMN_Y = (8.0, -7.0)
N_WORKERS = 3
OBSTACLE_RADIUS = (0.8, 0.75, 0.85, 0.9, 0.85)  # pallet, drums, trolley, cones, bale
N_OBSTACLES = len(OBSTACLE_RADIUS)
PARK = np.array([30.0, -25.0, -5.0])  # where unused people/obstacles wait

# loader
L_FRONT = 1.2  # articulation -> front axle
L_REAR = 1.2  # articulation -> rear axle
WHEEL_R = 0.55
BOOM_PIVOT = np.array([0.6, 0.0, 1.75])  # in front-frame coords
BOOM_LEN = 2.6
BOOM_RANGE = (-0.95, 0.56)  # + lowers the boom
BUCKET_RANGE = (-1.7, 2.3)  # relative to boom; world pitch = boom + bucket
MAX_SPEED = 3.0  # m/s
MAX_STEER = 0.65  # rad
BOOM_RATE = 0.3  # rad/s at full action
BUCKET_RATE = 0.6  # rad/s at full action
PIECE_DAMPING = 1.0  # N s/m and N m s/rad on every cardboard free joint
MAX_ACCEL = 1.0  # m/s^2, ground speed change limit (no jerky starts / stops)
STEER_RATE = 0.8  # rad/s, articulation swing limit
# goal-reaching eval (world model planning): goal counts as reached within these
GOAL_POS_TOL = 0.5  # m, articulation point
GOAL_YAW_TOL = 0.25  # rad, rear frame heading
GOAL_ARM_TOL = 0.15  # rad, boom and bucket joints

PIECE_HALF = (0.30, 0.22, 0.025)

YELLOW = '0.93 0.72 0.08 1'
DARK = '0.12 0.12 0.13 1'


def _lookat_xyaxes(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return ' '.join(f'{v:.4f}' for v in (*x, *y))


def _box(name, pos, size, rgba=None, material=None, group='static', vgroup=None, **extra):
    """MJCF box geom. group: static / loader / visual / piece."""
    col = {
        'static': 'contype="1" conaffinity="3"',
        'loader': 'contype="2" conaffinity="1"',
        'visual': 'contype="0" conaffinity="0"',
    }[group]
    look = f'material="{material}"' if material else f'rgba="{rgba}"'
    if vgroup is not None:
        extra['group'] = vgroup  # geom group 3 = roof, hidden in the top view
    ex = ' '.join(f'{k}="{v}"' for k, v in extra.items())
    nm = f'name="{name}"' if name else ''
    p = ' '.join(f'{v:.4f}' for v in pos)
    s = ' '.join(f'{v:.4f}' for v in size)
    return f'<geom {nm} type="box" pos="{p}" size="{s}" {look} {col} {ex}/>'


def build_xml(n_pieces: int, L=None, M=None) -> str:
    """Build the full MJCF scene as a string (L: Layout, M: Machine)."""
    L = L or LAYOUT_V0
    M = M or MACHINE_V0
    L_FRONT, L_REAR, WHEEL_R, BOOM_PIVOT, BOOM_LEN = M.l_front, M.l_rear, M.wheel_r, M.boom_pivot, M.boom_len
    BOOM_RANGE, BUCKET_RANGE, MAX_STEER = M.boom_range, M.bucket_range, M.max_steer
    BACK_WALL_Y, COLUMN_X, COLUMN_Y = L.back_wall_y, L.column_x, L.column_y
    HOPPER_CENTER, HOPPER_HALF, HOPPER_WALL_H = L.hopper_center, L.hopper_half, L.hopper_wall_h
    g = []  # static world geoms

    # --- building: back wall, side wall, roof structure
    g.append(_box('back_wall', (1, BACK_WALL_Y + 0.2, 4), (22, 0.2, 4), material='wall'))
    g.append(_box('back_wall_low', (1, BACK_WALL_Y + 0.05, 0.9), (22, 0.05, 0.9), material='concrete_wall'))
    g.append(_box('left_wall', (L.left_wall[0], L.left_wall[1], 4), (0.2, L.left_wall[2], 4), material='wall'))
    for cx in COLUMN_X:
        rusty = cx > 0
        mat = 'rust' if rusty else 'green'
        for cy in COLUMN_Y:
            g.append(_box(f'col_{cx:g}_{cy:g}', (cx, cy, 3.5), (L.column_half, L.column_half, 3.5), material=mat))
            # H-beam flanges for looks
            g.append(_box(None, (cx, cy, 3.5), (0.3, 0.04, 3.5), material=mat, group='visual'))
        # cross beam along y at roof height
        g.append(_box(None, (cx, 0.5, 7.1), (0.18, 8.0, 0.3), material='rust', group='visual', vgroup=3))
    for cy in COLUMN_Y:
        g.append(_box(None, (2, cy, 6.6), (17, 0.2, 0.35), material='rust', group='visual', vgroup=3))
    # purlins + corrugated roof
    for k in range(-7, 15, 2):
        g.append(_box(None, (2, k, 7.55), (19, 0.06, 0.12), material='steel_dark', group='visual', vgroup=3))
    g.append(_box(None, (2, 3, 8.0), (20, 11.5, 0.05), material='roof', group='visual', vgroup=3))
    # wall girts on the back wall
    for z in (2.2, 3.6, 5.0):
        g.append(_box(None, (1, BACK_WALL_Y - 0.05, z), (21, 0.06, 0.08), material='steel_dark', group='visual'))

    # --- hopper (target container)
    hx, hy = HOPPER_CENTER
    ex, ey = HOPPER_HALF
    t = 0.08
    h = HOPPER_WALL_H / 2
    g.append(_box('hopper_floor', (hx, hy, 0.02), (ex, ey, 0.02), material='hopper'))
    g.append(_box('hopper_w0', (hx + ex + t, hy, h), (t, ey + 2 * t, h), material='hopper'))
    g.append(_box('hopper_w1', (hx - ex - t, hy, h), (t, ey + 2 * t, h), material='hopper'))
    g.append(_box('hopper_w2', (hx, hy + ey + t, h), (ex, t, h), material='hopper'))
    g.append(_box('hopper_w3', (hx, hy - ey - t, h), (ex, t, h), material='hopper'))
    g.append(_box(None, (hx + ex + t, hy, HOPPER_WALL_H + 0.03), (t + 0.02, ey + 0.2, 0.04), material='stripe', group='visual'))

    # --- paper rolls on the far left (static)
    for i, x in enumerate(np.arange(-19.0, -15.0, 1.3)):
        for j, y in enumerate(np.arange(5.0, 13.0, 1.3)):
            for k in range(2):
                g.append(
                    f'<geom type="cylinder" pos="{x:.2f} {y:.2f} {0.7 + 1.4 * k:.2f}" size="0.6 0.7" '
                    f'material="paper_roll" contype="1" conaffinity="3"/>'
                )

    # --- background cardboard heaps (visual only, far from the work area)
    rng = np.random.default_rng(7)
    for i in range(70):
        x = rng.uniform(11.5, 20.0)
        y = rng.uniform(10.5, 13.4)
        z = rng.uniform(0.05, max(0.1, 1.6 - 0.12 * abs(x - 15.5)))
        a, b, c = rng.uniform(-0.5, 0.5, 3)
        g.append(
            f'<geom type="box" pos="{x:.2f} {y:.2f} {z:.2f}" size="0.32 0.24 0.02" '
            f'euler="{a:.2f} {b:.2f} {c:.2f}" material="cardboard{i % 3}" contype="0" conaffinity="0"/>'
        )
    # floor litter (visual only)
    for i in range(140):
        x, y = rng.uniform(-18, 20), rng.uniform(-12, 13)
        g.append(
            f'<geom type="box" pos="{x:.2f} {y:.2f} 0.004" size="{rng.uniform(0.05, 0.18):.2f} '
            f'{rng.uniform(0.03, 0.08):.2f} 0.003" euler="0 0 {rng.uniform(0, 3.14):.2f}" '
            f'material="cardboard{i % 3}" contype="0" conaffinity="0"/>'
        )

    world_static = '\n    '.join(g)

    # --- cardboard pieces (free bodies)
    pieces = []
    px, py, pz = PIECE_HALF
    for i in range(n_pieces):
        pieces.append(
            f'<body name="piece{i}" pos="0 0 {-5 - i:.1f}">'
            # light damping calms contact jitter of stacked cardboard (~3 s linear time constant)
            f'<joint name="piece{i}_j" type="free" damping="{PIECE_DAMPING}"/>'
            f'<geom name="piece{i}_g" type="box" size="{px} {py} {pz}" mass="1.5" '
            f'material="cardboard{i % 3}" contype="1" conaffinity="3" friction="0.9 0.01 0.001" '
            f'solref="0.01 1"/></body>'
        )
    pieces_xml = '\n    '.join(pieces)

    # --- loader
    ly = 'group="loader"'
    def lb(name, pos, size, color, **kw):
        if color in ('yellow', 'dark', 'glass', 'bucket_steel', 'rim', 'tire'):
            return _box(name, pos, size, material=color, group='loader', **kw)
        return _box(name, pos, size, rgba=color, group='loader', **kw)

    def wheel(name, x, y):
        return (
            f'<body name="{name}" pos="{x} {y} {WHEEL_R + 0.01}">'
            f'<joint name="{name}_spin" type="hinge" axis="0 1 0" damping="0"/>'
            f'<geom name="{name}_tire" type="cylinder" size="{WHEEL_R} 0.21" euler="1.5708 0 0" '
            f'material="tire" contype="2" conaffinity="1" mass="150"/>'
            f'<geom type="cylinder" size="0.30 0.215" euler="1.5708 0 0" material="rim" '
            f'contype="0" conaffinity="0" mass="1"/>'
            f'<geom type="box" size="0.40 0.22 0.06" material="tire" contype="0" conaffinity="0" mass="1"/>'
            f'<geom type="box" size="0.06 0.22 0.40" material="tire" contype="0" conaffinity="0" mass="1"/>'
            f'</body>'
        )

    chase = _lookat_xyaxes((-9.0, -3.5, 5.0), (2.5, 0.0, 1.0))
    cab_cam = _lookat_xyaxes((-0.4, 0.0, 2.3), (6.0, 0.0, 0.0))
    rear_cam = _lookat_xyaxes((-1.0, 0.0, 2.35), (-8.0, 0.0, 0.0))

    loader = f"""
    <body name="rear" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="0"/>
      <camera name="chase" pos="-9 -3.5 5" xyaxes="{chase}" fovy="55"/>
      <camera name="cab" pos="-0.4 0 2.3" xyaxes="{cab_cam}" fovy="75"/>
      <camera name="rear" pos="-1.0 0 2.35" xyaxes="{rear_cam}" fovy="75"/>
      {lb('rear_frame', (-1.0, 0, 0.75), (1.0, 0.45, 0.25), 'dark', mass=1200)}
      {lb('engine', (-1.75, 0, 1.25), (0.7, 0.62, 0.42), 'yellow', mass=1200)}
      {lb('counterweight', (-2.42, 0, 0.95), (0.12, 0.7, 0.35), 'yellow', mass=600)}
      {lb('fender_rl', (-1.2, 0.85, 1.12), (0.65, 0.24, 0.04), 'yellow', mass=10)}
      {lb('fender_rr', (-1.2, -0.85, 1.12), (0.65, 0.24, 0.04), 'yellow', mass=10)}
      {lb('cab_base', (-0.45, 0, 1.15), (0.55, 0.62, 0.15), 'yellow', mass=200)}
      {lb('cab_glass', (-0.45, 0, 1.85), (0.5, 0.58, 0.55), 'glass', mass=100)}
      {lb('cab_post_fl', (0.06, 0.6, 1.85), (0.05, 0.05, 0.56), 'yellow', mass=5)}
      {lb('cab_post_fr', (0.06, -0.6, 1.85), (0.05, 0.05, 0.56), 'yellow', mass=5)}
      {lb('cab_post_rl', (-0.96, 0.6, 1.85), (0.05, 0.05, 0.56), 'yellow', mass=5)}
      {lb('cab_post_rr', (-0.96, -0.6, 1.85), (0.05, 0.05, 0.56), 'yellow', mass=5)}
      {lb('cab_roof', (-0.45, 0, 2.44), (0.6, 0.66, 0.05), 'yellow', mass=50)}
      {lb('exhaust', (-1.5, 0.35, 1.85), (0.05, 0.05, 0.25), 'dark', mass=5)}
      {wheel('wheel_rl', -L_REAR, 0.85)}
      {wheel('wheel_rr', -L_REAR, -0.85)}
      <body name="front" pos="0 0 0">
        <joint name="steer" type="hinge" axis="0 0 1" range="{-MAX_STEER - 0.05} {MAX_STEER + 0.05}" damping="2000"/>
        {lb('front_frame', (0.75, 0, 0.75), (0.75, 0.42, 0.22), 'dark', mass=900)}
        {lb('boom_tower_l', (0.6, 0.5, 1.25), (0.14, 0.06, 0.55), 'yellow', mass=80)}
        {lb('boom_tower_r', (0.6, -0.5, 1.25), (0.14, 0.06, 0.55), 'yellow', mass=80)}
        {lb('fender_fl', (L_FRONT - 0.05, 0.85, 1.12), (0.5, 0.24, 0.04), 'yellow', mass=10)}
        {lb('fender_fr', (L_FRONT - 0.05, -0.85, 1.12), (0.5, 0.24, 0.04), 'yellow', mass=10)}
        {wheel('wheel_fl', L_FRONT, 0.85)}
        {wheel('wheel_fr', L_FRONT, -0.85)}
        <body name="boom" pos="{BOOM_PIVOT[0]} 0 {BOOM_PIVOT[1 + 1]}">
          <joint name="boom" type="hinge" axis="0 1 0" range="{BOOM_RANGE[0]} {BOOM_RANGE[1]}" damping="30000" armature="500"/>
          {lb('boom_l', (BOOM_LEN / 2, 0.58, 0), (BOOM_LEN / 2, 0.07, 0.13), 'yellow', mass=200)}
          {lb('boom_r', (BOOM_LEN / 2, -0.58, 0), (BOOM_LEN / 2, 0.07, 0.13), 'yellow', mass=200)}
          {lb('boom_cross', (1.5, 0, 0.0), (0.12, 0.55, 0.1), 'yellow', mass=60)}
          {lb('tilt_link', (1.3, 0, 0.35), (0.9, 0.08, 0.08), 'yellow', mass=40)}
          <body name="bucket" pos="{BOOM_LEN} 0 0">
            <joint name="bucket" type="hinge" axis="0 1 0" range="{BUCKET_RANGE[0]} {BUCKET_RANGE[1]}" damping="15000" armature="100"/>
            {lb('bucket_bottom', (0.55, 0, -0.33), (0.6, 1.08, 0.03), 'bucket_steel', mass=120, friction='1.0 0.01 0.001')}
            {lb('bucket_back', (0.0, 0, 0.12), (0.04, 1.08, 0.48), 'bucket_steel', mass=90)}
            {lb('bucket_side_l', (0.5, 1.06, 0.05), (0.55, 0.03, 0.38), 'bucket_steel', mass=40)}
            {lb('bucket_side_r', (0.5, -1.06, 0.05), (0.55, 0.03, 0.38), 'bucket_steel', mass=40)}
            {lb('bucket_lip', (1.13, 0, -0.3), (0.05, 1.08, 0.05), 'dark', mass=40)}
            {lb('bucket_grill', (-0.02, 0, 0.75), (0.03, 0.9, 0.16), 'dark', mass=20)}
            <site name="bucket_center" pos="0.55 0 -0.05" size="0.05" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>
    </body>"""

    photo = _lookat_xyaxes((0.5, -4.8, 1.7), (5.5, 7.0, 2.4))
    top = _lookat_xyaxes((-0.5, 3.0, 24.0), (-0.5, 5.6, 0.0))

    # --- people (mocap, moved by the env)
    looks = [('worker_suit', 'vest'), ('suit2', 'vest2'), ('worker_suit', 'vest2')]
    people = []
    for i in range(N_WORKERS):
        suit, vest = looks[i % len(looks)]
        st = 'contype="1" conaffinity="3"'
        people.append(f"""
    <body name="worker{i}" mocap="true" pos="{30 + 2 * i} -20 -5">
      <geom name="worker{i}_leg_l" type="capsule" fromto="0 0.1 0.08 0 0.1 0.88" size="0.09" material="{suit}" {st}/>
      <geom name="worker{i}_leg_r" type="capsule" fromto="0 -0.1 0.08 0 -0.1 0.88" size="0.09" material="{suit}" {st}/>
      <geom name="worker{i}_torso" type="capsule" fromto="0 0 0.95 0 0 1.42" size="0.2" material="{suit}" {st}/>
      <geom name="worker{i}_vest" type="box" pos="0.02 0 1.2" size="0.19 0.22 0.2" material="{vest}" contype="0" conaffinity="0"/>
      <geom name="worker{i}_arm_l" type="capsule" fromto="0 0.27 1.4 0.05 0.29 0.95" size="0.06" material="{suit}" {st}/>
      <geom name="worker{i}_arm_r" type="capsule" fromto="0 -0.27 1.4 0.05 -0.29 0.95" size="0.06" material="{suit}" {st}/>
      <geom name="worker{i}_head" type="sphere" pos="0 0 1.62" size="0.11" material="skin" {st}/>
      <geom name="worker{i}_helmet" type="sphere" pos="0 0 1.68" size="0.125" material="helmet" contype="0" conaffinity="0"/>
    </body>""")
    people_xml = '\n'.join(people)

    # --- obstacles (mocap, placed at reset; parked under the floor when unused)
    st = 'contype="1" conaffinity="3"'
    obs = [
        # 0 pallet stack
        f'<geom name="obs0_a" type="box" pos="0 0 0.07" size="0.6 0.5 0.07" material="wood" {st}/>'
        f'<geom name="obs0_b" type="box" pos="0 0 0.5" size="0.55 0.45 0.36" material="cardboard1" {st}/>'
        f'<geom name="obs0_c" type="box" pos="0.05 0 1.0" size="0.45 0.4 0.15" material="cardboard2" {st}/>',
        # 1 oil drums
        f'<geom name="obs1_a" type="cylinder" pos="-0.33 0.2 0.45" size="0.29 0.45" material="drum_blue" {st}/>'
        f'<geom name="obs1_b" type="cylinder" pos="0.33 0.2 0.45" size="0.29 0.45" material="drum_red" {st}/>'
        f'<geom name="obs1_c" type="cylinder" pos="0 -0.35 0.45" size="0.29 0.45" material="drum_blue" {st}/>',
        # 2 trolley
        f'<geom name="obs2_a" type="box" pos="0 0 0.4" size="0.7 0.45 0.04" material="cart" {st}/>'
        f'<geom name="obs2_b" type="box" pos="0 0 0.2" size="0.66 0.4 0.2" group="4" {st}/>'
        f'<geom name="obs2_c" type="box" pos="-0.68 0 0.75" size="0.03 0.42 0.35" material="cart" {st}/>'
        f'<geom name="obs2_d" type="box" pos="0.1 0 0.6" size="0.4 0.35 0.16" material="cardboard0" {st}/>',
        # 3 traffic cones in a row
        ''.join(
            f'<geom name="obs3_{k}" type="cylinder" pos="{dx} 0 0.3" size="0.15 0.3" material="cone" {st}/>'
            f'<geom type="box" pos="{dx} 0 0.02" size="0.22 0.22 0.02" material="cone" contype="0" conaffinity="0"/>'
            for k, dx in enumerate((-0.65, 0.0, 0.65))
        ),
        # 4 paper bale
        f'<geom name="obs4_a" type="box" pos="0 0 0.55" size="0.62 0.55 0.55" material="bale" {st}/>'
        f'<geom type="box" pos="0 0 0.55" size="0.63 0.03 0.56" material="steel_dark" contype="0" conaffinity="0"/>'
        f'<geom type="box" pos="0.3 0 0.55" size="0.03 0.56 0.56" material="steel_dark" contype="0" conaffinity="0"/>',
    ]
    obstacles_xml = '\n'.join(
        f'<body name="obstacle{i}" mocap="true" pos="{30 + 3 * i} -25 -5">{g}</body>'
        for i, g in enumerate(obs)
    )

    return f"""
<mujoco model="cardboard_yard">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" iterations="40" ls_iterations="20" cone="pyramidal">
    <flag warmstart="enable"/>
  </option>
  <size memory="200M"/>
  <visual>
    <global offwidth="1280" offheight="960" azimuth="120" elevation="-20"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <map znear="0.02" zfar="200" haze="0.15"/>
    <rgba haze="0.75 0.78 0.82 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.6 0.65 0.7" width="256" height="256"/>
    <texture name="concrete" type="2d" builtin="flat" rgb1="0.27 0.27 0.28" rgb2="0.22 0.22 0.23" mark="random" random="0.06" markrgb="0.55 0.48 0.38" width="1024" height="1024"/>
    <material name="floor" texture="concrete" texrepeat="12 12" reflectance="0.02" roughness="0.9"/>
    <texture name="corrugated" type="2d" builtin="gradient" rgb1="0.78 0.78 0.76" rgb2="0.52 0.52 0.5" width="64" height="64"/>
    <material name="wall" texture="corrugated" texrepeat="60 1" rgba="0.95 0.95 0.95 1"/>
    <material name="roof" texture="corrugated" texrepeat="1 80" rgba="0.6 0.6 0.62 1"/>
    <material name="concrete_wall" rgba="0.62 0.62 0.6 1"/>
    <material name="green" rgba="0.32 0.62 0.45 1" specular="0.2"/>
    <material name="rust" rgba="0.42 0.28 0.18 1" specular="0.1"/>
    <material name="steel_dark" rgba="0.32 0.3 0.28 1"/>
    <material name="hopper" rgba="0.18 0.32 0.55 1" specular="0.3"/>
    <material name="stripe" rgba="0.95 0.8 0.1 1"/>
    <material name="paper_roll" rgba="0.86 0.8 0.68 1"/>
    <material name="cardboard0" rgba="0.72 0.55 0.33 1"/>
    <material name="cardboard1" rgba="0.80 0.64 0.42 1"/>
    <material name="cardboard2" rgba="0.62 0.47 0.28 1"/>
    <material name="yellow" rgba="{YELLOW}" specular="0.4" shininess="0.5"/>
    <material name="dark" rgba="{DARK}"/>
    <material name="glass" rgba="0.15 0.2 0.25 0.55" specular="0.8" shininess="0.9"/>
    <material name="bucket_steel" rgba="0.62 0.64 0.66 1" specular="0.6" shininess="0.6"/>
    <material name="tire" rgba="0.08 0.08 0.08 1"/>
    <material name="rim" rgba="{YELLOW}"/>
    <material name="worker_suit" rgba="0.1 0.13 0.25 1"/>
    <material name="helmet" rgba="0.98 0.78 0.05 1"/>
    <material name="skin" rgba="0.85 0.68 0.55 1"/>
    <material name="vest" rgba="0.95 0.45 0.05 1"/>
    <material name="vest2" rgba="0.75 0.95 0.15 1"/>
    <material name="suit2" rgba="0.35 0.35 0.38 1"/>
    <material name="wood" rgba="0.55 0.4 0.22 1"/>
    <material name="drum_blue" rgba="0.1 0.3 0.7 1" specular="0.4"/>
    <material name="drum_red" rgba="0.7 0.12 0.1 1" specular="0.4"/>
    <material name="cone" rgba="1.0 0.42 0.05 1"/>
    <material name="cart" rgba="0.45 0.47 0.5 1" specular="0.5"/>
    <material name="bale" rgba="0.82 0.74 0.58 1"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="10 -20 30" dir="-0.3 0.6 -1" diffuse="0.55 0.55 0.52" castshadow="true"/>
    <light name="fill" directional="true" pos="-10 10 20" dir="0.3 -0.3 -1" diffuse="0.2 0.2 0.22" castshadow="false"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="floor" contype="1" conaffinity="3" friction="0.8 0.01 0.001"/>
    <camera name="photo" pos="0.5 -4.8 1.7" xyaxes="{photo}" fovy="70"/>
    <camera name="top" pos="-0.5 3.0 24" xyaxes="{top}" fovy="60"/>
    {world_static}
    {people_xml}
    {obstacles_xml}
    {loader}
    {pieces_xml}
  </worldbody>
  <actuator>
    <velocity name="drive_x" joint="base_x" kv="120000" forcerange="-70000 70000"/>
    <velocity name="drive_y" joint="base_y" kv="120000" forcerange="-70000 70000"/>
    <velocity name="drive_yaw" joint="base_yaw" kv="400000" forcerange="-400000 400000"/>
    <position name="steer" joint="steer" kp="400000" kv="30000" ctrlrange="{-MAX_STEER} {MAX_STEER}"/>
    <position name="boom" joint="boom" kp="900000" kv="60000" ctrlrange="{BOOM_RANGE[0]} {BOOM_RANGE[1]}"/>
    <position name="bucket" joint="bucket" kp="200000" kv="12000" ctrlrange="{BUCKET_RANGE[0]} {BUCKET_RANGE[1]}"/>
  </actuator>
</mujoco>
"""


class Machine:
    """The loader's size and motion limits. MACHINE_V0 is swm/CardboardYard-v0 (the module constants above);
    CardboardYardReal's WA100 profile uses the Komatsu WA100-8 numbers.

    Arm angles: boom = angle of the boom pivot -> bucket hinge line below horizontal (+ lowers); bucket = joint
    angle relative to the boom; bucket world pitch = boom + bucket (0: floor level, + lip down).
    poses: expert arm poses (boom joint, bucket world pitch): carry, dig, curl, high, dump."""

    def __init__(self, **kw):
        self.name = kw.get('name', 'v0')
        self.l_front = kw.get('l_front', L_FRONT)
        self.l_rear = kw.get('l_rear', L_REAR)
        self.wheel_r = kw.get('wheel_r', WHEEL_R)
        self.boom_pivot = np.array(kw.get('boom_pivot', BOOM_PIVOT), float)
        self.boom_len = kw.get('boom_len', BOOM_LEN)
        self.boom_range = tuple(kw.get('boom_range', BOOM_RANGE))
        self.bucket_range = tuple(kw.get('bucket_range', BUCKET_RANGE))
        self.max_speed = kw.get('max_speed', MAX_SPEED)
        self.max_steer = kw.get('max_steer', MAX_STEER)
        self.boom_rate = kw.get('boom_rate', BOOM_RATE)
        self.bucket_rate = kw.get('bucket_rate', BUCKET_RATE)
        self.poses = dict(kw.get('poses', dict(carry=(0.2, -0.95), dig=(0.53, 0.04), curl=(0.36, -1.05), high=(-0.93, -0.6),
                                               dump=(-0.93, 1.3))))
        self.reset_arm = tuple(kw.get('reset_arm', (0.2, -0.9)))  # boom, bucket pitch at the episode start
        self.raised_boom = kw.get('raised_boom', -0.5)  # boom below this: the bucket passes over people / obstacles
        self.footprint_rear = tuple(kw.get('footprint_rear', (-2.0, -1.0, 0.0)))  # circle centres along the rear frame
        self.footprint_front = tuple(kw.get('footprint_front', (1.0, 1.75)))
        self.bucket_reach = kw.get('bucket_reach', 0.4)  # bucket circle beyond the hinge (footprint)
        self.plan_strict = kw.get('plan_strict', False)  # planner: check every sub-step, no blind escape (nav)
        self.bucket_side = kw.get('bucket_side', 0.0)  # > 0: two bucket circles this far left / right (wide bucket)
        self.rear_side = kw.get('rear_side', 0.0)  # > 0: two circles at the rear axle this far left / right (wide tyres)
        self.rear_axle = kw.get('rear_axle', -1.3)  # rear axle x from the hinge (rear body frame)
        self.push_limit_dy = kw.get('push_limit_dy', 4.15)  # articulation this far from the back wall: lip at the wall
        self.plan_steer_frac = kw.get('plan_steer_frac', 1.0)  # share of max_steer the path planner's arcs use
        self.plan_col_margin = kw.get('plan_col_margin', 0.0)  # extra room the planner keeps around columns
        self.dump_stop_dx = kw.get('dump_stop_dx', 0.0)  # dump this much further from the hopper than layout.hopper_stop_x
        self.bucket_box = tuple(kw.get('bucket_box', (-0.1, 1.2, 1.1, -0.45, 0.9)))  # pieces 'in the bucket': x0, x1, |y|, z0, z1

    @property
    def r_turn(self):
        """Minimum turning radius of the articulation point at full steering lock."""
        return (self.l_front * math.cos(self.max_steer) + self.l_rear) / math.sin(self.max_steer)


MACHINE_V0 = Machine()

HOPPER_STOP_X = HOPPER_CENTER[0] + HOPPER_HALF[0] + 2.15  # articulation x when dumping
APPROACH_Y = 4.6  # articulation y where digging starts


class Layout:
    """Where things are in the yard. LAYOUT_V0 is swm/CardboardYard-v0; other layouts (CardboardYardReal) only
    change these numbers. The hopper is always approached driving in -x along y = hopper_center[1]."""

    def __init__(self, **kw):
        self.pile_center = np.array(kw.get('pile_center', PILE_CENTER), float)
        self.pile_half = np.array(kw.get('pile_half', PILE_HALF), float)
        self.hopper_center = np.array(kw.get('hopper_center', HOPPER_CENTER), float)
        self.hopper_half = np.array(kw.get('hopper_half', HOPPER_HALF), float)
        self.hopper_wall_h = kw.get('hopper_wall_h', HOPPER_WALL_H)
        self.back_wall_y = kw.get('back_wall_y', BACK_WALL_Y)
        self.column_x = tuple(kw.get('column_x', COLUMN_X))
        self.column_y = tuple(kw.get('column_y', COLUMN_Y))
        self.column_half = kw.get('column_half', 0.22)  # half side of the (square) column cross-section
        self.approach_y = kw.get('approach_y', APPROACH_Y)
        self.left_wall = kw.get('left_wall', (-21.0, 3.0, 11.0))  # x, y centre, half length
        self.area = kw.get('area', (-11.0, 15.0, -5.0, 7.5))  # x0, x1, y0, y1 for obstacles
        self.people_area = kw.get('people_area', (-11.0, 15.0, -5.5, 7.0))
        self.people_bounds = kw.get('people_bounds', (-13.0, 17.0, -6.0, 13.0))
        self.spawn_area = kw.get('spawn_area', (-4.0, 12.0, -5.0, 5.0))

    @property
    def hopper_stop_x(self):
        return self.hopper_center[0] + self.hopper_half[0] + 2.15  # articulation x when dumping

    @property
    def hopper_line_y(self):
        return self.hopper_center[1]


LAYOUT_V0 = Layout()


def _in_rect(p, x0, x1, y0, y1, pad=0.0):
    return (x0 - pad) <= p[0] <= (x1 + pad) and (y0 - pad) <= p[1] <= (y1 + pad)


class CardboardYardEnv(gym.Env):
    """Wheel loader moves cardboard from the pile into the hopper.

    Every episode is randomised: pile size and position, loader start pose,
    0-4 obstacles (often on the route between pile and hopper) and 1-3 people
    who cross the route, walk a beat or stand still. People step aside or
    turn back when the loader is right next to them. Touching a person ends
    the episode.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    W_FIELDS = ('ax', 'ay', 'bx', 'by', 'speed', 's', 'dir', 'pause_left', 'pause_time', 'active', 'block', 'kind')
    # control tunables (subclasses may change them, e.g. CardboardYardReal's smoother physics; v0 keeps these)
    boom_rate, bucket_rate = BOOM_RATE, BUCKET_RATE
    max_accel, steer_rate = MAX_ACCEL, STEER_RATE
    arm_lead = 0.1  # rad: arm target may lead the joint by this much
    layout = LAYOUT_V0  # where pile, hopper, walls and columns are (Layout)
    machine = MACHINE_V0  # loader size and limits (Machine)
    drive_sub = 5  # substeps between drive target updates
    # service / parking brake (physics v5). brake_decel None = no brake (v0-v4: unchanged). Else m/s^2 the commanded
    # speed drops at with the pedal fully down; _brake (0..1) and _park are set by the subclass before each step.
    brake_decel = None
    _brake, _park = 0.0, False
    KIND_IDLE, KIND_CROSS, KIND_PATROL = 0, 1, 2

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        camera: str = 'chase',
        width: int = 224,
        height: int = 224,
        n_pieces: int = 90,
        frame_skip: int = 25,
        success_pieces: int = 25,
        max_obstacles: int = 4,
        max_workers: int = N_WORKERS,
        init_value: dict | None = None,
    ):
        self.render_mode = render_mode
        self.camera = camera
        self.width, self.height = width, height
        self.n_pieces = n_pieces
        self.frame_skip = frame_skip
        self.success_pieces = success_pieces
        self.max_obstacles = min(max_obstacles, N_OBSTACLES)
        self.max_workers = min(max_workers, N_WORKERS)
        self.env_name = 'CardboardYard'

        self.model = mujoco.MjModel.from_xml_string(self._build_xml(n_pieces))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * frame_skip
        self._renderer = None

        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        names = ('base_x', 'base_y', 'base_yaw', 'steer', 'boom', 'bucket')
        self._qadr = {n: m.jnt_qposadr[jid(n)] for n in names}
        self._dadr = {n: m.jnt_dofadr[jid(n)] for n in names}
        self._wheel_q = [m.jnt_qposadr[jid(f'wheel_{w}_spin')] for w in ('rl', 'rr', 'fl', 'fr')]
        self._act = {
            n: aid(n) for n in ('drive_x', 'drive_y', 'drive_yaw', 'steer', 'boom', 'bucket')
        }
        self._piece_q = np.array([m.jnt_qposadr[jid(f'piece{i}_j')] for i in range(n_pieces)])
        self._bucket_body = bid('bucket')
        self._w_mocap = [m.body_mocapid[bid(f'worker{i}')] for i in range(N_WORKERS)]
        self._o_mocap = [m.body_mocapid[bid(f'obstacle{i}')] for i in range(N_OBSTACLES)]

        loader_bodies = {
            bid(n)
            for n in ('rear', 'front', 'boom', 'bucket', 'wheel_rl', 'wheel_rr', 'wheel_fl', 'wheel_fr')
        }
        worker_bodies = {bid(f'worker{i}') for i in range(N_WORKERS)}
        obstacle_bodies = {bid(f'obstacle{i}') for i in range(N_OBSTACLES)}
        bod = m.geom_bodyid
        self._loader_geoms = {g for g in range(m.ngeom) if bod[g] in loader_bodies}
        self._worker_geoms = {g for g in range(m.ngeom) if bod[g] in worker_bodies}
        self._obstacle_geoms = {g for g in range(m.ngeom) if bod[g] in obstacle_bodies}
        self._static_hit_geoms = {
            g
            for g in range(m.ngeom)
            if bod[g] == 0
            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith(
                ('col_', 'hopper_w', 'back_wall', 'left_wall')
            )
        }

        # obs: proprio(10) + workers(3x3) + obstacles(5x3) + pieces(3N) + counters(2)
        self.proprio_dim = 10
        obs_dim = self.proprio_dim + 3 * N_WORKERS + 3 * N_OBSTACLES + 3 * n_pieces + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.variation_space = None  # randomisation is done in reset()

        self.workers = np.zeros((N_WORKERS, len(self.W_FIELDS)))
        self.obstacles = np.zeros((N_OBSTACLES, 4))  # x, y, yaw, active
        self.pile_center = self.layout.pile_center.copy()
        self.pile_half = self.layout.pile_half.copy()
        self._boom_target = 0.0
        self._bucket_target = 0.0
        self._v_cmd = 0.0
        self._steer_cmd = 0.0
        self._wheel_angle = 0.0
        self._delivered = 0
        self._t = 0
        self._grid = None
        self._goal = None  # goal proprio in goal-reaching eval, else None

    # ------------------------------------------------------------ helpers
    def loader_pose(self):
        """(x, y, yaw) of the articulation point / rear frame."""
        q = self.data.qpos
        return (
            float(q[self._qadr['base_x']]),
            float(q[self._qadr['base_y']]),
            float(q[self._qadr['base_yaw']]),
        )

    def joint(self, name):
        return float(self.data.qpos[self._qadr[name]])

    def footprint(self):
        from . import nav

        x, y, yaw = self.loader_pose()
        return nav.footprint(x, y, yaw, self.joint('steer'), self.joint('boom'), M=self.machine)

    def bucket_frame(self):
        b = self._bucket_body
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def pieces_pos(self):
        idx = self._piece_q[:, None] + np.arange(3)[None]
        return self.data.qpos[idx].copy()

    def pieces_in_hopper(self):
        p = self.pieces_pos()
        L = self.layout
        d = np.abs(p[:, :2] - L.hopper_center)
        inside = (d[:, 0] < L.hopper_half[0]) & (d[:, 1] < L.hopper_half[1]) & (p[:, 2] < L.hopper_wall_h)
        return int(inside.sum())

    def pieces_in_bucket(self):
        p = self.pieces_pos()
        pos, R = self.bucket_frame()
        local = (p - pos) @ R
        x0, x1, ym, z0, z1 = self.machine.bucket_box
        inside = (
            (local[:, 0] > x0) & (local[:, 0] < x1)
            & (np.abs(local[:, 1]) < ym)
            & (local[:, 2] > z0) & (local[:, 2] < z1)
        )
        return int(inside.sum())

    def worker_positions(self):
        """(N, 2) positions of the active people and their kinds."""
        act = self.workers[:, 9] > 0
        pos = np.array([self._worker_xy(i) for i in range(N_WORKERS)])
        return pos[act], self.workers[act, 11].astype(int)

    def worker_velocities(self):
        """(N, 2) current walking velocities of the active people (0 while standing, pausing or waiting)."""
        act = self.workers[:, 9] > 0
        v = np.zeros((N_WORKERS, 2))
        for i in range(N_WORKERS):
            w = self.workers[i]
            if w[9] <= 0 or w[4] <= 0 or w[7] > 0 or w[10] > 0:
                continue
            u = w[2:4] - w[0:2]
            v[i] = w[4] * w[6] * u / max(np.linalg.norm(u), 1e-3)
        return v[act]

    def obstacle_circles(self):
        return [
            (o[0], o[1], OBSTACLE_RADIUS[i])
            for i, o in enumerate(self.obstacles)
            if o[3] > 0
        ]

    def nav_grid(self, near=None, ahead=0.0, skip_close=0.0, boxes=()):
        """Occupancy grid of the yard incl. obstacles and people standing still. near=(x, y, r): also the walking
        people within r of (x, y), where they are now (not cached, they move); ahead > 0 (s): and where they will be
        after ahead/2 and ahead seconds at their current velocity. skip_close > 0: people walking (not waiting) this close to
        the loader's footprint are left out (they would put the plan's start in collision; the safety stop has them).
        boxes: extra blocked boxes (cx, cy, hx, hy) for this plan only (not cached)."""
        from . import nav

        if near is not None:
            pos, kind = self.worker_positions()
            vel = self.worker_velocities() if (ahead or skip_close) else np.zeros((len(pos), 2))
            ppl = []
            fp = self.footprint() if skip_close else None
            for p, k, v in zip(pos, kind, vel):
                if skip_close and v.any() and np.min(np.hypot(fp[:, 0] - p[0], fp[:, 1] - p[1])) < skip_close:
                    continue
                if k == self.KIND_IDLE or math.hypot(p[0] - near[0], p[1] - near[1]) < near[2]:
                    ppl += [(p[0] + v[0] * t, p[1] + v[1] * t, 1.3) for t in ((0.0, ahead / 2, ahead) if ahead and v.any() else (0.0,))]
            pile = [(*self.pile_center, self.pile_half[0] + 0.4, self.pile_half[1] + 0.6)] + list(boxes)
            return nav.Grid(self.obstacle_circles(), ppl, pile, layout=self.layout, machine=self.machine)
        if boxes:
            pos, kind = self.worker_positions()
            idle = [(p[0], p[1], 1.3) for p, k in zip(pos, kind) if k == self.KIND_IDLE]
            pile = [(*self.pile_center, self.pile_half[0] + 0.4, self.pile_half[1] + 0.6)] + list(boxes)
            return nav.Grid(self.obstacle_circles(), idle, pile, layout=self.layout, machine=self.machine)
        if self._grid is None:
            pos, kind = self.worker_positions()
            idle = [(p[0], p[1], 1.3) for p, k in zip(pos, kind) if k == self.KIND_IDLE]
            pile = [(*self.pile_center, self.pile_half[0] + 0.4, self.pile_half[1] + 0.6)]
            self._grid = nav.Grid(self.obstacle_circles(), idle, pile, layout=self.layout, machine=self.machine)
        return self._grid

    def corridor(self):
        """Endpoints of the pile -> hopper transport corridor."""
        return (
            np.array([self.pile_center[0], 1.0]),
            np.array([self.layout.hopper_stop_x + 4.0, self.layout.hopper_line_y]),
        )

    def _forbidden(self, p, pad):
        """Areas kept free: the digging lane and the hopper approach."""
        cx = self.pile_center[0]
        return _in_rect(p, cx - 3.2, cx + 3.2, -1.6, 14.0, pad) or _in_rect(
            p, self.layout.hopper_stop_x - 1.0, self.layout.hopper_stop_x + 5.5, self.layout.hopper_line_y - 2.8,
            self.layout.hopper_line_y + 2.8, pad
        )

    def _worker_xy(self, i):
        w = self.workers[i]
        if w[9] <= 0:
            return PARK[:2] + [2.0 * i, 0.0]
        a, b = w[0:2], w[2:4]
        return a + (b - a) * w[5]

    def _place_mocaps(self):
        d = self.data
        for i in range(N_WORKERS):
            w = self.workers[i]
            if w[9] <= 0:
                d.mocap_pos[self._w_mocap[i]] = PARK + [2.0 * i, 0.0, 0.0]
                continue
            p = self._worker_xy(i)
            heading = (w[2:4] - w[0:2]) * (w[6] if w[11] != self.KIND_IDLE else 1.0)
            yaw = math.atan2(heading[1], heading[0])
            d.mocap_pos[self._w_mocap[i]] = [p[0], p[1], 0.0]
            d.mocap_quat[self._w_mocap[i]] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i in range(N_OBSTACLES):
            o = self.obstacles[i]
            if o[3] <= 0:
                d.mocap_pos[self._o_mocap[i]] = PARK + [0.0, 3.0 * i, 0.0]
            else:
                d.mocap_pos[self._o_mocap[i]] = [o[0], o[1], 0.0]
                d.mocap_quat[self._o_mocap[i]] = [math.cos(o[2] / 2), 0, 0, math.sin(o[2] / 2)]

    def _update_workers(self, dt):
        fp = self.footprint()
        for i in range(N_WORKERS):
            w = self.workers[i]
            if w[9] <= 0 or w[4] <= 0:
                continue
            if w[7] > 0:  # pausing at the end of the beat
                w[7] -= dt
                continue
            a, b = w[0:2], w[2:4]
            length = max(np.linalg.norm(b - a), 1e-3)
            s_new = float(np.clip(w[5] + w[4] * dt / length * w[6], 0.0, 1.0))
            old = a + (b - a) * w[5]
            new = a + (b - a) * s_new
            d_old = np.min(np.linalg.norm(fp - old, axis=1)) - 1.1
            d_new = np.min(np.linalg.norm(fp - new, axis=1)) - 1.1
            if d_new < 1.2 and d_new < d_old:  # loader right there: wait, then turn back
                w[10] += dt
                if w[10] > 3.0:
                    w[6] *= -1
                    w[10] = 0.0
                continue
            w[10] = 0.0
            w[5] = s_new
            if s_new <= 0.0 or s_new >= 1.0:
                w[6] *= -1
                w[7] = w[8]

    def _contacts(self):
        hit_worker = hit_static = hit_obstacle = False
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if g1 in self._loader_geoms or g2 in self._loader_geoms:
                other = g2 if g1 in self._loader_geoms else g1
                if other in self._worker_geoms:
                    hit_worker = True
                elif other in self._obstacle_geoms:
                    hit_obstacle = True
                elif other in self._static_hit_geoms:
                    hit_static = True
        return hit_worker, hit_static, hit_obstacle

    # ------------------------------------------------------------ gym api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # fresh GL context every episode: on Windows (glfw) a renderer kept across
        # swm.World episodes returned flipped / garbage frames after the first reset
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        mujoco.mj_resetData(self.model, self.data)
        self._grid = None
        self._goal = None if options.get('goal') is None else np.asarray(options['goal'], np.float64)
        if 'state' in options:
            self._set_state(np.asarray(options['state'], np.float64))
        else:
            self._randomize(self.np_random)
        self._delivered = self.pieces_in_hopper()
        self._t = 0
        return self._get_obs(), self._get_info()

    def _randomize(self, rng):
        from . import nav

        q = self.data.qpos
        L = self.layout
        # pile
        self.pile_half = np.array([rng.uniform(1.5, 2.2), rng.uniform(1.3, 1.7)])
        self.pile_center = np.array(
            [L.pile_center[0] + rng.uniform(-1.2, 1.2), L.back_wall_y - 0.6 - self.pile_half[1]]
        )
        static = nav.Grid(layout=L, machine=self.machine)

        # obstacles, mostly on the pile -> hopper route
        self.obstacles[:] = 0.0
        k = int(rng.choice(np.arange(5), p=[0.15, 0.3, 0.3, 0.15, 0.1]))
        k = min(k, self.max_obstacles)
        a, b = self.corridor()
        placed = []
        for typ in rng.choice(N_OBSTACLES, size=k, replace=False):
            r = OBSTACLE_RADIUS[typ]
            for _ in range(60):
                if rng.random() < 0.65:
                    t = rng.uniform(0.15, 0.85)
                    perp = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
                    p = a + t * (b - a) + perp * rng.uniform(-1.2, 1.2)
                else:
                    p = np.array([rng.uniform(L.area[0], L.area[1]), rng.uniform(L.area[2], L.area[3])])
                if self._forbidden(p, r + 0.3) or static.clearance(*p) < r + 0.5:
                    continue
                if any(np.hypot(p[0] - x, p[1] - y) < r + rr + 0.6 for x, y, rr in placed):
                    continue
                self.obstacles[typ] = [p[0], p[1], rng.uniform(-math.pi, math.pi), 1.0]
                placed.append((p[0], p[1], r))
                break
        # keep the task solvable: pile <-> hopper must stay connected
        while True:
            g = nav.Grid(self.obstacle_circles(), layout=L, machine=self.machine)
            p_pile = np.array([self.pile_center[0], L.approach_y - 3.0])
            p_hop = np.array([L.hopper_stop_x + 4.0, L.hopper_line_y])
            if g.astar(p_pile, p_hop) is not None or not self.obstacle_circles():
                break
            last = max(i for i in range(N_OBSTACLES) if self.obstacles[i, 3] > 0)
            self.obstacles[last] = 0.0

        # loader start pose
        g = nav.Grid(self.obstacle_circles(), layout=L, machine=self.machine)
        boom0, pitch0 = self.machine.reset_arm
        for _ in range(300):
            x, y = rng.uniform(L.spawn_area[0], L.spawn_area[1]), rng.uniform(L.spawn_area[2], L.spawn_area[3])
            yaw = rng.uniform(-math.pi, math.pi)
            fp = nav.footprint(x, y, yaw, 0.0, boom0, M=self.machine)
            if min(g.clearance(px, py) for px, py in fp) < nav.R_LOADER + 0.4:
                continue
            if g.astar((x, y), p_pile) is None:
                continue
            break
        q[self._qadr['base_x']], q[self._qadr['base_y']], q[self._qadr['base_yaw']] = x, y, yaw
        self._boom_target, self._bucket_target = boom0, pitch0 - boom0
        self._v_cmd, self._steer_cmd = 0.0, 0.0
        self._wheel_angle = 0.0  # new episode: do not carry the last episode's wheel turn (same seed -> same start)
        q[self._qadr['boom']] = self._boom_target
        q[self._qadr['bucket']] = self._bucket_target

        # people
        start_fp = nav.footprint(x, y, yaw, 0.0, boom0, M=self.machine)
        self._spawn_people(rng, g, a, b, start_fp)
        self._place_mocaps()
        self._spawn_pile(rng)

    def _spawn_people(self, rng, g, a, b, start_fp):
        """1-3 people: crossing the pile -> hopper route (a -> b), walking a beat or standing still."""
        L = self.layout
        self.workers[:] = 0.0
        n = int(rng.choice([1, 2, 3], p=[0.25, 0.4, 0.35]))
        n = min(n, self.max_workers)
        for i in range(n):
            kind = int(rng.choice([self.KIND_CROSS, self.KIND_PATROL, self.KIND_IDLE], p=[0.45, 0.35, 0.2]))
            for _ in range(120):
                if kind == self.KIND_CROSS:
                    t = rng.uniform(0.1, 0.9)
                    ang = math.atan2(*(b - a)[::-1]) + math.pi / 2 + rng.uniform(-0.4, 0.4)
                    u = np.array([math.cos(ang), math.sin(ang)])
                    mid = a + t * (b - a)
                    pa, pb = mid - u * rng.uniform(3.0, 5.5), mid + u * rng.uniform(3.0, 5.5)
                elif kind == self.KIND_PATROL:
                    pa = np.array([rng.uniform(L.people_area[0], L.people_area[1]), rng.uniform(L.people_area[2], L.people_area[3])])
                    ang = rng.uniform(-math.pi, math.pi)
                    pb = pa + rng.uniform(4.0, 10.0) * np.array([math.cos(ang), math.sin(ang)])
                else:
                    pa = np.array([rng.uniform(L.people_area[0], L.people_area[1]), rng.uniform(L.people_area[2], L.people_area[3])])
                    ang = rng.uniform(-math.pi, math.pi)
                    pb = pa + 0.01 * np.array([math.cos(ang), math.sin(ang)])
                pts = [pa + s * (pb - pa) for s in np.linspace(0, 1, max(2, int(np.linalg.norm(pb - pa) / 0.3)))]
                if any(
                    not (L.people_bounds[0] < p[0] < L.people_bounds[1] and L.people_bounds[2] < p[1] < L.people_bounds[3])
                    or g.clearance(*p) < 0.6
                    or self._forbidden(p, 0.5)
                    for p in pts
                ):
                    continue
                s0 = rng.uniform(0, 1)
                p0 = pa + s0 * (pb - pa)
                if np.min(np.linalg.norm(start_fp - p0, axis=1)) < 4.0:
                    continue
                speed = 0.0 if kind == self.KIND_IDLE else rng.uniform(0.6, 1.3)
                self.workers[i] = [*pa, *pb, speed, s0, rng.choice([-1.0, 1.0]), 0.0,
                                   rng.uniform(0.5, 3.0), 1.0, 0.0, kind]
                break


    def _spawn_pile(self, rng):
        """Drop cardboard sheets in a pyramid of layers and let them settle."""
        q = self.data.qpos
        placed = 0
        layer = 0
        while placed < self.n_pieces:
            shrink = 1.0 - 0.17 * layer
            hx, hy = self.pile_half * max(shrink, 0.25)
            nx = max(1, int(2 * hx / 0.72))
            ny = max(1, int(2 * hy / 0.52))
            xs = np.linspace(-hx + 0.36, hx - 0.36, nx) if nx > 1 else [0.0]
            ys = np.linspace(-hy + 0.26, hy - 0.26, ny) if ny > 1 else [0.0]
            cy = self.pile_center[1] + 0.25 * layer  # lean toward the back wall
            for x in xs:
                for y in ys:
                    if placed >= self.n_pieces:
                        break
                    a = self._piece_q[placed]
                    q[a : a + 3] = [
                        self.pile_center[0] + x + rng.uniform(-0.08, 0.08),
                        min(cy + y + rng.uniform(-0.06, 0.06), self.layout.back_wall_y - 0.4),
                        0.03 + 0.075 * layer + rng.uniform(0, 0.01),
                    ]
                    quat = np.zeros(4)
                    mujoco.mju_euler2Quat(
                        quat, np.array([rng.uniform(-0.1, 0.1), 0.0, rng.uniform(-0.15, 0.15)]), 'xyz'
                    )
                    q[a + 3 : a + 7] = quat
                    placed += 1
            layer += 1
        self.data.ctrl[:] = 0
        self.data.ctrl[self._act['boom']] = self._boom_target
        self.data.ctrl[self._act['bucket']] = self._bucket_target
        for _ in range(250):  # 1 s of settling
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64).reshape(-1), -1.0, 1.0)
        M = self.machine
        v_goal = a[0] * M.max_speed
        boom0, bucket0 = self._boom_target, self._bucket_target
        self._boom_target = float(np.clip(self._boom_target - a[2] * self.boom_rate * self.dt, *M.boom_range))
        self._bucket_target = float(
            np.clip(self._bucket_target - a[3] * self.bucket_rate * self.dt, *M.bucket_range)
        )
        # targets may not run far ahead of a joint held back by the load / the pile,
        # otherwise it snaps after them once free (jerk that throws cardboard out)
        lead = self.arm_lead
        jb, jk = self.joint('boom'), self.joint('bucket')
        self._boom_target = float(np.clip(self._boom_target, jb - lead, jb + lead))
        self._bucket_target = float(np.clip(self._bucket_target, jk - lead, jk + lead))
        d = self.data
        ctrl = d.ctrl
        # ramp the commanded speed / articulation (kept as state, not the measured
        # motion, so pushing into the pile still gets full drive force)
        v = self._v_cmd
        steer = self._steer_cmd
        steer_goal = a[1] * M.max_steer

        sub = self.drive_sub  # recompute drive targets / move people a few times per step
        h = sub * self.model.opt.timestep
        dv_max = self.max_accel * h  # largest change of the commanded speed per drive update
        if self.brake_decel is not None and (self._brake > 0.0 or self._park):
            # brake pedal: the drive is cut (speed goal 0) and the speed drops faster the further the pedal is down
            # (max_accel with the pedal just touched .. brake_decel fully down). Parking brake = full brake, held.
            b = 1.0 if self._park else self._brake
            v_goal = 0.0
            dv_max = (self.max_accel + b * (self.brake_decel - self.max_accel)) * h
        for k in range(self.frame_skip):
            f = (k + 1) / self.frame_skip  # arm targets slide over the step, no jumps
            ctrl[self._act['boom']] = boom0 + f * (self._boom_target - boom0)
            ctrl[self._act['bucket']] = bucket0 + f * (self._bucket_target - bucket0)
            if k % sub == 0:
                v += float(np.clip(v_goal - v, -dv_max, dv_max))
                steer += float(np.clip(steer_goal - steer, -self.steer_rate * h, self.steer_rate * h))
                ctrl[self._act['steer']] = steer
                self._v_cmd, self._steer_cmd = v, steer
                _, _, yaw = self.loader_pose()
                gamma = self.joint('steer')
                omega = v * math.sin(gamma) / (M.l_front * math.cos(gamma) + M.l_rear)
                c, s = math.cos(yaw), math.sin(yaw)
                vy_local = omega * M.l_rear
                ctrl[self._act['drive_x']] = v * c - vy_local * s
                ctrl[self._act['drive_y']] = v * s + vy_local * c
                ctrl[self._act['drive_yaw']] = omega
                self._update_workers(sub * self.model.opt.timestep)
                self._place_mocaps()
            mujoco.mj_step(self.model, d)

        speed = math.hypot(d.qvel[self._dadr['base_x']], d.qvel[self._dadr['base_y']])
        self._wheel_angle += (1.0 if v >= 0 else -1.0) * speed * self.dt / M.wheel_r
        for qa in self._wheel_q:
            d.qpos[qa] = self._wheel_angle
        mujoco.mj_kinematics(self.model, d)

        self._t += 1
        hit_worker, hit_static, hit_obstacle = self._contacts()
        delivered = self.pieces_in_hopper()
        reward = float(delivered - self._delivered)
        self._delivered = delivered
        terminated = False
        if hit_worker:
            reward -= 10.0
            terminated = True
        if delivered >= self.success_pieces:
            terminated = True
        info = self._get_info()
        if self._goal is not None:
            # goal-reaching eval: the episode ends (success) only when the goal pose is reached
            err = self.goal_error()
            terminated = bool(err['pos'] < GOAL_POS_TOL and err['yaw'] < GOAL_YAW_TOL and err['arm'] < GOAL_ARM_TOL)
        info['hit_worker'] = hit_worker
        info['hit_obstacle'] = hit_obstacle
        info['hit_static'] = hit_static
        return self._get_obs(), reward, terminated, False, info

    # ------------------------------------------------------------ obs
    def _proprio(self):
        x, y, yaw = self.loader_pose()
        d = self.data
        return np.array(
            [
                x, y, math.cos(yaw), math.sin(yaw),
                d.qvel[self._dadr['base_x']], d.qvel[self._dadr['base_y']],
                self.joint('steer'), self.joint('boom'), self.joint('bucket'),
                d.qvel[self._dadr['base_yaw']],
            ],
            np.float32,
        )

    def _get_obs(self):
        people = np.array([[*self._worker_xy(i), self.workers[i, 9]] for i in range(N_WORKERS)])
        obst = self.obstacles[:, [0, 1, 3]]
        return np.concatenate(
            [
                self._proprio(),
                people.reshape(-1),
                obst.reshape(-1),
                self.pieces_pos().reshape(-1),
                [self._delivered, self.pieces_in_bucket()],
            ]
        ).astype(np.float32)

    def get_episode_data(self):
        """Episode-level columns for swm.World.collect: expert mode of the episode (1 weak, 0 expert, -1 unknown)."""
        w = getattr(self, '_expert_weak', None)
        return {'expert_weak': -1 if w is None else int(w)}

    def _get_info(self):
        return {
            'env_name': self.env_name,
            'proprio': self._proprio(),
            'state': self._full_state(),
            'pieces_in_hopper': self._delivered,
            'pieces_in_bucket': self.pieces_in_bucket(),
            # (N_WORKERS, 3): x, y, active (1/0) of every person; lets a planner keep away from people
            'people_xy': np.array(
                [[*self._worker_xy(i), float(self.workers[i, 9] > 0)] for i in range(N_WORKERS)], np.float32
            ),
            'hit_worker': False,
            'hit_obstacle': False,
            'hit_static': False,
        }

    def _full_state(self):
        """Everything needed to restore the episode exactly."""
        d = self.data
        return np.concatenate(
            [
                d.qpos,
                d.qvel,
                [self._boom_target, self._bucket_target, self._wheel_angle, self._v_cmd, self._steer_cmd],
                self.workers.reshape(-1),
                self.obstacles.reshape(-1),
                self.pile_center,
                self.pile_half,
            ]
        ).astype(np.float32)

    def _set_state(self, state):
        nq, nv = self.model.nq, self.model.nv
        d = self.data
        d.qpos[:] = state[:nq]
        d.qvel[:] = state[nq : nq + nv]
        k = nq + nv
        self._boom_target, self._bucket_target, self._wheel_angle, self._v_cmd, self._steer_cmd = (
            float(v) for v in state[k : k + 5]
        )
        k += 5
        nw = self.workers.size
        self.workers[:] = np.asarray(state[k : k + nw], np.float64).reshape(self.workers.shape)
        k += nw
        no = self.obstacles.size
        self.obstacles[:] = np.asarray(state[k : k + no], np.float64).reshape(self.obstacles.shape)
        k += no
        self.pile_center = np.asarray(state[k : k + 2], np.float64)
        self.pile_half = np.asarray(state[k + 2 : k + 4], np.float64)
        self._grid = None
        d.ctrl[self._act['boom']] = self._boom_target
        d.ctrl[self._act['bucket']] = self._bucket_target
        self._place_mocaps()
        mujoco.mj_forward(self.model, d)

    def goal_error(self, goal=None):
        """Distance of the loader to a goal proprio: position (m), heading and worst arm joint (rad)."""
        g = self._goal if goal is None else np.asarray(goal, np.float64)
        p = self._proprio()
        dyaw = math.atan2(p[3], p[2]) - math.atan2(g[3], g[2])
        return {
            'pos': float(math.hypot(p[0] - g[0], p[1] - g[1])),
            'yaw': float(abs((dyaw + math.pi) % (2 * math.pi) - math.pi)),
            'arm': float(max(abs(p[7] - g[7]), abs(p[8] - g[8]))),
        }

    def reset_options_from_dataset(self, init_row, goal_row):
        opts = {'state': np.asarray(init_row['state'])}
        if 'goal_proprio' in goal_row:
            opts['goal'] = np.asarray(goal_row['goal_proprio'])
        return opts

    # ------------------------------------------------------------ render
    def render(self, camera=None, width=None, height=None):
        w, h = width or self.width, height or self.height
        if self._renderer is None or self._renderer.width != w or self._renderer.height != h:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = camera or self.camera
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0 if cam == 'top' else 1  # roof hides the top view
        self._renderer.update_scene(self.data, camera=cam, scene_option=opt)
        return self._renderer.render().copy()

    def _build_xml(self, n_pieces):
        """MJCF of the scene. Subclasses may change the looks here (CardboardYardReal does)."""
        return build_xml(n_pieces, self.layout, self.machine)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None







