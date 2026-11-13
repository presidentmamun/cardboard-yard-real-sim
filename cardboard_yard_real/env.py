"""swm/CardboardYardReal-v0: CardboardYard with the looks of the real site (photos of the yard and the loader).

Only the looks change. The physics, actions, proprio, rewards, the expert and the camera poses are those of
CardboardYardEnv: this class edits the MJCF of the parent scene and
  - hides the plain box geoms of the loader and of the building (geom group 4 is not rendered; their
    collision, size, position and mass stay exactly as they are),
  - draws the loader with procedural meshes (meshes.py: rounded cab with glass, hood, dog-leg lift arms,
    curved bucket with mesh extension, treaded tires) and the shed as in the photos (block wall with rusty
    rails, corrugated upper wall and roof, roof trusses, green/red H-beam columns, open sides, big cardboard
    heaps along the back wall, feeder conveyor + pulper behind the hopper),
  - every added geom is visual only: contype=conaffinity=0 and mass=0.
test_same_physics() checks this: same seed + same actions -> identical simulation state, step by step.
Textures: assets/*.png (make_textures.py: crops of the site photos + procedural ones); meshes: assets/meshes/*.obj
(written from meshes.py when the scene is built).
"""

import math
import os
import tempfile
import xml.etree.ElementTree as ET
import zlib

import mujoco
import numpy as np
from gymnasium import spaces

from cardboard_yard.env import (
    BACK_WALL_Y,
    BOOM_LEN,
    COLUMN_X,
    COLUMN_Y,
    HOPPER_CENTER,
    HOPPER_HALF,
    HOPPER_WALL_H,
    PILE_CENTER,
    CardboardYardEnv,
    build_xml,
)

from . import hydraulics as HY
from . import meshes as MS
from . import physics as PH
from .people import WalkingPeople
from cardboard_yard.env import LAYOUT_V0

_L = LAYOUT_V0  # layout of the scene being built (build_real_xml sets it)
MACH_DY = 0.0

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
MESH_DIR = os.path.join(ASSETS, 'meshes')
N_CARD_LOOKS = 6
LOADER_LOOK = 'model'  # 'model': hand-built loader (loader_model.py); 'scan': textured scan of a real Komatsu (assets/komatsu_scan); 'procedural': meshes.py
HIDDEN = '4'  # geom group that is never rendered (env.render only switches group 3)

# material -> (texture: 'png name' = 2d, 'png name:cube' = same image on every face, or None; attributes)
MATERIALS = {
    'floor': ('floor_proc', dict(texrepeat='0.125 0.125', texuniform='true', rgba='1 1 1 1', reflectance='0', specular='0.05')),
    'hopper': (None, dict(rgba='0.42 0.4 0.36 1', specular='0.1')),
    'paper_roll': (None, dict(rgba='0.9 0.86 0.76 1')),
    'yellow': (None, dict(rgba='0.96 0.73 0.03 1', specular='0.35', shininess='0.45')),
    'rim': (None, dict(rgba='0.96 0.73 0.03 1', specular='0.3')),
    'glass': (None, dict(rgba='0.16 0.2 0.22 0.55', specular='0.9', shininess='0.95', reflectance='0.15')),
    'tire': (None, dict(rgba='0.07 0.07 0.07 1', specular='0.05')),
    'dark': (None, dict(rgba='0.13 0.13 0.14 1')),
}
NEW_MATERIALS = {
    'blocks': ('blocks_proc', dict(texrepeat='0.5 0.5', texuniform='true', rgba='1 1 1 1')),
    'corrugated': ('corrugated_proc', dict(texrepeat='1 1', texuniform='true', rgba='0.74 0.68 0.62 1')),
    'roof_sheet': ('corrugated_proc', dict(texrepeat='1 1', texuniform='true', rgba='0.86 0.89 0.93 1')),
    'col_green': ('paint_green:cube', dict(rgba='1 1 1 1', specular='0.15')),
    'col_red': ('paint_red:cube', dict(rgba='1 1 1 1', specular='0.1')),
    'wall_cap': (None, dict(rgba='0.55 0.55 0.53 1')),
    'rust_beam': ('rust_proc:cube', dict(rgba='1 1 1 1', specular='0.05')),
    'column_paint': ('column:cube', dict(rgba='1 1 1 1', specular='0.1')),
    'pile': ('pile', dict(rgba='0.95 0.82 0.62 1')),
    'mesh': ('mesh:cube', dict(rgba='1 1 1 1')),
    'stainless': (None, dict(rgba='0.5 0.52 0.54 1', specular='0.5', shininess='0.5')),
    'conveyor': ('conveyor:cube', dict(rgba='1 1 1 1', specular='0.1')),
    'bucket_paint': (None, dict(rgba='0.3 0.29 0.26 1', specular='0.3', shininess='0.3')),
    'black': (None, dict(rgba='0.05 0.05 0.05 1')),
    'chrome': (None, dict(rgba='0.78 0.79 0.8 1', specular='0.9', shininess='0.9')),
    'lamp': (None, dict(rgba='1.0 0.96 0.85 1', emission='0.6')),
    'taillight': (None, dict(rgba='0.8 0.05 0.05 1', emission='0.3')),
    'beacon': (None, dict(rgba='1.0 0.5 0.05 1', emission='0.4')),
    'bale': (None, dict(rgba='0.74 0.66 0.52 1')),
    'building': (None, dict(rgba='0.52 0.54 0.57 1')),
    'girder_grey': (None, dict(rgba='0.5 0.5 0.49 1', specular='0.1')),
    'tank_rust': (None, dict(rgba='0.42 0.4 0.36 1', specular='0.1')),
    'stainless_in': (None, dict(rgba='0.72 0.73 0.75 1', specular='0.8', shininess='0.8', reflectance='0.15')),
    'screen': (None, dict(rgba='0.35 0.36 0.38 1', specular='0.4')),
    'green_machine': (None, dict(rgba='0.45 0.7 0.42 1', specular='0.15')),
    'pipe': (None, dict(rgba='0.72 0.73 0.74 1', specular='0.6')),
    'window': (None, dict(rgba='0.9 0.92 0.95 1', emission='0.5')),
    'red_roof': (None, dict(rgba='0.62 0.26 0.18 1')),
    'wood': (None, dict(rgba='0.62 0.48 0.3 1')),
    'forklift': (None, dict(rgba='0.9 0.38 0.08 1', specular='0.3')),
    'decal': ('decal_komatsu', dict(texrepeat='1 1', rgba='1 1 1 1', specular='0.35')),
    'decal_model': ('decal_wa100', dict(texrepeat='1 1', rgba='1 1 1 1', specular='0.2')),
}
for _k in range(N_CARD_LOOKS):
    NEW_MATERIALS[f'card{_k}'] = (f'card{_k}:cube', dict(rgba='1 1 1 1'))

VIS = dict(contype='0', conaffinity='0', mass='0')
LOADER_PARTS = {
    'rear': ('hood', 'counterweight', 'rear_chassis', 'cab_base', 'cab_glass', 'cab_roof', 'cab_frame',
             'fenders_rear', 'rear_details', 'taillights', 'worklamps', 'beacon'),
    'front': ('front_chassis', 'towers', 'fenders_front', 'lift_cyl', 'lift_rod'),
    'boom': ('boom', 'tilt_lever', 'tilt_cyl', 'tilt_link'),
    'bucket': ('bucket_shell', 'bucket_sides', 'bucket_edge', 'bucket_mesh', 'bucket_rail'),
    'wheel_rl': ('tire', 'rim', 'hub'), 'wheel_rr': ('tire', 'rim', 'hub'),
    'wheel_fl': ('tire', 'rim', 'hub'), 'wheel_fr': ('tire', 'rim', 'hub'),
}


def _geom(parent, **attr):
    a = {**VIS, **attr}
    return ET.SubElement(parent, 'geom', {k: str(v) for k, v in a.items()})


def _f(*v):
    return ' '.join(f'{x:.4f}' for x in v)


def _texture_and_material(asset, name, tex, attrs):
    m = asset.find(f"material[@name='{name}']")
    if m is None:
        m = ET.SubElement(asset, 'material', {'name': name})
    if tex is not None:
        img, _, kind = tex.partition(':')
        kind = kind or '2d'
        tname = f'tex_{img}_{kind}'
        if asset.find(f"texture[@name='{tname}']") is None:
            ET.SubElement(asset, 'texture', {'name': tname, 'type': kind,
                                             'file': os.path.join(ASSETS, f'{img}{"_sq" if kind == "cube" else ""}.png').replace('\\', '/')})
        m.set('texture', tname)
    for k, v in attrs.items():
        m.set(k, v)


def _write_mesh(name, mesh):
    """Write assets/meshes/<name>.obj if it changed (atomic: parallel workers may build at the same time)."""
    os.makedirs(MESH_DIR, exist_ok=True)
    path = os.path.join(MESH_DIR, name + '.obj')
    text = mesh.obj()
    try:
        with open(path) as f:
            if f.read() == text:
                return path
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=MESH_DIR, suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def _mesh_asset(asset, name, mesh):
    path = _write_mesh(name, mesh).replace('\\', '/')
    ET.SubElement(asset, 'mesh', {'name': name, 'file': path, 'inertia': 'shell'})


def _hide(g):
    g.set('group', HIDDEN)


def build_real_xml(n_pieces: int, physics: str = 'v0', looks: bool = True) -> str:
    """MJCF: base scene, physics profile (physics.py), then the real-site looks (visual only)."""
    global _L, ROOF_BACK, ROOF_FRONT, MACH_DY
    _L = PH.layout(physics)
    ROOF_BACK = (_L.back_wall_y, 4.5)
    ROOF_FRONT = (min(_L.column_y) - 7.0, 8.2)  # the roof ends 7 m beyond the far column row
    MACH_DY = min(_L.column_y) + 7.0  # machinery side shifts with the far column row (v0: 0)
    root = ET.fromstring(build_xml(n_pieces, _L))
    wa100 = PH.PROFILES[physics].get('machine') == 'wa100'
    if wa100:  # Komatsu WA100-8 bodies instead of the plain v0 loader (physics, wa100_sim.py)
        from . import wa100_sim as WS

        WS.replace_loader(root)
    PH.apply_xml(root, physics)  # after the loader swap, so the profile's joint settings reach the WA100 joints too
    if PH.PROFILES[physics].get('people') == 'walk':  # articulated walking people (people.py)
        from cardboard_yard.env import N_WORKERS

        from . import people as PE

        PE.replace_people(root, N_WORKERS)
    if _L is not LAYOUT_V0:  # real site: the v0 top camera (24 m over the v0 yard) missed the hopper 23 m south of the
        # pile. Straight down over the middle of the work area, 30 m square on the ground (back wall to the south edge
        # of the work area is 28.6 m), from 15 m with a 90 deg lens: the loader looks ~20% bigger than from 26 m / 60
        # deg (its top is nearer); lower (10 m / 113 deg) made the 6.3 m columns lean over large parts of the floor
        # (2026-10-28) any perspective lens low enough to show the loader big either cut the loader at the pile / hopper
        # (15 m, 90 deg) or bent the 6.3 m columns and the hopper plant across the floor (12 m, 110 deg). Orthographic
        # instead (a map: same scale everywhere, nothing leans, height only for clipping): the work area 1 m wider on each
        # side, from the hopper's south edge to the back wall (28 x 29.6 m) in a 30 m square, the loader ~40 px of 224
        x0, x1, y0, y1 = _L.area[0] - 1.0, _L.area[1] + 1.0, _L.area[2] - 1.0, _L.back_wall_y
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        root.find(".//camera[@name='top']").attrib.update(pos=f'{cx:.2f} {cy:.2f} 20', xyaxes='1 0 0 0 1 0',
                                                         projection='orthographic', fovy=f'{max(x1 - x0, y1 - y0) + 0.4:.1f}')
    if not looks:
        return ET.tostring(root, encoding='unicode')
    asset = root.find('asset')
    world = root.find('worldbody')
    for name, (tex, attrs) in {**MATERIALS, **NEW_MATERIALS}.items():
        _texture_and_material(asset, name, tex, attrs)

    # the big hall makes the model extent ~145 m, and the near clip plane (znear x extent) ~2.9 m: the cab and rear
    # cameras then lost the ground and the loader's own front. 0.002 x extent = 0.3 m (render setting only)
    root.find('visual/map').set('znear', '0.002')
    # cab camera at the driver's eye inside the (lower) scanned cab: the plain loader's cab was 0.3 m higher
    if LOADER_LOOK in ('scan', 'model') and not wa100:  # driver's eye inside the drawn cab (under its roof)
        from cardboard_yard.env import _lookat_xyaxes

        cab = root.find(".//camera[@name='cab']")
        cab.set('pos', '-0.25 0 1.85')
        cab.set('xyaxes', _lookat_xyaxes((-0.25, 0.0, 1.85), (6.0, 0.0, 0.2)))
    # daylight from the open sides of the shed (looks only)
    root.find('visual/headlight').attrib.update(ambient='0.42 0.42 0.41', diffuse='0.3 0.3 0.29')
    sky = asset.find("texture[@name='sky']")
    sky.attrib.update(rgb1='0.72 0.82 0.95', rgb2='0.45 0.6 0.82')
    sun = world.find("light[@name='sun']")
    sun.attrib.update(dir='0.25 0.7 -0.65', diffuse='0.65 0.63 0.58')

    # --- hide the plain geoms of the building (they keep colliding exactly as before)
    for g in world.findall('geom'):
        nm, mat = g.get('name', ''), g.get('material', '')
        if nm in ('back_wall', 'back_wall_low', 'left_wall') or nm.startswith('col_'):
            _hide(g)
        elif not nm and g.get('contype') == '0' and mat in ('green', 'rust', 'steel_dark', 'roof', 'stripe'):
            _hide(g)
        elif not nm and mat.startswith('cardboard') and g.get('size') == '0.32 0.24 0.02':
            _hide(g)  # small background heaps -> replaced by the big heap meshes below
    # cardboard: 6 looks (pieces by index, floor litter by position)
    for g in root.iter('geom'):
        nm, mat = g.get('name', ''), g.get('material', '')
        if nm.startswith('piece') and nm.endswith('_g'):
            g.set('material', f'card{int(nm[5:-2]) % N_CARD_LOOKS}')
        elif mat.startswith('cardboard') and g.get('contype') == '0':
            g.set('material', f'card{zlib.crc32(g.get("pos", "").encode()) % N_CARD_LOOKS}')

    _shed(world)
    _heaps(asset, world)
    _hopper_plant(world)
    _mesh_asset(asset, 'heap_far', _outside(world))
    _machinery(world)
    _machinery_hall(world)
    _open_side(world, -1, -21.0)  # left (photo)
    _mesh_asset(asset, 'heap_end', _hall_end(world, 63.0))  # right: no photo; the hall goes on, ends in a wall
    _geom(world, type='mesh', mesh='heap_end', material='pile')
    if wa100:
        WS.add_looks(asset, root, _mesh_asset, _geom, _f)
    else:
        _loader(asset, root)
    return ET.tostring(root, encoding='unicode')


ROOF_BACK = (13.6, 4.5)  # (y, z): the roof is low over the back wall ...
ROOF_FRONT = (-14.0, 8.2)  # ... and rises towards the machinery side, as in the photos (both set per layout)


def roof_z(y):
    (y0, z0), (y1, z1) = ROOF_BACK, ROOF_FRONT
    return z0 + (y - y0) * (z1 - z0) / (y1 - y0)


def _shed(world):
    by = _L.back_wall_y
    wall_top = ROOF_BACK[1]
    # back wall: same box as the hidden collision wall (plain, seen from the top), faced with planes:
    # concrete blocks up to 3.6 m with rusty rails and caught cardboard, a strip of corrugated sheet above
    _geom(world, type='box', pos=_f(1, by + 0.2, wall_top / 2), size=_f(22, 0.2, wall_top / 2), material='wall_cap')
    _geom(world, type='plane', pos=_f(1, by - 0.03, 1.8), size='22 1.8 0.1', euler='1.5708 0 0', material='blocks')
    _geom(world, type='plane', pos=_f(1, by - 0.03, (3.6 + wall_top) / 2), size=_f(22, (wall_top - 3.6) / 2, 0.1),
          euler='1.5708 0 0', material='corrugated')
    for z in (0.9, 1.5, 2.1, 2.7, 3.3, 3.9):
        _geom(world, type='box', pos=_f(1, by - 0.09, z), size='22 0.05 0.05', material='rust_beam')
    for x in np.arange(-20.0, 23.0, 2.0):
        _geom(world, type='box', pos=_f(x, by - 0.09, wall_top / 2), size=_f(0.05, 0.05, wall_top / 2), material='rust_beam')
    for x in np.arange(-18.0, 23.0, 6.0):  # wall columns (H-beam) up to the roof
        _geom(world, type='box', pos=_f(x, by - 0.2, wall_top / 2), size=_f(0.18, 0.03, wall_top / 2), material='rust_beam')
        _geom(world, type='box', pos=_f(x, by - 0.1, wall_top / 2), size=_f(0.03, 0.1, wall_top / 2), material='rust_beam')
    rng = np.random.default_rng(5)
    for _ in range(160):
        x = rng.uniform(-19, 22)
        z = rng.choice([0.9, 1.5, 2.1, 2.7, 3.3, 3.9]) + rng.uniform(-0.05, 0.25)
        _geom(world, type='box', pos=_f(x, by - 0.16, z), size=_f(rng.uniform(0.15, 0.4), 0.01, rng.uniform(0.08, 0.22)),
              euler=f'{rng.uniform(-0.4, 0.4):.3f} 0 {rng.uniform(-0.3, 0.3):.3f}', material=f'card{int(rng.integers(N_CARD_LOOKS))}')
    # free-standing columns at the (hidden) collision columns: H-beams up to the roof girder,
    # red-brown paint at the bottom, green above
    for cy in _L.column_y:
        top = roof_z(cy) - 0.55
        for cx in _L.column_x:
            for z0, z1, mat in ((0.0, 1.9, 'col_red'), (1.9, top, 'col_green')):
                zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
                if getattr(_L, 'column_look', 'h') == 'box':  # square hollow section (laser scan), rounded corners
                    a = _L.column_half
                    _geom(world, type='box', pos=_f(cx, cy, zc), size=_f(a - 0.04, a, hz), material=mat)
                    _geom(world, type='box', pos=_f(cx, cy, zc), size=_f(a, a - 0.04, hz), material=mat)
                    for sx in (-1, 1):
                        for sy in (-1, 1):
                            _geom(world, type='cylinder', pos=_f(cx + sx * (a - 0.04), cy + sy * (a - 0.04), zc),
                                  size=_f(0.04, hz), material=mat)
                    continue
                for dy in (-0.2, 0.2):
                    _geom(world, type='box', pos=_f(cx, cy + dy, zc), size=_f(0.24, 0.025, hz), material=mat)
                _geom(world, type='box', pos=_f(cx, cy, zc), size=_f(0.025, 0.2, hz), material=mat)
        # grey girder along the column row
        _geom(world, type='box', pos=_f(1.5, cy, top + 0.25), size='20.5 0.14 0.25', material='girder_grey', group='3')
    # sloped roof: rafters (rust) along y every 3 m, light purlins along x, silver sheet (group 3: not in the top view)
    (y0, z0), (y1, z1) = ROOF_BACK, ROOF_FRONT
    L = float(np.hypot(y1 - y0, z1 - z0))
    th = float(np.arctan2(z1 - z0, y0 - y1))  # rise towards -y
    yc, zc = (y0 + y1) / 2, (z0 + z1) / 2
    # heavy rusty girders under the tin roof (as in the photos): deep rafters along y every 6 m,
    # medium beams along x every 3.5 m, light purlins in between
    for x in np.arange(-20.0, 47.0, 6.0):
        _geom(world, type='box', pos=_f(x, yc, zc - 0.42), size=_f(0.16, L / 2, 0.36), euler=f'{-th:.4f} 0 0',
              material='rust_beam', group='3')
        _geom(world, type='box', pos=_f(x, yc, zc - 0.76), size=_f(0.22, L / 2, 0.035), euler=f'{-th:.4f} 0 0',
              material='rust_beam', group='3')  # bottom flange
    for k in range(int(L / 3.5)):
        y = y0 + (y1 - y0) * (k + 0.5) / int(L / 3.5)
        _geom(world, type='box', pos=_f(21, y, roof_z(y) - 0.25), size='42 0.1 0.2', material='rust_beam', group='3')
    for k in range(int(L / 1.1)):
        y = y0 + (y1 - y0) * (k + 0.5) / int(L / 1.1)
        _geom(world, type='box', pos=_f(21, y, roof_z(y) - 0.07), size='42 0.05 0.05', material='girder_grey', group='3')
    _geom(world, type='plane', pos=_f(13, yc, zc), size=_f(34, L / 2, 0.1), euler=f'{np.pi - th:.4f} 0 0',
          material='roof_sheet', group='3')
    # third layer of paper rolls on the stacks at the far left (visual)
    for x in np.arange(-19.0, -15.0, 1.3):
        for y in np.arange(5.0, 13.0, 1.3):
            _geom(world, type='cylinder', pos=_f(x, y, 3.5), size='0.6 0.7', material='paper_roll')
    # more floor litter (visual): small scraps everywhere, denser towards the heaps
    for _ in range(700):
        x, y = rng.uniform(-18, 21), min(rng.uniform(-12, 13.2), rng.uniform(-12, 13.2) + 6)
        _geom(world, type='box', pos=_f(x, y, 0.003), size=_f(rng.uniform(0.03, 0.15), rng.uniform(0.02, 0.07), 0.002),
              euler=f'0 0 {rng.uniform(0, 3.14):.2f}', material=f'card{int(rng.integers(N_CARD_LOOKS))}')


def _heaps(asset, world):
    """Big cardboard heaps along the back wall, left and right of the work pile (whose area stays free)."""
    rng = np.random.default_rng(11)
    px = _L.pile_center[0]
    spans = [(-14.0, px - 3.8), (px + 4.0, 21.0)]
    for k, (x0, x1) in enumerate(spans):
        m = MS.heap(x0, x1, _L.back_wall_y - 3.6, _L.back_wall_y - 0.02, 4.0, rng)
        name = f'heap{k}'
        _mesh_asset(asset, name, m)
        _geom(world, type='mesh', mesh=name, material='pile')
        # loose sheets on and in front of the heap
        for _ in range(int((x1 - x0) * 40)):
            x = rng.uniform(x0 + 0.2, x1 - 0.2)
            y = rng.uniform(_L.back_wall_y - 4.2, _L.back_wall_y - 0.3)
            ty = (y - (_L.back_wall_y - 3.6)) / 3.6
            tx = min(x - x0, x1 - x) / max(1e-6, 0.25 * (x1 - x0))
            z = max(0.02, 4.0 * min(1.0, tx) ** 0.7 * max(0.0, ty) ** 0.8 * rng.uniform(0.75, 1.1))
            a, b, c = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(0, 3.14)
            if rng.random() < 0.25:  # folded / half-open boxes
                size = _f(rng.uniform(0.2, 0.4), rng.uniform(0.15, 0.3), rng.uniform(0.06, 0.18))
            else:  # flattened sheets
                size = _f(rng.uniform(0.2, 0.5), rng.uniform(0.15, 0.4), 0.012)
            _geom(world, type='box', pos=_f(x, y, z), size=size, euler=f'{a:.3f} {b:.3f} {c:.3f}',
                  material=f'card{int(rng.integers(N_CARD_LOOKS))}')


def _hopper_plant(world):
    """Dump point as in the site video: the (hidden-look) hopper walls are drawn as a dirty rusty steel tank,
    inside a green steel frame (corner columns, braces, top beams on the far and side faces; the loader side
    stays open), a green feeder rising from the far side, an orange forklift parked nearby (all visual)."""
    hx, hy = _L.hopper_center
    ex, ey = _L.hopper_half
    H = 4.0
    cxs, cys = (hx - ex - 0.4, hx + ex + 0.4), (hy - ey - 0.4, hy + ey + 0.4)
    for x in cxs:
        for y in cys:
            _geom(world, type='box', pos=_f(x, y, H / 2), size=_f(0.12, 0.12, H / 2), material='col_green')
    for y in cys:  # side faces: top beam + X braces
        _geom(world, type='box', pos=_f(hx, y, H - 0.1), size=_f(ex + 0.5, 0.1, 0.1), material='col_green')
        d = float(np.hypot(2 * ex + 0.8, H - 1.6))
        a = float(np.arctan2(H - 1.6, 2 * ex + 0.8))
        for sgn in (-1, 1):
            _geom(world, type='box', pos=_f(hx, y, (H + 1.6) / 2), size=_f(d / 2, 0.05, 0.05),
                  euler=f'0 {sgn * a:.4f} 0', material='col_green')
    x = cxs[0]  # far face: beam + braces + splash plate above the tank
    _geom(world, type='box', pos=_f(x, hy, H - 0.1), size=_f(0.1, ey + 0.5, 0.1), material='col_green')
    _geom(world, type='box', pos=_f(hx - ex - 0.12, hy, _L.hopper_wall_h + 1.0), size=_f(0.03, ey, 1.0), material='conveyor')
    # inside: stainless liner and the pulper rotor at the bottom (where the paper goes, photo of the vat)
    t = 0.08
    for sx in (-1, 1):
        _geom(world, type='box', pos=_f(hx + sx * (ex - 0.006), hy, _L.hopper_wall_h / 2), size=_f(0.005, ey, _L.hopper_wall_h / 2),
              material='stainless_in')
    for sy in (-1, 1):
        _geom(world, type='box', pos=_f(hx, hy + sy * (ey - 0.006), _L.hopper_wall_h / 2), size=_f(ex, 0.005, _L.hopper_wall_h / 2),
              material='stainless_in')
    _geom(world, type='box', pos=_f(hx, hy, 0.047), size=_f(ex - 0.012, ey - 0.012, 0.002), material='stainless_in')
    _geom(world, type='cylinder', pos=_f(hx, hy, 0.055), size='0.75 0.012', material='screen')
    _geom(world, type='cylinder', pos=_f(hx, hy, 0.09), size='0.28 0.05', material='chrome')
    for k in range(6):
        a = 2 * np.pi * k / 6
        _geom(world, type='box', pos=_f(hx + 0.45 * np.cos(a), hy + 0.45 * np.sin(a), 0.08), size='0.24 0.07 0.035',
              euler=f'0 0 {a + 0.5:.3f}', material='chrome')
    _geom(world, type='box', pos=_f(hx + 0.9, hy - 0.9, 0.044), size='0.25 0.18 0.003', material='black')  # outlet
    # green feeder rising away from the far side
    x0 = hx - ex - 0.5
    x1, z1 = x0 - 4.0, _L.hopper_wall_h + 2.6
    L = float(np.hypot(x1 - x0, z1 - _L.hopper_wall_h))
    pitch = float(np.arctan2(z1 - _L.hopper_wall_h, x0 - x1))
    cx, cz = (x0 + x1) / 2, (_L.hopper_wall_h + z1) / 2
    _geom(world, type='box', pos=_f(cx, hy, cz), size=_f(L / 2, ey * 0.75, 0.05), euler=f'0 {-pitch:.4f} 0', material='conveyor')
    for side in (-1, 1):
        _geom(world, type='box', pos=_f(cx, hy + side * ey * 0.78, cz + 0.35), size=_f(L / 2, 0.04, 0.4),
              euler=f'0 {-pitch:.4f} 0', material='conveyor')
    for xs in (x0 - 1.4, x0 - 3.0):
        zt = _L.hopper_wall_h + (x0 - xs) * np.tan(pitch)
        for side in (-1, 1):
            _geom(world, type='box', pos=_f(xs, hy + side * ey * 0.7, zt / 2), size=_f(0.08, 0.08, zt / 2), material='conveyor')
    _geom(world, type='cylinder', pos=_f(x1 - 1.9, hy, 1.6), size='1.8 1.6', material='tank_rust')
    # orange forklift (Toyota-style) parked to the side
    fx, fy = hx + 0.5, hy - 11.6  # beside the machinery (v0: -7.5, -9.6)
    _geom(world, type='box', pos=_f(fx, fy, 0.65), size='0.95 0.5 0.35', material='forklift')
    _geom(world, type='box', pos=_f(fx - 0.75, fy, 0.75), size='0.25 0.5 0.4', material='black')  # counterweight
    _geom(world, type='box', pos=_f(fx - 0.1, fy, 1.55), size='0.45 0.48 0.03', material='black')  # roof guard
    for dx in (0.3, -0.5):
        for dy in (-0.45, 0.45):
            _geom(world, type='box', pos=_f(fx + dx, fy + dy, 1.2), size='0.03 0.03 0.36', material='black')
            _geom(world, type='cylinder', pos=_f(fx + (dx * 1.6), fy + dy * 1.1, 0.3), size='0.3 0.12',
                  euler='1.5708 0 0', material='tire')
    _geom(world, type='box', pos=_f(fx + 1.05, fy, 1.3), size='0.06 0.35 1.25', material='black')  # mast
    for dy in (-0.25, 0.25):
        _geom(world, type='box', pos=_f(fx + 1.55, fy + dy, 0.08), size='0.5 0.06 0.03', material='chrome')  # forks


def _machinery(world):
    """Machinery side of the hall (photo 'back side of the driving area'), at y < -10: green inclined feeder,
    dirty grey vats, green steel frame towers, overhead pipes and crane girder, paper roll stacks, crates,
    a parked forklift. Visual only; outside the area where the loader and the people work."""
    world = ET.SubElement(world, 'body', {'name': 'machinery', 'pos': _f(0, MACH_DY, 0)})  # static, moves with the far column row
    y = -12.0
    # long green feeder rising towards -x
    x0, z0, x1, z1 = -1.0, 0.8, -9.5, 3.4
    L = float(np.hypot(x1 - x0, z1 - z0))
    p = float(np.arctan2(z1 - z0, x0 - x1))
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    _geom(world, type='box', pos=_f(cx, y, cz), size=_f(L / 2, 1.2, 0.06), euler=f'0 {-p:.4f} 0', material='green_machine')
    for sy in (-1, 1):
        _geom(world, type='box', pos=_f(cx, y + sy * 1.2, cz + 0.6), size=_f(L / 2, 0.05, 0.65), euler=f'0 {-p:.4f} 0',
              material='green_machine')
    for xs in (-3.0, -5.5, -8.0):
        zt = z0 + (x0 - xs) * np.tan(p)
        _geom(world, type='box', pos=_f(xs, y, zt / 2), size=_f(0.1, 1.0, zt / 2), material='green_machine')
    _geom(world, type='box', pos=_f(0.6, y, 0.6), size='1.6 1.4 0.6', material='green_machine')  # receiving end
    # dirty grey vats and frame towers
    for vx, w in ((-12.0, 1.6), (4.5, 1.4), (13.0, 1.8)):
        _geom(world, type='box', pos=_f(vx, y + 0.3, 1.0), size=_f(w, 1.3, 1.0), material='tank_rust')
    for fx in (7.5, 9.5):
        for fy in (y - 1.0, y + 1.0):
            _geom(world, type='box', pos=_f(fx, fy, 2.6), size='0.09 0.09 2.6', material='green_machine')
        _geom(world, type='box', pos=_f(fx, y, 5.1), size='0.09 1.1 0.09', material='green_machine')
        _geom(world, type='box', pos=_f(fx, y, 2.6), size='0.05 1.4 0.05', euler='1.0 0 0', material='green_machine')
    _geom(world, type='box', pos=_f(8.5, y, 3.2), size='1.1 1.1 0.08', material='green_machine')
    x0, z0, x1, z1 = 11.0, 2.2, 18.0, 1.2  # second, lower conveyor
    L = float(np.hypot(x1 - x0, z1 - z0))
    p = float(np.arctan2(z0 - z1, x1 - x0))
    _geom(world, type='box', pos=_f((x0 + x1) / 2, y, (z0 + z1) / 2 + 0.4), size=_f(L / 2, 1.0, 0.5), euler=f'0 {p:.4f} 0',
          material='green_machine')
    # overhead pipes and a rusty crane girder
    for k, z in enumerate((5.4, 5.7)):
        _geom(world, type='cylinder', pos=_f(12.5, -10.6 - 0.3 * k, z), size='0.12 34', euler='0 1.5708 0', material='pipe')
    _geom(world, type='box', pos=_f(12.5, -9.8, 6.4), size='34 0.2 0.35', material='rust_beam', group='3')
    for vx, w in ((24.0, 1.6), (36.0, 1.5)):
        _geom(world, type='box', pos=_f(vx, y + 0.3, 1.0), size=_f(w, 1.3, 1.0), material='tank_rust')
    for fx in (29.5, 31.5):
        for fy in (y - 1.0, y + 1.0):
            _geom(world, type='box', pos=_f(fx, fy, 2.6), size='0.09 0.09 2.6', material='green_machine')
        _geom(world, type='box', pos=_f(fx, y, 5.1), size='0.09 1.1 0.09', material='green_machine')
    _geom(world, type='box', pos=_f(30.5, y, 3.2), size='1.1 1.1 0.08', material='green_machine')
    _geom(world, type='box', pos=_f(33.5, y, 1.6), size='3.6 1.1 0.06', euler='0 0.35 0', material='green_machine')
    for x in np.arange(40.0, 46.5, 1.15):
        for yy in (-13.0, -11.8):
            _geom(world, type='cylinder', pos=_f(x, yy, 0.6), size='0.55 0.6', material='paper_roll')
            _geom(world, type='cylinder', pos=_f(x, yy, 1.8), size='0.55 0.6', material='paper_roll')
    # paper roll stacks, wooden crates, mesh bins
    rng = np.random.default_rng(21)
    for x in np.arange(-20.0, -14.5, 1.15):
        for yy in (-13.0, -11.8):
            for k in range(int(rng.integers(1, 4))):
                _geom(world, type='cylinder', pos=_f(x, yy, 0.6 + 1.2 * k), size='0.55 0.6', material='paper_roll')
    for cx in (-14.0, -13.0):
        _geom(world, type='box', pos=_f(cx, -10.3, 0.75), size='0.5 0.6 0.75', material='wood')
    _geom(world, type='box', pos=_f(-15.6, -10.2, 0.6), size='0.9 0.6 0.6', material='mesh')


def _machinery_hall(world):
    """Behind the machinery: hall wall with a bright window band, upper steel deck, a tall silo."""
    world = ET.SubElement(world, 'body', {'name': 'machinery_hall', 'pos': _f(0, MACH_DY, 0)})  # static, moves with the far column row
    yb = -15.5
    _geom(world, type='box', pos=_f(21, yb, 3.0), size='43 0.15 3.0', material='corrugated')
    _geom(world, type='box', pos=_f(21, yb, 6.6), size='43 0.1 0.6', material='window')
    _geom(world, type='box', pos=_f(21, yb + 0.2, 8.0), size='43 0.15 0.8', material='corrugated')
    for x in np.arange(-20.0, 47.0, 6.0):
        _geom(world, type='box', pos=_f(x, yb + 0.3, 4.4), size='0.18 0.18 4.4', material='rust_beam')
    _geom(world, type='box', pos=_f(12.5, -14.6, 4.6), size='34 1.0 0.12', material='rust_beam')  # upper deck
    _geom(world, type='box', pos=_f(12.5, -13.65, 5.0), size='34 0.03 0.4', material='rust_beam')  # deck rail
    _geom(world, type='cylinder', pos=_f(-4.0, -14.0, 4.5), size='2.2 4.5', material='pipe')  # silo


def _hall_end(world, x_end):
    """Right part of the hall (no photo): same columns, roof girders, wall and heaps as the photographed part,
    closed by an end wall with blocks below and corrugated sheet above."""
    # the collision floor plane ends at |x| = 40: same floor drawn beyond it (visual)
    for xc in (-60.0, 60.0):
        _geom(world, type='plane', pos=_f(xc, 0, 0), size='20 40 0.1', material='floor')
    for cx in (50.0, 58.0):
        for cy in _L.column_y:
            top = roof_z(cy) - 0.55
            for z0, z1, mat in ((0.0, 1.9, 'col_red'), (1.9, top, 'col_green')):
                _geom(world, type='box', pos=_f(cx, cy, (z0 + z1) / 2), size=_f(0.24, 0.2, (z1 - z0) / 2), material=mat)
    (y0, z0), (y1, z1) = ROOF_BACK, ROOF_FRONT
    L = float(np.hypot(y1 - y0, z1 - z0))
    th = float(np.arctan2(z1 - z0, y0 - y1))
    yc, zc = (y0 + y1) / 2, (z0 + z1) / 2
    xc, hx = (47.0 + x_end) / 2, (x_end - 47.0) / 2
    _geom(world, type='plane', pos=_f(xc, yc, zc), size=_f(hx, L / 2, 0.1), euler=f'{np.pi - th:.4f} 0 0',
          material='roof_sheet', group='3')
    for x in np.arange(50.0, x_end, 6.0):
        _geom(world, type='box', pos=_f(x, yc, zc - 0.42), size=_f(0.16, L / 2, 0.36), euler=f'{-th:.4f} 0 0',
              material='rust_beam', group='3')
    _geom(world, type='box', pos=_f(xc, _L.back_wall_y + 0.2, z0 / 2), size=_f(hx, 0.2, z0 / 2), material='blocks')
    _geom(world, type='box', pos=_f(xc, -15.5, 4.4), size=_f(hx, 0.15, 4.4), material='corrugated')
    # end wall: blocks up to 3.6 m, corrugated sheet up to the roof
    zt = max(z0, z1)
    _geom(world, type='box', pos=_f(x_end, -0.9, 1.8), size='0.2 14.5 1.8', material='blocks')
    _geom(world, type='box', pos=_f(x_end, -0.9, (3.6 + zt) / 2), size=_f(0.2, 14.5, (zt - 3.6) / 2), material='corrugated')
    for y in np.arange(-14.0, 14.0, 6.0):
        _geom(world, type='box', pos=_f(x_end - 0.3, y, zt / 2), size=_f(0.18, 0.18, zt / 2), material='rust_beam')
    m = MS.heap(47.0, x_end - 0.5, _L.back_wall_y - 3.6, _L.back_wall_y - 0.02, 4.0, np.random.default_rng(17))
    return m


def _open_side(world, sgn, x_edge):
    """Open end of the hall (photo 'left side'; the right end, with no photo, is built the same way):
    the green roof structure continues outside, bale stacks beyond it, a red-roofed shed far away."""
    xs = [x_edge + sgn * d for d in (3.0, 8.0, 13.0)]
    for x in xs:
        for y in (-7.0, 8.0):
            _geom(world, type='box', pos=_f(x, y, 3.0), size='0.15 0.15 3.0', material='green_machine')
        _geom(world, type='box', pos=_f(x, 0.5, 6.0), size='0.12 9.5 0.15', material='green_machine', group='3')
    xc = x_edge + sgn * 8.0
    for y in np.arange(-8.0, 9.5, 1.5):
        _geom(world, type='box', pos=_f(xc, y, 6.2), size='6 0.05 0.08', material='green_machine', group='3')
    _geom(world, type='box', pos=_f(xc, 0.5, 6.35), size='6.2 10 0.04', material='roof_sheet', group='3')
    _geom(world, type='box', pos=_f(x_edge + sgn * 29.0, 4, 4.2), size='6 9 0.15', euler=f'0 {sgn * 0.12:.2f} 0',
          material='red_roof')  # far shed
    for y in (-4.0, 12.0):
        _geom(world, type='box', pos=_f(x_edge + sgn * 29.0, y, 2.0), size='0.15 0.15 2.0', material='rust_beam')
    rng = np.random.default_rng(31 if sgn < 0 else 32)
    for y in np.arange(-6.0, 8.0, 1.3):
        for k in range(int(rng.integers(1, 4))):
            _geom(world, type='box', pos=_f(x_edge + sgn * 17.0, y, 0.55 + 1.1 * k), size='0.62 0.6 0.55', material='bale')


def _outside(world):
    """Beyond the open sides: bale stacks and a neighbouring building."""
    for cx in (26.0, 34.0, 42.0):  # the long shed continues to the right
        for cy in _L.column_y:
            top = roof_z(cy) - 0.55
            for z0, z1, mat in ((0.0, 1.9, 'col_red'), (1.9, top, 'col_green')):
                _geom(world, type='box', pos=_f(cx, cy, (z0 + z1) / 2), size=_f(0.24, 0.2, (z1 - z0) / 2), material=mat)
    (y0, z0), (y1, z1) = ROOF_BACK, ROOF_FRONT
    L = float(np.hypot(y1 - y0, z1 - z0))
    th = float(np.arctan2(z1 - z0, y0 - y1))
    _geom(world, type='plane', pos=_f(34, (y0 + y1) / 2, (z0 + z1) / 2), size=_f(12, L / 2, 0.1),
          euler=f'{np.pi - th:.4f} 0 0', material='roof_sheet', group='3')
    _geom(world, type='box', pos=_f(34, _L.back_wall_y + 0.2, z0 / 2), size=_f(12, 0.2, z0 / 2), material='blocks')
    m = MS.heap(23.0, 46.0, _L.back_wall_y - 3.6, _L.back_wall_y - 0.02, 4.0, np.random.default_rng(13))
    _geom(world, type='mesh', mesh='heap_far', material='pile')
    return m


SCAN_DIR = os.path.join(ASSETS, 'komatsu_scan')  # made by make_scan_loader.py (CC BY-NC-SA scan, see LICENSE.txt)


_SCAN_CACHE = {}


def _scan_part(name):
    """(V, UV, F) of a scan part OBJ in its body frame (cached)."""
    if name not in _SCAN_CACHE:
        V, T, F = [], [], []
        for l in open(os.path.join(SCAN_DIR, name + '.obj')):
            if l.startswith('v '):
                V.append([float(x) for x in l.split()[1:4]])
            elif l.startswith('vt'):
                T.append([float(x) for x in l.split()[1:3]])
            elif l.startswith('f '):
                F.append([int(t.split('/')[0]) - 1 for t in l.split()[1:]])
        _SCAN_CACHE[name] = (np.array(V), np.array(T), np.array(F))
    return _SCAN_CACHE[name]


def scan_textured(mesh, parts, max_edge=0.06, up_only=False):
    """Give a filler mesh the scan's own texture: split it into small triangles, and map each one onto the
    nearest scanned triangle of the same body (its corners projected onto that triangle, barycentric
    coordinates clamped inside it, so the texture never bleeds into another island of the atlas).
    The filler then shows the rust / dust / paint that surrounds it instead of a flat colour."""
    import hashlib
    import pickle

    # cached on disk: same filler mesh + same scan parts (by content) + same options -> same result
    h = hashlib.sha1(np.asarray(mesh.v, np.float64).tobytes() + np.asarray(mesh.f, np.int64).tobytes())
    for p in parts:
        h.update(open(os.path.join(SCAN_DIR, p + '.obj'), 'rb').read())
    h.update(repr((max_edge, up_only, 1)).encode())
    cache = os.path.join(MESH_DIR, 'cache', h.hexdigest() + '.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
            return pickle.load(fh)
    out = _scan_textured(mesh, parts, max_edge, up_only)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + f'.{os.getpid()}.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(out, fh)
    os.replace(tmp, cache)
    return out


def _scan_textured(mesh, parts, max_edge, up_only):
    import trimesh
    from scipy.spatial import cKDTree

    tm = trimesh.Trimesh(np.asarray(mesh.v), np.asarray(mesh.f), process=False)
    v, f = trimesh.remesh.subdivide_to_size(tm.vertices, tm.faces, max_edge=max_edge, max_iter=8)
    SV, ST, SF, offs = [], [], [], 0
    for p in parts:
        pv, pt, pf = _scan_part(p)
        SV.append(pv)
        ST.append(pt)
        SF.append(pf + offs)
        offs += len(pv)
    SV, ST, SF = np.concatenate(SV), np.concatenate(ST), np.concatenate(SF)
    if up_only:  # only faces looking up (paint on top surfaces; no pipes, glass, sides)
        n = np.cross(SV[SF[:, 1]] - SV[SF[:, 0]], SV[SF[:, 2]] - SV[SF[:, 0]])
        SF = SF[np.abs(n[:, 2]) > 0.85 * np.linalg.norm(n, axis=1)]
    A, B, C = SV[SF[:, 0]], SV[SF[:, 1]], SV[SF[:, 2]]
    tree = cKDTree((A + B + C) / 3)
    out = MS.Mesh()
    _, idx = tree.query(v[f].mean(1))
    for (i0, i1, i2), k in zip(f, idx):
        a, b, c = A[k], B[k], C[k]
        e0, e1 = b - a, c - a
        d00, d01, d11 = e0 @ e0, e0 @ e1, e1 @ e1
        den = d00 * d11 - d01 * d01 or 1e-12
        uvs = []
        for vi in (i0, i1, i2):
            w = v[vi] - a
            d20, d21 = w @ e0, w @ e1
            l1 = np.clip((d11 * d20 - d01 * d21) / den, 0, 1)
            l2 = np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
            if l1 + l2 > 1:
                l1, l2 = l1 / (l1 + l2), l2 / (l1 + l2)
            ta, tb, tc = ST[SF[k, 0]], ST[SF[k, 1]], ST[SF[k, 2]]
            uvs.append(tuple((1 - l1 - l2) * ta + l1 * tb + l2 * tc))
        out.tri(tuple(v[i0]), tuple(v[i1]), tuple(v[i2]), *uvs)
    return out


def _loader_scan(asset, root):
    """Loader drawn with the textured photogrammetry scan of a real Komatsu, one mesh per moving body."""
    body = {b.get('name'): b for b in root.iter('body')}
    tex = os.path.join(SCAN_DIR, 'texture.png').replace(os.sep, '/')
    ET.SubElement(asset, 'texture', {'name': 'tex_scan', 'type': '2d', 'file': tex})
    ET.SubElement(asset, 'material', {'name': 'scan', 'texture': 'tex_scan', 'specular': '0.2', 'shininess': '0.2'})
    blur = os.path.join(SCAN_DIR, 'texture_blur.png').replace(os.sep, '/')
    ET.SubElement(asset, 'texture', {'name': 'tex_scan_blur', 'type': '2d', 'file': blur})
    ET.SubElement(asset, 'material', {'name': 'scan_fill', 'texture': 'tex_scan_blur', 'specular': '0.2', 'shininess': '0.2'})
    for bname in LOADER_PARTS:
        for g in body[bname].findall('geom'):
            _hide(g)  # plain boxes / cylinders of the loader: collision unchanged, not drawn
        part = bname
        path = os.path.join(SCAN_DIR, part + '.obj').replace(os.sep, '/')
        ET.SubElement(asset, 'mesh', {'name': f'scan_{part}', 'file': path, 'inertia': 'shell'})
        _geom(body[bname], type='mesh', mesh=f'scan_{part}', material='scan')
        fill = os.path.join(SCAN_DIR, part + '_fill.obj')
        if os.path.exists(fill) and os.path.getsize(fill) > 0:  # holes filled from the mirrored other side
            ET.SubElement(asset, 'mesh', {'name': f'scan_{part}_fill', 'file': fill.replace(os.sep, '/'), 'inertia': 'shell'})
            _geom(body[bname], type='mesh', mesh=f'scan_{part}_fill', material='scan_fill')
    # dark chassis under the scan (meshes.py): the scan has no underside, this is what shows through
    # cab roof: the scan's roof is a white smear (reflections, never seen from above) -> yellow ROPS roof with
    # a dark rim and two work lamps, sized to the scanned cab (rear.obj top: x -0.83..0.29, |y| < 0.6, z 2.18)
    roof = MS.rounded_box(0.6, 0.62, 0.055, e=0.3).moved(-0.27, 0.0, 2.17)
    rim = MS.box(0.6, 0.62, 0.012).moved(-0.27, 0.0, 2.12)
    lamps = MS.Mesh()
    for y in (-0.38, 0.38):
        lamps.add(MS.box(0.025, 0.05, 0.025).moved(0.31, y, 2.14))
    ET.SubElement(asset, 'material', {'name': 'roof_worn', 'rgba': '0.80 0.62 0.16 1', 'specular': '0.15'})
    # roof texture borrowed from the hood top (same sun-faded yellow paint; the cab top itself is glass/glare):
    # look it up with the roof shifted onto the hood, then shift it back
    hood_dx, hood_dz = -1.5, -0.52
    roof_tex = scan_textured(roof.moved(hood_dx, 0, hood_dz), ['rear'], up_only=True).moved(-hood_dx, 0, -hood_dz)
    for name, m, mat in (('scan_roof', roof_tex, 'scan'), ('scan_roof_rim', rim, 'black'), ('scan_roof_lamps', lamps, 'chrome')):
        _mesh_asset(asset, name, m)
        _geom(body['rear'], type='mesh', mesh=name, material=mat)
    # complete wheels and bucket under the scan (the scan has only the outer tread of the tires and a ragged
    # bucket): a full tire with sidewalls + rim per wheel and a full bucket shell, slightly smaller than the scan,
    # coloured like it (dusty grey rubber, rusty rims and bucket). They show only where the scan has holes.
    ET.SubElement(asset, 'material', {'name': 'tire_dusty', 'rgba': '0.32 0.31 0.3 1', 'specular': '0.05'})
    ET.SubElement(asset, 'material', {'name': 'rim_rust', 'rgba': '0.55 0.33 0.17 1', 'specular': '0.2'})
    ET.SubElement(asset, 'material', {'name': 'bucket_rust', 'texture': 'tex_rust_proc_cube', 'rgba': '0.765 0.496 0.358 1', 'specular': '0.15'})
    shrink = lambda k: [[k, 0, 0], [0, k, 0], [0, 0, k]]
    for w in ('wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr'):
        # sized and centred on this wheel's scanned tire (radius ~0.56, about 0.35 m wide, off-centre in y)
        wv = np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(SCAN_DIR, w + '.obj')) if l.startswith('v ')])
        y0, y1 = np.percentile(wv[:, 1], 3), np.percentile(wv[:, 1], 97)
        yc, hw = (y0 + y1) / 2, 0.42 * (y1 - y0)
        core = MS.Mesh().add(MS.tire(0.5, hw, lug_h=0.02)).add(MS.rim(0.31).moved(rot=[[1, 0, 0], [0, hw / 0.19, 0], [0, 0, 1]]))
        _mesh_asset(asset, f'scan_wheel_core_{w}', scan_textured(core.moved(0, yc, 0), [w]))
        _geom(body[w], type='mesh', mesh=f'scan_wheel_core_{w}', material='scan')
    parts = MS.loader_meshes(BOOM_LEN)
    shell = MS.Mesh().add(parts['bucket_shell'][1]).add(parts['bucket_sides'][1])
    # fit it just inside the scanned bucket (bucket.obj extents, bucket frame): no part of it may stick out
    sv = np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(SCAN_DIR, 'bucket.obj')) if l.startswith('v ')])
    # x / y: just inside the scan (2..98 %); z: up to the very top of the scanned back wall (99.5 %), so the
    # torn, tooth-like top edge of the scan has solid wall behind it instead of a gap
    lo, hi = np.percentile(sv, 2, axis=0), np.percentile(sv, 98, axis=0)
    hi[2] = np.percentile(sv[:, 2], 99.5)
    pv = np.asarray(shell.v)
    plo, phi = pv.min(0), pv.max(0)
    k = (hi - lo) / (phi - plo) * np.array([0.96, 0.96, 1.0])
    ctr = (lo + hi) / 2
    ctr[2] = lo[2] + (hi[2] - lo[2]) / 2
    shell.v = [tuple(p) for p in ((pv - (plo + phi) / 2) * k + ctr)]
    _mesh_asset(asset, 'scan_bucket_core', scan_textured(shell, ['bucket']))
    _geom(body['bucket'], type='mesh', mesh='scan_bucket_core', material='scan')
    # boom: the scan's lift arms are open on their inner faces -> two solid arms + cross tube inside them,
    # following the scanned arm profile (boom.obj: arms at |y| ~0.4, centre height rising to the middle)
    arms = MS.Mesh()
    for y in (-0.4, 0.4):
        # down to the bucket pin at BOOM_LEN (z 0), so arm and bucket meet without a gap
        arms.add(MS.extrude([(0.0, 0.02), (0.9, 0.16), (1.7, 0.16), (2.3, -0.06), (BOOM_LEN + 0.08, -0.1),
                             (BOOM_LEN + 0.08, 0.1), (2.3, 0.18), (1.7, 0.36), (0.9, 0.36), (0.0, 0.26)],
                            y - 0.025, y + 0.025))
        arms.add(MS.revolve_y([(0.11, y - 0.07), (0.11, y + 0.07)], n=16).moved(BOOM_LEN, 0, 0))  # pin boss
    arms.add(MS.revolve_y([(0.08, -0.4), (0.08, 0.4)], n=16).moved(1.25, 0, 0.26))
    _mesh_asset(asset, 'scan_boom_core', scan_textured(arms, ['boom']))
    _geom(body['boom'], type='mesh', mesh='scan_boom_core', material='scan')
    # narrow and low, so it stays inside the scanned body and only fills the view under it
    for name, bname, m in (('scan_chassis_rear', 'rear', MS.box(0.9, 0.26, 0.16).moved(-1.15, 0, 0.66)),
                           ('scan_chassis_front', 'front', MS.box(0.65, 0.26, 0.15).moved(0.85, 0, 0.66))):
        _mesh_asset(asset, name, scan_textured(m, [bname]))
        _geom(body[bname], type='mesh', mesh=name, material='scan')


def _loader_model(asset, root):
    """Hand-built loader (loader_model.py), shapes and sizes taken from the scan and the site photos."""
    from . import loader_model as LM
    from cardboard_yard.env import WHEEL_R

    body = {b.get('name'): b for b in root.iter('body')}
    tex = LM.make_textures()
    for name, (tk, rgba, spec, shin) in LM.MATERIALS.items():
        attrs = {'name': name, 'rgba': rgba, 'specular': str(spec), 'shininess': str(shin)}
        if tk:
            ET.SubElement(asset, 'texture', {'name': 'tex_' + name, 'type': 'cube', 'file': tex[tk].replace(os.sep, '/')})
            attrs['texture'] = 'tex_' + name
        ET.SubElement(asset, 'material', attrs)
    for bname in LOADER_PARTS:
        for g in body[bname].findall('geom'):
            _hide(g)  # plain boxes / cylinders of the loader: collision unchanged, not drawn
    parts = LM.build(BOOM_LEN, WHEEL_R)
    for bname, items in parts.items():
        targets = ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr'] if bname == 'wheel' else [bname]
        for name, mat, m in items:
            mname = f'lm_{bname}_{name}'
            _mesh_asset(asset, mname, m)
            for t in targets:
                _geom(body[t], type='mesh', mesh=mname, material=mat)
    # KOMATSU on both lift arms and the counterweight, WA100 on the hood sides
    decals = [('boom', (1.6, 0.565, 0.2), 0.6, 0.11, '+y', 'decal'), ('boom', (1.6, -0.565, 0.2), 0.6, 0.11, '-y', 'decal'),
              ('rear', (-2.705, 0.0, 0.76), 0.4, 0.075, '-x', 'decal'),
              ('rear', (-1.75, 0.606, 1.27), 0.15, 0.055, '+y', 'decal_model'), ('rear', (-1.75, -0.606, 1.27), 0.15, 0.055, '-y', 'decal_model')]
    for k, (bname, c, hw, hh, nrm, mat) in enumerate(decals):
        _mesh_asset(asset, f'lm_decal{k}', MS.decal(c, hw, hh, nrm))
        _geom(body[bname], type='mesh', mesh=f'lm_decal{k}', material=mat)
    _hydraulic_bodies(root.find('worldbody'))


def _hydraulic_bodies(world):
    """Lift cylinders, tilt cylinder, bell crank and links as mocap bodies (not in the physics state; placed by
    CardboardYardRealEnv._update_hydraulics before every render). Each body's local x runs along the part
    (crank: local frame = boom frame at the crank pivot)."""
    sz = HY._sizes()

    def cyl(b, p0, p1, r, mat):
        _geom(b, type='cylinder', fromto=_f(*p0, *p1), size=f'{r:.4f}', material=mat)

    def body(name):
        return ET.SubElement(world, 'body', {'name': f'hyd_{name}', 'mocap': 'true', 'pos': '0 0 -20'})

    for kind, ys, rb, rr in (('lift', (-HY.LIFT_Y, HY.LIFT_Y), 0.068, 0.036), ('tilt', (0.0,), 0.08, 0.045)):
        barrel, rod = sz[kind][:2]
        bb, br = body(f'{kind}_barrel'), body(f'{kind}_rod')
        for y in ys:
            cyl(bb, (0, y - 0.06, 0), (0, y + 0.06, 0), 0.055, 'ld_black')  # base eye
            cyl(bb, (0.05, y, 0), (barrel - 0.06, y, 0), rb, 'ld_paint')  # barrel
            cyl(bb, (0.0, y, 0), (0.09, y, 0), rb * 0.8, 'ld_black')  # base cap
            cyl(bb, (barrel - 0.07, y, 0), (barrel, y, 0), rb + 0.008, 'ld_black')  # gland
            cyl(br, (0, y - 0.05, 0), (0, y + 0.05, 0), rr + 0.02, 'ld_black')  # rod eye
            cyl(br, (0.03, y, 0), (rod, y, 0), rr, 'ld_chrome')  # rod
    cr = body('crank')
    t, c = HY.CRANK_T, HY.CRANK_C
    for y in (-0.06, 0.06):  # two plates
        _geom(cr, type='capsule', fromto=_f(0, y, 0, t[0], y, t[1]), size='0.055', material='ld_paint')
        _geom(cr, type='capsule', fromto=_f(0, y, 0, c[0], y, c[1]), size='0.06', material='ld_paint')
    for x, z, r in ((0, 0, 0.065), (t[0], t[1], 0.04), (c[0], c[1], 0.04)):
        cyl(cr, (x, -0.1, z), (x, 0.1, z), r, 'ld_black')
    lk = body('links')
    for y in (-0.12, 0.12):
        _geom(lk, type='capsule', fromto=_f(0, y, 0, HY.LINK_LEN, y, 0), size='0.038', material='ld_paint')
    cyl(lk, (0, -0.16, 0), (0, 0.16, 0), 0.04, 'ld_black')
    cyl(lk, (HY.LINK_LEN, -0.16, 0), (HY.LINK_LEN, 0.16, 0), 0.04, 'ld_black')


def _loader(asset, root):
    if LOADER_LOOK == 'model':
        return _loader_model(asset, root)
    if LOADER_LOOK == 'scan' and os.path.exists(os.path.join(SCAN_DIR, 'texture.png')):
        return _loader_scan(asset, root)
    body = {b.get('name'): b for b in root.iter('body')}
    parts = MS.loader_meshes(BOOM_LEN)
    for name, (_, m) in parts.items():
        _mesh_asset(asset, f'ld_{name}', m)
    for bname, names in LOADER_PARTS.items():
        b = body[bname]
        for g in b.findall('geom'):
            _hide(g)  # plain boxes / cylinders of the loader: collision unchanged, not drawn
        for name in names:
            mat, _ = parts[name]
            _geom(b, type='mesh', mesh=f'ld_{name}', material=mat)
    # KOMATSU decals where the real machine has them: outer side of both lift arms, rear counterweight,
    # both sides of the engine hood (textured quads, visual only)
    decals = {
        'boom': [((1.75, 0.665, 0.17), 0.62, 0.12, '+y'), ((1.75, -0.665, 0.17), 0.62, 0.12, '-y')],
        'rear': [((-2.56, 0.0, 0.85), 0.42, 0.08, '-x'), ((-1.25, 0.625, 1.25), 0.2, 0.04, '+y'),
                 ((-1.25, -0.625, 1.25), 0.2, 0.04, '-y'), ((-0.5, 0.65, 1.15), 0.22, 0.045, '+y'),
                 ((-0.5, -0.65, 1.15), 0.22, 0.045, '-y')],
    }
    for bname, items in decals.items():
        for k, (c, hw, hh, nrm) in enumerate(items):
            name = f'decal_{bname}{k}'
            _mesh_asset(asset, name, MS.decal(c, hw, hh, nrm))
            _geom(body[bname], type='mesh', mesh=name, material='decal')
    for k, (c, nrm) in enumerate((((-1.95, 0.626, 1.52), '+y'), ((-1.95, -0.626, 1.52), '-y'))):
        name = f'decal_model{k}'  # WA100 plate high on the hood side, towards the rear
        _mesh_asset(asset, name, MS.decal(c, 0.17, 0.065, nrm))
        _geom(body['rear'], type='mesh', mesh=name, material='decal_model')


class CardboardYardRealEnv(WalkingPeople, CardboardYardEnv):
    """CardboardYardEnv with the looks of the real site.

    physics: 'v1' (default, smoother loader, see physics.py), 'v2' (v1 on the real site layout from the laser scan)
    or 'v0' (identical to swm/CardboardYard-v0).
    looks=False draws the plain scene with the same physics (used by test_same_physics).
    """

    def __init__(self, *args, physics='v1', looks=True, **kwargs):
        self.physics, self.looks = physics, looks
        self.layout = PH.layout(physics)
        for k, v in PH.control(physics).items():
            setattr(self, k, v)
        self.machine = PH.machine(physics)
        hyd = PH.hydraulic(physics)
        self.hydraulic = HY.HydraulicModel(**hyd) if hyd else None
        # profile 'proprio_pressure' (v4): proprio gets the lift and tilt cylinder pressures (MPa) after the usual 10
        self.proprio_pressure = bool(PH.PROFILES[physics].get('proprio_pressure')) and self.hydraulic is not None
        self.walk_people = PH.PROFILES[physics].get('people') == 'walk'  # v4: people.py
        super().__init__(*args, **kwargs)
        if self.walk_people:
            self._people_init()
        if self.proprio_pressure:
            self.proprio_dim = 12
            n = self.observation_space.shape[0] + 2
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.env_name = 'CardboardYardReal'

    def _proprio(self):
        p = super()._proprio()
        if not self.proprio_pressure:
            return p
        return np.concatenate([p, np.array(self.hydraulic.pressures(self), np.float32)])

    def _build_xml(self, n_pieces):
        return build_real_xml(n_pieces, self.physics, self.looks)

    def step(self, action):
        if self.hydraulic is not None:  # physics v3: force / flow / power limits of the hydraulics
            action = self.hydraulic.limit(self, action)
        return super().step(action)

    def render(self, camera=None, width=None, height=None):
        self._update_hydraulics()
        return super().render(camera, width, height)

    def _update_hydraulics(self):
        """Place the cylinder / Z-bar mocap bodies on the current boom and bucket angles. Writes only the mocap
        bodies' poses (and their geoms' world poses, so this frame shows them): no effect on the physics."""
        m, d = self.model, self.data
        if not hasattr(self, '_hyd_ids'):
            self._hyd_ids = {}
            for name in ('lift_barrel', 'lift_rod', 'tilt_barrel', 'tilt_rod', 'crank', 'links', 'lever', 'link'):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'hyd_{name}')
                if bid >= 0:
                    g0 = m.body_geomadr[bid]
                    self._hyd_ids[name] = (bid, m.body_mocapid[bid], range(g0, g0 + m.body_geomnum[bid]))
            self._front_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'front')
        if not self._hyd_ids:
            return
        fb = self._front_bid
        fp, fq, fR = d.xpos[fb], d.xquat[fb], d.xmat[fb].reshape(3, 3)
        q, R, G = np.empty(4), np.empty(9), np.empty(9)
        if self.machine.name == 'wa100':
            from . import wa100_sim as WS

            parts = WS.poses(self.joint('boom'), self.joint('bucket'))
        else:
            parts = HY.poses(self.joint('boom'), self.joint('bucket'))
        for name, ((x, z), beta) in parts.items():
            bid, mid, geoms = self._hyd_ids[name]
            pos = fp + fR @ np.array([x, 0.0, z])
            mujoco.mju_mulQuat(q, fq, np.array([math.cos(beta / 2), 0.0, math.sin(beta / 2), 0.0]))
            mujoco.mju_quat2Mat(R, q)
            Rm = R.reshape(3, 3)
            d.mocap_pos[mid], d.mocap_quat[mid] = pos, q
            d.xpos[bid], d.xquat[bid], d.xmat[bid] = pos, q, R
            for g in geoms:
                d.geom_xpos[g] = pos + Rm @ m.geom_pos[g]
                mujoco.mju_quat2Mat(G, m.geom_quat[g])
                d.geom_xmat[g] = (Rm @ G.reshape(3, 3)).ravel()


def test_same_physics(seeds=(0, 1, 2), steps=300, physics='v1'):
    """Looks never change the physics: same seed + same expert actions with and without the real-site looks
    -> identical qpos/qvel/ctrl/mocap at every step. Also: profile 'v0' without looks == swm/CardboardYard-v0."""
    from cardboard_yard.expert_policy import LoaderExpertPolicy

    pairs = [(CardboardYardRealEnv(physics=physics, looks=False), CardboardYardRealEnv(physics=physics), physics + ' looks'),
             (CardboardYardEnv(), CardboardYardRealEnv(physics='v0', looks=False), 'v0 == CardboardYard-v0')]
    for a, b, what in pairs:
        # bodies matched by name: the looks add visual-only mocap bodies (hyd_*) after the physics bodies
        ids = [mujoco.mj_name2id(b.model, mujoco.mjtObj.mjOBJ_BODY, a.model.body(i).name) for i in range(a.model.nbody)]
        assert min(ids) >= 0, what
        for f in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
            assert np.array_equal(getattr(a.model, f), getattr(b.model, f)[ids]), (what, f)
        for f in ('jnt_range', 'actuator_gainprm'):
            assert np.array_equal(getattr(a.model, f), getattr(b.model, f)), (what, f)
        nmoc = a.model.nmocap
        for seed in seeds:
            a.reset(seed=seed)
            b.reset(seed=seed)
            pol = LoaderExpertPolicy(seed=seed)
            for t in range(steps):
                act = pol.act_single(a)
                a.step(act)
                b.step(act)
                if t % 50 == 0:
                    b.render()  # rendering places the hyd_* mocap bodies: must not change anything either
                for f in ('qpos', 'qvel', 'ctrl', 'mocap_pos'):
                    x, y = getattr(a.data, f), getattr(b.data, f)
                    if f == 'mocap_pos':
                        x, y = x[:nmoc], y[:nmoc]
                    if not np.array_equal(x, y):
                        d = np.abs(x - y).max()
                        raise AssertionError(f'{what}: seed {seed} step {t}: {f} differs (max {d:g})')
            print(f'{what}: seed {seed}: {steps} steps identical')
    return True
