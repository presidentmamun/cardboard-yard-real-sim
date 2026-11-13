"""Komatsu WA100-8 in the simulator (CardboardYardReal physics profile 'v4'): Machine (sizes, limits, expert arm
poses), the physics bodies replacing the plain loader of cardboard_yard/env.py, and the looks (wa100_model.py
meshes on those bodies, cylinders / lever / link as mocap bodies following the joints, wa100_linkage.py).

Body frames as in cardboard_yard/env.py: rear (origin on the ground at the articulation; slides / turns), front
(steer hinge at the origin), boom (at the boom pivot; x along the pivot -> hinge line, joint + lowers), bucket (at
the hinge; joint relative to the boom, world pitch = boom + bucket, 0 = floor level), wheel_* (hub).
Sizes: Komatsu's dimension drawing (wheelbase 2600, tread 1780, tyre 0.635 m, axle 0.61 m, boom pivot / hinge
pins, bucket profile, steering 38 deg). Masses: operating weight ~7.3 t (estimate: no data sheet).
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

from cardboard_yard.env import Machine

from . import wa100_dims as D
from . import wa100_linkage as WL
from . import wa100_model as WM

BOOM0 = WL.BOOM0
H_HINGE0 = float(WM.HINGE[1])  # hinge height of the drawing pose, floor on the ground (0.114)
LIP = (3.21 - WM.HINGE[0], -H_HINGE0)  # cutting edge in the bucket frame (x along the floor, z)


def _boom_for_hinge_z(z):
    """Boom joint that puts the bucket hinge at height z."""
    return math.asin((WM.PIVOT[1] - z) / WL.BOOM_LEN)


MACHINE_WA100 = Machine(
    name='wa100',
    l_front=D.WHEELBASE / 2, l_rear=D.WHEELBASE / 2,
    wheel_r=D.TYRE_R,
    boom_pivot=(WM.PIVOT[0], 0.0, WM.PIVOT[1]),
    boom_len=WL.BOOM_LEN,
    boom_range=(-0.77, 0.78),  # drawing: hinge 3.485 m at full raise, bucket on the ground at 0.772
    bucket_range=(-1.6, 1.7),  # rack back limited near the ground (the tilt linkage's lever arm vanishes beyond),
    max_speed=3.0,             # dump 45 deg at full raise (drawing)
    max_steer=math.radians(38),
    boom_rate=0.3,  # rad/s: full raise ~5.2 s (WA100: 5.5 s)
    bucket_rate=0.6,
    poses=dict(
        carry=(_boom_for_hinge_z(0.5), -0.8),   # bucket ~0.4 m off the ground, racked back 46 deg (drawing)
        dig=(_boom_for_hinge_z(0.16), 0.04),    # floor just above the ground
        curl=(_boom_for_hinge_z(0.75), -0.85),
        high=(-0.76, -0.6),
        dump=(-0.76, 0.8),                      # 45 deg dump at full raise
    ),
    reset_arm=(_boom_for_hinge_z(0.5), -0.8),
    raised_boom=-0.5,
    footprint_rear=(-2.0, -1.0, 0.0),
    footprint_front=(1.0, 1.75),
    bucket_reach=0.45,
    plan_strict=True,  # people test seed 27: the unchecked escape arc went into the column it touched
    rear_side=0.0,  # 0.25 covered the rear tyre corners (0.03-0.07 m outside the circles) but cost 110/126/132 3-6 loads; 0 column hits without it
    rear_axle=-D.WHEELBASE / 2,
    bucket_side=0.3,  # 2.34 m bucket: one 1.1 m circle left its corners 0.13-0.15 m out (column / hopper rubs)
    push_limit_dy=3.4,  # lip (0.84 m ahead of the hinge) at the wall in the dig pose (3.7: smaller loads, seeds 2-4 < 25)
    plan_steer_frac=0.92,  # headroom at full lock (0.85: wide U-turns into col_17.75; 1.0: more spill, slower)
    plan_col_margin=0.0,  # 0.25 did not stop the bucket-corner column rubs (tracking, not the plan) and cost seed 0 a load
    dump_stop_dx=0.3,  # front tyres / frame reach further than the v0 loader's: they touched the hopper wall
    bucket_box=(-0.05, 0.95, 1.12, -0.2, 0.95),
)


# ------------------------------------------------------------------ physics
def _box(name, pos, size, mat, mass, extra=''):
    nm = f'name="{name}" ' if name else ''
    return (f'<geom {nm}type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" size="{size[0]:.4f} {size[1]:.4f} '
            f'{size[2]:.4f}" material="{mat}" group="1" mass="{mass}" {extra}/>')


def loader_body(chase, cab_cam, rear_cam):
    """The loader as an MJCF <body name="rear"> subtree (collision boxes, masses, joints, cameras, bucket site)."""
    hx = D.WHEELBASE / 2
    ty = D.TREAD / 2
    zw = D.TYRE_R + 0.01

    def wheel(name, x, y):
        return (f'<body name="{name}" pos="{x} {y} {zw}">'
                f'<joint name="{name}_spin" type="hinge" axis="0 1 0" damping="0"/>'
                f'<geom name="{name}_tire" type="cylinder" size="{D.TYRE_R} {D.TYRE_W / 2}" euler="1.5708 0 0" '
                f'material="tire" group="1" contype="2" conaffinity="1" mass="170"/></body>')

    W = D.BUCKET_W / 2
    bucket = '\n'.join([  # bucket frame: hinge at the origin, floor level at joint 0 (drawing profile)
        _box('bucket_bottom', (0.45, 0, -0.095), (0.39, W - 0.04, 0.02), 'bucket_steel', 140, 'friction="1.0 0.01 0.001"'),
        _box('bucket_back', (0.09, 0, 0.26), (0.03, W - 0.04, 0.36), 'bucket_steel', 110),
        _box('bucket_side_l', (0.46, W - 0.02, 0.28), (0.42, 0.02, 0.39), 'bucket_steel', 45),
        _box('bucket_side_r', (0.46, -W + 0.02, 0.28), (0.42, 0.02, 0.39), 'bucket_steel', 45),
        _box('bucket_lip', (LIP[0] - 0.04, 0, LIP[1] + 0.02), (0.05, W - 0.04, 0.025), 'dark', 50),
        _box('bucket_guard', (0.46, 0, 0.76), (0.2, W - 0.04, 0.02), 'bucket_steel', 40,
             f'euler="0 {math.radians(-38):.4f} 0"'),
    ])
    L = WL.BOOM_LEN
    return f"""
    <body name="rear" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="0"/>
      <camera name="chase" pos="-9 -3.5 5" xyaxes="{chase}" fovy="55"/>
      <camera name="cab" pos="-0.55 0 2.6" xyaxes="{cab_cam}" fovy="75"/>
      <camera name="rear" pos="-1.2 0 2.6" xyaxes="{rear_cam}" fovy="75"/>
      {_box('rear_frame', (-1.4, 0, 0.88), (1.15, 0.4, 0.3), 'dark', 1300)}
      {_box('engine', (-2.05, 0, 1.74), (0.82, 0.7, 0.53), 'yellow', 1500)}
      {_box('counterweight', (-2.65, 0, 0.99), (0.33, D.BUMPER_HW, 0.18), 'dark', 900)}
      {_box('cab_base', (-0.5, 0, 1.68), (0.7, 0.64, 0.32), 'yellow', 250)}
      {_box('cab_glass', (-0.55, 0, 2.5), (0.66, 0.64, 0.5), 'glass', 150)}
      {wheel('wheel_rl', -hx, ty)}
      {wheel('wheel_rr', -hx, -ty)}
      <body name="front" pos="0 0 0">
        <joint name="steer" type="hinge" axis="0 0 1" range="{-MACHINE_WA100.max_steer - 0.05} {MACHINE_WA100.max_steer + 0.05}" damping="2000"/>
        {_box('front_frame', (1.0, 0, 0.95), (0.95, 0.38, 0.3), 'dark', 1100)}
        {_box('boom_tower_l', (0.6, 0.405, 1.45), (0.3, 0.025, 0.5), 'dark', 80)}
        {_box('boom_tower_r', (0.6, -0.405, 1.45), (0.3, 0.025, 0.5), 'dark', 80)}
        {_box('fender_fl', (0.9, 0.84, 1.48), (0.2, 0.24, 0.02), 'yellow', 10)}
        {_box('fender_fr', (0.9, -0.84, 1.48), (0.2, 0.24, 0.02), 'yellow', 10)}
        {wheel('wheel_fl', hx, ty)}
        {wheel('wheel_fr', hx, -ty)}
        <body name="boom" pos="{WM.PIVOT[0]} 0 {WM.PIVOT[1]}">
          <joint name="boom" type="hinge" axis="0 1 0" range="{MACHINE_WA100.boom_range[0]} {MACHINE_WA100.boom_range[1]}" damping="30000" armature="500"/>
          {_box('boom_l', (L / 2, WM.ARM_Y, 0.02), (L / 2, 0.045, 0.15), 'yellow', 230)}
          {_box('boom_r', (L / 2, -WM.ARM_Y, 0.02), (L / 2, 0.045, 0.15), 'yellow', 230)}
          {_box('boom_cross', (1.75, 0, 0.0), (0.1, WM.ARM_Y, 0.1), 'yellow', 80)}
          <body name="bucket" pos="{L} 0 0">
            <joint name="bucket" type="hinge" axis="0 1 0" range="{MACHINE_WA100.bucket_range[0]} {MACHINE_WA100.bucket_range[1]}" damping="15000" armature="100"/>
            {bucket}
            <site name="bucket_center" pos="0.45 0 0.1" size="0.05" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>
    </body>"""


def replace_loader(root):
    """Swap the plain loader of the base scene for the WA100 bodies; actuator ranges to the WA100 limits."""
    world = root.find('worldbody')
    old = world.find("body[@name='rear']")
    cams = {c.get('name'): c.get('xyaxes') for c in old.findall('camera')}
    from cardboard_yard.env import _lookat_xyaxes

    cab = _lookat_xyaxes((-0.55, 0.0, 2.6), (6.0, 0.0, 0.0))  # driver eye: 0.7 m above the seat (x -0.62)
    rear_cam = _lookat_xyaxes((-1.2, 0.0, 2.6), (-8.0, 0.0, 0.0))
    new = ET.fromstring(loader_body(cams['chase'], cab, rear_cam))
    idx = list(world).index(old)
    world.remove(old)
    world.insert(idx, new)
    M = MACHINE_WA100
    # the lift arms swing past the cab's front corners when articulated (the coarse boxes would touch): no contacts
    # between the boom and the rear frame
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    ET.SubElement(contact, 'exclude', {'body1': 'rear', 'body2': 'boom'})
    for name, rng in (('steer', (-M.max_steer, M.max_steer)), ('boom', M.boom_range), ('bucket', M.bucket_range)):
        a = root.find(f".//actuator/position[@name='{name}']")
        a.set('ctrlrange', f'{rng[0]} {rng[1]}')


# ------------------------------------------------------------------ looks
PARTS = ('lift_barrel', 'lift_rod', 'tilt_barrel', 'tilt_rod', 'lever', 'link')


def add_looks(asset, root, mesh_asset, geom, f):
    """WA100 meshes on the physics bodies (boom meshes turned from the drawing pose into the boom frame) and the
    hydraulic parts as mocap bodies (placed by place_hydraulics)."""
    from . import meshes as MS

    body = {b.get('name'): b for b in root.iter('body')}
    for name, (tf, rgba, spec, shin) in WM.MATERIALS.items():
        attrs = {'name': name, 'rgba': rgba, 'specular': str(spec), 'shininess': str(shin)}
        if tf:
            ET.SubElement(asset, 'texture', {'name': 'tex_' + name, 'type': '2d', 'file': WM.make_textures()[tf].replace('\\', '/')})
            attrs['texture'] = 'tex_' + name
        ET.SubElement(asset, 'material', attrs)
    for bname in ('rear', 'front', 'boom', 'bucket', 'wheel_rl', 'wheel_rr', 'wheel_fl', 'wheel_fr'):
        for g in body[bname].findall('geom'):
            g.set('group', '4')  # plain physics boxes: still colliding, not drawn
    to_boom = MS.rot_y(-BOOM0)  # drawing pose -> boom frame (joint 0 = pivot -> hinge line horizontal)
    for bname, items in WM.build().items():
        targets = ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr'] if bname == 'wheel' else [bname]
        for name, mat, m in items:
            if bname == 'boom':
                m = m.moved(0, 0, 0, rot=to_boom)
            mname = f'wa_{bname}_{name}'
            mesh_asset(asset, mname, m)
            if bname == 'wheel':  # the mesh's outer face is +y: the right (-y) wheels get a copy turned round about z
                # (vertices turned, not a geom euler: the scene compiles angles in radians, '0 0 180' tilted the axis)
                mesh_asset(asset, mname + '_r', m.moved(0, 0, 0, rot=MS.rot_z(math.pi)))
            for t in targets:  # wheels: drawn on the spin axis (an offset there made them wobble round the hub)
                right = bname == 'wheel' and t in ('wheel_fr', 'wheel_rr')
                geom(body[t], type='mesh', mesh=mname + ('_r' if right else ''), material=mat)
    _hydraulic_mocaps(asset, root.find('worldbody'), mesh_asset, geom, f)


def _hydraulic_mocaps(asset, world, mesh_asset, geom, f):
    from . import meshes as MS

    lk = WL.Linkage()
    sizes = {}
    for kind in ('lift', 'tilt'):
        ls = []
        for tb in np.linspace(-1.55, 0.02, 30):
            for tk in np.linspace(-0.85, 2.45, 16):
                s = lk.solve(tb, tk)
                ls.append(np.linalg.norm(s[f'{kind}_rod'][0] - s[f'{kind}_barrel'][0]))
        lo, hi = min(ls), max(ls)
        barrel = 0.93 * lo
        sizes[kind] = (barrel, min(hi - barrel + 0.12, lo - 0.04))

    def mb(name):
        return ET.SubElement(world, 'body', {'name': f'hyd_{name}', 'mocap': 'true', 'pos': '0 0 -20'})

    def cyl(b, p0, p1, r, mat):
        geom(b, type='cylinder', fromto=f(*p0, *p1), size=f'{r:.4f}', material=mat)

    for kind, ys, rb, rr, barrel_mat in (('lift', (-WM.LIFT_Y, WM.LIFT_Y), 0.062, 0.035, 'wa_black'), ('tilt', (0.0,), 0.072, 0.04, 'wa_paint')):
        barrel, rod = sizes[kind]
        bb, br = mb(f'{kind}_barrel'), mb(f'{kind}_rod')
        for y in ys:
            cyl(bb, (0, y - 0.07, 0), (0, y + 0.07, 0), 0.05, 'wa_black')
            cyl(bb, (0.04, y, 0), (barrel - 0.05, y, 0), rb, barrel_mat)
            cyl(bb, (barrel - 0.06, y, 0), (barrel, y, 0), rb + 0.008, 'wa_black')
            cyl(br, (0, y - 0.06, 0), (0, y + 0.06, 0), rr + 0.02, 'wa_black')
            cyl(br, (0.03, y, 0), (rod, y, 0), rr, 'wa_chrome')
    tb_ = world.find("body[@name='hyd_tilt_barrel']")
    for y in (-0.05, 0.05):  # tilt cylinder hoses (photo 03)
        pts = [(-0.06, y, -0.06), (0.02, y, 0.07), (0.12, y, 0.13), (0.24, y, 0.14), (0.4, y, 0.11), (0.56, y, 0.06)]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            geom(tb_, type='capsule', fromto=f(*p0, *p1), size='0.022', material='wa_black')
    tr = world.find("body[@name='hyd_tilt_rod']")
    cyl(tr, (0.06, 0.0, 0.075), (0.5, 0.0, 0.075), 0.013, 'wa_steel')  # bucket level indicator
    lv = mb('lever')
    t, c = WM.LEVER_T - WM.LEVER_P, WM.LEVER_C - WM.LEVER_P
    area = lambda p: 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))
    lever_poly = [(x - WM.LEVER_P[0], z - WM.LEVER_P[1]) for x, z in WM._PARTS['lever'][:-1]]
    lever_poly = lever_poly if area(lever_poly) > 0 else lever_poly[::-1]
    mesh_asset(asset, 'wa_lever_plate', MS.extrude(lever_poly, -0.013, 0.013))
    for y in (-0.05, 0.05):
        g = geom(lv, type='mesh', mesh='wa_lever_plate', material='wa_paint')
        g.set('pos', f'0 {y} 0')
    for x, z, r in ((0, 0, 0.05), (t[0], t[1], 0.032), (c[0], c[1], 0.032)):
        cyl(lv, (x, -0.075, z), (x, 0.075, z), r, 'wa_paint')
    li = mb('link')
    b0 = WL.ang(WM.LINK_D - WM.LEVER_C)
    link_poly = [tuple(WL.rot(-b0, np.array([x, z]) - WM.LEVER_C)) for x, z in WM._PARTS['link'][:-1]]
    link_poly = link_poly if area(link_poly) > 0 else link_poly[::-1]
    mesh_asset(asset, 'wa_link_plate', MS.extrude(link_poly, -0.012, 0.012))
    for y in (-0.1, 0.1):
        g = geom(li, type='mesh', mesh='wa_link_plate', material='wa_paint')
        g.set('pos', f'0 {y} 0')


_LK = None


def poses(boom, bucket):
    """{part: ((x, z), angle)} in the front frame for simulator joint angles (bucket clipped to the closing range)."""
    global _LK
    if _LK is None:
        _LK = WL.Linkage()
    tb, tk = WL.sim_to_drawing(boom, bucket)
    return _LK.solve(tb, min(max(tk, -0.85), 2.45))
