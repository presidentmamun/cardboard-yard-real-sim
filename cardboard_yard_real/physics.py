"""Physics profiles of CardboardYardReal. 'v0' = exactly the physics of swm/CardboardYard-v0 (the office
training data); 'v1' = smoother loader (less shaking, damping, driving), tuned with
scripts/examples/cardboard_yard_physics_metrics.py (results in E:/office-projects/physics_tuning/).

A profile has
  xml:     {(tag, name): {attribute: value}} applied to the MJCF of build_xml (joints, actuators, geoms)
  pieces:  attributes set on every cardboard piece geom / free joint
  control: env attributes (boom_rate, bucket_rate, max_accel, steer_rate, arm_lead, drive_sub)
  hydraulic: optional HydraulicModel settings (hydraulics.py): force / flow / power limits of boom and bucket
"""

from cardboard_yard.env import LAYOUT_V0, Layout

PROFILES = {
    'v0': dict(xml={}, pieces={}, piece_joint={}, option={}, control={}),
    # v1 (tuning 2026-10-26, physics_tuning/e*.json): drive targets updated every substep (smoother ground
    # acceleration, less body / yaw shaking), more damping on the bucket (less chatter) and on the boom servo
    # (no overshoot of the boom rate). Cardboard contact softer / more damped / condim 4 / noslip did not help.
    # impratio 4 looked good on short runs (less jitter and spill) but the bucket then barely enters the pile
    # (2 pieces per scoop, physics_tuning/v1_final.json) -> not used. Final check: physics_tuning/v1b_phys_v1expert.json.
    'v1': dict(
        xml={('position', 'bucket'): {'kv': 24000}, ('joint', 'bucket'): {'damping': 25000},
             ('position', 'boom'): {'kv': 90000}},
        pieces={}, piece_joint={}, option={},
        control={'drive_sub': 1},
    ),
}


# real site layout (from the laser scan, E:/office-projects/las_site_plan.png): pile against the north wall, the dump
# point 23 m south of it, column rows 21 m apart with 7.5 m spacing, the near row 11 m in front of the back wall
LAYOUT_REAL = Layout(
    hopper_center=(-8.0, -11.6),
    column_x=(-12.75, -5.25, 2.25, 9.75, 17.25),
    column_y=(2.6, -18.4),
    left_wall=(-21.0, -3.0, 17.0),
    area=(-11.0, 15.0, -15.0, 7.5),
    people_area=(-11.0, 15.0, -15.0, 7.0),
    people_bounds=(-13.0, 17.0, -17.0, 13.0),
    spawn_area=(-4.0, 12.0, -14.0, 5.0),
)
LAYOUT_REAL.nav_bounds = (-14.0, 18.0, -18.0, 13.8)


PROFILES['v2'] = dict(PROFILES['v1'], layout=LAYOUT_REAL)  # v1 physics on the real site layout


# v3 (2026-10-27): v2 + hydraulics (hydraulics.py). The boom / bucket servos keep their gains but get the limits
# of the WA100-like hydraulics: torque <= relief pressure x piston area x lever arm (force-limited actuators),
# shared pump flow (each function alone keeps its full rate, both together at full rate are slowed), and a
# hydraulic power limit (heavy loads lift slower). Cylinder sizes, relief pressure and power are estimates
# (bores 100 mm x 2 lift / 125 mm tilt, 20.6 MPa, 30 kW), not from a Komatsu data sheet.
PROFILES['v3'] = dict(
    PROFILES['v2'],
    xml={**PROFILES['v1']['xml'],
         ('position', 'boom'): {'kv': 90000, 'forcelimited': 'true', 'forcerange': '-1e9 1e9'},
         ('position', 'bucket'): {'kv': 24000, 'forcelimited': 'true', 'forcerange': '-1e9 1e9'}},
    hydraulic={'p_max': 30e3},  # q_max: see hydraulic()
)


# v4 (2026-10-27): the Komatsu WA100-8 (wa100_sim.py: sizes from Komatsu's dimension drawing, its linkage for the
# hydraulics), on the site layout with the columns measured in the laser scan (LAS, 24.5 M points): square hollow
# columns 0.73 m wide, rows at y 2.70 / -18.30 (the scan's rows are 21.0 m apart, rotated -9.16 deg in the LAS frame),
# x every 7.5 m except the last bay (8.0 m): -20.25 .. 17.75. Columns at x -20.25 are inside the left wall line.
LAYOUT_V4 = Layout(
    hopper_center=(-8.0, -11.6),
    column_x=(-20.25, -12.75, -5.3, 2.25, 9.75, 17.75),
    column_y=(2.70, -18.30),
    column_half=0.365,
    left_wall=(-21.0, -3.0, 17.0),
    area=(-11.0, 15.0, -15.0, 7.5),
    people_area=(-11.0, 15.0, -15.0, 7.0),
    people_bounds=(-13.0, 17.0, -17.0, 13.0),
    spawn_area=(-4.0, 12.0, -14.0, 5.0),
)
LAYOUT_V4.nav_bounds = (-14.0, 18.0, -18.0, 13.8)
LAYOUT_V4.column_look = 'box'  # square hollow sections (scan), not H-beams

PROFILES['v4'] = dict(
    PROFILES['v1'],
    # v1 gains + force-limited boom / bucket servos (hydraulic limits); bucket joint damping 40000 (tuning
    # physics_tuning/v4/s6_*.log, seeds 0-2: bucket chatter 0.021 -> 0.017, spill 5.3 -> 4.7, delivered 11 -> 17)
    xml={**PROFILES['v3']['xml'], ('joint', 'bucket'): {'damping': 40000}},
    layout=LAYOUT_V4,
    machine='wa100',
    hydraulic={'p_max': 30e3, 'geometry': 'wa100'},
    proprio_pressure=True,  # proprio 12: + lift / tilt cylinder pressure (MPa)
    people='walk',  # articulated people walking between waypoints (people.py; v0: a 3-11 m beat or standing)
)


def machine(profile):
    """Machine of a profile (cardboard_yard Machine): v0 loader unless the profile names another."""
    from cardboard_yard.env import MACHINE_V0

    if PROFILES[profile].get('machine') == 'wa100':
        from .wa100_sim import MACHINE_WA100

        return MACHINE_WA100
    return MACHINE_V0


def hydraulic(profile):
    """HydraulicModel kwargs of a profile, or None. Pump flow q_max = 1.05 x the largest flow boom or bucket alone
    need at their full rate anywhere in their range (so a single function is never flow limited)."""
    h = PROFILES[profile].get('hydraulic')
    if not h:
        return None
    h = dict(h)
    if h.pop('geometry', None) == 'wa100':
        import numpy as np
        from . import wa100_linkage as WL

        M = machine(profile)
        lk = WL.Linkage()
        bs = np.linspace(*M.boom_range, 31)
        qb = max(WL.A_LIFT * lk.lever_arms(b, 0.0)[0] * M.boom_rate for b in bs)
        qk = max(WL.A_TILT * lk.lever_arms(b, p - b)[1] * M.bucket_rate for b in bs[::5] for p in np.linspace(-0.8, 0.8, 17))
        lk.A_LIFT, lk.A_TILT = WL.A_LIFT, WL.A_TILT
        return {'q_max': 1.05 * max(qb, qk), 'geometry': lk, 'relief': WL.RELIEF, **h}
    from cardboard_yard.env import BOOM_RANGE, BOOM_RATE, BUCKET_RATE
    import numpy as np
    from . import hydraulics as HY

    bs = np.linspace(BOOM_RANGE[0], BOOM_RANGE[1], 31)
    ks = np.linspace(HY.K_MIN, HY.K_MAX, 31)
    qb = max(HY.A_LIFT * HY.lever_arms(b, 0.0)[0] * BOOM_RATE for b in bs)
    qk = max(HY.A_TILT * HY.lever_arms(b, k)[1] * BUCKET_RATE for b in bs[::5] for k in ks)
    return {'q_max': 1.05 * max(qb, qk), **h}


def layout(profile):
    return PROFILES[profile].get('layout', LAYOUT_V0)


def apply_xml(root, profile):
    p = PROFILES[profile]
    for (tag, name), attrs in p['xml'].items():
        el = root.find(f".//{tag}[@name='{name}']")
        assert el is not None, (tag, name)
        el.attrib.update({k: str(v) for k, v in attrs.items()})
    if p['option']:
        root.find('option').attrib.update({k: str(v) for k, v in p['option'].items()})
    for g in root.iter('geom'):
        nm = g.get('name', '')
        if nm.startswith('piece') and nm.endswith('_g') and p['pieces']:
            g.attrib.update({k: str(v) for k, v in p['pieces'].items()})
    for j in root.iter('joint'):
        nm = j.get('name', '')
        if nm.startswith('piece') and nm.endswith('_j') and p['piece_joint']:
            j.attrib.update({k: str(v) for k, v in p['piece_joint'].items()})


def control(profile):
    return PROFILES[profile]['control']
