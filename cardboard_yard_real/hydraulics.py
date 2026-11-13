"""Hydraulic cylinders and Z-bar linkage of the loader (Komatsu WA100-like), used twice:

- looks: the lift cylinders, the tilt cylinder, the bell crank and the links are drawn as mocap bodies that follow
  the boom and bucket joints every step (CardboardYardRealEnv._update_hydraulics). Mocap bodies are not part of
  the physics state, so the looks do not change the physics.
- physics (profile 'v3'): cylinder force limit (relief pressure x area x lever arm), shared pump flow and
  hydraulic power (see HydraulicModel).

Geometry is 2-D in the boom plane (x forward, z up), in three frames:
  front  - front frame of the loader (origin at the articulation, as in cardboard_yard/env.py)
  boom   - origin at the boom pivot BOOM_PIVOT, x along the lift arms at boom angle 0
  bucket - origin at the bucket pin (boom x = BOOM_LEN), bucket angle 0 = aligned with the boom
Joint angles as in the env: boom + lowers the boom, bucket + dumps; R(th) is the rotation of the hinge (axis +y).
The points below were chosen with scripts/examples/design_loader_linkage.py: every boom / bucket pair the expert
uses (plus a margin) keeps the Z-bar assembled with good transmission angles, and the cylinder strokes stay
within a real stroke ratio.
"""

from __future__ import annotations

import math

import numpy as np

from cardboard_yard.env import BOOM_LEN, BOOM_PIVOT

P_BOOM = np.array([BOOM_PIVOT[0], BOOM_PIVOT[2]])  # boom pivot in the front frame (x, z)
PIN = np.array([BOOM_LEN, 0.0])  # bucket pin in the boom frame

# ---- linkage points (x, z) -------------------------------------------------------------------------------------
LIFT_A = np.array([0.714, 1.293])  # lift cylinder base, front frame (y = +-LIFT_Y)
LIFT_B = np.array([1.357, -0.069])  # lift cylinder rod end, boom frame (lug under the arms)
LIFT_Y = 0.27  # between the towers (|y| 0.38-0.43)
TILT_A = np.array([0.716, 1.882])  # tilt cylinder base, front frame (bracket between the towers, y = 0)
CRANK_P = np.array([1.682, 0.208])  # bell crank pivot, boom frame (on the cross tube)
CRANK_T = np.array([-0.578, 0.635])  # crank top arm (to the tilt cylinder rod) at crank angle 0, from the pivot
CRANK_C = np.array([0.421, -0.267])  # crank bottom arm (to the links) at crank angle 0, from the pivot
LINK_D = np.array([-0.051, 0.254])  # link pin on the bucket, bucket frame (ears on the bucket back)
LINK_LEN = None  # set below: the link length that assembles the nominal pose (crank 0, bucket NOMINAL_BUCKET)
NOMINAL_BUCKET = 1.214
BRANCH = None  # side of the crank bottom pin relative to the line pivot -> bucket pin (fixed by the nominal pose)

# ---- cylinders: bore (m), number, relief pressure (Pa) ---------------------------------------------------------
LIFT_BORE, LIFT_N = 0.100, 2
TILT_BORE, TILT_N = 0.125, 1
RELIEF = 20.6e6
A_LIFT = LIFT_N * math.pi * (LIFT_BORE / 2) ** 2
A_TILT = TILT_N * math.pi * (TILT_BORE / 2) ** 2


def rot(th, v):
    """Hinge rotation about +y by th (positive turns +x towards -z), v (..., 2) as (x, z)."""
    c, s = math.cos(th), math.sin(th)
    v = np.asarray(v, float)
    return np.stack([v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c], -1)


def ang(v):
    """Angle th with rot(th, (|v|, 0)) = v."""
    return -math.atan2(v[1], v[0])


def boom_to_front(th_b, p):
    return P_BOOM + rot(th_b, p)


def bucket_to_boom(th_k, q):
    return PIN + rot(th_k, q)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _init_nominal():
    global LINK_LEN, BRANCH
    d = bucket_to_boom(NOMINAL_BUCKET, LINK_D)
    c = CRANK_P + CRANK_C
    LINK_LEN = float(np.linalg.norm(d - c))
    BRANCH = 1.0 if _cross(d - CRANK_P, c - CRANK_P) >= 0 else -1.0


_init_nominal()


def crank(th_k):
    """Bell crank angle for bucket angle th_k: (phi, C, T, D, ok) with C, T, D in the boom frame.
    ok=False when the linkage cannot assemble (the pin is then put at the closest reachable point)."""
    d = bucket_to_boom(th_k, LINK_D)
    rc, ln = float(np.linalg.norm(CRANK_C)), LINK_LEN
    v = d - CRANK_P
    dist = float(np.linalg.norm(v))
    ok = abs(rc - ln) <= dist <= rc + ln
    e = v / max(dist, 1e-9)
    if ok:
        a = (rc * rc - ln * ln + dist * dist) / (2 * dist)
        h = math.sqrt(max(rc * rc - a * a, 0.0))
        n = np.array([-e[1], e[0]])  # left normal of pivot -> bucket pin
        c = CRANK_P + a * e + BRANCH * h * n * (1.0 if _cross(e, n) >= 0 else -1.0)
    else:
        c = CRANK_P + rc * e
    phi = ang(c - CRANK_P) - ang(CRANK_C)
    t = CRANK_P + rot(phi, CRANK_T)
    return phi, c, t, d, ok


def lift_length(th_b):
    return float(np.linalg.norm(boom_to_front(th_b, LIFT_B) - LIFT_A))


def tilt_length(th_b, th_k):
    return float(np.linalg.norm(boom_to_front(th_b, crank(th_k)[2]) - TILT_A))


def lever_arms(th_b, th_k, h=1e-3):
    """|d length / d angle| of the lift cylinders (boom joint) and of the tilt cylinder (bucket joint), in m/rad."""
    rb = abs(lift_length(th_b + h) - lift_length(th_b - h)) / (2 * h)
    rk = abs(tilt_length(th_b, th_k + h) - tilt_length(th_b, th_k - h)) / (2 * h)
    return rb, rk


# ---- drawn cylinders ---------------------------------------------------------------------------------------------
# bucket angles the linkage is drawn for (expert range + margin); outside it the drawing stops at the limit
K_MIN, K_MAX = -1.27, 2.06


def _sizes():
    """(barrel length, rod length) of the lift and tilt cylinders: the rod never leaves the barrel at full stroke
    (>= 0.1 m inside) and never comes out of the barrel's back at full retraction."""
    from cardboard_yard.env import BOOM_RANGE

    bs = np.linspace(BOOM_RANGE[0], BOOM_RANGE[1], 25)
    ks = np.linspace(K_MIN, K_MAX, 25)
    out = {}
    for name, ls in (('lift', [lift_length(b) for b in bs]), ('tilt', [tilt_length(b, k) for b in bs for k in ks])):
        lo, hi = min(ls), max(ls)
        barrel = 0.95 * lo
        rod = min(hi - barrel + 0.1, lo - 0.03)
        out[name] = (barrel, rod, lo, hi)
    return out


def poses(th_b, th_k):
    """{part: ((x, z) origin, angle)} in the front frame; each part's local x axis points along the part.
    Parts: lift_barrel, lift_rod, tilt_barrel, tilt_rod (from their own pin, along the cylinder), crank
    (at the crank pivot, rotated like the boom frame by boom + crank angle) and links (from the crank pin to the
    bucket pin)."""
    th_k = min(max(th_k, K_MIN), K_MAX)
    phi, c, t, d, _ = crank(th_k)
    lb = boom_to_front(th_b, LIFT_B)
    tf = boom_to_front(th_b, t)
    cf, df = boom_to_front(th_b, c), boom_to_front(th_b, d)
    return {
        'lift_barrel': (LIFT_A, ang(lb - LIFT_A)),
        'lift_rod': (lb, ang(LIFT_A - lb)),
        'tilt_barrel': (TILT_A, ang(tf - TILT_A)),
        'tilt_rod': (tf, ang(TILT_A - tf)),
        'crank': (boom_to_front(th_b, CRANK_P), th_b + phi),
        'links': (cf, ang(df - cf)),
    }


# ---- physics (profile v3) ----------------------------------------------------------------------------------------
class HydraulicModel:
    """Limits of a real hydraulic system on top of the boom / bucket position servos.

    - force: the servo torque is capped at relief pressure x piston area x lever arm (lever arm = d cylinder
      length / d joint angle, changes with the angles); a heavy load or the pile can stall the arm.
    - flow: the pump delivers at most q_max; boom and bucket share it. Each one alone at full lever gets its full
      rate, both together at full lever are slowed down in proportion.
    - power: the cylinders that work against their load (torque and motion in the same direction) need
      pressure x flow; above p_max (the engine's share for the hydraulics) both rates are scaled down.
    The commanded action stays what the agent sent; the machine just follows it more slowly.
    """

    def __init__(self, q_max, p_max, relief=RELIEF, geometry=None):
        """geometry: None = the Z-bar of this module (loader_model look, v3); else an object with lever_arms(boom,
        bucket), A_LIFT, A_TILT (e.g. wa100_linkage: WA100 profile v4)."""
        self.q_max, self.p_max, self.relief = q_max, p_max, relief
        self.geom = geometry
        self.scale = 1.0  # last rate scale (for metrics)

    def _arms(self, tb, tk):
        if self.geom is None:
            return lever_arms(tb, min(max(tk, K_MIN), K_MAX)), A_LIFT, A_TILT
        return self.geom.lever_arms(tb, tk), self.geom.A_LIFT, self.geom.A_TILT

    def pressures(self, env):
        """Lift and tilt cylinder pressures (MPa, 0 .. relief) from the servo torques of the last step:
        pressure = |torque| / (piston area x lever arm). Sign-free (which cylinder side is not modelled)."""
        tb, tk = env.joint('boom'), env.joint('bucket')
        (rb, rk), A_LIFT, A_TILT = self._arms(tb, tk)
        f = env.data.actuator_force
        pl = abs(f[env._act['boom']]) / (A_LIFT * max(rb, 0.02))
        pt = abs(f[env._act['bucket']]) / (A_TILT * max(rk, 0.02))
        return min(pl, self.relief) / 1e6, min(pt, self.relief) / 1e6

    def limit(self, env, action):
        a = np.array(action, np.float64).reshape(-1).copy()
        m, d = env.model, env.data
        tb, tk = env.joint('boom'), env.joint('bucket')
        (rb, rk), A_LIFT, A_TILT = self._arms(tb, tk)
        ib, ik = env._act['boom'], env._act['bucket']
        # force limits for the coming step
        for i, area, r in ((ib, A_LIFT, rb), (ik, A_TILT, rk)):
            tmax = self.relief * area * max(r, 0.02)
            m.actuator_forcerange[i] = (-tmax, tmax)
        # rates the agent asks for (the env moves the targets by -a * rate * dt)
        wb = -np.clip(a[2], -1, 1) * env.boom_rate
        wk = -np.clip(a[3], -1, 1) * env.bucket_rate
        qb, qk = A_LIFT * rb * abs(wb), A_TILT * rk * abs(wk)
        s = 1.0
        if qb + qk > self.q_max:
            s = self.q_max / (qb + qk)
        # power: pressure from the torque the servo needed in the last step, only when it does work
        power = 0.0
        for tau, w, q, area, r in ((d.actuator_force[ib], wb, qb, A_LIFT, rb), (d.actuator_force[ik], wk, qk, A_TILT, rk)):
            if tau * w > 0:
                power += abs(tau) / (area * max(r, 0.02)) * q * s
        if power > self.p_max:
            s *= self.p_max / power
        self.scale = s
        a[2] *= s
        a[3] *= s
        return a
