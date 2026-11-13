"""WA100-8 lift and tilt linkage (pins from Komatsu's dimension drawing, wa100_model.py): closed-form 4-bar for the
tilt lever, cylinder end points and lever arms. Shared by the showroom renders, the simulator looks (cylinders,
lever and link drawn as mocap bodies) and the hydraulic physics (lever arms for the force / flow limits).

Two angle conventions:
  drawing angles (tb, tk): 0, 0 = the drawing's lowered pose (bucket floor on the ground)
  simulator joints (boom, bucket): boom = angle of the pivot -> hinge line below horizontal, bucket world pitch
    = boom + bucket (0: floor level). sim_to_drawing() converts.
All points in the front frame (x forward, z up, origin on the ground at the articulation).
"""

import math

import numpy as np

from . import wa100_model as WM

BOOM0 = math.atan2(WM.PIVOT[1] - WM.HINGE[1], WM.HINGE[0] - WM.PIVOT[0])  # boom joint of the drawing pose (+0.772)
BOOM_LEN = float(np.linalg.norm(WM.HINGE - WM.PIVOT))  # 2.424 m
A_LIFT = 2 * math.pi * 0.05 ** 2  # 2 lift cylinders, 100 mm bore (estimate, no WA100-8 data sheet)
A_TILT = math.pi * 0.0625 ** 2  # 1 tilt cylinder, 125 mm bore (estimate)
RELIEF = 20.6e6  # Pa (estimate)


def rot(th, v):
    """Hinge rotation about +y (positive turns +x towards -z), v = (x, z)."""
    c, s = math.cos(th), math.sin(th)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def ang(v):
    return -math.atan2(v[1], v[0])


def sim_to_drawing(boom, bucket):
    return boom - BOOM0, bucket + BOOM0


class Linkage:
    """Tilt lever 4-bar (lever pivot on the boom, link to the bucket) and the cylinder end points."""

    def __init__(self):
        self.rc = float(np.linalg.norm(WM.LEVER_C - WM.LEVER_P))
        self.ln = float(np.linalg.norm(WM.LINK_D - WM.LEVER_C))
        v, c = WM.LINK_D - WM.LEVER_P, WM.LEVER_C - WM.LEVER_P
        self.branch = 1.0 if v[0] * c[1] - v[1] * c[0] >= 0 else -1.0

    def boom(self, tb, p):
        """Boom-frame point given in drawing-pose coordinates -> front frame, boom turned by tb from the drawing."""
        return WM.PIVOT + rot(tb, np.asarray(p) - WM.PIVOT)

    def lever(self, tk):
        """Lever bottom pin C, top pin T, bucket link pin D (drawing-pose boom coordinates) for bucket angle tk."""
        d = WM.HINGE + rot(tk, WM.LINK_D - WM.HINGE)
        v = d - WM.LEVER_P
        dist = float(np.linalg.norm(v))
        e = v / dist
        a = (self.rc ** 2 - self.ln ** 2 + dist ** 2) / (2 * dist)
        hh = math.sqrt(max(self.rc ** 2 - a * a, 0.0))  # 0 when the 4-bar cannot close (pin at the nearest point)
        c = WM.LEVER_P + a * e + self.branch * hh * np.array([-e[1], e[0]])
        phi = ang(c - WM.LEVER_P) - ang(WM.LEVER_C - WM.LEVER_P)
        t = WM.LEVER_P + rot(phi, WM.LEVER_T - WM.LEVER_P)
        return c, t, d, phi

    def solve(self, tb, tk):
        """{part: ((x, z) origin, angle)} of the moving hydraulic parts (local x along the part; lever frame = boom
        frame turned by the lever angle)."""
        c, t, d, phi = self.lever(tk)
        B = lambda p: self.boom(tb, p)
        lb, tf = B(WM.LIFT_B), B(t)
        return {
            'lift_barrel': (WM.LIFT_A, ang(lb - WM.LIFT_A)),
            'lift_rod': (lb, ang(WM.LIFT_A - lb)),
            'tilt_barrel': (WM.TILT_A, ang(tf - WM.TILT_A)),
            'tilt_rod': (tf, ang(WM.TILT_A - tf)),
            'lever': (B(WM.LEVER_P), tb + phi),
            'link': (B(c), ang(B(d) - B(c))),
        }

    def lift_length(self, tb):
        return float(np.linalg.norm(self.boom(tb, WM.LIFT_B) - WM.LIFT_A))

    def tilt_length(self, tb, tk):
        return float(np.linalg.norm(self.boom(tb, self.lever(tk)[1]) - WM.TILT_A))

    def lever_arms(self, boom, bucket, h=1e-3):
        """|d length / d joint| (m/rad) of the lift cylinders (boom joint) and of the tilt cylinder (bucket joint),
        simulator joint angles."""
        tb, tk = sim_to_drawing(boom, bucket)
        rb = abs(self.lift_length(tb + h) - self.lift_length(tb - h)) / (2 * h)
        rk = abs(self.tilt_length(tb, tk + h) - self.tilt_length(tb, tk - h)) / (2 * h)
        return rb, rk
