"""Hand-built wheel loader for CardboardYardReal (LOADER_LOOK = 'model').

Shapes and sizes follow the photogrammetry scan (assets/komatsu_scan, measured in the body frames of
cardboard_yard/env.py) and the site photos of the Komatsu WA100: short tall engine hood, cab with large glass and
black pillars, black front frame and towers, dog-leg lift arms with a Z-bar, light-material bucket with a mesh
extension, L3 chevron tires on yellow rims. Everything is visual only; the physics bodies are untouched.

Bodies (frames as in env.py): rear (articulation at x=0, ground z=0), front, boom (pivot at origin, arm to the
bucket pin at BOOM_LEN), bucket (pin at origin), wheel_* (hub at origin, axis y).
"""

import math
import os

import numpy as np
from PIL import Image

from . import hydraulics as HY
from . import meshes as MS

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


# ------------------------------------------------------------------ smooth shapes
def _rrect(hw, z0, z1, r, n=6):
    """Rounded rectangle in the y-z plane (counter-clockwise seen from +x), r clipped to fit."""
    r = min(r, hw * 0.99, (z1 - z0) / 2 * 0.99)
    pts = []
    for cy, cz, a0 in ((hw - r, z1 - r, 0.0), (-hw + r, z1 - r, 90.0), (-hw + r, z0 + r, 180.0), (hw - r, z0 + r, 270.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((cy + r * math.cos(a), cz + r * math.sin(a)))
    return pts


def loft(sections, n=6, cap=True):
    """Smooth body through cross sections [(x, half_width, z_bottom, z_top, corner_radius), ...] (x increasing).
    Vertices are shared along each ring and between rings, so MuJoCo shades it smoothly."""
    m = MS.Mesh()
    rings = []
    for x, hw, z0, z1, r in sections:
        ring = []
        for y, z in _rrect(hw, z0, z1, r, n):
            ring.append(len(m.v))
            m.v.append((x, y, z))
            m.t.append(None)
        rings.append(ring)
    k = len(rings[0])
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(k):
            j = (i + 1) % k
            m.f += [(a[i], a[j], b[j]), (a[i], b[j], b[i])]
    if cap:
        for ring, flip in ((rings[0], True), (rings[-1], False)):
            c = len(m.v)
            m.v.append(tuple(np.mean([m.v[i] for i in ring], 0)))
            m.t.append(None)
            for i in range(k):
                j = (i + 1) % k
                m.f.append((c, ring[j], ring[i]) if flip else (c, ring[i], ring[j]))
    return m


def cyl_x(r, x0, x1, y, z, n=20):
    """Cylinder along x."""
    return MS.revolve_y([(0.0, 0.0), (r, 0.0), (r, x1 - x0), (0.0, x1 - x0)], n=n).moved(0, 0, 0, rot=MS.rot_z(-math.pi / 2)).moved(x0, y, z)


def cyl_between(p0, p1, r, n=16):
    """Cylinder from point p0 to p1 (body frame)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    m = MS.revolve_y([(0.0, 0.0), (r, 0.0), (r, L), (0.0, L)], n=n)  # along +y
    d = (p1 - p0) / L
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    s, c = np.linalg.norm(v), float(y @ d)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
        a = math.atan2(s, c)
        R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return m.moved(*p0, rot=R)


# ------------------------------------------------------------------ textures
def _noise(h, w, scale, rng):
    f = np.fft.fft2(rng.normal(0, 1, (h, w)))
    ky, kx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij')
    f *= np.exp(-(kx ** 2 + ky ** 2) * scale ** 2)
    n = np.real(np.fft.ifft2(f))
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


def make_textures():
    """Paint / steel / rubber textures with wear and grime (square, used as cube textures)."""
    out = {}
    rng = np.random.default_rng(42)
    S = 512
    jobs = {
        # Komatsu yellow, slightly faded, dust patches and fine scratches
        'ld_paint': ((236, 178, 22), 6, 0.1),  # low contrast: cube mapping on big panels would show a star pattern
        'ld_black': ((28, 28, 30), 10, 0.15),
        'ld_steel': ((70, 62, 54), 25, 0.5),  # worn bucket steel, rust tint
        'ld_rubber': ((32, 32, 33), 8, 0.2),
        'ld_paint_dirty': ((222, 166, 26), 22, 0.55),  # lower panels: more mud and dust
    }
    for name, (base, var, dirt) in jobs.items():
        path = os.path.join(ASSETS, name + '.png')
        out[name] = path
        if os.path.exists(path):
            continue
        n1, n2, n3 = _noise(S, S, 40, rng), _noise(S, S, 8, rng), rng.random((S, S))
        img = np.array(base, float)[None, None, :] * (1 + (var / 255) * (n2[..., None] - 0.5) * 4)
        grime = np.clip((n1 - 0.55) * 2.5, 0, 1)[..., None] * dirt
        if name == 'ld_paint_dirty':
            grime = np.clip(grime + np.linspace(0, 0.45, S)[:, None, None] * _noise(S, S, 12, rng)[..., None], 0, 0.85)
        img = img * (1 - grime) + np.array([110, 95, 75])[None, None, :] * grime  # dust / mud
        img += (n3[..., None] - 0.5) * 6
        if name == 'ld_steel':
            rust = np.clip((_noise(S, S, 15, rng) - 0.5) * 3, 0, 1)[..., None]
            img = img * (1 - 0.6 * rust) + np.array([120, 60, 30])[None, None, :] * 0.6 * rust
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return out


MATERIALS = {
    # name: (texture key or None, rgba, specular, shininess)
    'ld_paint': ('ld_paint', '1 1 1 1', 0.35, 0.45),
    'ld_black': ('ld_black', '1 1 1 1', 0.25, 0.3),
    'ld_paint_dirty': ('ld_paint_dirty', '1 1 1 1', 0.2, 0.25),
    'ld_steel': ('ld_steel', '1 1 1 1', 0.2, 0.2),
    'ld_rubber': ('ld_rubber', '1 1 1 1', 0.05, 0.05),
    'ld_glass': (None, '0.16 0.22 0.26 0.45', 0.9, 0.95),
    'ld_chrome': (None, '0.8 0.81 0.82 1', 0.9, 0.9),
    'ld_lamp': (None, '1.0 0.97 0.88 1', 0.8, 0.9),
    'ld_tail': (None, '0.75 0.05 0.04 1', 0.6, 0.7),
    'ld_amber': (None, '1.0 0.55 0.05 1', 0.6, 0.7),
    'ld_grey': (None, '0.42 0.43 0.44 1', 0.3, 0.3),
}


# ------------------------------------------------------------------ wheel parts
def tire_l3(R, hw, n=72, bars=20):
    """Carcass (smooth surface of revolution with bulging sidewalls and rounded shoulders, shared vertices so it
    shades smoothly) + tread bars (chevron: two staggered rows of angled bars that also wrap the shoulder)."""
    lug = 0.045
    rb = R - lug  # carcass outer radius under the bars
    prof = []  # (radius, y) from the inner bead around the outside to the outer bead
    r_bead = 0.34
    for k in range(9):  # inner sidewall, bulging out
        t = k / 8
        prof.append((r_bead + (rb - 0.07 - r_bead) * t, -hw * (0.86 + 0.14 * math.sin(math.pi * t))))
    for k in range(1, 6):  # inner shoulder
        a = math.pi / 2 * k / 5
        prof.append((rb - 0.07 + 0.07 * math.sin(a), -hw + 0.07 * (1 - math.cos(a))))
    prof.append((rb, 0.0))
    for k in range(5, 0, -1):
        a = math.pi / 2 * k / 5
        prof.append((rb - 0.07 + 0.07 * math.sin(a), hw - 0.07 * (1 - math.cos(a))))
    for k in range(8, -1, -1):
        t = k / 8
        prof.append((r_bead + (rb - 0.07 - r_bead) * t, hw * (0.86 + 0.14 * math.sin(math.pi * t))))
    car = MS.Mesh()
    rings = []
    for k in range(n):
        a = 2 * math.pi * k / n
        ring = []
        for r, y in prof:
            ring.append(len(car.v))
            car.v.append((r * math.cos(a), y, r * math.sin(a)))
            car.t.append(None)
        rings.append(ring)
    for k in range(n):
        A, B = rings[k], rings[(k + 1) % n]
        for i in range(len(prof) - 1):
            car.f += [(A[i], B[i + 1], B[i]), (A[i], A[i + 1], B[i + 1])]
    car.f += [(i, k, j) for i, j, k in car.f]  # both sides (the profile winding flips across the tread)
    bars_m = MS.Mesh()
    pitch = 2 * math.pi / bars
    for k in range(bars):
        for side in (-1, 1):
            a0 = k * pitch + (pitch / 2 if side > 0 else 0.0)
            # bar footprint on the tread: from near the centre line out to the shoulder, angled ~25 deg (chevron)
            pts = []
            for y, da in ((0.02, 0.0), (hw - 0.04, 0.17), (hw - 0.04, 0.28), (0.02, 0.11)):
                pts.append((side * y, a0 + da))
            # top face at radius R, bottom at rb - 0.01 (into the carcass); and a shoulder lug going down
            def P(y, a, r):
                return (r * math.cos(a), y, r * math.sin(a))
            top = [P(y, a, R) for y, a in pts]
            bot = [P(y, a, rb - 0.01) for y, a in pts]
            q = [0, 1, 2, 3] if side > 0 else [0, 3, 2, 1]
            t_, b_ = [top[i] for i in q], [bot[i] for i in q]
            bars_m.tri(t_[0], t_[1], t_[2]); bars_m.tri(t_[0], t_[2], t_[3])
            bars_m.tri(b_[0], b_[2], b_[1]); bars_m.tri(b_[0], b_[3], b_[2])
            for i in range(4):
                j = (i + 1) % 4
                bars_m.tri(t_[i], b_[i], b_[j]); bars_m.tri(t_[i], b_[j], t_[j])
            # shoulder block continuing the bar over the edge of the tread
            ya = side * (hw - 0.02)
            # tilted about its own centre (not the wheel axis), then placed on the shoulder
            sh = MS.box(0.05, 0.03, lug * 0.8).moved(0, 0, 0, rot=MS.rot_x(side * 0.6)).moved(0, ya, rb - 0.005)
            sh = sh.moved(0, 0, 0, rot=MS.rot_y(-(a0 + 0.22)))
            bars_m.add(sh)
    return car, bars_m


def wheel_disc(r, hw):
    """Yellow disc wheel: rim flanges, dished face with a ring of holes, hub, 10 nuts, valve."""
    out = []
    flange = MS.revolve_y([(r - 0.02, -hw * 0.86), (r + 0.015, -hw * 0.86), (r + 0.015, -hw * 0.8), (r, -hw * 0.78),
                           (r, hw * 0.78), (r + 0.015, hw * 0.8), (r + 0.015, hw * 0.86), (r - 0.02, hw * 0.86)], n=48)
    out.append(('rim', 'ld_paint', flange))
    face = MS.Mesh()
    yo = hw * 0.62  # outer face plane (the dish sits inboard of the flange)
    face.add(MS.revolve_y([(r - 0.02, yo - 0.02), (0.24, yo), (0.2, yo + 0.035), (0.12, yo + 0.035), (0.0, yo + 0.035)], n=48))
    out.append(('disc', 'ld_paint', face.two_sided((0.0, -0.008, 0.0))))  # 8 mm plate: seen from both sides
    holes = MS.Mesh()
    for k in range(8):
        a = 2 * math.pi * k / 8 + 0.2
        holes.add(MS.revolve_y([(0.0, 0.0), (0.035, 0.0), (0.035, 0.004), (0.0, 0.004)], n=12)
                  .moved(0.215 * math.cos(a), yo + 0.008, 0.215 * math.sin(a)))
    out.append(('disc_holes', 'ld_black', holes))
    hub = MS.revolve_y([(0.0, yo + 0.035), (0.08, yo + 0.035), (0.08, yo + 0.09), (0.06, yo + 0.11), (0.0, yo + 0.11)], n=24)
    out.append(('hub', 'ld_grey', hub))
    nuts = MS.Mesh()
    for k in range(10):
        a = 2 * math.pi * k / 10
        nuts.add(MS.revolve_y([(0.0, 0.0), (0.018, 0.0), (0.018, 0.03), (0.0, 0.03)], n=6)
                 .moved(0.13 * math.cos(a), yo + 0.035, 0.13 * math.sin(a)))
    out.append(('nuts', 'ld_black', nuts))
    out.append(('valve', 'ld_chrome', cyl_between((0.26, yo + 0.01, 0.0), (0.26, yo + 0.05, 0.03), 0.007)))
    return out


# ------------------------------------------------------------------ the loader
def build(boom_len, wheel_r):
    """{body: [(name, material, Mesh), ...]}"""
    P = {'rear': [], 'front': [], 'boom': [], 'bucket': [], 'wheel': []}
    add = lambda body, name, mat, m: P[body].append((name, mat, m))

    # ===== rear frame: chassis, engine hood, counterweight, cab =====
    add('rear', 'chassis', 'ld_black', loft([(-2.45, 0.42, 0.42, 0.86, 0.06), (-0.15, 0.42, 0.42, 0.86, 0.06)]))
    # engine hood: tall and short, rounded top, slightly narrower at the back (scan top ~1.50, |y| <= 0.62)
    # WA100 hood: top about level with the bottom of the cab glass, sloping down to a rounded rear end,
    # slightly narrower at the back
    add('rear', 'hood', 'ld_paint', loft([
        (-2.54, 0.46, 0.86, 1.08, 0.12), (-2.5, 0.54, 0.84, 1.18, 0.16), (-2.38, 0.57, 0.84, 1.26, 0.18),
        (-2.1, 0.59, 0.84, 1.33, 0.17), (-1.6, 0.6, 0.84, 1.39, 0.15), (-1.08, 0.6, 0.84, 1.42, 0.12)], n=8))
    add('rear', 'counterweight', 'ld_black', loft([(-2.7, 0.66, 0.5, 1.0, 0.1), (-2.48, 0.7, 0.48, 1.02, 0.08)]))
    grille = MS.Mesh()
    for k in range(5):  # radiator grille on the sloping rear end of the hood, above the counterweight
        grille.add(MS.box(0.012, 0.34, 0.01).moved(-2.5 - 0.008 * k, 0, 1.17 - 0.03 * k))
    add('rear', 'grille', 'ld_black', grille)
    louv = MS.Mesh()
    for side in (-1, 1):
        for x0 in (-2.15, -1.75):
            for k in range(8):
                louv.add(MS.box(0.012, 0.006, 0.12).moved(x0 + 0.035 * k, side * 0.605, 1.12))
    add('rear', 'louvers', 'ld_black', louv)
    tails = MS.Mesh()
    amber = MS.Mesh()
    for side in (-1, 1):
        tails.add(MS.rounded_box(0.02, 0.07, 0.05, e=0.4).moved(-2.71, side * 0.55, 0.88))
        amber.add(MS.rounded_box(0.02, 0.04, 0.03, e=0.4).moved(-2.71, side * 0.42, 0.88))
    add('rear', 'taillights', 'ld_tail', tails)
    add('rear', 'indicators', 'ld_amber', amber)
    add('rear', 'exhaust', 'ld_black', cyl_between((-1.72, 0.3, 1.36), (-1.72, 0.3, 1.88), 0.055))
    add('rear', 'exhaust_cap', 'ld_chrome', cyl_between((-1.72, 0.3, 1.88), (-1.72, 0.3, 1.92), 0.06))
    add('rear', 'air_intake', 'ld_black', cyl_between((-1.45, -0.32, 1.38), (-1.45, -0.32, 1.66), 0.07))
    # rear fenders over the rear wheels (wheel hub x = -1.2, r = wheel_r)
    fend = MS.Mesh()
    for s in (-1, 1):
        y0, y1 = (0.6, 1.1) if s > 0 else (-1.1, -0.6)
        # thin fender following the tire, front part only (the hood covers the rear part), with a mud flap
        fend.add(MS.arc_band(-1.2, wheel_r + 0.01, wheel_r + 0.07, wheel_r + 0.095, math.radians(40), math.radians(150), y0 + 0.04, y1 - 0.02, n=28))
    add('rear', 'fenders', 'ld_paint_dirty', fend)
    flaps = MS.Mesh()
    for s_ in (-1, 1):
        flaps.add(MS.box(0.006, 0.17, 0.11).moved(-1.2 + (wheel_r + 0.08) * math.cos(math.radians(150)), s_ * 0.85, 0.5))
    add('rear', 'mud_flaps', 'ld_rubber', flaps)
    # cab (WA100): yellow base, black ROPS pillars, front pillars leaning forward, rounded upper rear corner,
    # large glass on all sides with a door frame on the left, roof with a small overhang
    cx0, cx1, chw, cz0, cz1 = -1.02, 0.18, 0.62, 1.22, 2.1
    lean = 0.12  # front pillars lean forward by this much at the top
    side = [(cx0, cz0), (cx1, cz0), (cx1 + lean, cz1), (cx0 + 0.12, cz1)]
    for k in range(1, 7):  # rounded rear top corner, radius 0.12
        a = math.pi / 2 + (math.pi / 2) * k / 6
        side.append((cx0 + 0.12 + 0.12 * math.cos(a), cz1 - 0.12 + 0.12 * math.sin(a)))
    # under the cab: dark deck on the frame, yellow lower door panels, fuel tank (right) and steps (left)
    add('rear', 'cab_deck', 'ld_black', loft([(cx0, 0.5, 0.84, cz0 - 0.02, 0.04), (cx1, 0.5, 0.84, cz0 - 0.02, 0.04)]))
    lower = MS.Mesh()
    for s_ in (-1, 1):
        lower.add(MS.rounded_box((cx1 - cx0) / 2, 0.025, 0.12, e=0.3).moved((cx0 + cx1) / 2, s_ * (chw - 0.02), cz0 - 0.1))
    add('rear', 'cab_lower', 'ld_paint_dirty', lower)
    add('rear', 'fuel_tank', 'ld_paint_dirty', loft([(-0.95, 0.24, 0.5, 0.95, 0.08), (-0.1, 0.24, 0.5, 0.95, 0.08)]).moved(0, -0.62, 0))
    add('rear', 'fuel_cap', 'ld_black', cyl_between((-0.3, -0.62, 0.95), (-0.3, -0.62, 0.99), 0.05))
    pil = MS.Mesh()
    yp = chw - 0.035
    for y in (-yp, yp):
        for (xa, za), (xb, zb) in zip(side, side[1:] + side[:1]):  # frame tubes along the side outline
            pil.add(cyl_between((xa, y, za), (xb, y, zb), 0.035, n=10))
        pil.add(cyl_between((-0.4, y, cz0), (-0.38, y, cz1), 0.03, n=10))  # door / B-pillar
        pil.add(cyl_between((-0.4, y, 1.62), (cx1 + lean * 0.45, y, 1.62), 0.018, n=8))  # door window rail
    for (xa, za) in ((cx0, cz0), (cx1, cz0), (cx1 + lean, cz1), (cx0 + 0.12, cz1), (cx0, cz1 - 0.12)):
        pil.add(cyl_between((xa, -yp, za), (xa, yp, za), 0.03, n=10))  # cross members
    add('rear', 'cab_frame', 'ld_black', pil)
    glass = MS.Mesh()
    inset = [(x + (0.03 if x < -0.4 else -0.03), z + (0.03 if z < 1.6 else -0.03)) for x, z in side]
    for y in (-yp, yp):
        glass.add(MS.extrude(inset, y - 0.004, y + 0.004))
    fz0, fz1 = cz0 + 0.04, cz1 - 0.04
    for x0_, x1_, s_ in ((cx1, cx1 + lean, -0.01), (cx0, cx0, 0.01)):  # front (leaning) and rear glass
        g = MS.Mesh()
        a_, b_ = (x0_ + (x1_ - x0_) * 0.04 / 0.88, fz0), (x0_ + (x1_ - x0_) * 0.84 / 0.88, fz1 - (0.08 if s_ > 0 else 0))
        q = [(a_[0], -yp + 0.03, a_[1]), (a_[0], yp - 0.03, a_[1]), (b_[0], yp - 0.03, b_[1]), (b_[0], -yp + 0.03, b_[1])]
        g.tri(q[0], q[1], q[2]); g.tri(q[0], q[2], q[3]); g.tri(q[0], q[2], q[1]); g.tri(q[0], q[3], q[2])
        glass.add(g)
    add('rear', 'cab_glass', 'ld_glass', glass)
    add('rear', 'cab_roof', 'ld_paint', loft([(cx0 + 0.02, chw + 0.04, cz1 - 0.02, cz1 + 0.08, 0.05),
                                              (cx1 + lean + 0.1, chw + 0.06, cz1 - 0.02, cz1 + 0.08, 0.05)], n=6))
    add('rear', 'cab_wiper', 'ld_black', cyl_between((cx1 + 0.03, -0.1, 1.32), (cx1 + 0.09, 0.25, 1.75), 0.008, n=6))
    inside = MS.Mesh()  # seat and steering column seen through the glass
    inside.add(MS.rounded_box(0.22, 0.24, 0.06, e=0.5).moved(-0.6, 0, 1.45))
    inside.add(MS.rounded_box(0.06, 0.22, 0.26, e=0.5).moved(-0.82, 0, 1.68))
    inside.add(cyl_between((-0.05, 0, 1.3), (-0.2, 0, 1.62), 0.035))
    inside.add(MS.revolve_y([(0.0, -0.015), (0.17, -0.015), (0.17, 0.015), (0.0, 0.015)], n=24).moved(0, 0, 0, rot=MS.rot_x(math.pi / 2)).moved(0, 0, 0, rot=MS.rot_y(-0.5)).moved(-0.22, 0, 1.65))
    add('rear', 'cab_inside', 'ld_black', inside)
    lamps = MS.Mesh()
    for y in (-0.4, 0.4):
        lamps.add(MS.rounded_box(0.04, 0.07, 0.045, e=0.5).moved(cx1 + 0.06, y, cz1 + 0.05))
        lamps.add(MS.rounded_box(0.04, 0.07, 0.045, e=0.5).moved(cx0 - 0.06, y, cz1 + 0.05))
    add('rear', 'work_lamps', 'ld_lamp', lamps)
    add('rear', 'beacon', 'ld_amber', MS.revolve_y([(0.0, 0.0), (0.06, 0.0), (0.06, 0.08), (0.0, 0.1)], n=16)
        .moved(0, 0, 0, rot=MS.rot_x(-math.pi / 2)).moved(-0.75, 0.42, cz1 + 0.1))
    mir = MS.Mesh()
    for s in (-1, 1):
        mir.add(cyl_between((cx1 - 0.02, s * chw, 1.95), (cx1 + 0.1, s * (chw + 0.18), 1.95), 0.012))
        mir.add(MS.rounded_box(0.02, 0.07, 0.1, e=0.4).moved(cx1 + 0.1, s * (chw + 0.2), 1.92))
    add('rear', 'mirrors', 'ld_black', mir)
    # steps and grab rail on the left (door side)
    steps = MS.Mesh()
    for z in (0.5, 0.8):
        steps.add(MS.box(0.18, 0.12, 0.015).moved(-0.42, 0.78, z))
    steps.add(MS.box(0.012, 0.012, 0.2).moved(-0.6, 0.86, 0.65))
    steps.add(MS.box(0.012, 0.012, 0.2).moved(-0.24, 0.86, 0.65))
    add('rear', 'steps', 'ld_black', steps)
    add('rear', 'grab_rail', 'ld_black', cyl_between((-0.2, 0.66, 1.3), (-0.2, 0.66, 1.95), 0.015))
    # articulation hinge (rear half)
    add('rear', 'hinge', 'ld_black', MS.box(0.12, 0.3, 0.12).moved(-0.1, 0, 0.62))

    # ===== front frame: black frame and towers, cylinder mounts, fenders, head lights =====
    # (the cylinders themselves, the bell crank and the links move with the joints: mocap bodies, see hydraulics.py)
    add('front', 'frame', 'ld_black', loft([(0.0, 0.4, 0.42, 0.9, 0.06), (1.35, 0.4, 0.42, 0.9, 0.06), (1.6, 0.34, 0.5, 0.82, 0.06)]))
    towers = MS.Mesh()
    for s in (-1, 1):
        # inside the lift arms (arms |y| 0.44-0.56): no overlap at the pivot, the boom pin goes through both
        towers.add(MS.extrude([(0.15, 0.85), (1.05, 0.85), (0.85, 1.3), (0.72, 1.86), (0.45, 1.86), (0.3, 1.3)],
                              s * 0.405 - 0.025, s * 0.405 + 0.025))
    add('front', 'towers', 'ld_black', towers)
    fend = MS.Mesh()
    for s in (-1, 1):
        y0, y1 = (0.6, 1.1) if s > 0 else (-1.1, -0.6)
        fend.add(MS.arc_band(1.2, wheel_r + 0.01, wheel_r + 0.08, wheel_r + 0.11, math.radians(55), math.radians(135), y0, y1, n=16))
    add('front', 'fenders', 'ld_black', fend)
    mounts = MS.Mesh()
    ax, az = HY.LIFT_A
    for s_ in (-1, 1):  # lift cylinder base: clevis on the inside of each tower
        y = s_ * HY.LIFT_Y
        for yy in (y - 0.085, y + 0.085):
            mounts.add(MS.extrude([(ax - 0.13, az - 0.16), (ax + 0.1, az - 0.16), (ax + 0.09, az + 0.06), (ax - 0.06, az + 0.1)],
                                  yy - 0.018, yy + 0.018))
        mounts.add(MS.box(0.12, (0.385 - (abs(y) + 0.1)) / 2, 0.1).moved(ax - 0.02, s_ * (0.385 + abs(y) + 0.1) / 2, az - 0.08))
        mounts.add(cyl_between((ax, y - 0.11, az), (ax, y + 0.11, az), 0.035))
    tx, tz = HY.TILT_A  # tilt cylinder base: cross plate between the towers with a clevis on top
    # (behind the base pin: the raised lift cylinders pass in front of it)
    mounts.add(MS.extrude([(tx - 0.22, 1.62), (tx + 0.04, 1.62), (tx + 0.04, 1.84), (tx - 0.18, 1.84)], -0.38, 0.38))
    for yy in (-0.1, 0.1):
        mounts.add(MS.extrude([(tx - 0.13, 1.8), (tx + 0.11, 1.8), (tx + 0.07, tz + 0.06), (tx - 0.08, tz + 0.08)], yy - 0.02, yy + 0.02))
    mounts.add(cyl_between((tx, -0.13, tz), (tx, 0.13, tz), 0.035))
    add('front', 'cyl_mounts', 'ld_black', mounts)
    hl = MS.Mesh()
    for s in (-1, 1):
        hl.add(MS.rounded_box(0.035, 0.035, 0.05, e=0.5).moved(0.845, s * 0.395, 1.44))  # on the tower front face
    add('front', 'head_lights', 'ld_lamp', hl)
    add('front', 'hinge', 'ld_black', MS.box(0.12, 0.25, 0.1).moved(0.1, 0, 0.62))

    # ===== boom: dog-leg lift arms, cross tube, Z-bar =====
    L = boom_len
    arm = [(-0.16, 0.14), (0.75, 0.4), (1.5, 0.36), (L - 0.02, 0.1), (L + 0.02, 0.0), (L - 0.02, -0.1), (1.5, 0.06), (0.75, 0.12), (-0.16, -0.14)]
    arms = MS.Mesh()
    for y in (-0.5, 0.5):
        arms.add(MS.extrude(arm, y - 0.06, y + 0.06))
    add('boom', 'arms', 'ld_paint', arms)
    bosses = MS.Mesh()
    for y in (-0.5, 0.5):
        for x, z in ((0.0, 0.0), (L, 0.0)):
            bosses.add(cyl_between((x, y - 0.08, z), (x, y + 0.08, z), 0.085))
    add('boom', 'pin_bosses', 'ld_black', bosses)
    # Z-bar: the bell crank turns on the cross tube (its pivot); crank, links and cylinders are mocap bodies
    px, pz = HY.CRANK_P
    add('boom', 'cross_tube', 'ld_paint', cyl_between((px, -0.5, pz), (px, 0.5, pz), 0.1))
    lugs = MS.Mesh()
    bx, bz = HY.LIFT_B
    for s_ in (-1, 1):  # lift cylinder rod end: clevis under each arm
        y = s_ * HY.LIFT_Y
        for yy in (y - 0.085, y + 0.085):
            lugs.add(MS.extrude([(bx - 0.14, 0.08), (bx + 0.14, 0.08), (bx + 0.08, bz - 0.07), (bx - 0.08, bz - 0.07)], yy - 0.018, yy + 0.018))
        lugs.add(MS.box(0.14, (0.44 - (abs(y) + 0.1)) / 2, 0.05).moved(bx, s_ * (0.44 + abs(y) + 0.1) / 2, 0.03))
        lugs.add(cyl_between((bx, y - 0.11, bz), (bx, y + 0.11, bz), 0.035))
    add('boom', 'cyl_lugs', 'ld_black', lugs)

    # ===== bucket (WA100 general purpose bucket; physics bucket: pin at origin, floor z -0.33, lip x 1.13,
    # width 2.16): J-profile shell, yellow outside / worn steel inside, side plates, spill guard, cutting edge
    # the boom pins sit just behind the back wall: the whole shell is shifted forward by SHIFT (lip stays at the
    # physics lip, x ~1.16)
    SHIFT = 0.12
    _add = add
    add = lambda b, n, mat, m: _add(b, n, mat, m.moved(SHIFT, 0, 0))
    W = 1.06  # half width
    prof = [(1.04, -0.36), (0.35, -0.36)]  # floor, from the lip back
    for k in range(1, 13):
        a = -math.pi / 2 - (math.pi / 2) * k / 12
        prof.append((0.35 + 0.36 * math.cos(a), 0.0 + 0.36 * math.sin(a)))  # rounded heel
    prof += [(-0.02, 0.3), (0.02, 0.56), (0.12, 0.66)]  # back wall up to the spill guard lip
    t = 0.035
    # offset_band: positive = to the left of the path = outside of the bucket here
    outer = MS.extrude(MS.offset_band(prof, t), -W, W)  # plate, thickness outwards
    inner = MS.extrude(MS.offset_band(prof, -0.008), -W + 0.025, W - 0.025)  # worn liner on the inside
    add('bucket', 'shell', 'ld_paint_dirty', outer)
    add('bucket', 'liner', 'ld_steel', inner)
    side = [(1.04, -0.36), (1.04, -0.3), (0.9, 0.1), (0.42, 0.56), (0.12, 0.68), (-0.04, 0.58), (-0.06, 0.0)]
    sides = MS.Mesh()
    for y in (-W - 0.02, W + 0.02):
        sides.add(MS.extrude(side, y - 0.022, y + 0.022))
    add('bucket', 'sides', 'ld_paint_dirty', sides)
    wear = MS.Mesh()  # side wear strips along the front edge of the side plates
    for y in (-W - 0.02, W + 0.02):
        wear.add(MS.extrude(MS.offset_band([(1.04, -0.3), (0.9, 0.1), (0.42, 0.56)], 0.04), y - 0.026, y + 0.026))
    add('bucket', 'side_edges', 'ld_grey', wear)
    add('bucket', 'edge', 'ld_grey', MS.extrude([(0.86, -0.37), (1.12, -0.37), (1.12, -0.35), (0.86, -0.32)], -W - 0.04, W + 0.04))
    ribs = MS.Mesh()  # vertical stiffeners on the back
    for y in (-0.7, 0.7):
        ribs.add(MS.extrude(MS.offset_band([(0.6, -0.4), (0.35, -0.385), (0.12, -0.31), (-0.02, -0.12), (-0.06, 0.1), (-0.03, 0.55)], 0.05), y - 0.03, y + 0.03))
    add('bucket', 'ribs', 'ld_paint_dirty', ribs)
    add = _add
    ears = MS.Mesh()  # pin ears (boom pins at the origin) and Z-bar link ears
    for y in (-0.5, 0.5):
        ears.add(MS.extrude([(-0.17, -0.1), (-0.03, -0.2), (-0.03, 0.14), (-0.15, 0.12)], y - 0.06, y + 0.06))
        ears.add(cyl_between((0.0, y - 0.08, 0.0), (0.0, y + 0.08, 0.0), 0.07))
    dx, dz = HY.LINK_D  # link ears on the back wall, either side of the two links
    for yy in (-0.165, 0.165):
        ears.add(MS.extrude([(dx - 0.09, dz - 0.07), (dx + 0.1, dz - 0.16), (dx + 0.12, dz + 0.16), (dx - 0.06, dz + 0.08)], yy - 0.02, yy + 0.02))
    ears.add(cyl_between((dx, -0.19, dz), (dx, 0.19, dz), 0.04))
    add('bucket', 'ears', 'ld_black', ears)
    # light-material extension as on the site's loader: expanded-metal grid of thin dark bars. The back panel has
    # a notch in the middle (|y| < NOTCH): the Z-bar crank and the tilt rod pass there when the bucket is racked
    # back (hydraulics.py; without it the crank went through the grid in most carry poses)
    NOTCH = 0.24
    ext = MS.Mesh()
    for k in range(9):  # back panel: vertical + horizontal bars, left and right of the notch
        z = 0.7 + 0.04 * k
        for s_ in (-1, 1):
            ext.add(MS.box(0.006, (W - NOTCH) / 2, 0.006).moved(0.13, s_ * (W + NOTCH) / 2, z))
    for k in range(25):
        y = -W + 2 * W * k / 24
        if abs(y) > NOTCH + 0.02:
            ext.add(MS.box(0.006, 0.006, 0.17).moved(0.13, y, 0.87))
    for y in (-W - 0.02, W + 0.02):  # side panels: triangle filled with slanted bars
        for k in range(8):
            x = 0.15 + 0.035 * k
            ztop = 1.04 - (x - 0.13) * (0.32 / 0.29)
            zbot = 0.68 - (x - 0.12) * (0.12 / 0.3)
            ext.add(MS.box(0.004, 0.004, (ztop - zbot) / 2).moved(x, y, (ztop + zbot) / 2))
    add('bucket', 'mesh', 'ld_black', ext.moved(SHIFT, 0, 0))
    frame = MS.Mesh()
    for s_ in (-1, 1):  # top tube in two pieces, and the notch edges
        frame.add(cyl_between((0.13, s_ * NOTCH, 1.04), (0.13, s_ * (W + 0.02), 1.04), 0.025))
        frame.add(cyl_between((0.13, s_ * NOTCH, 0.68), (0.13, s_ * NOTCH, 1.04), 0.022))
    for y in (-W - 0.02, W + 0.02):
        frame.add(cyl_between((0.12, y, 0.68), (0.13, y, 1.04), 0.022))
        frame.add(cyl_between((0.13, y, 1.04), (0.42, y, 0.72), 0.022))
        frame.add(cyl_between((0.42, y, 0.72), (0.42, y, 0.56), 0.022))
    add('bucket', 'mesh_frame', 'ld_paint', frame.moved(SHIFT, 0, 0))

    # ===== wheel (same mesh for all four; hub at origin, axis y). 15.5-25 L3 style tire: rounded carcass,
    # deep chevron bars running onto the shoulders; yellow disc wheel: dished face, hub, 10 nuts, valve
    R, hw = wheel_r, 0.21
    tire_m, lugs_m = tire_l3(R, hw)
    add('wheel', 'tire', 'ld_rubber', tire_m)
    add('wheel', 'tread', 'ld_rubber', lugs_m)
    for nm, mat, m in wheel_disc(0.335, hw):
        add('wheel', nm, mat, m)
    return P
