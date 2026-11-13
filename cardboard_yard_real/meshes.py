"""Procedural meshes for CardboardYardReal (wheel loader parts, cardboard heaps), written as OBJ text.

All meshes are visual only (the env gives their geoms contype=conaffinity=0, mass=0), so their exact shape
never touches the physics. Coordinates are in the frame of the body the geom is attached to.
"""

import math

import numpy as np


class Mesh:
    def __init__(self):
        self.v = []  # vertices (x, y, z)
        self.t = []  # texcoords (u, v) or None
        self.f = []  # triangles (i, j, k) into v (same index into t when t is used)

    def tri(self, a, b, c, ta=None, tb=None, tc=None):
        """Add one triangle with its own vertices (flat shading)."""
        n = len(self.v)
        self.v += [a, b, c]
        self.t += [ta, tb, tc]
        self.f.append((n, n + 1, n + 2))

    def quad(self, a, b, c, d, **kw):
        self.tri(a, b, c)
        self.tri(a, c, d)

    def add(self, other):
        n = len(self.v)
        self.v += other.v
        self.t += other.t
        self.f += [(i + n, j + n, k + n) for i, j, k in other.f]
        return self

    def two_sided(self, shift=(0.0, 0.0, 0.0)):
        """Copy with every triangle also in reverse order (a thin plate): open surfaces (a dished wheel face) are
        drawn from both sides (the renderer culls back faces). shift moves the back faces off the front ones; at
        the same place they shadow the front faces (shadow acne)."""
        m = Mesh()
        n = len(self.v)
        m.v = list(self.v) + [tuple(np.add(p, shift)) for p in self.v]
        m.t = list(self.t) * 2
        m.f = list(self.f) + [(i + n, k + n, j + n) for i, j, k in self.f]
        return m

    def moved(self, dx=0.0, dy=0.0, dz=0.0, rot=None):
        m = Mesh()
        V = np.asarray(self.v, float)
        if rot is not None:
            V = V @ np.asarray(rot).T
        m.v = [tuple(p) for p in V + [dx, dy, dz]]
        m.t, m.f = list(self.t), list(self.f)
        return m

    def normals(self):
        """Vertex normals: area-weighted average of the faces using the vertex (a vertex that belongs to
        one triangle only, as in the flat-shaded parts, gets that face's normal)."""
        V = np.asarray(self.v, float)
        N = np.zeros_like(V)
        for i, j, k in self.f:
            n = np.cross(V[j] - V[i], V[k] - V[i])
            N[i] += n
            N[j] += n
            N[k] += n
        L = np.linalg.norm(N, axis=1, keepdims=True)
        return N / np.where(L > 0, L, 1)

    def obj(self):
        lines = [f'v {x:.5f} {y:.5f} {z:.5f}' for x, y, z in self.v]
        # no 'vn': MuJoCo moves the mesh into its inertial frame and would keep our normals unrotated;
        # it computes them itself (per face here, since flat-shaded faces do not share vertices)
        uv = all(t is not None for t in self.t) and self.t
        if uv:
            lines += [f'vt {u:.5f} {v:.5f}' for u, v in self.t]
            lines += [f'f {i + 1}/{i + 1} {j + 1}/{j + 1} {k + 1}/{k + 1}' for i, j, k in self.f]
        else:
            lines += [f'f {i + 1} {j + 1} {k + 1}' for i, j, k in self.f]
        return '\n'.join(lines) + '\n'


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _area2(poly):
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly)))


def triangulate(poly):
    """Ear clipping for a simple polygon (list of (a, b)); returns index triples, counter-clockwise."""
    idx = list(range(len(poly)))
    if _area2(poly) < 0:
        idx.reverse()
    P = [np.asarray(p, float) for p in poly]

    def inside(p, a, b, c):
        d = lambda u, v, w: (u[0] - w[0]) * (v[1] - w[1]) - (v[0] - w[0]) * (u[1] - w[1])
        d1, d2, d3 = d(p, a, b), d(p, b, c), d(p, c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    out = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = P[i0], P[i1], P[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue  # reflex
            if any(inside(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def extrude(poly_xz, y0, y1):
    """Closed prism: polygon in the x-z plane, extruded along y from y0 to y1 (flat shaded)."""
    m = Mesh()
    n = len(poly_xz)
    ccw = _area2(poly_xz) > 0
    for i in range(n):
        (x0, z0), (x1, z1) = poly_xz[i], poly_xz[(i + 1) % n]
        a, b, c, d = (x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)
        if ccw:
            m.quad(a, d, c, b)
        else:
            m.quad(a, b, c, d)
    for i, j, k in triangulate(poly_xz):
        (ax, az), (bx, bz), (cx, cz) = poly_xz[i], poly_xz[j], poly_xz[k]
        m.tri((ax, y0, az), (bx, y0, bz), (cx, y0, cz))
        m.tri((ax, y1, az), (cx, y1, cz), (bx, y1, bz))
    return m


def box(hx, hy, hz):
    return extrude([(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)], -hy, hy)


def rounded_box(hx, hy, hz, e=0.25, n=24):
    """Superellipsoid: a box with rounded edges and corners (e -> 0 is sharper). Smooth shading."""
    m = Mesh()
    sp = lambda w, p: math.copysign(abs(w) ** p, w)
    grid = []
    for i in range(n + 1):
        th = -math.pi / 2 + math.pi * i / n
        row = []
        for j in range(2 * n + 1):
            ph = -math.pi + 2 * math.pi * j / (2 * n)
            ce, se = sp(math.cos(th), e), sp(math.sin(th), e)
            row.append((hx * ce * sp(math.cos(ph), e), hy * ce * sp(math.sin(ph), e), hz * se))
        grid.append(row)
    base = len(m.v)
    for row in grid:
        for p in row:
            m.v.append(p)
            m.t.append(None)
    w = 2 * n + 1
    for i in range(n):
        for j in range(2 * n):
            a, b, c, d = base + i * w + j, base + i * w + j + 1, base + (i + 1) * w + j + 1, base + (i + 1) * w + j
            m.f += [(a, b, c), (a, c, d)]
    return m


def revolve_y(profile_rz, n=40):
    """Surface of revolution about the y axis; profile = [(radius, y), ...] from one end to the other."""
    m = Mesh()
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        for (r0, y0), (r1, y1) in zip(profile_rz[:-1], profile_rz[1:]):
            p = lambda r, y, a: (r * math.cos(a), y, r * math.sin(a))
            m.quad(p(r0, y0, a0), p(r0, y0, a1), p(r1, y1, a1), p(r1, y1, a0))
    return m


def tire(radius, half_width, lugs=20, lug_h=0.05):
    """Tire: ring with rounded shoulders (the rim sits inside) and chunky tread blocks in a chevron
    pattern that run over the shoulders, as on loader L2/L3 tires; axis along y."""
    r_in = radius * 0.62
    w = half_width
    prof = [(r_in, -w * 0.85), (radius - 0.13, -w), (radius - lug_h - 0.03, -w * 0.97), (radius - lug_h, -w * 0.8),
            (radius - lug_h, w * 0.8), (radius - lug_h - 0.03, w * 0.97), (radius - 0.13, w), (r_in, w * 0.85)]
    m = revolve_y(prof, n=60)
    pitch = 2 * math.pi / lugs
    for k in range(lugs):
        for side in (-1, 1):
            a = k * pitch + (pitch / 2 if side > 0 else 0)
            blk = box(0.075, w * 0.36, lug_h / 2 + 0.01).moved(0, 0, 0, rot=rot_x(side * 0.45))
            blk = blk.moved(0, side * w * 0.5, radius - lug_h / 2)
            sh = box(0.06, 0.03, 0.06).moved(0, side * (w - 0.01), radius - lug_h - 0.04)  # over the shoulder
            m.add(blk.moved(0, 0, 0, rot=rot_y(a)))
            m.add(sh.moved(0, 0, 0, rot=rot_y(a)))
    return m


def rim(r):
    """Yellow wheel: flat dish, raised lip, centre hub with a ring of nuts (both sides)."""
    m = revolve_y([(0.0, -0.13), (0.12, -0.13), (0.14, -0.1), (r - 0.04, -0.1), (r, -0.17), (r, 0.17),
                   (r - 0.04, 0.1), (0.14, 0.1), (0.12, 0.13), (0.0, 0.13)], n=40)
    return m


def hub_nuts():
    m = Mesh()
    for k in range(10):
        a = 2 * math.pi * k / 10
        for side in (-1, 1):
            m.add(revolve_y([(0.0, 0.0), (0.018, 0.0), (0.018, 0.025), (0.0, 0.025)], n=8)
                  .moved(0.085 * math.cos(a), side * 0.13 + (0 if side > 0 else -0.025), 0.085 * math.sin(a)))
    return m


def decal(center, half_w, half_h, normal):
    """Textured quad (u along the text, v up) a few mm thick, facing +y, -y or -x; text reads correctly
    when seen from the side it faces. center = (x, y, z) in the body frame."""
    cx, cy, cz = center
    m = Mesh()
    if normal in ('+y', '-y'):
        s = 1 if normal == '+y' else -1
        # seen from +y the text runs towards -x (u increases to the viewer's right = -x)
        p = lambda u, v: (cx - s * u, cy, cz + v)
    else:  # '-x': seen from behind, viewer's right = +y... text runs towards -y
        p = lambda u, v: (cx, cy - u, cz + v)
    a, b_, c, d = p(-half_w, -half_h), p(half_w, -half_h), p(half_w, half_h), p(-half_w, half_h)
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    m.tri(a, b_, c, uv[0], uv[1], uv[2])
    m.tri(a, c, d, uv[0], uv[2], uv[3])
    m.tri(a, c, b_, uv[0], uv[2], uv[1])  # back side too (MuJoCo may cull back faces)
    m.tri(a, d, c, uv[0], uv[3], uv[2])
    return m


def arc_band(cx, cz, r_in, r_out, a0, a1, y0, y1, n=16):
    """Curved plate (fender): annulus sector in x-z around (cx, cz), angles in rad, extruded along y."""
    outer = [(cx + r_out * math.cos(a0 + (a1 - a0) * i / n), cz + r_out * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
    inner = [(cx + r_in * math.cos(a1 - (a1 - a0) * i / n), cz + r_in * math.sin(a1 - (a1 - a0) * i / n)) for i in range(n + 1)]
    return extrude(outer + inner, y0, y1)


def offset_band(path, thick):
    """Closed polygon of a thick polyline: path on one side, offset by 'thick' to the left."""
    P = np.asarray(path, float)
    nrm = []
    for i in range(len(P)):
        a = P[max(i - 1, 0)]
        b = P[min(i + 1, len(P) - 1)]
        t = b - a
        t /= np.linalg.norm(t)
        nrm.append(np.array([-t[1], t[0]]))
    Q = [tuple(P[i] + thick * nrm[i]) for i in range(len(P))]
    return [tuple(p) for p in P] + Q[::-1]


def heap(x0, x1, y0, y1, height, rng, nx=60, ny=16, uv_scale=2.5):
    """Cardboard heap: bumpy height field (high at y1 = against the wall), textured with u, v in metres."""
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    bumps = rng.normal(0, 1, (ny, nx))
    k = np.ones(5) / 5
    for _ in range(2):
        bumps = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, bumps)
        bumps = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, bumps)
    m = Mesh()
    H = np.zeros((ny, nx))
    for i, y in enumerate(ys):
        ty = (y - y0) / (y1 - y0)
        for j, x in enumerate(xs):
            tx = min(x - x0, x1 - x) / max(1e-6, 0.25 * (x1 - x0))
            H[i, j] = max(0.0, height * min(1.0, tx) ** 0.7 * ty ** 0.8 * (1 + 0.35 * bumps[i, j]))
    uv = lambda x, y: (x / uv_scale, y / uv_scale)
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = (xs[j], ys[i], H[i, j])
            b = (xs[j + 1], ys[i], H[i, j + 1])
            c = (xs[j + 1], ys[i + 1], H[i + 1, j + 1])
            d = (xs[j], ys[i + 1], H[i + 1, j])
            m.tri(a, b, c, uv(*a[:2]), uv(*b[:2]), uv(*c[:2]))
            m.tri(a, c, d, uv(*a[:2]), uv(*c[:2]), uv(*d[:2]))
    return m


# ------------------------------------------------------------------ wheel loader (WA100-class) parts
# Frames follow cardboard_yard/env.py: rear body (articulation at x=0, ground z=0), front body (same origin),
# boom body (origin at the boom pivot, arm along +x, length BOOM_LEN), bucket body (origin at the bucket pin),
# wheel bodies (origin at the hub, axis y).


def loader_meshes(boom_len):
    M = {}
    # rear body ---------------------------------------------------------------
    # engine area as on the WA100 photo: short, tall hood right behind the cab (flat top, vertical rear
    # with a rounded upper corner), black frame / counterweight below it, louver panels on both sides
    hood = [(-0.98, 1.08), (-0.98, 1.66)]
    for k in range(7):
        a = math.pi / 2 - (math.pi / 2) * k / 6
        hood.append((-2.06 + 0.16 * math.cos(a), 1.5 + 0.16 * math.sin(a)))
    hood += [(-2.22, 1.08)]
    M['hood'] = ('yellow', extrude(hood, -0.62, 0.62))
    M['counterweight'] = ('black', box(0.5, 0.6, 0.28).moved(-2.05, 0, 0.8)
                          .add(box(0.12, 0.62, 0.05).moved(-2.45, 0, 0.5)))
    M['rear_chassis'] = ('dark', box(1.05, 0.46, 0.24).moved(-1.05, 0, 0.72))
    M['cab_base'] = ('yellow', extrude([(0.12, 0.95), (0.12, 1.32), (-0.98, 1.32), (-0.98, 0.95)], -0.64, 0.64))
    M['cab_glass'] = ('glass', rounded_box(0.52, 0.6, 0.55, e=0.18).moved(-0.43, 0, 1.86))
    M['cab_roof'] = ('yellow', rounded_box(0.62, 0.68, 0.06, e=0.3).moved(-0.43, 0, 2.45))
    frame = Mesh()
    for x in (0.08, -0.95):
        for y in (-0.61, 0.61):
            frame.add(box(0.045, 0.035, 0.56).moved(x, y, 1.87))
    for y in (-0.61, 0.61):  # door / window frame bars
        frame.add(box(0.53, 0.03, 0.03).moved(-0.43, y, 1.62))
        frame.add(box(0.03, 0.03, 0.52).moved(-0.43, y, 1.86))
    M['cab_frame'] = ('black', frame)
    fend = Mesh()
    for s in (-1, 1):
        y0, y1 = (0.62, 1.08) if s > 0 else (-1.08, -0.62)
        fend.add(arc_band(-1.2, 0.56, 0.62, 0.66, math.radians(55), math.radians(160), y0, y1))  # front-top
    M['fenders_rear'] = ('black', fend)
    det = Mesh()
    det.add(box(0.01, 0.5, 0.17).moved(-2.225, 0, 1.3))  # rear grille
    for x0 in (-1.48, -1.72):  # two louver panels per side
        for k in range(6):
            for y in (-0.625, 0.625):
                det.add(box(0.008, 0.006, 0.15).moved(x0 - 0.035 * k, y, 1.32))
    for y in (-0.95, 0.95):
        det.add(box(0.2, 0.13, 0.015).moved(-0.55, y, 0.6))  # steps
        det.add(box(0.2, 0.13, 0.015).moved(-0.55, y, 0.92))
    det.add(revolve_y([(0.055, -0.25), (0.055, 0.25)], n=12).moved(-1.6, 0.35, 1.95, rot=rot_x(math.pi / 2)))  # exhaust
    for y in (-0.72, 0.72):
        det.add(box(0.015, 0.015, 0.3).moved(0.15, y, 2.1))  # mirror arms
        det.add(box(0.02, 0.11, 0.08).moved(0.15, y * 1.12, 2.3))  # mirrors
    M['rear_details'] = ('black', det)
    lights = Mesh()
    for y in (-0.5, 0.5):
        lights.add(box(0.015, 0.06, 0.04).moved(-2.23, y, 1.55))
    M['taillights'] = ('taillight', lights)
    lamps = Mesh()
    for y in (-0.42, 0.42):
        lamps.add(box(0.03, 0.08, 0.04).moved(0.17, y, 2.43))
    M['worklamps'] = ('lamp', lamps)
    M['beacon'] = ('beacon', revolve_y([(0.0, 0), (0.07, 0), (0.07, 0.1), (0.0, 0.12)], n=16)
                   .moved(-0.75, 0.45, 2.5, rot=rot_x(-math.pi / 2)))
    # front body --------------------------------------------------------------
    M['front_chassis'] = ('dark', box(0.78, 0.42, 0.22).moved(0.78, 0, 0.75))
    tower = Mesh()
    for y in (-0.5, 0.5):
        tower.add(extrude([(0.25, 0.6), (1.0, 0.6), (0.85, 1.25), (0.72, 1.86), (0.45, 1.86), (0.3, 1.2)], y - 0.07, y + 0.07))
    M['towers'] = ('dark', tower)  # black front frame / towers, as on the WA100
    fend = Mesh()
    for s in (-1, 1):
        y0, y1 = (0.62, 1.08) if s > 0 else (-1.08, -0.62)
        fend.add(arc_band(1.2, 0.56, 0.64, 0.69, math.radians(40), math.radians(140), y0, y1))
    M['fenders_front'] = ('dark', fend)
    cyl = Mesh()
    rod = Mesh()
    for y in (-0.3, 0.3):  # lift cylinders: body on the frame, rod up to the boom
        cyl.add(revolve_y([(0.075, 0), (0.075, 0.6)], n=14).moved(0.0, 0, 0.0, rot=rot_x(-math.pi / 2))
                .moved(0, 0, 0, rot=rot_y(math.radians(48))).moved(0.95, y, 0.92))
        rod.add(revolve_y([(0.04, 0), (0.04, 0.45)], n=12).moved(0, 0, 0, rot=rot_x(-math.pi / 2))
                .moved(0, 0, 0, rot=rot_y(math.radians(48))).moved(1.38, y, 1.31))
    M['lift_cyl'] = ('black', cyl)
    M['lift_rod'] = ('chrome', rod)
    # boom body: dog-leg lift arms, cross tube, Z-bar tilt lever --------------
    L = boom_len
    arm = [(-0.18, 0.16), (0.8, 0.42), (1.55, 0.36), (L + 0.12, 0.1), (L + 0.12, -0.13), (1.55, 0.06), (0.8, 0.13), (-0.18, -0.16)]
    boom = Mesh()
    for y in (-0.58, 0.58):
        boom.add(extrude(arm, y - 0.08, y + 0.08))
    boom.add(revolve_y([(0.11, -0.6), (0.11, 0.6)], n=16).moved(1.45, 0, 0.24))
    M['boom'] = ('yellow', boom)
    lever = extrude([(1.25, 0.25), (1.45, 0.25), (1.2, 0.95), (1.0, 0.95)], -0.09, 0.09)
    M['tilt_lever'] = ('yellow', lever)
    tilt = Mesh()
    tilt.add(revolve_y([(0.075, 0), (0.075, 0.9)], n=14).moved(0, 0, 0, rot=rot_z(math.pi / 2))
             .moved(0, 0, 0, rot=rot_y(math.radians(-6))).moved(0.15, 0, 0.82))
    M['tilt_cyl'] = ('black', tilt)
    link = Mesh()
    link.add(box(0.7, 0.05, 0.04).moved(0, 0, 0, rot=rot_y(math.radians(28))).moved(1.75, 0, 0.62))
    M['tilt_link'] = ('yellow', link)
    # bucket body: curved shell, side plates, cutting edge ---------------------
    shell_path = [(1.2, -0.36), (0.32, -0.36)]
    for k in range(1, 9):
        a = -math.pi / 2 - (math.pi / 2) * k / 8
        shell_path.append((0.32 + 0.34 * math.cos(a), -0.02 + 0.34 * math.sin(a)))
    shell_path += [(-0.02, 0.62)]
    shell = extrude(offset_band(shell_path, -0.05), -1.04, 1.04)
    M['bucket_shell'] = ('bucket_paint', shell)  # dark worn steel, as in the site video
    side = [(1.2, -0.36), (1.2, -0.24), (0.85, 0.22), (0.35, 0.5), (-0.02, 0.62), (-0.04, -0.02), (0.3, -0.38)]
    sides = Mesh()
    for y in (-1.06, 1.06):
        sides.add(extrude(side, y - 0.025, y + 0.025))
    M['bucket_sides'] = ('bucket_paint', sides)
    M['bucket_edge'] = ('dark', box(0.05, 1.09, 0.035).moved(1.2, 0, -0.34))
    ext = Mesh()  # high mesh extension of the light-material bucket
    ext.add(box(0.015, 1.06, 0.3).moved(-0.03, 0, 0.92))
    for y in (-1.07, 1.07):
        ext.add(extrude([(0.0, 0.62), (0.85, 0.24), (0.85, 0.42), (0.0, 1.22)], y - 0.01, y + 0.01))
    M['bucket_mesh'] = ('mesh', ext)
    rail = Mesh()
    rail.add(box(0.035, 1.1, 0.035).moved(-0.03, 0, 1.23))
    for y in (-1.07, 1.07):
        rail.add(box(0.035, 0.035, 0.32).moved(-0.03, y, 0.92))
        rail.add(box(0.46, 0.03, 0.03).moved(0.42, y, 0.0).moved(0, 0, 0)  # top frame of the side mesh
                 .moved(0, 0, 0, rot=None).moved(0, 0, 0.83))
    M['bucket_rail'] = ('yellow', rail)
    # wheels ------------------------------------------------------------------
    M['tire'] = ('tire', tire(0.56, 0.22))
    M['rim'] = ('rim', rim(0.35))
    M['hub'] = ('dark', hub_nuts())
    return M
