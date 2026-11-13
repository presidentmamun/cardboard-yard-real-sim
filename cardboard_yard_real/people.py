"""Walking people for CardboardYardReal (physics profile key people='walk', v4).

The v0 people slid along a 3-11 m line back and forth (or stood still) as stiff capsule figures, so most of the
time they looked like they were standing. Here:

- they walk from waypoint to waypoint over the whole people area (1.0-1.5 m/s, a pause of 1-5 s at about half of
  the waypoints), some cross the pile -> hopper route again and again, a few stand and watch the loader
- they turn smoothly (no snapping at waypoints), wait when the loader is right there and then walk away from it
- at the roadside they wait while the loader is near and their walk would pass within 3 m of it (up to
  10 s, then they walk elsewhere; within 4 m of it they walk away at once)
- they stand / pause only off the pile -> hopper route (4.5 m idle, 3 m waypoints) and clear of the digging lane
  and the hopper approach; crossing the route on the way is fine
- articulated figure (1.75 m): pelvis/torso/head on the root mocap body, thighs, shins, feet, upper and fore arms as
  their own mocap bodies, placed every step by a walking cycle (leg and opposite arm swing, knee bend in the swing
  phase, body bob); hi-vis vest with reflective bands, helmet with brim, boots, hands
- every part collides like the v0 figure (touching any of it is a hit)

State: the walking state stays in env.workers (same 12 fields, kinds IDLE / CROSS / WALK), so the obs, the
expert's people queries and get/set state are unchanged. The gait phase and the facing are looks only.
"""

import math

import numpy as np

KIND_WALK = 3
LIMBS = ('thigh_l', 'thigh_r', 'shin_l', 'shin_r', 'foot_l', 'foot_r', 'uarm_l', 'uarm_r', 'farm_l', 'farm_r')
HIP_Z, HIP_Y, THIGH, SHIN = 0.93, 0.1, 0.44, 0.42
SHOULDER_Z, SHOULDER_Y, UPPER, FORE = 1.43, 0.22, 0.29, 0.26
STRIDE = 1.4  # m per gait cycle (two steps)

MATERIALS = {  # name: rgba, specular
    'p_trousers': ('0.13 0.15 0.22 1', 0.1),
    'p_boot': ('0.08 0.07 0.06 1', 0.3),
    'p_band': ('0.85 0.86 0.88 1', 0.9),  # reflective bands on the vest
    'p_glove': ('0.55 0.45 0.3 1', 0.1),
}
LOOKS = [('worker_suit', 'vest', 'helmet'), ('suit2', 'vest2', 'helmet'), ('worker_suit', 'vest2', 'helmet')]


def _q_yaw(a):
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def _q_pitch(t):
    """Rotation that turns the body's -z axis (a hanging limb) forward by t (towards +x)."""
    return np.array([math.cos(-t / 2), 0.0, math.sin(-t / 2), 0.0])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _down(t):
    return np.array([math.sin(t), 0.0, -math.cos(t)])


def replace_people(root, n_workers):
    """Swap the v0 worker bodies in the MJCF tree for articulated ones (same root names, plus limb bodies)."""
    import xml.etree.ElementTree as ET

    asset, world = root.find('asset'), root.find('worldbody')
    for name, (rgba, spec) in MATERIALS.items():
        if asset.find(f"material[@name='{name}']") is None:
            ET.SubElement(asset, 'material', {'name': name, 'rgba': rgba, 'specular': str(spec)})
    st = {'contype': '1', 'conaffinity': '3'}
    vis = {'contype': '0', 'conaffinity': '0'}

    def g(body, **kw):
        a = {k: str(v) for k, v in kw.items()}
        ET.SubElement(body, 'geom', a)

    for i in range(n_workers):
        old = world.find(f"body[@name='worker{i}']")
        idx = list(world).index(old)
        world.remove(old)
        suit, vest, helmet = LOOKS[i % len(LOOKS)]
        park = f'{30 + 2 * i} -20 -5'
        b = ET.Element('body', {'name': f'worker{i}', 'mocap': 'true', 'pos': park})
        g(b, name=f'worker{i}_pelvis', type='capsule', fromto=f'0 -0.12 {HIP_Z + 0.03} 0 0.12 {HIP_Z + 0.03}', size='0.12',
          material='p_trousers', **st)
        # torso and vest as ellipsoids (wider than deep; the v0-style box vest looked like a crate)
        zc = (HIP_Z + SHOULDER_Z) / 2 + 0.02
        g(b, name=f'worker{i}_torso', type='ellipsoid', pos=f'0 0 {zc:.3f}', size='0.13 0.19 0.29', material=suit, **st)
        g(b, name=f'worker{i}_vest', type='ellipsoid', pos=f'0 0 {zc + 0.06:.3f}', size='0.145 0.205 0.24', material=vest, **vis)
        for z in (zc - 0.04, zc + 0.1):  # bands just outside the vest at their height
            k = math.sqrt(max(0.0, 1 - ((z - zc - 0.06) / 0.24) ** 2))
            g(b, type='ellipsoid', pos=f'0 0 {z:.3f}', size=f'{0.145 * k + 0.004:.3f} {0.205 * k + 0.004:.3f} 0.02',
              material='p_band', **vis)
        g(b, name=f'worker{i}_shoulders', type='capsule', fromto=f'0 {-SHOULDER_Y + 0.04} {SHOULDER_Z} 0 {SHOULDER_Y - 0.04} {SHOULDER_Z}',
          size='0.075', material=vest, **st)
        g(b, type='capsule', fromto=f'0 0 {SHOULDER_Z} 0 0 {SHOULDER_Z + 0.1}', size='0.05', material='skin', **vis)
        g(b, name=f'worker{i}_head', type='ellipsoid', pos=f'0.01 0 {SHOULDER_Z + 0.19}', size='0.095 0.085 0.11',
          material='skin', **st)
        g(b, type='ellipsoid', pos=f'0 0 {SHOULDER_Z + 0.26}', size='0.12 0.11 0.08', material=helmet, **vis)
        g(b, type='cylinder', pos=f'0.03 0 {SHOULDER_Z + 0.22}', size='0.135 0.006', material=helmet, **vis)
        new = [b]
        for limb in LIMBS:
            lb = ET.Element('body', {'name': f'worker{i}_{limb}', 'mocap': 'true', 'pos': park})
            kind, side = limb.split('_')
            if kind == 'thigh':
                g(lb, type='capsule', fromto=f'0 0 0 0 0 {-THIGH}', size='0.075', material='p_trousers', **st)
            elif kind == 'shin':
                g(lb, type='capsule', fromto=f'0 0 0 0 0 {-SHIN + 0.04}', size='0.06', material='p_trousers', **st)
            elif kind == 'foot':
                g(lb, type='box', pos='0.05 0 -0.045', size='0.13 0.055 0.045', material='p_boot', **st)
            elif kind == 'uarm':
                g(lb, type='capsule', fromto=f'0 0 0 0 0 {-UPPER}', size='0.052', material=suit, **st)
            else:
                g(lb, type='capsule', fromto=f'0 0 0 0 0 {-FORE}', size='0.045', material=suit, **st)
                g(lb, type='sphere', pos=f'0 0 {-FORE - 0.05}', size='0.048', material='p_glove', **vis)
            new.append(lb)
        for k, nb in enumerate(new):
            world.insert(idx + k, nb)


class WalkingPeople:
    """Mixin for CardboardYardRealEnv (people='walk'): spawn, walk and pose the articulated people."""

    def _people_init(self):
        import mujoco

        m = self.model
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        n = len(self._w_mocap)
        self._limb_mocap = [[m.body_mocapid[bid(f'worker{i}_{l}')] for l in LIMBS] for i in range(n)]
        bodies = {bid(f'worker{i}_{l}') for i in range(n) for l in LIMBS}
        self._worker_geoms |= {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}
        self._gait = np.zeros(n)  # phase (rad)
        self._amp = np.zeros(n)  # 0 standing .. 1 walking
        self._face = np.zeros(n)  # facing (rad), turns smoothly
        self._face_set = np.zeros(n, bool)
        self._people_dt = 0.02
        self._road_t = np.zeros(n)  # time waited at the roadside (s)
        self._leaving = np.zeros(n, bool)  # walking out of the loader's way (no roadside wait on that walk)

    def _walk(self):
        return getattr(self, 'walk_people', False) and hasattr(self, '_limb_mocap')

    def _set_state(self, state):
        """Restore a saved state (reset(options={'state': ...})): also rebuild what _spawn_people made at
        reset and the state vector does not hold, else the people's next waypoint crashes (no _route)."""
        if self._walk():
            self._face_set[:] = False  # _place_mocaps (in super) faces them again
        super()._set_state(state)
        if not self._walk():
            return
        import importlib

        nav = importlib.import_module(type(self).corridor.__module__.rsplit('.', 1)[0] + '.nav')
        a, b = self.corridor()
        self._people_grid = nav.Grid(self.obstacle_circles(), layout=self.layout, machine=self.machine)
        self._route = (np.asarray(a, float), np.asarray(b, float))
        self._road_t[:] = 0.0
        self._leaving[:] = False

    # ------------------------------------------------------------ spawn
    def _spawn_people(self, rng, g, a, b, start_fp):
        if not self._walk():
            return super()._spawn_people(rng, g, a, b, start_fp)
        L = self.layout
        self.workers[:] = 0.0
        self._people_grid, self._route = g, (np.asarray(a, float), np.asarray(b, float))
        self._face_set[:] = False
        # new episode: no roadside wait / gait left over from the last one (same seed -> same episode)
        self._road_t[:], self._leaving[:], self._gait[:], self._amp[:] = 0.0, False, 0.0, 0.0
        n = min(int(rng.choice([1, 2, 3], p=[0.15, 0.4, 0.45])), self.max_workers)
        for i in range(n):
            kind = int(rng.choice([KIND_WALK, self.KIND_CROSS, self.KIND_IDLE], p=[0.6, 0.25, 0.15]))
            for _ in range(200):
                p0 = np.array([rng.uniform(L.people_area[0], L.people_area[1]), rng.uniform(L.people_area[2], L.people_area[3])])
                if kind == self.KIND_CROSS:  # start beside the route, 3-5.5 m off it
                    t = rng.uniform(0.1, 0.9)
                    p0 = self._route[0] + t * (self._route[1] - self._route[0]) + self._route_normal() * rng.choice([-1, 1]) * rng.uniform(3.0, 5.5)
                if not self._walkable(p0) or np.min(np.linalg.norm(start_fp - p0, axis=1)) < 4.0:
                    continue
                if not self._stop_ok(p0, 4.5 if kind == self.KIND_IDLE else 3.0):
                    continue
                if kind == self.KIND_IDLE:
                    tgt = p0 + 0.01 * np.array([1.0, 0.0])
                else:
                    tgt = self._next_waypoint(rng, p0, kind)
                    if tgt is None:
                        continue
                speed = 0.0 if kind == self.KIND_IDLE else rng.uniform(1.0, 1.5)
                self.workers[i] = [*p0, *tgt, speed, 0.0, 1.0, rng.uniform(0.0, 2.0), 0.0, 1.0, 0.0, kind]
                break

    def _route_normal(self):
        a, b = self._route
        u = (b - a) / max(np.linalg.norm(b - a), 1e-3)
        return np.array([-u[1], u[0]])

    def _route_dist(self, p):
        a, b = self._route
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-6), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * (b - a))))

    def _stop_ok(self, p, route=4.5):
        """A place to stand or pause: off the pile -> hopper route and well clear of the digging lane / hopper approach
        (an idle worker in the lane, seed 4 of the people test, cost the loader 900 steps a trip)."""
        return self._route_dist(p) >= route and not self._forbidden(p, 2.0)

    def _walkable(self, p, pad=0.5):
        L = self.layout
        return (L.people_bounds[0] < p[0] < L.people_bounds[1] and L.people_bounds[2] < p[1] < L.people_bounds[3]
                and self._people_grid.clearance(*p) >= 0.6 and not self._forbidden(p, pad))

    def _segment_ok(self, p, q):
        n = max(2, int(np.linalg.norm(q - p) / 0.3))
        return all(self._walkable(p + s * (q - p)) for s in np.linspace(0, 1, n)[1:])

    def _next_waypoint(self, rng, p, kind, away=None):
        """A reachable point 4-14 m away (CROSS: to the other side of the route; away=(x, y): not towards it)."""
        L = self.layout
        for _ in range(60):
            if kind == self.KIND_CROSS:
                a, b = self._route
                nrm = self._route_normal()
                side = np.sign(np.dot(p - a, nrm)) or 1.0
                t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-6) + rng.uniform(-0.15, 0.15), 0.1, 0.9)
                q = a + t * (b - a) - side * nrm * rng.uniform(3.0, 5.5)
            else:
                ang = rng.uniform(-math.pi, math.pi)
                q = p + rng.uniform(4.0, 14.0) * np.array([math.cos(ang), math.sin(ang)])
                q = np.clip(q, [L.people_area[0], L.people_area[2]], [L.people_area[1], L.people_area[3]])
            if away is not None and np.dot(q - p, np.asarray(away) - p) > 0:
                continue
            if np.linalg.norm(q - p) > 1.0 and self._stop_ok(q, 3.0) and self._segment_ok(p, q):
                return q
        return None

    # ------------------------------------------------------------ walk
    def _update_workers(self, dt):
        if not self._walk():
            return super()._update_workers(dt)
        self._people_dt = dt
        fp = self.footprint()
        rng = self.np_random
        for i in range(len(self.workers)):
            w = self.workers[i]
            if w[9] <= 0:
                continue
            kind = int(w[11])
            if kind == self.KIND_IDLE:  # stands and watches the loader
                p = w[0:2]
                to = np.mean(fp, axis=0) - p
                w[2:4] = p + 0.01 * to / max(np.linalg.norm(to), 1e-3)
                continue
            if w[7] > 0:
                w[7] -= dt
                continue
            a, b = w[0:2].copy(), w[2:4].copy()
            length = max(np.linalg.norm(b - a), 1e-3)
            if w[5] < 0.02 and not self._leaving[i] and self._crosses_loader(a, b, fp):  # at the roadside
                close = np.min(np.linalg.norm(fp - a, axis=1)) < 4.0 + 1.1
                self._road_t[i] += 0.5
                if close or self._road_t[i] > 10.0:  # right by the loader, or it does not come past: get out of its
                    # way (waiting 3 m from it, the loader waited for them as well: seeds 110 / 126, 1000+ steps)
                    q = self._next_waypoint(rng, a, KIND_WALK, away=np.mean(fp, axis=0))
                    if q is not None:
                        w[2:4], self._leaving[i] = q, True
                    self._road_t[i] = 0.0
                    if q is not None:
                        continue
                w[7] = 0.5
                continue
            self._road_t[i] = 0.0
            s_new = float(min(w[5] + w[4] * dt / length, 1.0))
            old, new = a + (b - a) * w[5], a + (b - a) * s_new
            d_old = np.min(np.linalg.norm(fp - old, axis=1)) - 1.1
            d_new = np.min(np.linalg.norm(fp - new, axis=1)) - 1.1
            if d_new < 1.5 and d_new < d_old:  # loader right there: wait, then walk away from it
                w[10] += dt
                if w[10] > 2.5:
                    q = self._next_waypoint(rng, old, KIND_WALK, away=np.mean(fp, axis=0))
                    if q is not None:
                        w[0:2], w[2:4], w[5], self._leaving[i] = old, q, 0.0, True
                    w[10] = 0.0
                continue
            w[10] = 0.0
            w[5] = s_new
            self._gait[i] += 2 * math.pi * np.linalg.norm(new - old) / STRIDE
            if s_new >= 1.0:  # arrived: next waypoint, sometimes a pause first (turns towards it meanwhile)
                self._leaving[i] = False
                q = self._next_waypoint(rng, b, kind)
                if q is None:
                    q = a
                w[0:2], w[2:4], w[5] = b, q, 0.0
                w[7] = rng.uniform(1.0, 5.0) if rng.random() < 0.5 else 0.0

    @staticmethod
    def _crosses_loader(a, b, fp):
        """Loader within 8 m and the walk a -> b passing within 3 m of it (seed 110: a worker crossing back and forth
        in front of the stopped loader kept it waiting 1000 steps)."""
        if np.min(np.linalg.norm(fp - a, axis=1)) > 8.0:
            return False
        ts = np.linspace(0.0, 1.0, 12)
        pts = a[None] + ts[:, None] * (b - a)[None]
        return float(np.min(np.linalg.norm(pts[:, None] - fp[None], axis=2))) < 3.0 + 1.1

    def worker_velocities(self):
        if not self._walk():
            return super().worker_velocities()
        act = self.workers[:, 9] > 0
        v = np.zeros((len(self.workers), 2))
        for i, w in enumerate(self.workers):
            if w[9] <= 0 or w[4] <= 0 or w[7] > 0 or w[10] > 0 or int(w[11]) == self.KIND_IDLE:
                continue
            u = w[2:4] - w[0:2]
            v[i] = w[4] * u / max(np.linalg.norm(u), 1e-3)
        return v[act]

    # ------------------------------------------------------------ pose
    def _place_mocaps(self):
        super()._place_mocaps()  # obstacles (and the v0 placement of the root bodies, overwritten below)
        if not self._walk():
            return
        d, dt = self.data, self._people_dt
        for i, w in enumerate(self.workers):
            ids = self._limb_mocap[i]
            if w[9] <= 0:
                for k, mid in enumerate(ids):
                    d.mocap_pos[mid] = [30.0 + 2 * i, -20.0 - 0.3 * k, -5.0]
                continue
            p = self._worker_xy(i)
            u = w[2:4] - w[0:2]
            want = math.atan2(u[1], u[0])
            if not self._face_set[i]:
                self._face[i], self._face_set[i] = want, True
            err = (want - self._face[i] + math.pi) % (2 * math.pi) - math.pi
            self._face[i] += float(np.clip(err, -3.0 * dt, 3.0 * dt))  # ~170 deg/s at most
            walking = w[4] > 0 and w[7] <= 0 and w[10] <= 0 and int(w[11]) != self.KIND_IDLE
            self._amp[i] += float(np.clip((1.0 if walking else 0.0) - self._amp[i], -dt / 0.3, dt / 0.3))
            if not walking and self._amp[i] < 0.05:
                self._gait[i] = round(self._gait[i] / math.pi) * math.pi  # feet together when standing
            self._pose(i, p, self._face[i], self._gait[i], self._amp[i])

    def _pose(self, i, p, yaw, ph, amp):
        d = self.data
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        qy = _q_yaw(yaw)
        base = np.array([p[0], p[1], 0.0])
        bob = 0.025 * amp * (math.cos(2 * ph) - 1.0)
        d.mocap_pos[self._w_mocap[i]] = base + [0.0, 0.0, bob]
        d.mocap_quat[self._w_mocap[i]] = qy
        ids = dict(zip(LIMBS, self._limb_mocap[i]))

        def put(name, local, t):
            d.mocap_pos[ids[name]] = base + R @ local
            d.mocap_quat[ids[name]] = _qmul(qy, _q_pitch(t))

        for side, sy, off in (('l', 1.0, 0.0), ('r', -1.0, math.pi)):
            phi = ph + off
            hip_a = amp * 0.42 * math.sin(phi)  # forward swing of the thigh
            knee = amp * (0.08 + 0.6 * max(0.0, math.cos(phi)))  # bends while the leg swings forward
            hip = np.array([0.0, sy * HIP_Y, HIP_Z + bob])
            kn = hip + THIGH * _down(hip_a)
            an = kn + SHIN * _down(hip_a - knee)
            put(f'thigh_{side}', hip, hip_a)
            put(f'shin_{side}', kn, hip_a - knee)
            put(f'foot_{side}', an, 0.4 * (hip_a - knee))
            arm = -amp * 0.35 * math.sin(phi)  # with the opposite leg
            sh = np.array([0.0, sy * SHOULDER_Y, SHOULDER_Z + bob])
            el = sh + UPPER * _down(arm)
            put(f'uarm_{side}', sh, arm)
            put(f'farm_{side}', el, arm + 0.25 + 0.25 * amp)
