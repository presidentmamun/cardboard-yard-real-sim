"""2D navigation helpers for the CardboardYard loader (planning only, no MuJoCo).

- Grid: occupancy map of the yard (columns, walls, hopper, paper rolls,
  obstacles, standing workers) plus a distance field.
- Loader footprint as a few circles, and a kinematic rollout of the
  articulated loader, used to check turns and to stop for people.
- A* path search with a clearance cost and line-of-sight shortcutting.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import env as E

RES = 0.25
X0, X1 = -14.0, 18.0
Y0, Y1 = -6.6, 13.8
R_LOADER = 1.1  # footprint circle radius
R_INFLATE = 1.4  # clearance the path centre line keeps


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ grid
class Grid:
    def __init__(self, obstacles=(), people=(), boxes=(), layout=None, machine=None):
        """obstacles/people: iterables of (x, y, radius); boxes: (cx, cy, hx, hy); layout: env Layout (v0 if None);
        machine: env Machine (v0 if None), used by the planners that take this grid."""
        L = layout or E.LAYOUT_V0
        self.machine = machine or E.MACHINE_V0
        x0_, x1_, y0_, y1_ = getattr(L, 'nav_bounds', (X0, X1, Y0, Y1))
        self.x0, self.y0 = x0_, y0_
        self.nx = int(round((x1_ - x0_) / RES))
        self.ny = int(round((y1_ - y0_) / RES))
        xs = x0_ + (np.arange(self.nx) + 0.5) * RES
        ys = y0_ + (np.arange(self.ny) + 0.5) * RES
        self.gx, self.gy = np.meshgrid(xs, ys, indexing='ij')  # (nx, ny)
        occ = np.zeros((self.nx, self.ny), bool)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True

        def box(cx, cy, hx, hy):
            occ[(np.abs(self.gx - cx) <= hx) & (np.abs(self.gy - cy) <= hy)] = True

        def disk(cx, cy, r):
            occ[(self.gx - cx) ** 2 + (self.gy - cy) ** 2 <= r * r] = True

        for cx in L.column_x:
            for cy in L.column_y:
                # v0: 0.32 as before; scanned columns (v4) are wider, and the WA100 bucket wants more room
                h = getattr(L, 'column_half', 0.22) + 0.1 + self.machine.plan_col_margin
                box(cx, cy, h, h)
        hx, hy = L.hopper_center
        ex, ey = L.hopper_half
        box(hx, hy, ex + 0.16, ey + 0.16)  # whole hopper counts as blocked
        box(1.0, L.back_wall_y + 0.5, 30, 0.5)
        box(L.left_wall[0], L.left_wall[1], 0.3, L.left_wall[2] + 1.0)
        box(-17.0, 8.8, 2.6, 4.4)  # paper rolls
        for x, y, r in obstacles:
            disk(x, y, r)
        for x, y, r in people:
            disk(x, y, r)
        for cx, cy, hx, hy in boxes:
            box(cx, cy, hx, hy)
        self.occ = occ
        self.dist = distance_transform_edt(~occ) * RES

    def idx(self, x, y):
        i = int((x - self.x0) / RES)
        j = int((y - self.y0) / RES)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def clearance(self, x, y):
        i, j = self.idx(x, y)
        return float(self.dist[i, j]) - 0.5 * RES

    def segment_clear(self, p, q, r):
        n = max(2, int(np.hypot(*(q - p)) / 0.2))
        for t in np.linspace(0, 1, n):
            pt = p + t * (q - p)
            if self.clearance(pt[0], pt[1]) < r:
                return False
        return True

    # -------------------------------------------------------------- A*
    def astar(self, start, goal, r=R_INFLATE):
        """Shortest path keeping `r` clearance. Returns (K, 2) array or None."""
        free = self.dist > r
        s = self._nearest_free(self.idx(*start), free)
        g = self._nearest_free(self.idx(*goal), free)
        if s is None or g is None:
            return None
        pen = 1.0 + 2.0 * np.clip(2.6 - self.dist, 0, None)  # prefer clearance
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        best = {s: 0.0}
        parent = {s: None}
        pq = [(h(s), 0.0, s)]
        while pq:
            _, cost, cur = heapq.heappop(pq)
            if cur == g:
                break
            if cost > best.get(cur, 1e18):
                continue
            for di, dj, w in nbrs:
                n = (cur[0] + di, cur[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or not free[n]:
                    continue
                nc = cost + w * pen[n]
                if nc < best.get(n, 1e18):
                    best[n] = nc
                    parent[n] = cur
                    heapq.heappush(pq, (nc + h(n), nc, n))
        if g not in parent:
            return None
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = parent[c]
        cells.reverse()
        pts = [np.array(start, float)]
        pts += [np.array([self.x0 + (i + 0.5) * RES, self.y0 + (j + 0.5) * RES]) for i, j in cells]
        pts.append(np.array(goal, float))
        return self._shortcut(np.array(pts), r)

    def cost_to_go(self, goal, r=1.2):
        """Dijkstra field: path length from every cell to `goal` (inf = unreachable)."""
        free = self.dist > r
        g = self._nearest_free(self.idx(*goal), free)
        field = np.full((self.nx, self.ny), np.inf)
        if g is None:
            return field
        pen = 1.0 + 2.0 * np.clip(2.6 - self.dist, 0, None)
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        field[g] = 0.0
        pq = [(0.0, g)]
        while pq:
            c, cur = heapq.heappop(pq)
            if c > field[cur]:
                continue
            for di, dj, w in nbrs:
                n = (cur[0] + di, cur[1] + dj)
                if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny and free[n]:
                    nc = c + w * RES * pen[n]
                    if nc < field[n]:
                        field[n] = nc
                        heapq.heappush(pq, (nc, n))
        return field

    def field_at(self, field, x, y):
        i, j = self.idx(x, y)
        v = field[i, j]
        if np.isfinite(v):
            return float(v)
        # inside an obstacle margin: use the best neighbour plus a penalty
        sub = field[max(i - 6, 0) : i + 7, max(j - 6, 0) : j + 7]
        fin = sub[np.isfinite(sub)]
        return float(fin.min() + 3.0) if fin.size else 1e6

    def _nearest_free(self, ij, free, max_r=16):
        if free[ij]:
            return ij
        i0, j0 = ij
        for rad in range(1, max_r):
            best = None
            for i in range(i0 - rad, i0 + rad + 1):
                for j in (j0 - rad, j0 + rad) if abs(i - i0) != rad else range(j0 - rad, j0 + rad + 1):
                    if 0 <= i < self.nx and 0 <= j < self.ny and free[i, j]:
                        d = (i - i0) ** 2 + (j - j0) ** 2
                        if best is None or d < best[0]:
                            best = (d, (i, j))
            if best:
                return best[1]
        return None

    def _shortcut(self, pts, r):
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_clear(pts[i], pts[j], r - 0.15):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)


# ------------------------------------------------------------------ loader
def footprint(x, y, yaw, gamma, boom, M=None):
    """Circle centres (K, 2) covering the loader; bucket only when low. M: Machine (default v0)."""
    M = M or E.MACHINE_V0
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(a, 0.0) for a in M.footprint_rear]
    rs = getattr(M, 'rear_side', 0.0)
    if rs:  # wide rear axle: tyre corners left / right of the centre-line circles
        pts += [(M.rear_axle, rs), (M.rear_axle, -rs)]
    out = [(x + c * a - s * b, y + s * a + c * b) for a, b in pts]
    cf, sf = math.cos(yaw + gamma), math.sin(yaw + gamma)
    front = [(a, 0.0) for a in M.footprint_front]
    if boom > M.raised_boom:  # raised bucket passes over people/obstacles
        a = M.boom_pivot[0] + M.boom_len * math.cos(boom) + M.bucket_reach
        side = getattr(M, 'bucket_side', 0.0)
        front += [(a, side), (a, -side)] if side else [(a, 0.0)]
    out += [(x + cf * a - sf * b, y + sf * a + cf * b) for a, b in front]
    return np.array(out)


def step_kin(state, v, steer_cmd, dt=0.1, M=None):
    """One control step of the articulated-loader kinematics. M: Machine (default v0)."""
    M = M or E.MACHINE_V0
    x, y, yaw, gamma = state
    gamma += float(np.clip(steer_cmd - gamma, -0.15, 0.15))
    om = v * math.sin(gamma) / (M.l_front * math.cos(gamma) + M.l_rear)
    c, s = math.cos(yaw), math.sin(yaw)
    vy = om * M.l_rear
    return (x + (v * c - vy * s) * dt, y + (v * s + vy * c) * dt, yaw + om * dt, gamma)


def rollout_hits(grid, state, boom, v, steer_cmd, steps, margin=0.15):
    st = state
    for _ in range(steps):
        st = step_kin(st, v, steer_cmd, M=grid.machine)
        for px, py in footprint(st[0], st[1], st[2], st[3], boom, M=grid.machine):
            if grid.clearance(px, py) < R_LOADER + margin:
                return True
    return False


def plan_turn(grid, state, boom, target_yaw, tol=0.3, speed=1.5):
    """Find a collision-free manoeuvre that brings the heading near target_yaw.

    Returns a list of segments (v, steer, kind, amount) where kind is
    'dist' (straight for `amount` metres) or 'yaw' (arc until |yaw err| < tol),
    or None.
    """
    best = None
    for pre in (0.0, 1.5, 3.0):
        for pre_v in (-speed, speed) if pre > 0 else (0.0,):
            for v in (-speed, speed):
                for steer in (-grid.machine.max_steer, grid.machine.max_steer):
                    st = state
                    ok = True
                    n = 0
                    for _ in range(int(pre / abs(pre_v) / 0.1) if pre > 0 else 0):
                        st = step_kin(st, pre_v, 0.0, M=grid.machine)
                        n += 1
                        if _hit(grid, st, boom):
                            ok = False
                            break
                    if not ok:
                        continue
                    reached = False
                    for _ in range(90):
                        if abs(_wrap(st[2] - target_yaw)) < tol:
                            reached = True
                            break
                        st = step_kin(st, v, steer, M=grid.machine)
                        n += 1
                        if _hit(grid, st, boom):
                            ok = False
                            break
                    if ok and reached:
                        clear = min(
                            grid.clearance(px, py) for px, py in footprint(*st, boom, M=grid.machine)
                        )
                        score = n - 4.0 * min(clear, 2.0)
                        if best is None or score < best[0]:
                            segs = []
                            if pre > 0:
                                segs.append((pre_v, 0.0, 'dist', pre))
                            segs.append((v, steer, 'yaw', target_yaw))
                            best = (score, segs)
    return None if best is None else best[1]


def hybrid_astar(grid, start, goal, boom, goal_tol=(0.6, 1.5, 0.3), max_expand=12000, rev_cost=1.8, margin=0.1):
    """Hybrid A* for the articulated loader (forward and reverse).

    start: (x, y, yaw, gamma); goal: (x, y, yaw). Each expansion drives one
    1.6 m arc (5 steering settings x forward/reverse) with the loader
    kinematics and checks the footprint. Heuristic: obstacle-aware 2D
    distance to the goal. The goal region is a box around the goal pose:
    goal_tol = (lateral m, along-heading m, yaw rad). Returns a list of
    (x, y, yaw, dir, gamma) waypoints or None (gamma = planned articulation).
    """
    gx, gy, gyaw = goal
    field = grid.cost_to_go((gx, gy), r=1.15)
    ms = grid.machine.max_steer * grid.machine.plan_steer_frac  # headroom for the tracking controller
    steers = (-ms, -0.5 * ms, 0.0, 0.5 * ms, ms)
    sub, ds = 4, 0.4  # 4 sub-steps of 0.4 m = 1.6 m per arc
    nyaw = 36
    gu = np.array([math.cos(gyaw), math.sin(gyaw)])

    def key(s):
        return (int(round(s[0] / 0.6)), int(round(s[1] / 0.6)), int(round(s[2] / (2 * math.pi / nyaw))) % nyaw)

    def h(s):
        return grid.field_at(field, s[0], s[1]) + 0.8 * abs(_wrap(s[2] - gyaw)) * (1.0 if np.hypot(s[0] - gx, s[1] - gy) < 4 else 0.3)

    s0 = (start[0], start[1], start[2], start[3])
    start_hit = _hit(grid, s0, boom, margin=margin)  # already too close: let the first arcs escape
    # machine.plan_strict (WA100): every 0.4 m sub-step is checked (v0: every second one), and from a start that is
    # already too close only arcs that never get closer and end >= 0.1 m further out count as escapes (v0: the first arc was not
    # checked at all; seed 27 of the people test planned that arc into the column it was touching, again and again)
    strict = getattr(grid.machine, 'plan_strict', False)
    room0 = _room(grid, s0, boom) if strict else 0.0
    nodes = {0: (s0, None, 0, 0.0)}  # id -> (state, parent, dir, steer)
    pq = [(h(s0), 0.0, 0)]
    seen = {}
    nid = 1
    expanded = 0
    while pq and expanded < max_expand:
        f, g, cur = heapq.heappop(pq)
        st, parent, pdir, psteer = nodes[cur]
        k = key(st)
        if k in seen and seen[k] <= g:
            continue
        seen[k] = g
        expanded += 1
        rel = np.array([st[0] - gx, st[1] - gy])
        along, lat = rel @ gu, rel @ np.array([-gu[1], gu[0]])
        if abs(lat) < goal_tol[0] and abs(along) < goal_tol[1] and abs(_wrap(st[2] - gyaw)) < goal_tol[2]:
            out = []
            while cur is not None:
                s, parent, d, _ = nodes[cur]
                out.append((s[0], s[1], s[2], d, s[3]))
                cur = parent
            return out[::-1]
        for d in (1, -1):
            for steer in steers:
                s = (st[0], st[1], st[2], steer)
                ok = True
                for i in range(sub):
                    # one sub-step = 0.4 m; the articulation swings at a finite rate
                    # (step_kin limits it), so the plan only uses curves the loader can drive
                    s = step_kin(s, d * ds / 0.1, steer, M=grid.machine)
                    if strict:
                        if cur == 0 and start_hit:  # escape: never closer, and clearly out by the arc's end
                            r = _room(grid, s, boom)
                            if r < room0 - 0.02 or (i == sub - 1 and r < min(room0 + 0.1, 0.1)):
                                ok = False
                                break
                        elif _hit(grid, s, boom, margin=margin):
                            ok = False
                            break
                    elif i % 2 == 1 and not (cur == 0 and start_hit) and _hit(grid, s, boom, margin=margin):
                        ok = False
                        break
                if not ok:
                    continue
                cost = 1.6 * (1.0 if d > 0 else rev_cost) + 0.3 * abs(steer) + 0.2 * abs(steer - psteer)
                if pdir and d != pdir:
                    cost += 3.0  # direction change (cusp)
                ng = g + cost
                nk = key(s)
                if nk in seen and seen[nk] <= ng:
                    continue
                nodes[nid] = (s, cur, d, steer)
                heapq.heappush(pq, (ng + h(s), ng, nid))
                nid += 1
    return None


def plan_kturn(grid, state, boom, target_yaw, tol=0.4, speed=1.2, max_segs=9):
    """Multi-point turn (K-turn) towards target_yaw in tight spaces.

    Alternates forward and reverse arcs at full lock, each run until the
    heading is reached or the next step would come too close to something.
    Returns segments (v, steer, 'steps', n) or None.
    """
    best = None
    for first_v in (speed, -speed):
        st = state
        segs = []
        v = first_v
        ok = False
        for _ in range(max_segs):
            d = _wrap(target_yaw - st[2])
            if abs(d) < tol:
                ok = True
                break
            # steering sign that turns the heading towards the target
            s = math.copysign(grid.machine.max_steer, d) * (1 if v > 0 else -1)
            n = 0
            while n < 45:
                nxt = step_kin(st, v, s, M=grid.machine)
                if _hit(grid, nxt, boom, margin=0.15):
                    break
                st = nxt
                n += 1
                if abs(_wrap(target_yaw - st[2])) < tol:
                    break
            if n >= 3:
                segs.append((v, s, 'steps', n))
            elif segs and segs[-1][0] == -v:
                pass  # no room this way either: try flipping again
            v = -v
            if n < 3 and len(segs) == 0 and v == first_v:
                break
        if ok and segs:
            total = sum(sg[3] for sg in segs)
            if best is None or total < best[0]:
                best = (total, segs)
    return None if best is None else best[1]


PRIMITIVES = [(v, s) for v in (0.6, -0.35) for s in (-1.0, -0.5, 0.0, 0.5, 1.0)]


def _arc(grid, st, boom, thr, steer, steps):
    for k in range(steps):
        st = step_kin(st, thr * grid.machine.max_speed, steer * grid.machine.max_steer, M=grid.machine)
        if k % 3 == 2 and _hit(grid, st, boom, margin=0.05):
            return None
    return st


def choose_primitive(grid, field, state, boom, prev_steer, goal_yaw=None, steps=10):
    """Pick the (throttle, steer) arc that best lowers the cost-to-go.

    Two arcs are chained (each `steps` control steps, both forward and reverse
    with five steering settings) and rolled out with the loader kinematics;
    sequences that bring the footprint near an obstacle are dropped. The first
    arc of the best sequence is returned. Reversing and steering changes cost
    a little, so the loader prefers smooth forward driving.
    """
    best, best_j = None, np.inf
    for p1 in PRIMITIVES:
        s1 = _arc(grid, state, boom, p1[0], p1[1], steps)
        if s1 is None:
            continue
        base = 1.0 * (p1[0] < 0) + 0.3 * abs(p1[1] - prev_steer)
        j1 = _score(grid, field, s1, goal_yaw) + base
        if j1 < best_j:
            best, best_j = p1, j1
        for p2 in PRIMITIVES:
            s2 = _arc(grid, s1, boom, p2[0], p2[1], steps)
            if s2 is None:
                continue
            j = _score(grid, field, s2, goal_yaw) + base + 0.7 * (p2[0] < 0) + 0.15 * abs(p2[1] - p1[1])
            if j < best_j:
                best, best_j = p1, j
    return best

def _score(grid, field, st, goal_yaw):
    c = grid.field_at(field, st[0], st[1])
    if goal_yaw is not None and c < 5.0:  # near the goal also care about heading
        c += 1.6 * abs(_wrap(st[2] - goal_yaw)) * (1.0 - c / 5.0)
    return c


def _room(grid, st, boom):
    """Smallest clearance of the footprint beyond the footprint radius (m, negative: overlapping)."""
    return min(grid.clearance(px, py) for px, py in footprint(st[0], st[1], st[2], st[3], boom, M=grid.machine)) - R_LOADER


def _hit(grid, st, boom, margin=0.1):
    for px, py in footprint(st[0], st[1], st[2], st[3], boom, M=grid.machine):
        if grid.clearance(px, py) < R_LOADER + margin:
            return True
    return False
