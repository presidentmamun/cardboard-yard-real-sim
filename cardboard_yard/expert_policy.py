"""Scripted expert for CardboardYard: scoop at the pile, dump in the hopper.

One state machine per env:

    go_pile -> align_pile -> lower -> push -> curl -> back_out
    -> go_hopper -> approach_hopper -> dump -> leave_hopper -> go_pile ...

go_* phases plan an A* path around columns, obstacles and people standing
still, and first turn the loader with a collision-checked manoeuvre when it
faces the wrong way. A safety layer stops the loader whenever its next 1.2 s
would bring it close to a person or an obstacle (this also guards noisy and
random actions).

Data diversity (per episode):
    mode 'expert' (p = 1 - p_weak): small correlated noise, rare random bursts
    mode 'weak'   (p = p_weak):     strong noise, frequent random bursts that
                                    the expert then recovers from
"""

from __future__ import annotations

import collections
import math

import numpy as np

class BasePolicy:
    """Minimal stand-in for stable_worldmodel.policy.BasePolicy (standalone copy)."""

    def __init__(self, **kwargs):
        self.env = None
        self.type = 'base'
        for k, v in kwargs.items():
            setattr(self, k, v)

from . import env as E
from . import nav


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


PUSH_LIMIT_DY = 4.15  # v0 (E.MACHINE_V0.push_limit_dy): articulation this far in front of the back wall: the lip reaches the wall
# minimum turning radius of the articulation point at full steering lock
R_TURN = (E.L_FRONT * math.cos(E.MAX_STEER) + E.L_REAR) / math.sin(E.MAX_STEER)

# v0 arm poses (the expert uses env.machine.poses): (boom joint, bucket world pitch). boom + = lower, pitch + = lip down
POSE_CARRY = (0.2, -0.95)
POSE_DIG = (0.53, 0.04)
POSE_CURL = (0.36, -1.05)
POSE_HIGH = (-0.93, -0.6)
POSE_DUMP = (-0.93, 1.3)


class _Controller:
    def __init__(self, rng, p_weak, noise_scale):
        self.rng = rng
        self.p_weak = p_weak
        self.noise_scale = noise_scale
        self.reset()

    # ---------------------------------------------------------- episode
    def reset(self):
        rng = self.rng
        self.weak = rng.random() < self.p_weak
        if self.weak:
            self.sigma = rng.uniform(0.15, 0.35) * self.noise_scale
            self.p_burst, self.burst_len = 0.015, (10, 30)
        else:
            self.sigma = rng.uniform(0.0, 0.08) * self.noise_scale
            self.p_burst, self.burst_len = 0.002, (5, 15)
        if self.noise_scale <= 0:
            self.p_burst = 0.0
        self.ou = np.zeros(4)
        self.burst = 0
        self.burst_action = np.zeros(4)
        self.blocked = 0
        self.touch_t = 0
        self.static_n = 0
        self.static_recent = 0
        self.stats = collections.Counter()  # diagnostics: safety stops / slow steps / plans (hit_report)
        self.backoff, self.backoff_thr, self.backoff_steer = 0, -0.25, 0.0
        self.cycle = 0
        self.arm_prev = np.zeros(2)
        self.drive_prev = np.zeros(2)
        self._set('go_pile')

    def _set(self, phase):
        self.phase = phase
        self.t_phase = 0
        self.crowd_t = 0
        self.goal_retries = 0
        self.shook = 0
        self.stall = 0
        self.path = None
        self.turn = None

    # ---------------------------------------------------------- arm
    LOADED = ('back_out', 'go_hopper', 'approach_hopper')
    THR_SLEW = 0.025  # per step -> 0.75 m/s^2 with a full bucket
    STEER_SLEW = 0.08  # per step -> ~0.5 rad/s articulation swing with a full bucket
    LOADED_REVERSE = 0.72  # reverse slower with a full bucket

    # behaviour tunables (CardboardYardReal's expert v1 changes them; these values are the v0 expert)
    PURSUIT_GAIN = 1.8  # steering per rad of heading error
    CROWD = None  # (boom raise, bucket rack-back) in rad for lift-and-crowd when the pile stops the push early
    CROWD_MIN = 25  # ... and fewer pieces than this are in the bucket
    ESCAPE_REVERSE = False  # blocked for long: back straight away instead of the free-direction escape
    CURL_WALL_STOP = False  # no gas while curling once at the push limit
    REPLAN_DEV = 1.8  # m off the plan before re-planning
    PLAN_SKIP_CLOSE = 0.0  # > 0: the plan leaves out walking people this close to the loader (v0: keeps them)
    PLAN_PEOPLE_AHEAD = 0.0  # > 0 (s): the plan also blocks where walking people will be (v0: where they are)
    PILE_ROW_GAP = 0.0  # > 0: a column row near the go_pile goal moves the goal this far south of the row (align_pile then
    # drives straight through the gap); 0: goal at approach_y - 2.5 (v0)
    LANE_COL_CLEAR = 0.0  # > 0: keep the lane this far (bucket edge) from the columns of a row the loader passes
    BACKOUT_ROW_GAP = 0.0  # > 0: back out (straight down the lane) until the articulation is this far south of a column
    # row near the pile, so the turn to the hopper happens in the open; 0: stop backing at approach_y + 0.3 (v0)
    HOPPER_ROW_FENCE = False  # True: go_hopper plans made south of the pile-side column row block that row (the
    # loaded loader turns in the open towards the hopper instead of driving back north through the row to turn)
    CRUISE_V = None  # forward throttle on open stretches (no person within CRUISE_CLEAR, far from the goal); None: 0.5 (v0)
    CRUISE_CLEAR = 10.0
    BACKOUT_V = None  # reverse throttle once the bucket is off the pile (same people rule); None: 0.3 (v0)
    BACKOUT_TURN = False  # back out of the pile turning towards the hopper (v0: straight back along the lane)
    GOAL_YAW_TOL = 0.0  # > 0: at the goal (< 2 m) accept this heading error, or any after 3 re-plans (v0: 0.45 only)
    STATIC_SAFE = False  # stop + re-plan when the next 1.2 s would bring the bucket to a column / wall (v0: no)
    STATIC_SAFE_PHASES = ('go_pile', 'go_hopper')
    CONTACT_ESCAPE = False  # driving between pile and hopper: touching a static geom 5 steps -> back off, re-plan
    PLAN_LOW_BOOM = False  # plan with the bucket at carry height or lower (v0: the boom of the moment)
    SWING_CHECK = False  # check the bucket's sweep before swinging the articulation standing still (v0: no)
    PEOPLE_AHEAD = 0.0  # > 0 (s): the safety stop predicts the walking people that far (v0: where they are now)
    PLAN_NEAR_PEOPLE = 0.0  # > 0: plan around walking people within this distance too (v0: only people standing
    # still; all walking people sent the loader on long detours around people far away who would have moved)
    BLOCK_KEEP = False  # keep counting 'blocked' over steps without gas (else re-planning resets it: no escape)
    SLEW_ALWAYS = False  # rate-limit gas and articulation in every phase, not only with a full bucket
    THR_SLEW_EMPTY = None  # gas rate limit with an empty bucket when SLEW_ALWAYS (None: THR_SLEW)
    PUSH_STOP = 0.0  # m before PUSH_LIMIT_Y where pushing stops
    RAISE_FROM = 4.5  # m before HOPPER_STOP_X where the boom goes up
    STOP_EARLY = 0.0  # m before HOPPER_STOP_X where dumping starts
    DUMP_HOLD, DUMP_ROCK_PERIOD, DUMP_ROCK = 15, 12, 0.25  # steps tipped before rocking, rock half-period, rock size
    DUMP_LEAVE_MIN = 5  # steps tipped before leaving when (almost) empty
    REV_COST = 1.8  # planner cost factor for reversing (higher: turn around instead of long reverses)
    PLAN_EXPAND = 12000  # planner node budget

    ARM_GAIN = 2.0  # 1/s: slow down when close to the target pose (ease out)
    ARM_SLEW = 0.2  # max change of an arm action per step (ease in)

    def _arm(self, env, pose):
        boom, pitch = pose
        err = np.array([env._boom_target - boom, env._bucket_target - (pitch - boom)])
        rate = np.array([env.machine.boom_rate, env.machine.bucket_rate])
        a = np.clip(err * self.ARM_GAIN / rate, -1, 1)
        a = np.clip(a, self.arm_prev - self.ARM_SLEW, self.arm_prev + self.ARM_SLEW)
        self.arm_prev = a
        return float(a[0]), float(a[1])

    @staticmethod
    def _hstop(env):
        """Articulation x when dumping (layout, plus the machine's extra distance)."""
        return env.layout.hopper_stop_x + env.machine.dump_stop_dx

    @staticmethod
    def _arm_done(env, pose, tol=0.08):
        boom, pitch = pose
        return abs(env.joint('boom') - boom) < tol and abs(env.joint('boom') + env.joint('bucket') - pitch) < tol

    # ---------------------------------------------------------- driving
    def _pursue(self, env, target, speed):
        x, y, yaw = env.loader_pose()
        gamma = env.joint('steer')
        if speed >= 0:
            err = _wrap(math.atan2(target[1] - y, target[0] - x) - (yaw + 0.5 * gamma))
            steer = self.PURSUIT_GAIN * err / env.machine.max_steer
        else:
            err = _wrap(math.atan2(target[1] - y, target[0] - x) - (yaw + math.pi))
            steer = -self.PURSUIT_GAIN * err / env.machine.max_steer
        slow = max(0.5, 1.0 - abs(err) / 1.5)
        return speed * slow, float(np.clip(steer, -1, 1))

    def _follow_line(self, env, point, heading, speed, lookahead=2.6):
        x, y, _ = env.loader_pose()
        u = np.array([math.cos(heading), math.sin(heading)])
        along = (np.array([x, y]) - point) @ u
        tgt = point + (along + (lookahead if speed >= 0 else -lookahead)) * u
        return self._pursue(env, tgt, speed)

    def _start_turn(self, env, target_yaw, tol):
        x, y, yaw = env.loader_pose()
        st = (x, y, yaw, env.joint('steer'))
        grid = env.nav_grid((x, y, self.PLAN_NEAR_PEOPLE) if self.PLAN_NEAR_PEOPLE else None, ahead=self.PLAN_PEOPLE_AHEAD,
                               skip_close=self.PLAN_SKIP_CLOSE)
        self.turn = nav.plan_turn(grid, st, env.joint('boom'), target_yaw, tol=tol) or []
        self.turn_i, self.turn_t, self.turn_tol = 0, 0, tol
        self.turn_start = np.array([x, y])

    def _run_turn(self, env):
        """Execute the planned manoeuvre. Returns (thr, steer, finished)."""
        x, y, yaw = env.loader_pose()
        v, steer, kind, amount = self.turn[self.turn_i]
        if kind == 'dist':
            done = np.hypot(x - self.turn_start[0], y - self.turn_start[1]) >= amount
        elif kind == 'steps':
            done = self.turn_t >= amount or abs(_wrap(yaw - self.turn_target)) < self.turn_tol
            if abs(_wrap(yaw - self.turn_target)) < self.turn_tol:
                self.turn_i = len(self.turn) - 1  # target heading reached: finish
        else:
            done = abs(_wrap(yaw - amount)) < self.turn_tol
        self.turn_t += 1
        if done or self.turn_t > 120:
            self.turn_i += 1
            self.turn_t = 0
            self.turn_start = np.array([x, y])
            if self.turn_i >= len(self.turn):
                self.turn = []
                return 0.0, steer / env.machine.max_steer, True
            return 0.0, steer / env.machine.max_steer, False
        return v / env.machine.max_speed, steer / env.machine.max_steer, False

    def _go(self, env, goal, goal_yaw):
        """Drive to `goal` facing `goal_yaw` along a Hybrid A* plan.

        The plan is split into forward / reverse segments; each is tracked with
        pure pursuit (aiming the rear when reversing). Re-plans when the loader
        drifts off the plan, gets stuck, or a random burst moved it.
        Returns (thr, steer, done).
        """
        x, y, yaw = env.loader_pose()
        p = np.array([x, y])
        if self.backoff > 0:
            self.backoff -= 1
            return self.backoff_thr, self.backoff_steer, False
        if self.path is None:
            if getattr(self, 'replan_wait', 0) > 0:
                self.replan_wait -= 1
                return 0.0, 0.0, False
            st = (x, y, yaw, env.joint('steer'))
            fence = self._row_fence(env, y)
            grid = env.nav_grid((x, y, self.PLAN_NEAR_PEOPLE) if self.PLAN_NEAR_PEOPLE else None, ahead=self.PLAN_PEOPLE_AHEAD,
                               skip_close=self.PLAN_SKIP_CLOSE, boxes=fence)
            boom = env.joint('boom')
            if self.PLAN_LOW_BOOM:  # the arm comes down to carry on the way: a plan made with the bucket still raised
                # (after a dump) passed it over the hopper wall, then the lowered bucket stuck on it (seed 45)
                boom = max(boom, env.machine.poses['carry'][0])
            self.stats['plans'] += 1
            plan = nav.hybrid_astar(grid, st, (goal[0], goal[1], goal_yaw), boom,
                                    max_expand=self.PLAN_EXPAND, rev_cost=self.REV_COST,
                                    margin=0.4 if self.static_recent > 0 else 0.1)
            if plan is None and fence:  # boxed in south of the row (obstacles behind): any way will do
                grid = env.nav_grid((x, y, self.PLAN_NEAR_PEOPLE) if self.PLAN_NEAR_PEOPLE else None,
                                    ahead=self.PLAN_PEOPLE_AHEAD, skip_close=self.PLAN_SKIP_CLOSE)
                plan = nav.hybrid_astar(grid, st, (goal[0], goal[1], goal_yaw), boom,
                                        max_expand=self.PLAN_EXPAND, rev_cost=self.REV_COST,
                                        margin=0.4 if self.static_recent > 0 else 0.1)
            if plan is None:
                self.fails = getattr(self, 'fails', 0) + 1
                # move towards open space and try again (people move, too)
                self.backoff, (self.backoff_thr, self.backoff_steer) = 10, self._escape(env, 0.3)
                return 0.0, 0.0, False
            self.fails = 0
            segs, cur = [], [plan[0]]
            for wp in plan[1:]:
                if len(cur) > 1 and wp[3] != cur[-1][3]:
                    segs.append(cur)
                    cur = [cur[-1]]
                cur.append(wp)
            segs.append(cur)
            self.path = [
                (int(np.sign(s[-1][3]) or 1), np.array([[w[0], w[1], w[4]] for w in s])) for s in segs
            ]
            self.seg, self.stuck_t, self.seg_t = 0, 0, 0
        d, pts3 = self.path[self.seg]
        pts, gam = pts3[:, :2], pts3[:, 2]
        end = pts[-1]
        last = self.seg == len(self.path) - 1
        # segment finished?
        seg_dir = pts[-1] - pts[-2] if len(pts) > 1 else np.array([math.cos(yaw), math.sin(yaw)])
        # 'passed the end' only counts near it (a U-turn can start 'ahead' of its end)
        passed = (p - end) @ seg_dir > 0 and np.linalg.norm(p - end) < 2.0
        self.seg_t += 1
        if np.linalg.norm(p - end) < 0.6 or passed or self.seg_t > 150:
            if last:
                if abs(_wrap(goal_yaw - yaw)) < 0.45 and np.linalg.norm(p - goal) < 2.0:
                    return 0.0, 0.0, True
                if self.GOAL_YAW_TOL and np.linalg.norm(p - goal) < 2.0:
                    self.goal_retries = getattr(self, 'goal_retries', 0) + 1
                    if abs(_wrap(goal_yaw - yaw)) < self.GOAL_YAW_TOL or self.goal_retries > 3:
                        self.goal_retries = 0  # close enough: the next phase lines up (seed 1048: 500 steps of
                        return 0.0, 0.0, True  # tiny forward / reverse plans at the hopper goal, 35 deg off)
                self.path = None  # ended badly: plan again from here
                return 0.0, 0.0, False
            self.seg += 1
            self.seg_t = 0
            return 0.0, 0.0, False
        # drifted off the plan?
        dev = min(np.linalg.norm(pts - p, axis=1))
        if dev > self.REPLAN_DEV:
            self.path = None
            return 0.0, 0.0, False
        # touching a column / wall / the hopper on the way (stuck detection misses it: the loader keeps creeping)
        if self.CONTACT_ESCAPE:
            self.touch_t = self.touch_t + 1 if env._contacts()[1] else 0
            if self.touch_t > 5:
                self.touch_t = 0
                self.backoff, self.backoff_thr = 12, -0.3 * d  # straight back the way it came
                self.backoff_steer = env.joint('steer') / env.machine.max_steer
                self.path, self.replan_wait = None, 0
                return 0.0, 0.0, False
        # stuck (pushing into something)?
        speed = math.hypot(*env._proprio()[4:6])
        # (waiting for a person does not count as stuck)
        self.stuck_t = self.stuck_t + 1 if (speed < 0.08 and self.blocked == 0) else 0
        if self.stuck_t > 15:
            self.backoff, (self.backoff_thr, self.backoff_steer) = 10, self._escape(env, 0.3)
            self.path, self.replan_wait = None, 0
            return 0.0, 0.0, False
        ext = np.vstack([pts, end + 3.0 * seg_dir / max(np.linalg.norm(seg_dir), 1e-6)])
        tgt = self._point_along(ext, p, 1.6 if d > 0 else 1.4)
        near_end = np.linalg.norm(p - end) < 2.5
        if d > 0:
            v = 0.35 if near_end else (0.5 if not last or np.linalg.norm(p - goal) > 6 else 0.4)
            if (self.CRUISE_V and not near_end and np.linalg.norm(p - goal) > 6 and np.linalg.norm(p - end) > 4.0
                    and self._people_clear(env)):
                v = self.CRUISE_V  # open stretch, nobody near (trips took ~1000 steps: two per 2000-step episode at best)
        else:
            v = -(0.25 if near_end else 0.35)
        if d < 0 and self.phase in self.LOADED:
            v *= self.LOADED_REVERSE
        thr, fb = self._pursue(env, tgt, v)
        # feedforward: the articulation planned for the arc we are on
        # (waypoints are 1.6 m arc ends; gamma[k] is used while driving into k)
        i = int(np.argmin(np.linalg.norm(pts - p, axis=1)))
        ahead = i + 1 < len(pts) and (p - pts[i]) @ (pts[i + 1] - pts[i]) > 0
        j = min(i + 1, len(gam) - 1) if ahead else i
        ff = gam[j] / env.machine.max_steer
        steer = float(np.clip(0.6 * ff + 0.4 * fb, -1, 1))
        # like a driver: swing the articulation first, then give gas
        lag = abs(steer * env.machine.max_steer - env.joint('steer'))
        if self.SWING_CHECK and lag > 0.08:
            swing = self._swing_room(env, steer)
            if swing is not None:  # swinging here (almost standing) takes the bucket into a column / wall
                # swing on the move instead, as the plan assumed (0.15 rad per 0.4 m driven): small steps of the
                # articulation while driving; the static safety stop guards the actual sweep. (A rollout check here,
                # with step_kin's per-step swing rate, rejected the plan's own first metre: seed 103 re-planned every
                # step for 10 steps, then backed off, again and again)
                g = env.joint('steer')
                ms = env.machine.max_steer
                return d * max(abs(thr), 0.25), float(np.clip(g + np.clip(steer * ms - g, -0.05, 0.05), -ms, ms)) / ms, False
        thr *= float(np.clip(1.0 - lag / 0.5, 0.15, 1.0))
        return thr, steer, False

    def _solid_grid(self, env):
        """Walls, columns, the hopper and the obstacles only: the nav grid also blocks the pile (so plans keep off it),
        and the static stop then held a loaded loader at the pile it had just backed out of (seed 101: 1000 steps)."""
        key = (id(env), env._t == 0, tuple(map(tuple, env.obstacle_circles())))
        if getattr(self, '_solid_key', None) != key:
            self._solid_key = key
            self._solid = nav.Grid(env.obstacle_circles(), layout=env.layout, machine=env.machine)
        return self._solid

    def _static_block(self, env, thr, steer):
        """True if the next 1.2 s at this command make the footprint overlap a static geom (and closer than now). The plan was clear, but tracking it with the articulation off the plan swung the 2.7 m front frame's
        bucket up to a metre sideways (seed 103 of the fresh-seed people test: bucket into a column)."""
        x, y, yaw = env.loader_pose()
        grid, boom, M = self._solid_grid(env), env.joint('boom'), env.machine
        st = (x, y, yaw, env.joint('steer'))
        r0 = nav._room(grid, st, boom)
        for _ in range(12):
            st = nav.step_kin(st, thr * M.max_speed, steer * M.max_steer, M=M)
            r = nav._room(grid, st, boom)
            if r < 0.03 and r < r0 - 0.02:  # (0.05 also stopped the planner's own 0.1 m plans; 0.0 let the
                return True  # bucket reach columns in seeds 110 / 116)
        return False

    def _swing_room(self, env, steer):
        """None if the articulation can swing to `steer` in place; else the furthest safe command on the way. The
        plan lets the articulation swing while driving (0.15 rad per 0.4 m); swung standing still at a cusp, the
        bucket swept sideways into a column (seed 23)."""
        x, y, yaw = env.loader_pose()
        grid, boom, ms = self._solid_grid(env), env.joint('boom'), env.machine.max_steer
        g0, g1 = env.joint('steer'), steer * ms

        def room(g):
            fp = nav.footprint(x, y, yaw, g, boom, M=env.machine)
            return min(grid.clearance(px, py) for px, py in fp) - nav.R_LOADER

        r0, safe = room(g0), g0
        for k in range(1, 7):
            g = g0 + (g1 - g0) * k / 6
            r = room(g)
            if r < 0.05 and r < r0 - 0.01:
                return safe / ms
            safe = g
        return None
    @staticmethod
    def _point_along(path, p, ahead):
        """Point `ahead` metres past the projection of p onto the polyline."""
        if len(path) < 2:
            return path[-1]
        best = (1e18, 0, 0.0)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
            d = np.linalg.norm(a + t * ab - p)
            if d < best[0]:
                best = (d, i, t)
        _, i, t = best
        pos = path[i] + t * (path[i + 1] - path[i])
        remain = ahead
        for j in range(i + 1, len(path)):
            seg = np.linalg.norm(path[j] - pos)
            if seg >= remain:
                return pos + (path[j] - pos) * (remain / max(seg, 1e-9))
            remain -= seg
            pos = path[j]
        return path[-1]

    def _pick_lane(self, env):
        """x of the lane with the most cardboard left near the wall."""
        p = env.pieces_pos()
        free = p[(p[:, 1] > 7.0) & (p[:, 2] < 2.0)]
        cx, hx = env.pile_center[0], env.pile_half[0]
        if len(free) == 0:
            return float(cx)
        cands = np.linspace(cx - hx + 0.6, cx + hx - 0.6, 9)
        counts = [np.sum(np.abs(free[:, 0] - c) < 1.0) for c in cands]
        best = cands[int(np.argmax(counts))]
        return float(best + self.rng.uniform(-0.2, 0.2))

    def _pile_goal_y(self, env):
        """y of the go_pile goal. With PILE_ROW_GAP: south of a column row the loader would otherwise stand in (WA100
        layout: row at y 2.7, goal 2.1; seeds 110 / 1048 turned 500-900 steps between the columns trying to line up)."""
        L = env.layout
        gy = L.approach_y - 2.5
        if self.PILE_ROW_GAP:
            rows = [cy for cy in L.column_y if gy - self.PILE_ROW_GAP < cy < L.approach_y]
            if rows:
                gy = min(rows) - self.PILE_ROW_GAP
        return gy

    def _backout_end_y(self, env):
        """Articulation y where back_out ends. With BACKOUT_ROW_GAP: south of a column row between the pile and the
        open yard (WA100 layout: backed out to y 4.9 between the pile and the row at 2.7, the loaded loader U-turned
        east and stood 500 steps at col 17.75 (seeds 116 / 147, static stops))."""
        L = env.layout
        yb = L.approach_y + 0.3
        if self.BACKOUT_ROW_GAP:
            rows = [cy for cy in L.column_y if yb - self.BACKOUT_ROW_GAP - 2.0 < cy < L.approach_y]
            if rows:
                yb = min(rows) - self.BACKOUT_ROW_GAP
        return yb

    def _people_clear(self, env):
        """True if nobody is within CRUISE_CLEAR of the loader."""
        pos, _ = env.worker_positions()
        if not len(pos):
            return True
        x, y, _ = env.loader_pose()
        return float(np.min(np.hypot(pos[:, 0] - x, pos[:, 1] - y))) > self.CRUISE_CLEAR

    def _row_fence(self, env, y):
        """Extra blocked boxes for this plan (HOPPER_ROW_FENCE): the column row nearest the pile, when the loaded
        loader is already south of it (seeds 101 / 116 / 147: from y -0.8 the plan went north through the row, turned
        east of the pile and came back south: 35 m, 400-700 steps)."""
        if not (self.HOPPER_ROW_FENCE and self.phase == 'go_hopper'):
            return ()
        L = env.layout
        rows = [cy for cy in L.column_y if L.approach_y - 2.5 - self.PILE_ROW_GAP - 1.0 < cy < L.approach_y]
        # 2.5 m north of the row: room for a first forward arc (1.5: boxed in between it and a pallet 3.5 m behind, no
        # plan at all, seed 116), still too low to U-turn above the row
        return [(0.0, min(rows) + 2.5, 60.0, 0.3)] if rows and y < min(rows) - 1.0 else ()

    def _clear_lane(self, env, lx):
        """With LANE_COL_CLEAR: lane x kept so the bucket passes the columns of a row between the goal and the pile."""
        L = env.layout
        if not self.LANE_COL_CLEAR or not any(cy < L.approach_y + 2.0 and cy > L.approach_y - 2.5 - self.PILE_ROW_GAP - 1.0
                                               for cy in L.column_y):
            return lx
        half = env.machine.bucket_box[2] + 0.05  # outer bucket edge (WA100: 1.17)
        need = L.column_half + half + self.LANE_COL_CLEAR
        cols = sorted(L.column_x)
        lo = max([c + need for c in cols if c < lx] or [-1e9])
        hi = min([c - need for c in cols if c > lx] or [1e9])
        return float(np.clip(lx, lo, hi)) if lo <= hi else lx

    def _escape(self, env, mag):
        """(thr, steer) that the safety check allows and that leaves the most room
        after ~1.5 s: forward / reverse x straight / left / right."""
        x, y, yaw = env.loader_pose()
        grid = env.nav_grid()
        boom = env.joint('boom')
        best = (-1e9, (-mag, 0.0))
        for d in (1.0, -1.0):
            for sc in (0.0, 1.0, -1.0):
                if self._safe(env, d * mag, sc)[1]:
                    continue
                st = (x, y, yaw, env.joint('steer'))
                for _ in range(15):
                    st = nav.step_kin(st, d * mag * env.machine.max_speed, sc * env.machine.max_steer, M=env.machine)
                room = min(grid.clearance(px, py) for px, py in nav.footprint(*st, boom, M=env.machine))
                best = max(best, (room, (d * mag, sc)), key=lambda b: b[0])
        return best[1]

    @staticmethod
    def _free_dir(env):
        """+1 or -1: which way (forward / reverse) gives more room in the next 1.5 s."""
        x, y, yaw = env.loader_pose()
        grid = env.nav_grid()
        best = (-1e9, -1.0)
        for d in (1.0, -1.0):
            st = (x, y, yaw, env.joint('steer'))
            for _ in range(15):
                st = nav.step_kin(st, 0.9 * d, 0.0, M=env.machine)
            room = min(grid.clearance(px, py) for px, py in nav.footprint(*st, env.joint('boom'), M=env.machine))
            best = max(best, (room, d))
        return best[1]

    # ---------------------------------------------------------- safety
    def _safe(self, env, thr, steer):
        """Stop if the next 1.2 s (driving or swinging the front frame) would bring
        the loader closer to a person or an obstacle than allowed. Moves that
        increase the distance are always allowed, so it never locks itself in.
        PEOPLE_AHEAD > 0: the people walk on at their current velocity for that long (the loader too, at the same
        command), compared with the loader standing still; and slows down within 0.8 m of the limit."""
        v = thr * env.machine.max_speed
        x, y, yaw = env.loader_pose()
        st = (x, y, yaw, env.joint('steer'))
        boom = env.joint('boom')
        people, _ = env.worker_positions()
        vel = env.worker_velocities() if self.PEOPLE_AHEAD else np.zeros((len(people), 2))
        things = [(px, py, nav.R_LOADER + 1.1, vx, vy, True) for (px, py), (vx, vy) in zip(people, vel)]
        if not self.STATIC_SAFE:  # (STATIC_SAFE: the obstacles are in the nav grid, checked there with the planner's
            # geometry; the same R + r + 0.1 limit here stopped every plan that passed them closely: seed 103 waited
            # 800 steps between two pallets, stop / re-plan / back off)
            things += [(ox, oy, nav.R_LOADER + r + 0.1, 0.0, 0.0, False) for ox, oy, r in env.obstacle_circles()]
        if not things:
            return thr, False
        fp0 = nav.footprint(*st, boom, M=env.machine)
        n = max(12, int(round(self.PEOPLE_AHEAD / 0.1)))
        slow = False
        for k in range(n):
            st = nav.step_kin(st, v, steer * env.machine.max_steer, M=env.machine)
            fp = nav.footprint(*st, boom, M=env.machine)
            t = (k + 1) * 0.1
            for tx, ty, lim, vx, vy, person in things:
                px, py = tx + vx * t, ty + vy * t
                d = np.min(np.hypot(fp[:, 0] - px, fp[:, 1] - py))
                d_stay = np.min(np.hypot(fp0[:, 0] - px, fp0[:, 1] - py))  # where the loader is now
                if d < lim and d < d_stay - 0.02:
                    return 0.0, True
                if self.PEOPLE_AHEAD and person and d < lim + 0.8 and d < d_stay - 0.02:  # (1.5: half speed for
                    # 800-900 of 2000 steps in seeds 100 / 101 with people walking about)
                    slow = True
        self.stats['slow'] += int(slow)
        return (0.5 * thr if slow else thr), False

    # ---------------------------------------------------------- policy
    def act(self, env):
        P = env.machine.poses  # arm poses (boom joint, bucket world pitch) of this loader
        if env._t == 0 and (self.t_phase > 0 or self.cycle > 0):
            self.reset()
        x, y, yaw = env.loader_pose()
        speed = math.hypot(*env._proprio()[4:6])
        thr, steer, arm = 0.0, 0.0, P['carry']
        p = self.phase
        if p == 'go_pile' and self.t_phase == 0 and self.path is None:
            self.lane_x = self._clear_lane(env, self._pick_lane(env))
        lane = np.array([getattr(self, 'lane_x', env.pile_center[0]), 0.0])

        if p == 'go_pile':
            gy = self._pile_goal_y(env)
            if self.PILE_ROW_GAP:  # goal pose on a pallet (fresh seed 244: rear circles over one 1.8 m from the goal,
                g0 = env.layout.approach_y - 2.5  # no plan for 1000+ steps): move it north towards the v0 goal
                sg = self._solid_grid(env)
                while gy < g0 and nav._room(sg, (lane[0], gy, math.pi / 2, 0.0), P['carry'][0]) < 0.3:
                    gy = min(gy + 0.5, g0)
            if abs(x - lane[0]) < 1.0 and min(0.5, gy - 1.0) < y < env.layout.approach_y and abs(_wrap(yaw - math.pi / 2)) < 0.4:
                self._set('align_pile')  # already lined up
            thr, steer, done = self._go(env, np.array([lane[0], gy]), math.pi / 2)
            if done:
                self._set('align_pile')
        elif p == 'align_pile':
            arm = P['carry'] if y < env.layout.approach_y - 2.0 else P['dig']
            thr, steer = self._follow_line(env, lane, math.pi / 2, 0.45)
            if y >= env.layout.approach_y:
                self._set('lower')
            elif self.t_phase > 100 or abs(_wrap(yaw - math.pi / 2)) > 1.0:  # lost the line
                self._set('go_pile')
        elif p == 'lower':
            arm = P['dig']
            thr, steer = self._follow_line(env, lane, math.pi / 2, 0.15)
            if self._arm_done(env, P['dig']) or self.t_phase > 45:
                self._set('push')
        elif p == 'push':
            arm = P['dig']
            if self.crowd_t > 0:  # lift and crowd: raise the boom and rack the bucket back a little, keep pushing
                k = min(self.crowd_t / 20.0, 1.0)
                arm = (P['dig'][0] - self.CROWD[0] * k, P['dig'][1] - self.CROWD[1] * k)
                self.crowd_t += 1
            thr, steer = self._follow_line(env, lane, math.pi / 2, 0.38)
            self.stall = self.stall + 1 if (self.t_phase > 10 and speed < 0.15) else 0
            if self.stall > 6 and self.CROWD and self.crowd_t == 0 and env.pieces_in_bucket() < self.CROWD_MIN:
                self.crowd_t, self.stall = 1, 0  # stopped by the pile with a small load: crowd instead of curling
            if (y >= env.layout.back_wall_y - env.machine.push_limit_dy - self.PUSH_STOP or self.stall > 6
                    or self.t_phase > 90 or self.crowd_t > 30):
                self._set('curl')
        elif p == 'curl':
            arm = P['curl']
            thr = 0.08
            if self.CURL_WALL_STOP and y >= env.layout.back_wall_y - env.machine.push_limit_dy - self.PUSH_STOP - 0.1:
                thr = 0.0  # at the push limit: no creeping on while curling (the bucket pressed on the back wall)
            if self._arm_done(env, P['curl']) or self.t_phase > 45:
                self._set('back_out')
        elif p == 'back_out':
            # first let loose cardboard on top fall back onto the pile, then back out gently
            bv = -0.3
            if self.BACKOUT_V and y < env.layout.approach_y - 0.5 and self._people_clear(env):
                bv = -self.BACKOUT_V  # bucket off the pile, nobody near: back out at a normal pace
            thr, steer = self._follow_line(env, lane, math.pi / 2, bv)
            if self.t_phase < 16:
                # still over the pile: rock the bucket a little so a heaped top load
                # drops back onto the pile instead of onto the road
                thr = 0.0
                rock = (self.t_phase // 4) % 2 if self.t_phase < 12 else 0
                arm = (P['curl'][0], P['curl'][1] + 0.15 * rock)
            if self.BACKOUT_TURN and self.t_phase >= 16 and y <= env.layout.approach_y + 2.5:
                # bucket off the pile: keep reversing and swing the front round towards the hopper (V pattern), so
                # the transport starts facing it (straight back, the plan U-turned forward between the pile and
                # the columns: 30+ m and 400+ steps a trip, seed 100)
                goal = np.array([self._hstop(env) + 4.0, env.layout.hopper_line_y])
                err = _wrap(math.atan2(goal[1] - y, goal[0] - x) - yaw)
                if abs(err) > 0.6 and y > env.layout.approach_y - 4.5 and self.t_phase < 140:
                    thr, steer = -0.3, -float(np.sign(err)) * 0.8  # reversing: articulation opposite to the turn
                else:
                    self._set('go_hopper')
            elif y <= self._backout_end_y(env) or self.t_phase > (200 if self.BACKOUT_ROW_GAP else 80):
                self._set('go_hopper')
        elif p == 'go_hopper':
            thr, steer, done = self._go(env, np.array([self._hstop(env) + 4.0, env.layout.hopper_line_y]), math.pi)
            if done:
                self._set('approach_hopper')
        elif p == 'approach_hopper':
            near = x < self._hstop(env) + self.RAISE_FROM
            arm = P['high'] if near else P['carry']
            v = 0.5
            if near and env.joint('boom') > -0.7:
                v = 0.0 if x < self._hstop(env) + 1.5 else 0.2
            if x < self._hstop(env) + 1.5:
                v = min(v, 0.2)
            thr, steer = self._follow_line(env, np.array([0.0, env.layout.hopper_line_y]), math.pi, v)
            if x <= self._hstop(env) + self.STOP_EARLY:
                self._set('dump')
            elif self.t_phase > 150:
                self._set('go_hopper')
        elif p == 'dump':
            # tip the bucket, then shake it until (almost) empty
            arm = P['dump']
            # tip fully and hold; only if cardboard sticks, rock the bucket slowly
            if self.t_phase > 40 or self._arm_done(env, P['dump'], tol=0.12):
                self.shook += 1
                if self.shook > self.DUMP_HOLD and env.pieces_in_bucket() > 2:
                    rock = (self.shook // self.DUMP_ROCK_PERIOD) % 2
                    arm = (P['dump'][0], P['dump'][1] - self.DUMP_ROCK * rock)
            left = env.pieces_in_bucket()
            if (self.shook > self.DUMP_LEAVE_MIN and left <= 2) or self.t_phase > 110:
                self._set('leave_hopper')
                self.cycle += 1
        elif p == 'leave_hopper':
            arm = P['high'] if x < self._hstop(env) + 2.2 else P['carry']
            thr, steer = self._follow_line(env, np.array([0.0, env.layout.hopper_line_y]), math.pi, -0.6)
            if x >= self._hstop(env) + 3.0 or self.t_phase > 80:
                self._set('go_pile')

        a_boom, a_bucket = self._arm(env, arm)
        action = np.array([thr, steer, a_boom, a_bucket], np.float64)

        # data diversity: correlated noise + random bursts
        self.ou = 0.8 * self.ou + 0.6 * self.rng.normal(0, 1, 4)
        action += self.sigma * self.ou * np.array([1.0, 1.0, 0.6, 0.6])
        if self.burst == 0 and self.rng.random() < self.p_burst:
            self.burst = int(self.rng.integers(*self.burst_len))
            self.burst_action = np.array([
                self.rng.uniform(-0.6, 0.6), self.rng.uniform(-1, 1),
                self.rng.uniform(-1, 1), self.rng.uniform(-1, 1),
            ])
        if self.burst > 0:
            self.burst -= 1
            action = self.burst_action + 0.1 * self.rng.normal(0, 1, 4)
            if self.burst == 0 and self.phase in ('go_pile', 'go_hopper'):
                self.path, self.turn = None, None  # re-plan from where the burst left us
        action = np.clip(action, -1, 1)
        self.arm_prev = action[2:4].copy()  # keep ease-in honest after noise / bursts
        if (self.SLEW_ALWAYS or self.phase in self.LOADED) and self.burst == 0:
            # loaded bucket: gentle on gas / brake and on swinging the articulation
            ts = self.THR_SLEW if (self.phase in self.LOADED or self.THR_SLEW_EMPTY is None) else self.THR_SLEW_EMPTY
            action[0] = np.clip(action[0], self.drive_prev[0] - ts, self.drive_prev[0] + ts)
            action[1] = np.clip(action[1], self.drive_prev[1] - self.STEER_SLEW, self.drive_prev[1] + self.STEER_SLEW)

        # safety last: never drive into people or obstacles
        action[0], stopped = self._safe(env, action[0], action[1])
        self.static_recent = max(0, self.static_recent - 1)
        if self.STATIC_SAFE and self.phase in self.STATIC_SAFE_PHASES and action[0] != 0.0:
            # (also while backing off: a blind back-off put the bucket side on a pallet, seed 217 of the fresh seeds)
            if self._static_block(env, action[0], action[1]):  # the bucket would reach a column / wall / the hopper
                self.stats['static_stop'] += 1
                backing = self.backoff > 0
                self.backoff = 0  # (cancelled: plan again from here)
                self.static_n += 1
                self.static_recent = 100  # the next plans keep 0.4 m (0.1: they ran along the same obstacle again, seed 103)
                way = 1.0 if action[0] > 0 else -1.0
                action[0] = 0.0
                action[1] = env.joint('steer') / env.machine.max_steer
                self.path, self.turn = None, None  # plan again from here (strict planner: escapes only outwards)
                if self.phase == 'align_pile':  # off the straight line through the column row: plan the approach again
                    self.static_n = 0
                    self._set('go_pile')
                elif self.phase == 'back_out' and self.BACKOUT_ROW_GAP and y <= env.layout.approach_y + 0.3:
                    self.static_n = 0  # off the pile and something behind (seed 101: a pallet; waited 130 steps for
                    self._set('go_hopper')  # the back-out timeout): plan the transport from here
                elif self.static_n > 8 and not backing:  # the fresh plans start the same way: make room straight back first
                    self.static_n = 0  # (seed 103 of the fresh seeds: 1100 stop / re-plan cycles in one place)
                    g = env.joint('steer') / env.machine.max_steer
                    # a move the static check itself allows (straight back first): a blocked back-off was cancelled
                    # and the stop / re-plan loop went on for good (fresh seed 231: 2000 steps, 4.7 m from a pallet)
                    for t_, s_ in ((-0.25 * way, g), (0.25 * way, g), (-0.25 * way, 0.0), (-0.25 * way, 1.0), (-0.25 * way, -1.0)):
                        if not self._static_block(env, t_, s_):
                            self.backoff, self.backoff_thr, self.backoff_steer = 10, t_, s_
                            break
            elif abs(action[0]) > 0.05:
                self.static_n = 0
        if stopped:  # also hold the articulation so the bucket does not swing
            action[1] = env.joint('steer') / env.machine.max_steer
        if stopped:
            self.blocked += 1
            self.stats['people_stop'] += 1
        elif not self.BLOCK_KEEP or speed > 0.3:  # V1: only real driving ends a block (a fresh plan creeps, stops)
            self.blocked = 0
        if self.phase in ('go_pile', 'go_hopper'):
            if self.blocked == 30:  # re-plan: maybe there is a way around
                env._grid = None
                self.path, self.turn = None, None
                if self.BLOCK_KEEP:  # once: the count stays while creeping (BLOCK_KEEP), and the fresh plan was
                    self.blocked = 31  # thrown away again every step (seed 43 of the people test stood 900 steps)
            elif self.blocked > 50:  # still blocked: make room
                self.path, self.turn = None, None
                self.blocked = 0
                self.backoff, (self.backoff_thr, self.backoff_steer) = 15, self._escape(env, 0.25)
                if self.ESCAPE_REVERSE:  # back away, articulation held (the safety stop allows moves that open the
                    # distance; straightening it swung the bucket towards the person and was stopped as well)
                    self.backoff, self.backoff_thr = 20, -0.3
                    self.backoff_steer = env.joint('steer') / env.machine.max_steer
        self.drive_prev = action[:2].copy()
        if not stopped:
            self.t_phase += 1
        return action.astype(np.float32)


class LoaderExpertPolicy(BasePolicy):
    """Collection policy for swm/CardboardYard-v0 (works with swm.World).

    Args:
        p_weak: share of episodes recorded in 'weak' mode (noisy, many
            random bursts). 0.0 gives clean expert data only.
        noise_scale: multiplies every noise level (0 = no noise, no bursts).
    """

    controller_cls = _Controller  # per-env driver; CardboardYardReal's expert v1 uses a tuned subclass

    def __init__(self, p_weak: float = 0.3, noise_scale: float = 1.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.p_weak = p_weak
        self.noise_scale = noise_scale
        self.discrete = False
        self.set_seed(seed)
        self._ctrls: dict[int, _Controller] = {}

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        self.env = env

    def _envs(self):
        if hasattr(self.env, 'envs'):
            return [e.unwrapped for e in self.env.envs]
        return [self.env.unwrapped]

    def act_single(self, env, idx: int = 0):
        """Action for one unwrapped CardboardYardEnv (handy outside swm.World)."""
        if idx not in self._ctrls:
            self._ctrls[idx] = self.controller_cls(self.rng, self.p_weak, self.noise_scale)
        a = self._ctrls[idx].act(env)
        env._expert_weak = bool(self._ctrls[idx].weak)  # saved per episode by env.get_episode_data
        return a

    def get_action(self, info_dict, **kwargs):
        return np.stack([self.act_single(e, i) for i, e in enumerate(self._envs())])
